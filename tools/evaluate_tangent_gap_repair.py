"""Leakage-controlled evaluation of conservative topology repairs.

Repair parameters for each frame are selected only from the other two LOFO
frames.  Tangent-aware bridges add straight 3--5 px connectors only where both
endpoint directions agree; existing curve pixels are never moved or smoothed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.task_a import _repair_short_gaps, _thin
from tools.evaluate_dsv_detail_experiment import label
from tools.evaluate_residual_postprocess import metrics
from tools.fuse_geometry_color_lofo import (BOXES, FRAMES, assign, fixed_geometry,
                                            probabilities)
from tools.train_topology_residual import DATA, CURRENT, colorize, masks, read_rgb


NEIGHBOURS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if dy or dx)


def prune_tiny(line: np.ndarray, max_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), 8)
    keep = np.zeros_like(line, bool)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] > max_area:
            keep |= labels == index
    return keep


def endpoint_tangent(skeleton: np.ndarray, y: int, x: int,
                     walk: int = 4) -> np.ndarray | None:
    previous = None
    current = (y, x)
    for _ in range(walk):
        choices = []
        for dy, dx in NEIGHBOURS:
            point = (current[0] + dy, current[1] + dx)
            if (0 <= point[0] < skeleton.shape[0] and 0 <= point[1] < skeleton.shape[1]
                    and skeleton[point] and point != previous):
                choices.append(point)
        if not choices:
            break
        # Stop at branches; an ambiguous tangent is unsafe to bridge.
        if len(choices) > 1:
            return None
        previous, current = current, choices[0]
    vector = np.asarray((y - current[0], x - current[1]), np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm >= 1.5 else None


def tangent_bridge(line: np.ndarray, max_gap: float, max_angle: float) -> np.ndarray:
    skeleton = _thin(line)
    degree = cv2.filter2D(skeleton.astype(np.uint8), -1,
                          np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
    endpoints = np.argwhere(skeleton & (degree == 2))  # self + one neighbour
    tangents = [endpoint_tangent(skeleton, int(y), int(x)) for y, x in endpoints]
    cosine = math.cos(math.radians(max_angle))
    candidates = []
    for i, (p, tp) in enumerate(zip(endpoints, tangents)):
        if tp is None:
            continue
        for j in range(i + 1, len(endpoints)):
            tq = tangents[j]
            if tq is None:
                continue
            delta = endpoints[j].astype(np.float32) - p.astype(np.float32)
            distance = float(np.linalg.norm(delta))
            if not (1.5 < distance <= max_gap):
                continue
            direction = delta / distance
            alignment = min(float(np.dot(tp, direction)), float(np.dot(tq, -direction)))
            if alignment >= cosine:
                candidates.append((distance, -alignment, i, j))
    result = line.astype(np.uint8).copy()
    used = set()
    for _, _, i, j in sorted(candidates):
        if i in used or j in used:
            continue
        y1, x1 = (int(value) for value in endpoints[i])
        y2, x2 = (int(value) for value in endpoints[j])
        cv2.line(result, (x1, y1), (x2, y2), 1, 1, cv2.LINE_8)
        used.update((i, j))
    return result > 0


def objective(item: dict) -> float:
    return (item["exact_f1"] + 0.25 * item["f1_2px"]
            - 0.002 * item["local_tangent_mean_deg"]
            - 0.02 * abs(item["line_pixel_ratio"] - 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--backbone-root", type=Path, required=True)
    parser.add_argument("--colorreset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, dict[str, dict]] = {}
    for frame in FRAMES:
        geometry_probability = probabilities(args.geometry_root, frame)
        colour_probability = (0.60 * probabilities(args.backbone_root, frame)
                              + 0.40 * probabilities(args.colorreset_root, frame))
        base = fixed_geometry(geometry_probability)
        lines = {"base": base,
                 "prune1": prune_tiny(base, 1),
                 "prune2": prune_tiny(base, 2),
                 "repair5": _repair_short_gaps(base)}
        for gap in (3.0, 4.0, 5.0):
            for angle in (25.0, 35.0):
                key = f"bridge_g{int(gap)}_a{int(angle)}"
                lines[key] = tangent_bridge(base, gap, angle)
                lines[f"prune1_{key}"] = tangent_bridge(prune_tiny(base, 1), gap, angle)
        target = masks(read_rgb(DATA / "成品" / "描原" / f"{frame}.tga"))
        candidates[frame] = {name: assign(line, colour_probability.argmax(0))
                             for name, line in lines.items()}
        results[frame] = {name: metrics(value, target)
                          for name, value in candidates[frame].items()}

    selections = {}
    for heldout in FRAMES:
        train_frames = tuple(frame for frame in FRAMES if frame != heldout)
        scores = {name: float(np.mean([objective(results[frame][name])
                                      for frame in train_frames]))
                  for name in results[heldout]}
        chosen = max(scores, key=scores.get)
        selections[heldout] = {"chosen_on_other_frames": chosen,
                               "calibration_frames": train_frames,
                               "calibration_score": scores[chosen],
                               "heldout_metrics": results[heldout][chosen],
                               "base_metrics": results[heldout]["base"]}
        candidate = candidates[heldout][chosen]
        Image.fromarray(colorize(candidate)).save(args.output / f"{heldout}.tga")
        current = masks(read_rgb(CURRENT / f"{heldout}.tga"))
        target = masks(read_rgb(DATA / "成品" / "描原" / f"{heldout}.tga"))
        x, y, w, h = BOXES[heldout]
        views = [current, candidates[heldout]["base"], candidate, target]
        names = ["CURRENT", "FUSION BASE", f"REPAIR {chosen}", "REFERENCE"]
        crops = [label(cv2.cvtColor(colorize(value), cv2.COLOR_RGB2BGR)[y:y+h, x:x+w], name)
                 for value, name in zip(views, names)]
        panel = cv2.resize(np.hstack(crops), None, fx=2, fy=2,
                           interpolation=cv2.INTER_NEAREST)
        cv2.imencode(".png", panel)[1].tofile(args.output / f"{heldout}_detail_2x.png")

    report = {"selection_rule": "maximize fixed geometry objective on the other two frames",
              "objective": "exact_f1 + .25*f1_2px - .002*tangent - .02*abs(pixel_ratio-1)",
              "selections": selections, "all_metrics": results}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for frame, item in selections.items():
        before, after = item["base_metrics"], item["heldout_metrics"]
        print(frame, item["chosen_on_other_frames"],
              f"exact {before['exact_f1']:.6f}->{after['exact_f1']:.6f}",
              f"F1 {before['f1_2px']:.6f}->{after['f1_2px']:.6f}",
              f"tangent {before['local_tangent_mean_deg']:.4f}->{after['local_tangent_mean_deg']:.4f}",
              f"components {before['pred_topology']['components']}->{after['pred_topology']['components']}")


if __name__ == "__main__":
    main()
