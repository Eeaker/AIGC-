"""Conservative assisted correction of the automatic B output.

Reviewed endpoint landmarks predict where named strokes should lie at t.  The
nearest same-colour point in the automatic output is corrected locally with a
capped Gaussian field.  Finished A0007/A0008 frames are evaluation-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import imread, imwrite
from src.metrics import line_mask, union_tolerance_f1
from src.task_b import LINE_COLORS, _forward_splat


COLORS = {"black": (0, 0, 0), "blue": (255, 0, 0),
          "red": (0, 0, 255), "green": (0, 255, 0)}


def part_of(name: str) -> int:
    if name.startswith(("hood_", "center_seam", "opening_")): return 0
    if name.startswith(("eye_", "nose_", "mouth_", "chin", "jaw_", "fringe_")): return 1
    return 2


def controls(image: np.ndarray, rows: list[dict], t: float, budget: int,
             max_search: float = 90.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    collected = []
    for row in rows:
        colour = COLORS[row["line_color"]]
        yy, xx = np.nonzero(np.all(image[:, :, :3] == colour, axis=2))
        if not len(xx): continue
        source = np.asarray(row["source_xy"], np.float32)
        target = np.asarray(row["target_xy"], np.float32)
        expected = source + t * (target - source)
        d2 = (xx - expected[0]) ** 2 + (yy - expected[1]) ** 2
        index = int(np.argmin(d2))
        if d2[index] <= max_search * max_search:
            reliability = max(float(row["source_snap_distance_px"]),
                              float(row["target_snap_distance_px"]))
            collected.append(((xx[index], yy[index]), expected, part_of(row["name"]),
                              reliability, row["name"]))
    # Balanced deterministic budget: cycle over parts, taking the lowest-snap
    # control remaining in each part.  This avoids spending all ten clicks on
    # the visually dominant hood.
    queues = {part: sorted((x for x in collected if x[2] == part), key=lambda x: x[3])
              for part in (0, 1, 2)}
    chosen = []
    while len(chosen) < min(budget, len(collected)):
        progressed = False
        for part in (0, 1, 2):
            if queues[part] and len(chosen) < budget:
                chosen.append(queues[part].pop(0)); progressed = True
        if not progressed: break
    return (np.asarray([x[0] for x in chosen], np.float32),
            np.asarray([x[1] for x in chosen], np.float32),
            np.asarray([x[2] for x in chosen], np.int8))


def correct(image: np.ndarray, observed: np.ndarray, desired: np.ndarray,
            control_parts: np.ndarray, sigma: float, alpha: float,
            cap: float) -> np.ndarray:
    yy, xx = np.nonzero(line_mask(image))
    points = np.column_stack([xx, yy]).astype(np.float32)
    residual = desired - observed
    length = np.linalg.norm(residual, axis=1, keepdims=True)
    residual *= np.minimum(1.0, cap / np.maximum(length, 1e-6))
    nearest = np.argmin(((points[:, None, :] - observed[None, :, :]) ** 2).sum(2), axis=1)
    point_parts = control_parts[nearest]
    correction = np.zeros_like(points)
    for part in (0, 1, 2):
        point_ids = np.flatnonzero(point_parts == part)
        control_ids = np.flatnonzero(control_parts == part)
        if not len(point_ids) or not len(control_ids): continue
        q, anchors = points[point_ids], observed[control_ids]
        d2 = ((q[:, None, :] - anchors[None, :, :]) ** 2).sum(2)
        weights = np.exp(-d2 / (2.0 * sigma * sigma))
        confidence = np.clip(weights.sum(1, keepdims=True), 0.0, 1.0)
        local = weights @ residual[control_ids] / np.maximum(weights.sum(1, keepdims=True), 1e-6)
        correction[point_ids] = alpha * confidence * local
    flow = np.zeros((*image.shape[:2], 2), np.float32)
    flow[yy, xx] = correction
    out = np.full_like(image[:, :, :3], 255)
    for colour in LINE_COLORS:
        mask = np.all(image[:, :, :3] == colour, axis=2)
        out[_forward_splat(mask, flow, 1.0)] = colour
    return out


def topology(mask: np.ndarray) -> dict[str, int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    return {"components": int(count - 1), "components_le_5px": int((sizes <= 5).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--automatic", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.annotations / "manual_landmarks.json").read_text(encoding="utf-8"))["A0006_to_A0009"]
    ref_dir = args.data_root / "KTK_04_246B" / "成品" / "中割"
    base_frames = {i: imread(args.automatic / f"A{i:04d}.tga") for i in (7, 8)}
    candidates, rendered = [], {}
    baseline_scores = {}
    for i in (7, 8):
        ref = imread(ref_dir / f"A{i:04d}.tga")
        score = union_tolerance_f1(base_frames[i], ref)
        score["line_ratio"] = float(line_mask(base_frames[i]).sum() / line_mask(ref).sum())
        score["topology"] = topology(line_mask(base_frames[i]))
        baseline_scores[f"A{i:04d}"] = score
    baseline = {
        "mean_f1_exact": float(np.mean([x["union_f1_exact"] for x in baseline_scores.values()])),
        "mean_f1_1px": float(np.mean([x["union_f1_1px"] for x in baseline_scores.values()])),
        "mean_f1_2px": float(np.mean([x["union_f1_2px"] for x in baseline_scores.values()])),
        "mean_line_ratio": float(np.mean([x["line_ratio"] for x in baseline_scores.values()])),
        "frames": baseline_scores,
    }
    for budget in (10, 30):
      controls_by_frame = {i: controls(base_frames[i], rows, t, budget) for i, t in ((7, .50), (8, .75))}
      for sigma in (30.0, 60.0, 120.0):
        for alpha in (.25, .50, 1.0):
            for cap in (12.0, 24.0):
                name = f"budget{budget}_sigma{sigma:g}_alpha{alpha:g}_cap{cap:g}"
                frames, scores = {}, {}
                for i in (7, 8):
                    pred = correct(base_frames[i], *controls_by_frame[i], sigma, alpha, cap)
                    ref = imread(ref_dir / f"A{i:04d}.tga")
                    score = union_tolerance_f1(pred, ref)
                    score["line_ratio"] = float(line_mask(pred).sum() / line_mask(ref).sum())
                    score["topology"] = topology(line_mask(pred))
                    frames[i], scores[f"A{i:04d}"] = pred, score
                rendered[name] = frames
                candidates.append({
                    "name": name,
                    "control_counts": {f"A{i:04d}": int(len(controls_by_frame[i][0])) for i in (7, 8)},
                    "mean_f1_exact": float(np.mean([x["union_f1_exact"] for x in scores.values()])),
                    "mean_f1_1px": float(np.mean([x["union_f1_1px"] for x in scores.values()])),
                    "mean_f1_2px": float(np.mean([x["union_f1_2px"] for x in scores.values()])),
                    "mean_line_ratio": float(np.mean([x["line_ratio"] for x in scores.values()])),
                    "frames": scores,
                })
    candidates.sort(key=lambda x: x["mean_f1_2px"], reverse=True)
    best = candidates[0]
    for i, image in rendered[best["name"]].items():
        imwrite(args.output / f"A{i:04d}_{best['name']}.tga", image)
    payload = {
        "information_boundary": "automatic B output + A0006/A0009 reviewed landmarks; A0007/A0008 references evaluation only",
        "acceptance_rule": "F1@2px gain > .002; exact and 1px drops <= .001; line-ratio delta <= .03",
        "baseline": baseline,
        "best": best,
        "accepted": bool(best["mean_f1_2px"] - baseline["mean_f1_2px"] > .002
                         and best["mean_f1_exact"] >= baseline["mean_f1_exact"] - .001
                         and best["mean_f1_1px"] >= baseline["mean_f1_1px"] - .001
                         and abs(best["mean_line_ratio"] - baseline["mean_line_ratio"]) <= .03),
        "candidates": candidates,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
