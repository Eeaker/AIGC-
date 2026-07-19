"""Part-level landmark/visibility assisted ablation for Task B.

Construction reads A0006/A0009, reviewed landmarks and edge visibility only.
A0007/A0008 finished frames are opened after all candidates are rendered and
serve only as held-out evaluation.  This is an assisted experiment, not the
automatic formal branch.
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


PARTS = ("hood", "face_hair", "lower_body")


def part_of(name: str) -> str:
    if name.startswith(("hood_", "center_seam", "opening_")):
        return "hood"
    if name.startswith(("eye_", "nose_", "mouth_", "chin", "jaw_", "fringe_")):
        return "face_hair"
    return "lower_body"


def load_landmarks(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["A0006_to_A0009"]
    for row in rows:
        row["part"] = part_of(row["name"])
    return rows


def affine_with_residual(points: np.ndarray, source: np.ndarray, target: np.ndarray,
                         source_snap: np.ndarray, target_snap: np.ndarray,
                         parts: np.ndarray, direction: str, snap_limit: float,
                         residual_alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Robust per-part affine plus capped inverse-distance residual."""
    endpoint = source if direction == "forward" else target
    other = target if direction == "forward" else source
    snap = np.maximum(source_snap, target_snap)
    nearest_all = np.argmin(((points[:, None, :] - endpoint[None, :, :]) ** 2).sum(2), axis=1)
    point_parts = parts[nearest_all]
    displacement = np.zeros_like(points, np.float32)

    for part in PARTS:
        point_ids = np.flatnonzero(point_parts == part)
        landmark_ids = np.flatnonzero((parts == part) & (snap <= snap_limit))
        if len(landmark_ids) < 3:
            landmark_ids = np.flatnonzero(parts == part)
        p0 = np.ascontiguousarray(endpoint[landmark_ids], dtype=np.float32)
        p1 = np.ascontiguousarray(other[landmark_ids], dtype=np.float32)
        matrix, inliers = cv2.estimateAffinePartial2D(
            p0.reshape(-1, 1, 2), p1.reshape(-1, 1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=24.0,
            maxIters=3000, confidence=.995, refineIters=20)
        if matrix is None:
            matrix = cv2.getAffineTransform(p0[:3].astype(np.float32), p1[:3].astype(np.float32))
            inlier_mask = np.ones(len(p0), bool)
        else:
            inlier_mask = inliers.ravel().astype(bool)
        q = points[point_ids]
        base_mapped = q @ matrix[:, :2].T + matrix[:, 2]
        base_disp = base_mapped - q

        anchor_base = p0 @ matrix[:, :2].T + matrix[:, 2]
        residual = p1 - anchor_base
        residual_len = np.linalg.norm(residual, axis=1, keepdims=True)
        residual *= np.minimum(1.0, 40.0 / np.maximum(residual_len, 1e-6))
        reliability = np.exp(-snap[landmark_ids] / 25.0) * (.25 + .75 * inlier_mask)
        d2 = ((q[:, None, :] - p0[None, :, :]) ** 2).sum(2)
        order = np.argsort(d2, axis=1)[:, :min(4, len(p0))]
        chosen_d2 = np.take_along_axis(d2, order, axis=1)
        chosen_rel = reliability[order]
        weights = chosen_rel / np.maximum(chosen_d2, 64.0)
        chosen_residual = residual[order]
        local = (weights[:, :, None] * chosen_residual).sum(1) / np.maximum(
            weights.sum(1, keepdims=True), 1e-6)
        displacement[point_ids] = base_disp + residual_alpha * local
    return displacement, point_parts


def fields(image: np.ndarray, rows: list[dict], direction: str,
           snap_limit: float, residual_alpha: float) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray([r["source_xy"] for r in rows], np.float32)
    target = np.asarray([r["target_xy"] for r in rows], np.float32)
    source_snap = np.asarray([r["source_snap_distance_px"] for r in rows], np.float32)
    target_snap = np.asarray([r["target_snap_distance_px"] for r in rows], np.float32)
    parts = np.asarray([r["part"] for r in rows])
    yy, xx = np.nonzero(line_mask(image))
    points = np.column_stack([xx, yy]).astype(np.float32)
    displacement, point_parts = affine_with_residual(
        points, source, target, source_snap, target_snap, parts, direction,
        snap_limit, residual_alpha)
    flow = np.zeros((*image.shape[:2], 2), np.float32)
    labels = np.full(image.shape[:2], -1, np.int8)
    flow[yy, xx] = displacement
    labels[yy, xx] = np.asarray([PARTS.index(value) for value in point_parts], np.int8)
    return flow, labels


def render_parts(image: np.ndarray, flow: np.ndarray, labels: np.ndarray,
                 amount: float) -> list[np.ndarray]:
    outputs = [np.full_like(image[:, :, :3], 255) for _ in PARTS]
    for part_id, output in enumerate(outputs):
        for color in LINE_COLORS:
            mask = np.all(image[:, :, :3] == color, axis=2) & (labels == part_id)
            output[_forward_splat(mask, flow, amount)] = color
    return outputs


def compose(source_parts: list[np.ndarray], target_parts: list[np.ndarray], pattern: str) -> np.ndarray:
    if pattern == "source_all": owners = (0, 0, 0)
    elif pattern == "target_all": owners = (1, 1, 1)
    elif pattern == "target_head_source_body": owners = (1, 1, 0)
    elif pattern == "target_hood_source_face_target_body": owners = (1, 0, 1)
    else: raise ValueError(pattern)
    out = np.full_like(source_parts[0], 255)
    for part_id, owner in enumerate(owners):
        layer = target_parts[part_id] if owner else source_parts[part_id]
        ink = np.any(layer != 255, axis=2)
        out[ink] = layer[ink]
    return out


def topology(mask: np.ndarray) -> dict[str, int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    return {"components": int(count - 1), "components_le_5px": int((sizes <= 5).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    shot = args.data_root / "KTK_04_246B"
    line_dir, ref_dir = shot / "源文件" / "中割", shot / "成品" / "中割"
    a6, a9 = imread(line_dir / "A0006.tga"), imread(line_dir / "A0009.tga")
    rows = load_landmarks(args.annotations / "manual_landmarks.json")
    args.output.mkdir(parents=True, exist_ok=True)
    patterns = ("source_all", "target_all", "target_head_source_body",
                "target_hood_source_face_target_body")
    results = []
    rendered: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for snap_limit in (15.0, 30.0, 80.0):
        for residual_alpha in (0.0, 0.35):
            sf, sl = fields(a6, rows, "forward", snap_limit, residual_alpha)
            tf, tl = fields(a9, rows, "backward", snap_limit, residual_alpha)
            frame_layers = {}
            for index, time in ((7, .50), (8, .75)):
                frame_layers[index] = (render_parts(a6, sf, sl, time),
                                       render_parts(a9, tf, tl, 1.0 - time))
            for pattern in patterns:
                name = f"snap{snap_limit:g}_res{residual_alpha:g}_{pattern}"
                frames, scores = [], []
                for index in (7, 8):
                    pred = compose(*frame_layers[index], pattern)
                    ref = imread(ref_dir / f"A{index:04d}.tga")
                    score = union_tolerance_f1(pred, ref)
                    score["line_ratio"] = float(line_mask(pred).sum() / line_mask(ref).sum())
                    score["topology"] = topology(line_mask(pred))
                    frames.append(pred); scores.append(score)
                rendered[name] = (frames[0], frames[1])
                results.append({
                    "name": name,
                    "mean_f1_exact": float(np.mean([s["union_f1_exact"] for s in scores])),
                    "mean_f1_1px": float(np.mean([s["union_f1_1px"] for s in scores])),
                    "mean_f1_2px": float(np.mean([s["union_f1_2px"] for s in scores])),
                    "mean_line_ratio": float(np.mean([s["line_ratio"] for s in scores])),
                    "frames": {f"A{i:04d}": s for i, s in zip((7, 8), scores)},
                })
    results.sort(key=lambda x: x["mean_f1_2px"], reverse=True)
    best = results[0]
    for index, image in zip((7, 8), rendered[best["name"]]):
        imwrite(args.output / f"A{index:04d}_{best['name']}.tga", image)
    payload = {
        "information_boundary": "A0006/A0009 + reviewed landmarks/visibility; A0007/A0008 references evaluation only",
        "visibility_policy": "target topology retained for occluded face/hair candidates; four frozen owner patterns evaluated",
        "best": best, "candidates": results,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
