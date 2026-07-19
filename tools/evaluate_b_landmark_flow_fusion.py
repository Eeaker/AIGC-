"""Diagnostic: fuse reviewed A0006->A0009 landmarks into the raster flow.

The supplied A0007/A0008 finished frames are used only after rendering, to
compare candidates.  This script is deliberately an experiment, not part of
the formal inference path.
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
from src.stroke_regularization import regularize_flow
from src.task_b import LINE_COLORS, _flow, _forward_splat


def _landmarks(annotation_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads((annotation_dir / "manual_landmarks.json").read_text(encoding="utf-8"))
    rows = payload["A0006_to_A0009"]
    source = np.asarray([row["source_xy"] for row in rows], np.float32)
    target = np.asarray([row["target_xy"] for row in rows], np.float32)
    return source, target


def _correct_line_samples(flow: np.ndarray, image: np.ndarray, source: np.ndarray,
                          target: np.ndarray, sigma: float, alpha: float,
                          cap: float) -> np.ndarray:
    """Apply a capped Gaussian residual field only where endpoint ink exists."""
    h, w = flow.shape[:2]
    tx = np.clip(np.rint(target[:, 0]).astype(int), 0, w - 1)
    ty = np.clip(np.rint(target[:, 1]).astype(int), 0, h - 1)
    desired = source - target
    residual = desired - flow[ty, tx]
    length = np.linalg.norm(residual, axis=1, keepdims=True)
    residual *= np.minimum(1.0, cap / np.maximum(length, 1e-6))

    yy, xx = np.nonzero(line_mask(image))
    points = np.column_stack([xx, yy]).astype(np.float32)
    d2 = ((points[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
    weights = np.exp(-d2 / (2.0 * sigma * sigma))
    weight_sum = weights.sum(axis=1, keepdims=True)
    correction = (weights @ residual) / np.maximum(weight_sum, 1e-6)
    confidence = np.clip(weight_sum, 0.0, 1.0)
    out = flow.copy()
    out[yy, xx] += alpha * confidence * correction
    return out


def _render(endpoint: np.ndarray, flow: np.ndarray, amount: float) -> np.ndarray:
    out = np.full_like(endpoint[:, :, :3], 255)
    for color in LINE_COLORS:
        mask = np.all(endpoint[:, :, :3] == color, axis=2)
        out[_forward_splat(mask, flow, amount)] = color
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sigmas", type=float, nargs="+", default=(30.0, 60.0, 120.0))
    ap.add_argument("--alphas", type=float, nargs="+", default=(0.1, 0.25, 0.5))
    ap.add_argument("--caps", type=float, nargs="+", default=(4.0, 8.0, 16.0))
    args = ap.parse_args()

    shot = args.data_root / "KTK_04_246B"
    line_dir = shot / "源文件" / "中割"
    ref_dir = shot / "成品" / "中割"
    a6, a9 = imread(line_dir / "A0006.tga"), imread(line_dir / "A0009.tga")
    source, target = _landmarks(args.annotations)
    f10 = regularize_flow(a9, _flow(a9, a6), _flow(a6, a9), LINE_COLORS)
    f10 = np.dstack([cv2.GaussianBlur(f10[:, :, c], (0, 0), 3) for c in (0, 1)])
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    candidates = [("baseline", f10)]
    for sigma in args.sigmas:
        for alpha in args.alphas:
            for cap in args.caps:
                name = f"s{sigma:g}_a{alpha:g}_c{cap:g}"
                candidates.append((name, _correct_line_samples(
                    f10, a9, source, target, sigma, alpha, cap)))

    for name, field in candidates:
        frame_scores = []
        for index, t in ((7, 0.50), (8, 0.75)):
            pred = _render(a9, field, 1.0 - t)
            ref = imread(ref_dir / f"A{index:04d}.tga")
            score = union_tolerance_f1(pred, ref)
            score["line_ratio"] = float(line_mask(pred).sum() / line_mask(ref).sum())
            frame_scores.append(score)
        rows.append({
            "name": name,
            "mean_f1_exact": float(np.mean([x["union_f1_exact"] for x in frame_scores])),
            "mean_f1_1px": float(np.mean([x["union_f1_1px"] for x in frame_scores])),
            "mean_f1_2px": float(np.mean([x["union_f1_2px"] for x in frame_scores])),
            "mean_line_ratio": float(np.mean([x["line_ratio"] for x in frame_scores])),
            "frames": frame_scores,
        })
    rows.sort(key=lambda x: x["mean_f1_2px"], reverse=True)
    best_name = rows[0]["name"]
    best_field = dict(candidates)[best_name]
    for index, t in ((7, 0.50), (8, 0.75)):
        imwrite(args.output / f"A{index:04d}_{best_name}.tga", _render(a9, best_field, 1.0 - t))
    payload = {"best": rows[0], "candidates": rows}
    (args.output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
