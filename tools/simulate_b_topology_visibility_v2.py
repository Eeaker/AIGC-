"""Evaluate the reviewed topology/visibility v2 annotations on Task B.

Construction reads only A0006/A0009 and the annotation package.  Held-out
A0007/A0008 finished frames are opened after rendering for evaluation only.
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
from src.metrics import chamfer_and_f1, color_compliance, color_line_f1, union_tolerance_f1
from src.task_b import LINE_COLORS


def _controls(annotation_dir: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads((annotation_dir / "manual_landmarks.json").read_text(encoding="utf-8"))
    rows = payload["A0006_to_A0009"]
    source = np.asarray([row["source_xy"] for row in rows], np.float32)
    target = np.asarray([row["target_xy"] for row in rows], np.float32)
    h, w = shape
    boundary = np.asarray([
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w / 2, 0), (w / 2, h - 1), (0, h / 2), (w - 1, h / 2),
    ], np.float32)
    return np.vstack([source, boundary]), np.vstack([target, boundary])


def _warp(image: np.ndarray, source: np.ndarray, destination: np.ndarray,
          regularization: float) -> np.ndarray:
    matches = [cv2.DMatch(i, i, 0) for i in range(len(source))]
    transformer = cv2.createThinPlateSplineShapeTransformer(
        regularizationParameter=regularization)
    transformer.estimateTransformation(destination.reshape(1, -1, 2),
                                       source.reshape(1, -1, 2), matches)
    out = np.full_like(image[:, :, :3], 255)
    for color in LINE_COLORS:
        mask = np.all(image[:, :, :3] == color, axis=2).astype(np.uint8) * 255
        warped = transformer.warpImage(mask, flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[warped > 0] = color
    return out


def _cleanup(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    ink = np.any(out != 255, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] <= 2:
            out[labels == label] = 255
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=.01)
    args = parser.parse_args()
    shot = args.data_root / "KTK_04_246B"
    key_dir = shot / "源文件" / "中割"
    ref_dir = shot / "成品" / "中割"
    first, last = imread(key_dir / "A0006.tga"), imread(key_dir / "A0009.tga")
    p0, p1 = _controls(args.annotations, first.shape[:2])
    results = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for index, time in ((7, .50), (8, .75)):
        mid = p0 + time * (p1 - p0)
        source_owner = _cleanup(_warp(first, p0, mid, args.regularization))
        target_owner = _cleanup(_warp(last, p1, mid, args.regularization))
        ref = imread(ref_dir / f"A{index:04d}.tga")
        for owner, image in (("source", source_owner), ("target", target_owner)):
            name = f"A{index:04d}_{owner}.tga"
            imwrite(args.output / name, image)
            results[name] = {
                "five_color_compliance": color_compliance(image, include_green=True),
                **chamfer_and_f1(image, ref),
                **union_tolerance_f1(image, ref),
                **color_line_f1(image, ref, include_green=True),
            }
    (args.output / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
