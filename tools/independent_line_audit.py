"""Independent A/B union and per-colour line audit.

This intentionally does not import ``src.metrics`` or ``src.io_utils``.  It is
a second implementation for catching shared-code mistakes in the main runner
and validator, not another wrapper around the saved metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


COLORS = {
    "black": (0, 0, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
}


def read(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return rgb[:, :, ::-1].copy()


def distance_to(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def tolerant_f1(predicted: np.ndarray, reference: np.ndarray, tolerance: float) -> float:
    pred_distance = distance_to(predicted)
    ref_distance = distance_to(reference)
    precision = float((ref_distance[predicted] <= tolerance).mean()) if predicted.any() else 0.0
    recall = float((pred_distance[reference] <= tolerance).mean()) if reference.any() else 0.0
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def audit(prediction: Path, reference: Path) -> dict[str, float]:
    pred, ref = read(prediction), read(reference)
    masks = {"union": (np.any(pred != 255, axis=2), np.any(ref != 255, axis=2))}
    masks.update({name: (np.all(pred == color, axis=2), np.all(ref == color, axis=2))
                  for name, color in COLORS.items()})
    result = {}
    for name, (p, r) in masks.items():
        for tolerance, suffix in ((0.0, "exact"), (1.0, "1px"), (2.0, "2px")):
            result[f"{name}_f1_{suffix}"] = tolerant_f1(p, r, tolerance)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/line_semantic_audit.json"))
    args = parser.parse_args()
    shot = args.data_root / "KTK_04_246B"
    rows = {"A": {}, "A_strict_four_color": {}, "B": {}}
    for index in (1, 6, 9):
        name = f"A{index:04d}.tga"
        reference = shot / "成品" / "描原" / name
        rows["A"][name] = audit(args.outputs / "task_a" / name, reference)
        rows["A_strict_four_color"][name] = audit(
            args.outputs / "task_a_strict_four_color" / name, reference
        )
    for index in (2, 3, 4, 5, 7, 8):
        name = f"A{index:04d}.tga"
        rows["B"][name] = audit(
            args.outputs / "task_b" / name, shot / "成品" / "中割" / name
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote independent audit: {args.output}")


if __name__ == "__main__":
    main()
