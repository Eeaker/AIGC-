"""Curve-aware diagnostics: distance, tangent error and fragmentation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import imread
from src.metrics import line_mask


def tangent_map(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Principal local line orientation from windowed coordinate moments."""
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    binary = mask.astype(np.float32)
    count = cv2.boxFilter(binary, -1, (size, size), normalize=False)
    sx = cv2.filter2D(binary, -1, xx)
    sy = cv2.filter2D(binary, -1, yy)
    sxx = cv2.filter2D(binary, -1, xx * xx)
    syy = cv2.filter2D(binary, -1, yy * yy)
    sxy = cv2.filter2D(binary, -1, xx * yy)
    safe = np.maximum(count, 1)
    cxx = sxx / safe - (sx / safe) ** 2
    cyy = syy / safe - (sy / safe) ** 2
    cxy = sxy / safe - (sx * sy) / safe ** 2
    return 0.5 * np.arctan2(2 * cxy, cxx - cyy)


def topology(mask: np.ndarray) -> dict[str, int]:
    binary = mask.astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    neighbours = cv2.filter2D(binary, -1, np.ones((3, 3), np.uint8)) - binary
    return {
        "components": int(count - 1),
        "components_le_5px": int((sizes <= 5).sum()),
        "isolated_pixels": int(((binary > 0) & (neighbours == 0)).sum()),
    }


def evaluate(pred: np.ndarray, ref: np.ndarray) -> dict:
    pm, rm = line_mask(pred), line_mask(ref)
    pt, rt = tangent_map(pm), tangent_map(rm)
    distance, labels = cv2.distanceTransformWithLabels(
        (~rm).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ry, rx = np.nonzero(rm)
    lut_y = np.zeros(int(labels.max()) + 1, np.int32)
    lut_x = np.zeros_like(lut_y)
    lut_y[labels[ry, rx]], lut_x[labels[ry, rx]] = ry, rx
    py, px = np.nonzero(pm)
    nearest_angle = rt[lut_y[labels[py, px]], lut_x[labels[py, px]]]
    angle = np.abs(np.arctan2(np.sin(pt[py, px] - nearest_angle),
                              np.cos(pt[py, px] - nearest_angle)))
    angle = np.minimum(angle, np.pi - angle) * 180 / np.pi
    d = distance[py, px]
    return {
        "pred_topology": topology(pm), "ref_topology": topology(rm),
        "pred_line_pixels": int(pm.sum()), "ref_line_pixels": int(rm.sum()),
        "pred_to_ref_mean_px": float(d.mean()),
        "pred_to_ref_p95_px": float(np.percentile(d, 95)),
        "local_tangent_mean_deg": float(angle.mean()),
        "local_tangent_p90_deg": float(np.percentile(angle, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pred", type=Path)
    parser.add_argument("ref", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(imread(args.pred), imread(args.ref))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
