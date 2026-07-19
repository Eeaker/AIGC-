"""Fuse leakage-controlled LOFO geometry and colour predictions.

The full DSV-transfer model supplies a single fixed geometry mask.  Colour is
assigned by independent target-domain models or by nearest current ink.  No
reference pixel is consulted while constructing any candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.image_metric_utils import label
from tools.evaluate_residual_postprocess import metrics
from tools.train_topology_residual import PALETTE, colorize, masks, read_rgb


FRAMES = ("A0001", "A0006", "A0009")
BOXES = {
    "A0001": (720, 640, 512, 512),
    "A0006": (720, 640, 512, 512),
    "A0009": (880, 800, 512, 512),
}


def probabilities(root: Path, frame: str) -> np.ndarray:
    path = root / f"holdout_{frame}" / f"{frame}_probabilities.npz"
    return np.load(path)["probabilities"]


def fixed_geometry(probability: np.ndarray, threshold: float = 0.80) -> np.ndarray:
    line = (probability.max(0) >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(line, cv2.MORPH_CLOSE, kernel) > 0


def assign(line: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.zeros((3, *line.shape), np.float32)
    for channel in range(3):
        result[channel, line & (labels == channel)] = 1
    return result


def nearest_current_labels(current: np.ndarray) -> np.ndarray:
    distances = [cv2.distanceTransform((current[channel] <= 0.5).astype(np.uint8),
                                       cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                 for channel in range(3)]
    return np.argmin(np.stack(distances), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--backbone-root", type=Path, required=True)
    parser.add_argument("--colorreset-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "2026.07.13")
    parser.add_argument("--current-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "task_a")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report = {"construction": {
        "geometry": "full DSV transfer; max probability >= 0.80; 3x3 elliptical close",
        "colour": "independent target-domain prediction; reference-free",
    }, "frames": {}}
    for frame in FRAMES:
        geometry_probability = probabilities(args.geometry_root, frame)
        backbone_probability = probabilities(args.backbone_root, frame)
        colorreset_probability = probabilities(args.colorreset_root, frame)
        control_probability = probabilities(args.control_root, frame)
        line = fixed_geometry(geometry_probability)
        current = masks(read_rgb(args.current_root / f"{frame}.tga"))
        target = masks(read_rgb(args.data_root / "KTK_04_246B" / "成品" / "描原" / f"{frame}.tga"))

        variants = {
            "backbone_colour": assign(line, backbone_probability.argmax(0)),
            "colorreset_colour": assign(line, colorreset_probability.argmax(0)),
            "control_colour": assign(line, control_probability.argmax(0)),
            "ensemble_colour": assign(line, (0.60 * backbone_probability
                                               + 0.40 * colorreset_probability).argmax(0)),
            "nearest_current_colour": assign(line, nearest_current_labels(current)),
        }
        report["frames"][frame] = {}
        frame_dir = args.output / frame
        frame_dir.mkdir(exist_ok=True)
        for name, candidate in variants.items():
            result = metrics(candidate, target)
            report["frames"][frame][name] = result
            Image.fromarray(colorize(candidate)).save(frame_dir / f"{frame}_{name}.tga")
            x, y, w, h = BOXES[frame]
            views = [current, candidate, target]
            captions = ["CURRENT", name.upper(), "REFERENCE"]
            crops = [label(cv2.cvtColor(colorize(value), cv2.COLOR_RGB2BGR)[y:y+h, x:x+w], caption)
                     for value, caption in zip(views, captions)]
            panel = cv2.resize(np.hstack(crops), None, fx=2, fy=2,
                               interpolation=cv2.INTER_NEAREST)
            cv2.imencode(".png", panel)[1].tofile(frame_dir / f"{frame}_{name}_detail_2x.png")

    keys = ("f1_2px", "exact_f1", "local_tangent_mean_deg",
            "pred_to_ref_mean_px", "color_macro_f1_2px", "line_pixel_ratio")
    names = tuple(report["frames"][FRAMES[0]])
    report["mean"] = {
        name: {key: float(np.mean([report["frames"][frame][name][key]
                                  for frame in FRAMES])) for key in keys}
        for name in names
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, result in report["mean"].items():
        print(name, " ".join(f"{key}={result[key]:.6f}" for key in keys))


if __name__ == "__main__":
    main()
