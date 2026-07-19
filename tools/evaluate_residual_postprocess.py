"""Evaluate conservative one-pixel post-processing for a residual-model run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.task_a import _thin, _repair_short_gaps
from tools.evaluate_curve_quality import evaluate
from tools.image_metric_utils import tolerant_f1, label
from tools.train_topology_residual import (ROOT, DATA, CURRENT, PALETTE, masks,
                                           colorize, union_bgr, color_macro_f1)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def thin_channels(channels: np.ndarray) -> np.ndarray:
    result = np.zeros_like(channels)
    union = channels.max(0) > 0.5
    line = _thin(union)
    # Thin the union once so differently coloured strokes cannot create gaps
    # at crossings. Assign each skeleton pixel to its nearest source colour.
    distances = []
    for index in range(len(channels)):
        distances.append(cv2.distanceTransform((channels[index] <= 0.5).astype(np.uint8),
                                               cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
    nearest = np.argmin(np.stack(distances), axis=0)
    for index in range(len(channels)):
        result[index, line & (nearest == index)] = 1
    return result


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred_union, target_union = pred.max(0) > 0.5, target.max(0) > 0.5
    exact_intersection = int((pred_union & target_union).sum())
    exact_precision = exact_intersection / max(int(pred_union.sum()), 1)
    exact_recall = exact_intersection / max(int(target_union.sum()), 1)
    exact_f1 = 2 * exact_precision * exact_recall / max(exact_precision + exact_recall, 1e-12)
    return (evaluate(union_bgr(pred), union_bgr(target))
            | tolerant_f1(pred_union, target_union)
            | {"exact_f1": exact_f1,
               "color_macro_f1_2px": color_macro_f1(pred, target),
               "line_pixel_ratio": float(pred_union.sum() / max(target_union.sum(), 1))})


def threshold_channels(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    result = np.zeros_like(probabilities, np.float32)
    score = probabilities.max(0)
    index = probabilities.argmax(0)
    for channel in range(len(probabilities)):
        result[channel, (score >= threshold) & (index == channel)] = 1
    return result


def threshold_channels_per_color(probabilities: np.ndarray,
                                 thresholds: tuple[float, ...]) -> np.ndarray:
    result = np.zeros_like(probabilities, np.float32)
    score = probabilities.max(0)
    index = probabilities.argmax(0)
    for channel, threshold in enumerate(thresholds):
        result[channel, (score >= threshold) & (index == channel)] = 1
    return result


def repair_channels(channels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    union = channels.max(0) > 0.5
    repaired = _repair_short_gaps(union)
    result = np.zeros_like(channels)
    nearest = probabilities.argmax(0)
    for channel in range(len(channels)):
        result[channel, repaired & (nearest == channel)] = 1
    return result


def repair_small_channels(channels: np.ndarray, probabilities: np.ndarray,
                          square: bool = False) -> np.ndarray:
    union = (channels.max(0) > 0.5).astype(np.uint8)
    kernel = (np.ones((3, 3), np.uint8) if square else
              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    repaired = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel) > 0
    result = np.zeros_like(channels)
    nearest = probabilities.argmax(0)
    for channel in range(len(channels)):
        result[channel, repaired & (nearest == channel)] = 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", default="A0006")
    parser.add_argument("--neural", action="store_true")
    args = parser.parse_args()
    run = ROOT / "outputs" / "topology_residual" / f"holdout_{args.holdout}"
    current = masks(read_rgb(CURRENT / f"{args.holdout}.tga"))
    target = masks(read_rgb(DATA / "成品" / "描原" / f"{args.holdout}.tga"))
    raw = masks(read_rgb(run / f"{args.holdout}_candidate.tga"))
    gate = np.asarray(Image.open(run / f"{args.holdout}_gate.png").convert("L"), np.float32) / 255
    probability_path = run / f"{args.holdout}_probabilities.npz"
    probabilities = np.load(probability_path)["probabilities"] if probability_path.exists() else None
    thinned = thin_channels(raw)
    variants = {"current": current, "raw_model": raw, "thin_all": thinned}
    if probabilities is not None:
        for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
            variants[f"prob_{threshold:.2f}"] = threshold_channels(probabilities, threshold)
        for threshold in (0.70, 0.80, 0.85):
            key = f"prob_{threshold:.2f}"
            variants[f"{key}_repair"] = repair_channels(variants[key], probabilities)
            variants[f"{key}_close3"] = repair_small_channels(variants[key], probabilities)
            variants[f"{key}_square3"] = repair_small_channels(variants[key], probabilities, square=True)
        calibrated = threshold_channels_per_color(probabilities, (0.75, 0.84, 0.80))
        variants["channel_calibrated"] = repair_small_channels(calibrated, probabilities)
        if args.neural:
            from src.neural_thinning import predict_white_probability
            base = variants["prob_0.85_repair"]
            base_bgr = cv2.cvtColor(colorize(base), cv2.COLOR_RGB2BGR)
            weights = ROOT / "solution" / "models" / "line_thinning_siggraph2018.pth"
            white = predict_white_probability(base_bgr, weights)
            nearest = probabilities.argmax(0)
            for threshold in (0.25, 0.30, 0.35, 0.40, 0.50):
                line = _thin(white < threshold)
                candidate = np.zeros_like(base)
                for channel in range(len(base)):
                    candidate[channel, line & (nearest == channel)] = 1
                variants[f"neural_{threshold:.2f}"] = candidate
    for threshold in (0.35, 0.50, 0.65, 0.80):
        select = gate >= threshold
        candidate = current.copy()
        candidate[:, select] = thinned[:, select]
        variants[f"thin_gate_{threshold:.2f}"] = candidate
    results = {name: metrics(value, target) for name, value in variants.items()}
    (run / "postprocess_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    views = [label(cv2.cvtColor(colorize(value), cv2.COLOR_RGB2BGR), name)
             for name, value in variants.items()]
    views.append(label(cv2.cvtColor(colorize(target), cv2.COLOR_RGB2BGR), "REFERENCE"))
    cv2.imencode(".png", np.hstack(views))[1].tofile(run / f"{args.holdout}_postprocess.png")
    if probabilities is not None:
        for key in ("prob_0.70", "prob_0.80", "prob_0.80_close3",
                    "prob_0.80_repair", "prob_0.85_repair", "channel_calibrated"):
            candidate_rgb = colorize(variants[key])
            Image.fromarray(candidate_rgb).save(run / f"{args.holdout}_{key}.tga")
            focused = np.hstack((label(cv2.cvtColor(colorize(current), cv2.COLOR_RGB2BGR), "CURRENT"),
                                 label(cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2BGR), key),
                                 label(cv2.cvtColor(colorize(target), cv2.COLOR_RGB2BGR), "REFERENCE")))
            cv2.imencode(".png", focused)[1].tofile(run / f"{args.holdout}_{key}.png")
    for name, item in results.items():
        print(f"{name:16s} F1={item['f1_2px']:.5f} exact={item['exact_f1']:.5f} "
              f"tangent={item['local_tangent_mean_deg']:.3f} "
              f"color={item['color_macro_f1_2px']:.5f} "
              f"pixels={item['line_pixel_ratio']:.3f}x "
              f"components={item['pred_topology']['components']}")


if __name__ == "__main__":
    main()
