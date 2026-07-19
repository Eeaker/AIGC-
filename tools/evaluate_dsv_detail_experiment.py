"""Rasterize and evaluate official DSV SVGs on Task-A detail crops."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from svgpathtools import svg2paths

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_curve_quality import evaluate


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs" / "dsv_detail_experiment"
SVG = EXP / "dsv_light_refine" / "svg_full"
OUT = EXP / "evaluation"


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return np.repeat(np.where(mask[..., None], 0, 255).astype(np.uint8), 3, 2)


def rasterize(svg_path: Path, size: int = 512) -> np.ndarray:
    paths, _ = svg2paths(str(svg_path))
    canvas = np.zeros((size, size), np.uint8)
    # DSV emits a 2x coordinate canvas for a 512px input.
    scale = size / 1024.0
    for path in paths:
        # Raw/refined SVGs may store many disconnected M...L subpaths in one
        # path element.  Never join those discontinuities while rasterizing.
        for subpath in path.continuous_subpaths():
            points: list[tuple[int, int]] = []
            for segment in subpath:
                count = max(2, int(np.ceil(segment.length() * scale * 2)))
                for t in np.linspace(0.0, 1.0, count, endpoint=False):
                    z = segment.point(float(t))
                    points.append((round(z.real * scale), round(z.imag * scale)))
            if subpath:
                z = subpath[-1].end
                points.append((round(z.real * scale), round(z.imag * scale)))
            if len(points) >= 2:
                cv2.polylines(canvas, [np.asarray(points, np.int32)], False, 1,
                              thickness=1, lineType=cv2.LINE_8)
    return canvas.astype(bool)


def tolerant_f1(pred: np.ndarray, ref: np.ndarray, tolerance: float = 2.0) -> dict[str, float]:
    to_ref = cv2.distanceTransform((~ref).astype(np.uint8), cv2.DIST_L2,
                                   cv2.DIST_MASK_PRECISE)
    to_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2,
                                    cv2.DIST_MASK_PRECISE)
    precision = float((to_ref[pred] <= tolerance).mean()) if pred.any() else 0.0
    recall = float((to_pred[ref] <= tolerance).mean()) if ref.any() else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision_2px": precision, "recall_2px": recall, "f1_2px": f1}


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (245, 245, 245), -1)
    cv2.putText(result, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (25, 25, 25), 1, cv2.LINE_AA)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for input_path in sorted((EXP / "input").glob("*.png")):
        name = input_path.stem
        current = read_gray(input_path) < 128
        ref = read_gray(EXP / "reference" / input_path.name) < 128
        candidates = {
            stage: rasterize(SVG / f"{name}_{stage}.svg")
            for stage in ("raw", "refine", "final")
        }
        dsv = candidates["final"]
        current_bgr, dsv_bgr, ref_bgr = map(mask_to_bgr, (current, dsv, ref))
        current_metrics = evaluate(current_bgr, ref_bgr) | tolerant_f1(current, ref)
        stage_metrics = {}
        for stage, candidate in candidates.items():
            candidate_bgr = mask_to_bgr(candidate)
            stage_metrics[stage] = (evaluate(candidate_bgr, ref_bgr)
                                    | tolerant_f1(candidate, ref)
                                    | {f"fidelity_{key}": value for key, value
                                       in tolerant_f1(candidate, current).items()})
        dsv_metrics = stage_metrics["final"]
        fidelity = tolerant_f1(dsv, current)
        summary[name] = {"current_vs_reference": current_metrics,
                         "dsv_stages_vs_reference": stage_metrics,
                         "dsv_vs_reference": dsv_metrics,
                         "dsv_vs_current_fidelity": fidelity}

        error = np.full((*dsv.shape, 3), 255, np.uint8)
        error[current & dsv] = (30, 30, 30)
        error[current & ~dsv] = (255, 80, 0)   # current removed: blue
        error[~current & dsv] = (0, 50, 255)   # DSV added/moved: red
        panel = np.hstack((label(current_bgr, "CURRENT HYBRID"),
                           label(dsv_bgr, "DSV SIGGRAPH 2024"),
                           label(ref_bgr, "REFERENCE"),
                           label(error, "DSV vs CURRENT")))
        cv2.imencode(".png", panel)[1].tofile(OUT / f"{name}_comparison.png")

    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, item in summary.items():
        a, b, f = item["current_vs_reference"], item["dsv_vs_reference"], item["dsv_vs_current_fidelity"]
        print(name)
        print(f"  current F1={a['f1_2px']:.5f} tangent={a['local_tangent_mean_deg']:.3f} components={a['pred_topology']['components']}")
        for stage, metrics in item["dsv_stages_vs_reference"].items():
            print(f"  DSV {stage:6s} F1={metrics['f1_2px']:.5f} tangent={metrics['local_tangent_mean_deg']:.3f} components={metrics['pred_topology']['components']}")
        print(f"  fidelity to current F1={f['f1_2px']:.5f}")

    # The full UDF model is expensive on an 8GB GPU, so it may only exist for
    # the deliberately hardest crop.  Evaluate it alongside the light model
    # when available without making the experiment depend on it.
    full_svg = EXP / "dsv_full_refine" / "svg_full" / "A0006_eye_hair_final.svg"
    if full_svg.exists():
        name = "A0006_eye_hair"
        current = read_gray(EXP / "input" / f"{name}.png") < 128
        ref = read_gray(EXP / "reference" / f"{name}.png") < 128
        full = rasterize(full_svg)
        metrics = evaluate(mask_to_bgr(full), mask_to_bgr(ref)) | tolerant_f1(full, ref)
        fidelity = tolerant_f1(full, current)
        summary["A0006_eye_hair_full_model"] = {
            "full_vs_reference": metrics,
            "full_vs_current_fidelity": fidelity,
        }
        (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"A0006 full F1={metrics['f1_2px']:.5f} "
              f"tangent={metrics['local_tangent_mean_deg']:.3f} "
              f"components={metrics['pred_topology']['components']} "
              f"fidelity={fidelity['f1_2px']:.5f}")
        panel = np.hstack((label(mask_to_bgr(current), "CURRENT HYBRID"),
                           label(mask_to_bgr(full), "DSV FULL SIGGRAPH 2024"),
                           label(mask_to_bgr(ref), "REFERENCE")))
        cv2.imencode(".png", panel)[1].tofile(OUT / f"{name}_full_comparison.png")


if __name__ == "__main__":
    main()
