"""Validate KTK_05_140 advanced outputs, especially strict color/line constraints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import imread
from src.metrics import color_compliance, region_color_metrics, union_tolerance_f1


def _hierarchy_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Audit exact-colour recovery by automatic luminance strata."""
    colors, counts = np.unique(ref.reshape(-1, 3), axis=0, return_counts=True)
    colors = colors[np.any(colors != 255, axis=1) & (counts >= 8)]
    if not len(colors):
        return {}
    luminance = .0722 * colors[:, 0] + .7152 * colors[:, 1] + .2126 * colors[:, 2]
    low, high = np.quantile(luminance, [.33, .67])
    ref_code = (ref[:, :, 0].astype(np.uint32) |
                (ref[:, :, 1].astype(np.uint32) << 8) |
                (ref[:, :, 2].astype(np.uint32) << 16))
    pred_code = (pred[:, :, 0].astype(np.uint32) |
                 (pred[:, :, 1].astype(np.uint32) << 8) |
                 (pred[:, :, 2].astype(np.uint32) << 16))
    exact = pred_code == ref_code
    result = {}
    for name, select in (("shadow", luminance <= low),
                         ("base", (luminance > low) & (luminance < high)),
                         ("highlight", luminance >= high)):
        palette = colors[select]
        codes = (palette[:, 0].astype(np.uint32) |
                 (palette[:, 1].astype(np.uint32) << 8) |
                 (palette[:, 2].astype(np.uint32) << 16))
        target = np.isin(ref_code, codes)
        correct = target & exact
        predicted = np.isin(pred_code, codes)
        result[f"{name}_exact_precision"] = float(correct.sum() / max(predicted.sum(), 1))
        result[f"{name}_exact_coverage"] = float(correct.sum() / max(target.sum(), 1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    shot = args.data_root / "KTK_05_140"
    base = args.root / "outputs" / "supplementary" / "ktk05"
    result: dict[str, object] = {"A": {}, "B": {}, "C": {}}

    for layer, folder, names in (
        ("A", "task_a_A_strict_4color", ("A0001.tga", "A0005.tga")),
        ("B", "task_a_B", ("B0001.tga", "B0003.tga")),
    ):
        ref_dir = shot / "成品" / "描原" / layer
        for name in names:
            pred, ref = imread(base / folder / name), imread(ref_dir / name)
            compliance = color_compliance(pred, include_green=False)
            assert compliance == 1.0, (folder, name, compliance)
            result["A"][f"{layer}/{name}"] = {
                "four_color_compliance": compliance, **union_tolerance_f1(pred, ref)}

    for layer, folder, names in (
        ("A", "task_b_A", ("A0002.tga", "A0003.tga", "A0004.tga")),
        ("B", "task_b_B", ("B0002.tga",)),
    ):
        ref_dir = shot / "成品" / "中割" / layer
        for name in names:
            pred, ref = imread(base / folder / name), imread(ref_dir / name)
            result["B"][f"{layer}/{name}"] = union_tolerance_f1(pred, ref)

    for layer, folder, names in (
        ("A", "task_c_A_strict_lines", tuple(f"A{i:04d}.tga" for i in range(1, 6))),
        ("B", "task_c_B_strict_lines", tuple(f"B{i:04d}.tga" for i in range(1, 4))),
    ):
        line_dir, ref_dir = shot / "源文件" / "上色" / layer, shot / "成品" / "上色" / layer
        for name in names:
            line, pred, ref = imread(line_dir / name), imread(base / folder / name), imread(ref_dir / name)
            technical = np.any(line[:, :, :3] != 255, axis=2)
            changed = int(np.any(line[:, :, :3] != pred[:, :, :3], axis=2)[technical].sum())
            assert changed == 0, (folder, name, changed)
            row = {**region_color_metrics(line, pred, ref),
                   **_hierarchy_metrics(pred, ref),
                   "strict_line_pixels_changed": changed}
            result["C"][f"{layer}/{name}"] = row

    path = base / "validation.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    task_c = {layer: {
        key.split("/", 1)[1]: value for key, value in result["C"].items()
        if key.startswith(layer + "/")
    } for layer in ("A", "B")}
    (base / "task_c_metrics.json").write_text(
        json.dumps(task_c, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: KTK_05 A/B/C supplementary outputs validated")


if __name__ == "__main__":
    main()
