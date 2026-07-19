"""Independent, fail-closed validation of the submission hierarchy."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import imread
from src.metrics import (closure_metrics, color_compliance, color_line_f1,
                         region_color_metrics, union_tolerance_f1)


def _write_flat_csv(metrics: dict[str, object], output_path: Path) -> None:
    """Export the validated JSON payload to CSV from the same source of truth."""
    rows: list[dict[str, object]] = []
    metric_keys: set[str] = set()
    for task in ("A", "B", "C"):
        frames = metrics[task]
        assert isinstance(frames, dict)
        for frame, values in frames.items():
            assert isinstance(values, dict)
            metric_keys.update(values.keys())
            rows.append({"task": task, "frame": frame, **values})
    fieldnames = ["task", "frame", *sorted(metric_keys)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    shot = args.data_root / "KTK_04_246B"
    official = args.root / "outputs" / "official"
    metrics: dict[str, object] = {
        "metadata": {
            "formal_task_a": "strict_four_color",
            "formal_task_b": "endpoint_raster_flow_with_topology_regularization",
            "formal_task_b_uses_shot_cleanup": False,
            "formal_task_c": "assisted_clean_keyframe_labels",
            "held_out_A0007_A0008_used_for_B_rendering": False,
        }, "A": {}, "B": {}, "C": {},
    }

    for index in (1, 6, 9):
        name = f"A{index:04d}.tga"
        pred = imread(official / "task_a" / name)
        ref = imread(shot / "成品" / "描原" / name)
        assert pred.shape == ref.shape
        compliance = color_compliance(pred, include_green=False)
        assert compliance == 1.0, (name, compliance)
        metrics["A"][name] = {
            "four_color_compliance": compliance,
            **union_tolerance_f1(pred, ref), **color_line_f1(pred, ref, include_green=True),
            **closure_metrics(pred, ref),
        }

    for index in (2, 3, 4, 5, 7, 8):
        name = f"A{index:04d}.tga"
        pred = imread(official / "task_b" / name)
        ref = imread(shot / "成品" / "中割" / name)
        assert pred.shape == ref.shape
        compliance = color_compliance(pred, include_green=True)
        assert compliance == 1.0, (name, compliance)
        metrics["B"][name] = {
            "production_color_compliance": compliance,
            **union_tolerance_f1(pred, ref), **color_line_f1(pred, ref, include_green=True),
        }

    for index in range(1, 10):
        name = f"A{index:04d}.tga"
        line = imread(shot / "源文件" / "上色" / name)
        pred = imread(official / "task_c_assisted" / name)
        ref = imread(shot / "成品" / "上色" / name)
        assert pred.shape == line.shape == ref.shape
        technical = np.any(line[:, :, :3] != 255, axis=2)
        changed = int(np.any(pred[:, :, :3] != line[:, :, :3], axis=2)[technical].sum())
        assert changed == 0, (name, changed)
        metrics["C"][name] = region_color_metrics(line, pred, ref)

    seed_file = official / "task_c_assisted" / "semantic_seeds.json"
    seeds = json.loads(seed_file.read_text(encoding="utf-8"))
    assert len(seeds) == 153, len(seeds)
    metrics["metadata"]["task_c_seed_count"] = len(seeds)

    # Keep the zero-label baseline self-describing. Recompute this file from
    # images during validation instead of maintaining a hand-written copy.
    automatic_dir = args.root / "outputs" / "algorithm_only" / "task_c_automatic"
    automatic_frames: dict[str, object] = {}
    automatic_line_changes = 0
    for index in range(1, 10):
        name = f"A{index:04d}.tga"
        line = imread(shot / "源文件" / "上色" / name)
        pred = imread(automatic_dir / name)
        ref = imread(shot / "成品" / "上色" / name)
        technical = np.any(line[:, :, :3] != 255, axis=2)
        changed = int(np.any(pred[:, :, :3] != line[:, :, :3], axis=2)[technical].sum())
        automatic_line_changes += changed
        automatic_frames[name] = region_color_metrics(line, pred, ref)
    automatic_payload = {
        "profile": "automatic",
        "semantic_seed_count": 0,
        "area_precision_mean": float(np.mean([x["area_precision"] for x in automatic_frames.values()])),
        "correct_coverage_mean": float(np.mean([x["area_coverage"] for x in automatic_frames.values()])),
        "line_pixels_changed": automatic_line_changes,
        "frames": automatic_frames,
    }
    (automatic_dir / "metrics.json").write_text(
        json.dumps(automatic_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out = args.root / "outputs" / "summary" / "official_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_flat_csv(metrics, out.with_suffix(".csv"))
    for task, folder in (("A", "task_a"), ("B", "task_b"), ("C", "task_c_assisted")):
        (official / folder / "metrics.json").write_text(
            json.dumps({"metadata": metrics["metadata"], "frames": metrics[task]},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: 18 formal TGA files; metrics written to {out}")


if __name__ == "__main__":
    main()
