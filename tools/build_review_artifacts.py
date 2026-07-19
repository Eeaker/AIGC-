"""Build compact comparison figures and summary CSV files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import imread, imwrite


def _fit(image: np.ndarray, width: int = 520) -> np.ndarray:
    scale = width / image.shape[1]
    return cv2.resize(image[:, :, :3], (width, int(image.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((52, image.shape[1], 3), 247, np.uint8)
    cv2.putText(bar, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, .72, (35, 35, 35), 2,
                cv2.LINE_AA)
    return np.vstack([bar, image])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    shot = args.data_root / "KTK_04_246B"
    line = _fit(imread(shot / "源文件" / "上色" / "A0006.tga"))
    pred = _fit(imread(args.root / "outputs" / "official" / "task_c_assisted" / "A0006.tga"))
    ref = _fit(imread(shot / "成品" / "上色" / "A0006.tga"))
    panel = np.hstack([_label(line, "Input line"), _label(pred, "Assisted output (153 labels)"),
                       _label(ref, "Reference (evaluation only)")])
    imwrite(args.root / "outputs" / "comparisons" / "task_c.png", panel)

    metrics = json.loads((args.root / "outputs" / "summary" / "official_metrics.json").read_text(encoding="utf-8"))
    rows = []
    for task in ("A", "B", "C"):
        for frame, values in metrics[task].items():
            row = {"task": task, "frame": frame}
            row.update(values)
            rows.append(row)
    keys = sorted({k for row in rows for k in row})
    with (args.root / "outputs" / "summary" / "official_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
