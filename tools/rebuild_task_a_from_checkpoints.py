"""Rebuild formal Task A from the packaged LOFO checkpoints.

No held-out reference is used for inference.  The fixed post-processing
choices are recorded in experiments/task_a/tangent_gap_summary.json and were
selected on the other two folds before this reconstruction script was frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.io_utils import imread, imwrite
from tools import train_topology_residual as trainer
from tools.evaluate_tangent_gap_repair import prune_tiny, tangent_bridge
from tools.fuse_geometry_color_lofo import assign, fixed_geometry


FRAMES = ("A0001", "A0006", "A0009")
BRANCHES = ("geometry", "backbone", "colorreset")


def restore_production_green(strict_bgr: np.ndarray, rough_bgr: np.ndarray) -> np.ndarray:
    values = rough_bgr.astype(np.int16)
    blue, green, red = values[:, :, 0], values[:, :, 1], values[:, :, 2]
    dominance = green - np.maximum(blue, red)
    saturation = values.max(axis=2) - values.min(axis=2)
    gray = np.asarray(Image.fromarray(rough_bgr[:, :, ::-1]).convert("L"))
    mask = (dominance > 70) & (saturation > 35) & (gray < 220)
    out = strict_bgr[:, :, :3].copy()
    out[mask] = (0, 255, 0)
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    work = (args.work or root / "experiments" / "task_a" / "rebuild_cache").resolve()
    output = (args.output or root / "outputs" / "reconstructed" / "task_a").resolve()
    strict_dir, production_dir = output / "strict_four_color", output / "production_five_color"
    strict_dir.mkdir(parents=True, exist_ok=True)
    production_dir.mkdir(parents=True, exist_ok=True)

    trainer.DATA = args.data_root / "KTK_04_246B"
    trainer.CURRENT = root / "artifacts" / "task_a_baseline_input"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = {name: trainer.load_frame(name) for name in FRAMES}
    probabilities: dict[str, dict[str, np.ndarray]] = {name: {} for name in FRAMES}

    for branch in BRANCHES:
        for frame in FRAMES:
            checkpoint_path = (root / "models" / "task_a_lofo" / branch /
                               f"holdout_{frame}" / "model.pth")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = trainer.ResidualUNet().to(device)
            model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
            probability, gate = trainer.infer(model, frames[frame], device)
            probabilities[frame][branch] = probability
            cache = work / branch / f"holdout_{frame}"
            cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache / f"{frame}_probabilities.npz",
                                probabilities=probability, gate=gate)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rough_dir = args.data_root / "KTK_04_246B" / "源文件" / "描原"
    rough_names = {"A0001": "A1.jpg", "A0006": "A2.jpg", "A0009": "A3.jpg"}
    manifest = {"device": str(device), "frames": {}}
    for frame in FRAMES:
        p = probabilities[frame]
        line = fixed_geometry(p["geometry"])
        # Only A0009 passed the frozen multi-fold short-gap acceptance gate.
        if frame == "A0009":
            line = tangent_bridge(prune_tiny(line, 1), 5.0, 35.0)
        colour = (0.60 * p["backbone"] + 0.40 * p["colorreset"]).argmax(0)
        rgb = trainer.colorize(assign(line, colour))
        strict_path = strict_dir / f"{frame}.tga"
        Image.fromarray(rgb).save(strict_path)
        strict_bgr = imread(strict_path)[:, :, :3]
        production = restore_production_green(
            strict_bgr, imread(rough_dir / rough_names[frame])[:, :, :3])
        production_path = production_dir / f"{frame}.tga"
        imwrite(production_path, production)
        manifest["frames"][frame] = {
            "strict_sha256": sha256(strict_path),
            "production_sha256": sha256(production_path),
            "postprocess": "prune1_bridge_g5_a35" if frame == "A0009" else "fusion_base",
        }
    (output / "rebuild_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
