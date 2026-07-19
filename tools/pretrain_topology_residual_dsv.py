"""Pretrain the conservative topology residual model on official DeepSketch GT.

The raster brush image is used as the observation.  The supervision comes from
the supplied dual-contouring edge flags, not from thinning the raster input.
Only the black channel is supervised here; production red/blue separation is
learned later during target-domain fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from train_topology_residual import ResidualUNet, clipped_udf, losses


class DeepSketchResidualSampler:
    def __init__(self, root: Path, patch: int, seed: int):
        self.root = root
        self.patch = patch
        self.rng = random.Random(seed)
        self.brushes = sorted(p for p in (root / "png").iterdir()
                              if p.is_dir() and p.name.isnumeric())
        if not self.brushes:
            raise FileNotFoundError(f"no numeric brush folders under {root / 'png'}")
        index = root / "img_index.txt"
        if index.exists():
            names = [x.strip() for x in index.read_text(encoding="utf-8").splitlines()
                     if x.strip()]
        else:
            names = [p.name for p in self.brushes[0].glob("*.png")]
        self.names = [name for name in names if (root / "gt" / Path(name).with_suffix(".npz")).exists()]
        if not self.names:
            raise FileNotFoundError(f"no matched PNG/GT pairs under {root}")

    @staticmethod
    def _target_from_gt(path: Path, shape: tuple[int, int]) -> np.ndarray:
        with np.load(path, allow_pickle=True) as gt:
            candidates = []
            for level, (edge_x, edge_y) in enumerate(zip(gt["edge_x"], gt["edge_y"])):
                skel = np.logical_or(np.asarray(edge_x), np.asarray(edge_y)).astype(np.uint8)
                candidates.append((abs(skel.shape[0] - shape[0]) + abs(skel.shape[1] - shape[1]), level, skel))
        skel = min(candidates, key=lambda item: item[0])[2]
        if skel.shape != shape:
            skel = cv2.resize(skel, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return skel

    def sample(self) -> tuple[torch.Tensor, ...]:
        name = self.rng.choice(self.names)
        valid_brushes = [b for b in self.brushes if (b / name).exists()]
        raster_path = self.rng.choice(valid_brushes) / name
        rgba = np.asarray(Image.open(raster_path).convert("RGBA"))
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32)
        rough_rgb = rgb * alpha + 255.0 * (1.0 - alpha)
        rough_gray = cv2.cvtColor(rough_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        target_union = self._target_from_gt(
            self.root / "gt" / Path(name).with_suffix(".npz"), rough_gray.shape)

        current_union = (rough_gray < self.rng.randint(170, 235)).astype(np.uint8)
        # Controlled detector/thickness errors teach correction without bending
        # the exact GT curve itself.
        op = self.rng.random()
        if op < 0.45:
            current_union = cv2.dilate(current_union, np.ones((3, 3), np.uint8))
        elif op < 0.70:
            current_union = cv2.erode(current_union, np.ones((3, 3), np.uint8))
        elif op < 0.82:
            dx, dy = self.rng.choice((-1, 1)), self.rng.choice((-1, 1))
            current_union = cv2.warpAffine(current_union, np.float32([[1, 0, dx], [0, 1, dy]]),
                                           (current_union.shape[1], current_union.shape[0]),
                                           flags=cv2.INTER_NEAREST, borderValue=0)

        h, w = target_union.shape
        if h < self.patch or w < self.patch:
            scale = max(self.patch / h, self.patch / w)
            size = (int(round(w * scale)), int(round(h * scale)))
            rough_rgb = cv2.resize(rough_rgb, size, interpolation=cv2.INTER_AREA)
            current_union = cv2.resize(current_union, size, interpolation=cv2.INTER_NEAREST)
            target_union = cv2.resize(target_union, size, interpolation=cv2.INTER_NEAREST)
            h, w = target_union.shape

        focus = np.argwhere((current_union != target_union) | (target_union > 0))
        if len(focus) and self.rng.random() < 0.9:
            cy, cx = focus[self.rng.randrange(len(focus))]
            y = int(np.clip(cy - self.rng.randrange(self.patch), 0, h - self.patch))
            x = int(np.clip(cx - self.rng.randrange(self.patch), 0, w - self.patch))
        else:
            y = self.rng.randint(0, h - self.patch)
            x = self.rng.randint(0, w - self.patch)
        sl = np.s_[y:y + self.patch, x:x + self.patch]
        rough = rough_rgb[sl].astype(np.float32).transpose(2, 0, 1) / 255.0
        current = np.zeros((3, self.patch, self.patch), np.float32)
        target = np.zeros_like(current)
        current[0] = current_union[sl]
        target[0] = target_union[sl]
        difference = np.any(current != target, axis=0).astype(np.uint8)
        change = (cv2.dilate(difference, np.ones((5, 5), np.uint8)) > 0)[None].astype(np.float32)
        udf = clipped_udf(target_union[sl] > 0)[None]
        if self.rng.random() < 0.5:
            current, target, rough, udf, change = [np.flip(a, -1).copy()
                                                   for a in (current, target, rough, udf, change)]
        model_input = np.concatenate((current, rough), axis=0)
        return tuple(torch.from_numpy(a).float() for a in
                     (model_input, current, target, udf, change))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--patch", type=int, default=192)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = DeepSketchResidualSampler(args.root, args.patch, args.seed)
    model = ResidualUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        batch = [sampler.sample() for _ in range(args.batch)]
        tensors = [torch.stack(items).to(device, non_blocking=True) for items in zip(*batch)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            output = model(tensors[0])
            loss, values = losses(output, *tensors[1:])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 100 == 0:
            elapsed = time.perf_counter() - started
            values |= {"step": step, "steps_per_second": (step - start_step) / elapsed}
            history.append(values)
            print("step", step, " ".join(f"{k}={v:.4f}" for k, v in values.items()
                                           if k != "step"), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "step": step, "args": vars(args)}
            torch.save(checkpoint, args.output / f"checkpoint_{step:06d}.pth")
            torch.save(checkpoint, args.output / "latest.pth")
            (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
