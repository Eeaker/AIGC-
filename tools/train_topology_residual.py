"""Leakage-controlled pilot for topology-aware Task-A residual correction.

Train on two frames and evaluate on a completely held-out third frame.  The
network sees the current formal output plus the rough RGB image and predicts
three legal ink channels, a union unsigned-distance field, and a conservative
change gate.  Reference pixels from the held-out frame are used only after
inference for metrics and visualisation.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_curve_quality import evaluate
from tools.image_metric_utils import tolerant_f1, label


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
# Backward-compatible alias for older diagnostic scripts. New code should use
# PACKAGE_ROOT/WORKSPACE_ROOT explicitly.
ROOT = WORKSPACE_ROOT
DATA = WORKSPACE_ROOT / "2026.07.13" / "KTK_04_246B"
CURRENT = PACKAGE_ROOT / "outputs" / "task_a"
OUT = PACKAGE_ROOT / "experiments" / "topology_residual"
FRAME_MAP = {"A0001": "A1.jpg", "A0006": "A2.jpg", "A0009": "A3.jpg"}
# RGB ink palette. Green registration pixels are deliberately excluded.
PALETTE = np.asarray(((0, 0, 0), (0, 0, 255), (255, 0, 0)), np.uint8)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def masks(image: np.ndarray) -> np.ndarray:
    return np.stack([np.all(image == color, axis=2) for color in PALETTE]).astype(np.float32)


def union_bgr(channels: np.ndarray) -> np.ndarray:
    ink = channels.max(axis=0) > 0.5
    return np.repeat(np.where(ink[..., None], 0, 255).astype(np.uint8), 3, 2)


def colorize(channels: np.ndarray) -> np.ndarray:
    result = np.full((*channels.shape[1:], 3), 255, np.uint8)
    score = channels.max(axis=0)
    index = channels.argmax(axis=0)
    for channel, color in enumerate(PALETTE):
        result[(score >= 0.5) & (index == channel)] = color
    return result


def clipped_udf(union: np.ndarray, clip: float = 8.0) -> np.ndarray:
    distance = cv2.distanceTransform((~union).astype(np.uint8), cv2.DIST_L2,
                                     cv2.DIST_MASK_PRECISE)
    return np.minimum(distance, clip).astype(np.float32) / clip


@dataclass
class Frame:
    name: str
    rough: np.ndarray
    current: np.ndarray
    target: np.ndarray
    udf: np.ndarray
    change: np.ndarray
    focus: np.ndarray


def load_frame(name: str) -> Frame:
    rough = read_rgb(DATA / "源文件" / "描原" / FRAME_MAP[name]).astype(np.float32) / 255.0
    current = masks(read_rgb(CURRENT / f"{name}.tga"))
    target = masks(read_rgb(DATA / "成品" / "描原" / f"{name}.tga"))
    # Ignore non-task green registration strokes by construction (all target
    # channels are zero there). The training ROI also excludes production notes.
    current_union = current.max(0) > 0
    target_union = target.max(0) > 0
    difference = np.any(current != target, axis=0).astype(np.uint8)
    change = cv2.dilate(difference, np.ones((5, 5), np.uint8)) > 0
    focus = np.argwhere(change | target_union | current_union)
    return Frame(name, rough, current, target, clipped_udf(target_union),
                 change.astype(np.float32), focus)


class Block(nn.Module):
    def __init__(self, a: int, b: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualUNet(nn.Module):
    def __init__(self, base: int = 20):
        super().__init__()
        self.e1, self.e2, self.e3 = Block(6, base), Block(base, base * 2), Block(base * 2, base * 4)
        self.mid = Block(base * 4, base * 8)
        self.d3 = Block(base * 12, base * 4)
        self.d2 = Block(base * 6, base * 2)
        self.d1 = Block(base * 3, base)
        self.head = nn.Conv2d(base, 5, 1)  # 3 ink logits, UDF, change gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        m = self.mid(F.avg_pool2d(c, 2))
        c2 = self.d3(torch.cat((F.interpolate(m, size=c.shape[-2:], mode="bilinear", align_corners=False), c), 1))
        b2 = self.d2(torch.cat((F.interpolate(c2, size=b.shape[-2:], mode="bilinear", align_corners=False), b), 1))
        a2 = self.d1(torch.cat((F.interpolate(b2, size=a.shape[-2:], mode="bilinear", align_corners=False), a), 1))
        return self.head(a2)


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-image, 3, stride=1, padding=1)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, 3, stride=1, padding=1)


def soft_skeleton(image: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    opened = soft_dilate(soft_erode(image))
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = soft_dilate(soft_erode(image))
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_s, target_s = soft_skeleton(pred), soft_skeleton(target)
    precision = (pred_s * target).sum((2, 3)) / (pred_s.sum((2, 3)) + 1e-6)
    sensitivity = (target_s * pred).sum((2, 3)) / (target_s.sum((2, 3)) + 1e-6)
    return (1 - 2 * precision * sensitivity / (precision + sensitivity + 1e-6)).mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = 2 * (pred * target).sum((2, 3)) + 1
    denominator = pred.sum((2, 3)) + target.sum((2, 3)) + 1
    return (1 - numerator / denominator).mean()


def crop(frame: Frame, size: int, rng: random.Random) -> tuple[torch.Tensor, ...]:
    h, w = frame.target.shape[1:]
    xmin, xmax, ymin, ymax = 0, int(0.82 * w) - size, 120, int(0.86 * h) - size
    if rng.random() < 0.85 and len(frame.focus):
        y0, x0 = frame.focus[rng.randrange(len(frame.focus))]
        x = int(np.clip(x0 - rng.randrange(size), xmin, xmax))
        y = int(np.clip(y0 - rng.randrange(size), ymin, ymax))
    else:
        x, y = rng.randint(xmin, xmax), rng.randint(ymin, ymax)
    sl = np.s_[y:y + size, x:x + size]
    current = frame.current[:, sl[0], sl[1]].copy()
    target = frame.target[:, sl[0], sl[1]].copy()
    # Synthetic width/noise variation prevents memorising the two training
    # frames and teaches correction of the exact thick/thin failure mode.
    if rng.random() < 0.35:
        channel = rng.randrange(3)
        binary = current[channel].astype(np.uint8)
        if rng.random() < 0.7:
            binary = cv2.dilate(binary, np.ones((3, 3), np.uint8))
        else:
            binary = cv2.erode(binary, np.ones((3, 3), np.uint8))
        current[channel] = binary
        synthetic_difference = np.any(current != target, axis=0).astype(np.uint8)
        change = (cv2.dilate(synthetic_difference, np.ones((5, 5), np.uint8)) > 0)[None].astype(np.float32)
    rough = frame.rough[sl].transpose(2, 0, 1)
    udf = frame.udf[sl][None]
    change = frame.change[sl][None]
    if rng.random() < 0.5:
        current, target, rough, udf, change = [np.flip(a, axis=-1).copy()
                                               for a in (current, target, rough, udf, change)]
    model_input = np.concatenate((current, rough), axis=0)
    return tuple(torch.from_numpy(a).float() for a in (model_input, current, target, udf, change))


def losses(output: torch.Tensor, current: torch.Tensor, target: torch.Tensor,
           udf: torch.Tensor, change: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    logits, udf_pred, gate_logits = output[:, :3], output[:, 3:4], output[:, 4:5]
    prob, gate = torch.sigmoid(logits), torch.sigmoid(gate_logits)
    positives = target.sum().clamp_min(1)
    negatives = target.numel() - positives
    pos_weight = (negatives / positives).clamp(1, 25).detach()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    dice = dice_loss(prob, target)
    union_prob = 1 - torch.prod(1 - prob, dim=1, keepdim=True)
    union_target = target.max(1, keepdim=True).values
    topology = cldice_loss(union_prob, union_target)
    near = (udf < 1).float() * 2 + 0.25
    distance = (near * (torch.sigmoid(udf_pred) - udf).abs()).mean()
    gate_pos = ((change.numel() - change.sum()) / change.sum().clamp_min(1)).clamp(1, 20).detach()
    gate_loss = F.binary_cross_entropy_with_logits(gate_logits, change, pos_weight=gate_pos)
    identity = ((prob - current).abs() * (1 - change)).mean()
    total = bce + 0.55 * dice + 0.45 * topology + 0.20 * distance + 0.25 * gate_loss + 0.30 * identity
    values = {"total": total.item(), "bce": bce.item(), "dice": dice.item(),
              "cldice": topology.item(), "udf": distance.item(),
              "gate": gate_loss.item(), "identity": identity.item()}
    return total, values


@torch.inference_mode()
def infer(model: nn.Module, frame: Frame, device: torch.device, tile: int = 256,
          overlap: int = 48) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    h, w = frame.target.shape[1:]
    y0, y1, x0, x1 = 120, int(0.86 * h), 0, int(0.82 * w)
    accum = np.zeros((4, h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    step = tile - overlap
    ys = list(range(y0, max(y0 + 1, y1 - tile + 1), step)) + [y1 - tile]
    xs = list(range(x0, max(x0 + 1, x1 - tile + 1), step)) + [x1 - tile]
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            inp = np.concatenate((frame.current[:, y:y + tile, x:x + tile],
                                  frame.rough[y:y + tile, x:x + tile].transpose(2, 0, 1)), 0)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(torch.from_numpy(inp)[None].float().to(device))
            value = torch.sigmoid(torch.cat((out[:, :3], out[:, 4:5]), 1))[0].float().cpu().numpy()
            accum[:, y:y + tile, x:x + tile] += value
            weight[y:y + tile, x:x + tile] += 1
    valid = weight > 0
    averaged = np.zeros_like(accum)
    averaged[:, valid] = accum[:, valid] / weight[valid]
    gate = averaged[3]
    blended = frame.current.copy()
    blended[:, valid] = (gate[valid] * averaged[:3, valid]
                         + (1 - gate[valid]) * frame.current[:, valid])
    return blended, gate


def color_macro_f1(pred: np.ndarray, target: np.ndarray, tolerance: float = 2.0) -> float:
    return float(np.mean([tolerant_f1(pred[i] >= 0.5, target[i] > 0.5, tolerance)["f1_2px"]
                          for i in range(3)]))


def main() -> None:
    global DATA, CURRENT
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", choices=FRAME_MAP, default="A0006")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--patch", type=int, default=192)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--init", type=Path, default=None,
                        help="Optional external-pretraining checkpoint")
    parser.add_argument("--init-backbone-only", action="store_true",
                        help="Transfer geometry features but reset the domain-specific output head")
    parser.add_argument("--reset-color-head", action="store_true",
                        help="Keep pretrained black/UDF/gate outputs but reset red and blue logits")
    parser.add_argument("--output-root", type=Path, default=OUT,
                        help="Experiment root; keeps controls and pretrained runs separate")
    parser.add_argument("--data-root", type=Path,
                        default=WORKSPACE_ROOT / "2026.07.13",
                        help="Root containing KTK_04_246B")
    parser.add_argument("--current-root", type=Path,
                        default=PACKAGE_ROOT / "outputs" / "task_a",
                        help="Current Task-A TGA directory used as the residual input")
    args = parser.parse_args()
    DATA = args.data_root / "KTK_04_246B"
    CURRENT = args.current_root
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = {name: load_frame(name) for name in FRAME_MAP}
    train_frames = [frame for name, frame in frames.items() if name != args.holdout]
    heldout = frames[args.holdout]
    run_dir = args.output_root / f"holdout_{args.holdout}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "split.json").write_text(json.dumps({"train": [f.name for f in train_frames],
                                                     "holdout": args.holdout,
                                                     "seed": args.seed}, indent=2), encoding="utf-8")

    model = ResidualUNet().to(device)
    if args.init is not None:
        fresh_head_weight = model.head.weight.detach().clone()
        fresh_head_bias = model.head.bias.detach().clone()
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        if args.init_backbone_only:
            state = {key: value for key, value in state.items() if not key.startswith("head.")}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if set(missing) != {"head.weight", "head.bias"} or unexpected:
                raise RuntimeError(f"unexpected transfer keys: missing={missing}, unexpected={unexpected}")
        else:
            model.load_state_dict(state, strict=True)
            if args.reset_color_head:
                with torch.no_grad():
                    model.head.weight[1:3].copy_(fresh_head_weight[1:3])
                    model.head.bias[1:3].copy_(fresh_head_bias[1:3])
        print(f"loaded initialization: {args.init}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = random.Random(args.seed)
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        batch = [crop(train_frames[rng.randrange(len(train_frames))], args.patch, rng)
                 for _ in range(args.batch)]
        tensors = [torch.stack(items).to(device) for items in zip(*batch)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            output = model(tensors[0])
            loss, values = losses(output, *tensors[1:])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 25 == 0:
            values["step"] = step
            history.append(values)
            print("step", step, " ".join(f"{k}={v:.4f}" for k, v in values.items()
                                          if k not in {"step"}))

    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, run_dir / "model.pth")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    probabilities, gate = infer(model, heldout, device)
    np.savez_compressed(run_dir / f"{args.holdout}_probabilities.npz",
                        probabilities=probabilities, gate=gate)
    candidate = colorize(probabilities)
    current_rgb = colorize(heldout.current)
    target_rgb = colorize(heldout.target)
    Image.fromarray(candidate).save(run_dir / f"{args.holdout}_candidate.tga")
    Image.fromarray((gate * 255).astype(np.uint8)).save(run_dir / f"{args.holdout}_gate.png")

    current_eval = evaluate(union_bgr(heldout.current), union_bgr(heldout.target))
    candidate_eval = evaluate(cv2.cvtColor(candidate, cv2.COLOR_RGB2BGR),
                              cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR))
    current_union, candidate_union, target_union = (x.max(0) >= 0.5 for x in
                                                     (heldout.current, probabilities, heldout.target))
    metrics = {
        "split": {"train": [f.name for f in train_frames], "holdout": args.holdout},
        "current": current_eval | tolerant_f1(current_union, target_union)
        | {"color_macro_f1_2px": color_macro_f1(heldout.current, heldout.target)},
        "candidate": candidate_eval | tolerant_f1(candidate_union, target_union)
        | {"color_macro_f1_2px": color_macro_f1(probabilities, heldout.target)},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    panel = np.hstack((label(cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR), "CURRENT"),
                       label(cv2.cvtColor(candidate, cv2.COLOR_RGB2BGR), "RESIDUAL MODEL"),
                       label(cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR), "REFERENCE")))
    cv2.imencode(".png", panel)[1].tofile(run_dir / f"{args.holdout}_comparison.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
