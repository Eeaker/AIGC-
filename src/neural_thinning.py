"""Learned line-width normalisation used by the Task-A neural path.

Architecture and weights are from Simo-Serra et al., "Real-Time Data-Driven
Interactive Rough Sketch Inking", SIGGRAPH 2018.  See models/THIRD_PARTY.md.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def predict_white_probability(image_bgr: np.ndarray, weights: Path) -> np.ndarray:
    """Return the learned probability of white background for a BGR image."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Task-A neural cleanup requires PyTorch") from exc

    class Conv(nn.Module):
        def __init__(self, inputs: int, outputs: int, kernel: int = 3,
                     padding: int = 1) -> None:
            super().__init__()
            self.conv = nn.Conv2d(inputs, outputs, kernel_size=kernel,
                                  padding=padding)

        def forward(self, value):
            return functional.relu(self.conv(value), inplace=True)

    class ThinningNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.ReplicationPad2d(4),
                Conv(1, 64, 9, 0),
                Conv(64, 64), Conv(64, 64), Conv(64, 64),
                Conv(64, 64), Conv(64, 64), Conv(64, 64), Conv(64, 64),
                nn.Conv2d(64, 1, kernel_size=3, padding=1),
            )

        def forward(self, value):
            return torch.sigmoid(self.layers(value - 0.7))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThinningNet()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device).eval()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray)[None, None].to(device)
    with torch.no_grad():
        probability = model(tensor)[0, 0].float().cpu().numpy()
    return probability
