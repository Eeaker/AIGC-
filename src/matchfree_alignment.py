"""Raster-distance alignment for one fixed source line topology.

This is a compact, dependency-free analogue of MIBA's vector-to-raster
alignment stage: optical flow initializes source vertices, then the source
topology is attracted to the target raster distance field.  It intentionally
does not infer any source/target stroke correspondence.
"""
from __future__ import annotations

import cv2
import numpy as np


def _sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = np.empty(len(x), np.float32)
    field = field.astype(np.float32)
    for start in range(0, len(x), 30_000):
        end = min(start + 30_000, len(x))
        result[start:end] = cv2.remap(
            field, x[start:end].astype(np.float32)[:, None],
            y[start:end].astype(np.float32)[:, None], cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE).ravel()
    return result


def align_to_raster(source_mask: np.ndarray, target_mask: np.ndarray, initial_flow: np.ndarray,
                    iterations: int = 6, step: float = .65, cap: float = 4.0) -> np.ndarray:
    """Align source pixels to target ink while retaining source topology."""
    height, width = source_mask.shape
    ys, xs = np.nonzero(source_mask)
    base_x, base_y = xs.astype(np.float32), ys.astype(np.float32)
    qx, qy = base_x + initial_flow[ys, xs, 0], base_y + initial_flow[ys, xs, 1]
    distance = cv2.distanceTransform((~target_mask).astype(np.uint8), cv2.DIST_L2, 5)
    gx = cv2.Sobel(distance, cv2.CV_32F, 1, 0, ksize=3) * .5
    gy = cv2.Sobel(distance, cv2.CV_32F, 0, 1, ksize=3) * .5
    for index in range(iterations):
        d, dx, dy = _sample(distance, qx, qy), _sample(gx, qx, qy), _sample(gy, qx, qy)
        gain = step * (1 - .35 * index / max(iterations, 1)) * np.minimum(d, cap) / np.maximum(np.hypot(dx, dy), 1e-3)
        qx, qy = np.clip(qx - gain * dx, 0, width - 1), np.clip(qy - gain * dy, 0, height - 1)
    result = initial_flow.copy()
    result[ys, xs, 0], result[ys, xs, 1] = qx - base_x, qy - base_y
    return result
