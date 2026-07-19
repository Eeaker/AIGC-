"""Shared line-art optical-flow features without Task-B/SciPy dependencies."""
from __future__ import annotations

import cv2
import numpy as np


def distance_field_flow(src: np.ndarray, dst: np.ndarray, scale: float = 0.5) -> np.ndarray:
    def feature(image: np.ndarray) -> np.ndarray:
        nonwhite = np.any(image[:, :, :3] != 255, axis=2).astype(np.uint8)
        distance = cv2.distanceTransform(1 - nonwhite, cv2.DIST_L2, 5)
        distance = np.clip(distance, 0, 32)
        return np.rint(255 * (1 - distance / 32)).astype(np.uint8)

    a, b = feature(src), feature(dst)
    small = (int(a.shape[1] * scale), int(a.shape[0] * scale))
    a = cv2.resize(a, small, interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, small, interpolation=cv2.INTER_AREA)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    dis.setGradientDescentIterations(32)
    flow = dis.calc(a, b, None)
    return cv2.resize(flow, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR) / scale
