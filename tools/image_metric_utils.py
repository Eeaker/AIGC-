"""Small dependency-free helpers shared by experiment scripts."""
from __future__ import annotations

import cv2
import numpy as np


def tolerant_f1(pred: np.ndarray, ref: np.ndarray,
                tolerance: float = 2.0) -> dict[str, float]:
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
