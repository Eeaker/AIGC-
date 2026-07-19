from __future__ import annotations

import cv2
import numpy as np


FOUR_COLORS_BGR = np.array([(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255)], dtype=np.uint8)
PRODUCTION_LINE_COLORS_BGR = np.vstack([FOUR_COLORS_BGR, np.array([(0, 255, 0)], dtype=np.uint8)])
NAMED_LINE_COLORS_BGR = {
    "black": (0, 0, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
}


def line_mask(image: np.ndarray) -> np.ndarray:
    bgr = image[:, :, :3]
    return np.any(bgr != 255, axis=2)


def color_compliance(image: np.ndarray, include_green: bool = False) -> float:
    bgr = image[:, :, :3]
    palette = PRODUCTION_LINE_COLORS_BGR if include_green else FOUR_COLORS_BGR
    valid = np.any(np.all(bgr[:, :, None, :] == palette[None, None, :, :], axis=3), axis=2)
    return float(valid.mean())


def _distance_to(mask: np.ndarray) -> np.ndarray:
    # OpenCV computes distance to zero; make target pixels zero.
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def chamfer_and_f1(pred: np.ndarray, ref: np.ndarray, tolerance: float = 2.0) -> dict[str, float]:
    p, r = line_mask(pred), line_mask(ref)
    dp, dr = _distance_to(p), _distance_to(r)
    p_to_r = float(dr[p].mean()) if p.any() else float("inf")
    r_to_p = float(dp[r].mean()) if r.any() else float("inf")
    precision = float((dr[p] <= tolerance).mean()) if p.any() else 0.0
    recall = float((dp[r] <= tolerance).mean()) if r.any() else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "chamfer": (p_to_r + r_to_p) / 2,
        "pred_to_ref": p_to_r,
        "ref_to_pred": r_to_p,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def color_line_f1(pred: np.ndarray, ref: np.ndarray, *, include_green: bool = True) -> dict[str, float]:
    """Expose colour semantics that union-line F1 deliberately ignores.

    Exact/1 px/2 px scores make the tolerance sensitivity of one-pixel
    animation lines visible.  Flat keys keep saved metrics easy to diff and
    independently recompute.
    """
    rows: dict[str, float] = {}
    for name, color in NAMED_LINE_COLORS_BGR.items():
        if name == "green" and not include_green:
            continue
        p = np.all(pred[:, :, :3] == color, axis=2)
        r = np.all(ref[:, :, :3] == color, axis=2)
        if not p.any() and not r.any():
            continue
        for tolerance, suffix in ((0.0, "exact"), (1.0, "1px"), (2.0, "2px")):
            rows[f"{name}_f1_{suffix}"] = chamfer_and_f1(
                np.where(p[:, :, None], 0, 255).astype(np.uint8),
                np.where(r[:, :, None], 0, 255).astype(np.uint8),
                tolerance=tolerance,
            )["f1"]
    return rows


def union_tolerance_f1(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    return {
        "union_f1_exact": chamfer_and_f1(pred, ref, tolerance=0.0)["f1"],
        "union_f1_1px": chamfer_and_f1(pred, ref, tolerance=1.0)["f1"],
        "union_f1_2px": chamfer_and_f1(pred, ref, tolerance=2.0)["f1"],
    }


def closure_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    def enclosed(image: np.ndarray) -> np.ndarray:
        free = (~line_mask(image)).astype(np.uint8)
        _, labels = cv2.connectedComponents(free, connectivity=4)
        exterior = labels[0, 0]
        return (labels != exterior) & (free > 0)
    p, r = enclosed(pred), enclosed(ref)
    leaked = r & ~p
    return {
        "reference_enclosed_pixels": int(r.sum()),
        "predicted_enclosed_pixels": int(p.sum()),
        "reference_region_leak_rate": float(leaked.sum() / max(r.sum(), 1)),
    }


def region_color_metrics(line: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    white = np.all(line[:, :, :3] == 255, axis=2).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    bg = labels[0, 0]
    valid = (labels != bg) & (white > 0)
    pred_colored = valid & np.any(pred[:, :, :3] != 255, axis=2)
    ref_colored = valid & np.any(ref[:, :, :3] != 255, axis=2)
    correct = pred_colored & np.all(pred[:, :, :3] == ref[:, :, :3], axis=2)
    area_precision = float(correct.sum() / max(pred_colored.sum(), 1))
    coverage = float(correct.sum() / max(ref_colored.sum(), 1))
    region_total = region_pred = region_correct = 0
    for label in range(1, n):
        if label == bg or stats[label, cv2.CC_STAT_AREA] < 8:
            continue
        mask = labels == label
        ref_pixels = ref[mask, :3]
        ref_u, ref_c = np.unique(ref_pixels, axis=0, return_counts=True)
        ref_mode = ref_u[np.argmax(ref_c)]
        if np.all(ref_mode == 255):
            continue
        region_total += 1
        pred_pixels = pred[mask, :3]
        pred_u, pred_c = np.unique(pred_pixels, axis=0, return_counts=True)
        pred_mode = pred_u[np.argmax(pred_c)]
        if not np.all(pred_mode == 255):
            region_pred += 1
            region_correct += int(np.array_equal(pred_mode, ref_mode))
    source_line = line_mask(line)
    # Task C requires exact preservation, not merely avoiding erasure.
    line_changed = int(np.any(pred[source_line, :3] != line[source_line, :3], axis=1).sum())
    target = valid & np.any(ref[:, :, :3] != 255, axis=2)
    pixel_correct = target & np.all(pred[:, :, :3] == ref[:, :, :3], axis=2)
    pixel_unfilled = target & np.all(pred[:, :, :3] == 255, axis=2)
    pixel_wrong = target & ~pixel_correct & ~pixel_unfilled
    return {
        "area_precision": area_precision,
        "area_coverage": coverage,
        "region_precision": region_correct / max(region_pred, 1),
        "region_coverage": region_correct / max(region_total, 1),
        "regions_predicted": region_pred,
        "regions_total": region_total,
        "line_pixels_changed": line_changed,
        # Exact-pixel accounting exposes mixed-colour regions (especially the
        # iris) that a region-mode score can otherwise hide.
        "white_target_pixels": int(target.sum()),
        "white_target_correct_pixels": int(pixel_correct.sum()),
        "white_target_unfilled_pixels": int(pixel_unfilled.sum()),
        "white_target_wrong_color_pixels": int(pixel_wrong.sum()),
        "white_target_exact_fraction": float(pixel_correct.sum() / max(target.sum(), 1)),
    }


def motion_smoothness(frames: list[np.ndarray], times: list[float]) -> dict[str, float]:
    """Timing-aware global motion diagnostic for an inbetween sequence.

    The ink centroid is deliberately simple and deterministic.  It does not
    pretend to measure drawing quality; it catches timing reversals, stalls and
    sudden whole-character jumps that per-frame F1 cannot reveal.
    """
    if len(frames) != len(times) or len(frames) < 3:
        raise ValueError("motion_smoothness needs matching frames/times and >=3 samples")
    centers = []
    diagonal = float(np.hypot(*frames[0].shape[:2]))
    for frame in frames:
        y, x = np.nonzero(line_mask(frame))
        centers.append((float(x.mean()) / diagonal, float(y.mean()) / diagonal))
    centers_array = np.asarray(centers, np.float64)
    dt = np.diff(np.asarray(times, np.float64))[:, None]
    velocity = np.diff(centers_array, axis=0) / dt
    velocity_dt = ((np.asarray(times[2:]) - np.asarray(times[:-2])) / 2.0)[:, None]
    acceleration = np.diff(velocity, axis=0) / velocity_dt
    speed_rms = float(np.sqrt(np.mean(np.sum(velocity * velocity, axis=1))))
    acceleration_rms = float(np.sqrt(np.mean(np.sum(acceleration * acceleration, axis=1))))
    ratio = acceleration_rms / max(speed_rms, 1e-12)
    return {
        "centroid_speed_rms": speed_rms,
        "centroid_acceleration_rms": acceleration_rms,
        "centroid_acceleration_ratio": ratio,
        "smoothness_score": float(1.0 / (1.0 + ratio)),
    }
