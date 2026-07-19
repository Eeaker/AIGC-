from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .io_utils import imread, imwrite


GREEN_BGR = np.array((0, 255, 0), dtype=np.uint8)


def _thin(mask: np.ndarray) -> np.ndarray:
    binary = (mask.astype(np.uint8) * 255)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary) > 0
    # Morphological skeleton fallback (OpenCV builds without ximgproc.thinning).
    skel = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = binary.copy()
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(work, opened))
        work = eroded
    return skel > 0


def _repair_short_gaps(mask: np.ndarray, max_addition_area: int = 80) -> np.ndarray:
    """Close short gaps without square-kernel corner blobs.

    An ellipse supplies the conservative repair.  Small square-only additions
    are accepted only when they touch at least two locally separate stroke
    pieces, which preserves closure without thickening every near crossing.
    """
    binary = mask.astype(np.uint8)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    repaired = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ellipse) > 0
    square = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8)) > 0
    candidates = (square & ~repaired).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    halo_kernel = np.ones((3, 3), np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] > max_addition_area:
            continue
        component = (labels == label).astype(np.uint8)
        contact = (cv2.dilate(component, halo_kernel) > 0) & repaired
        contact_count, _ = cv2.connectedComponents(contact.astype(np.uint8), 8)
        if contact_count >= 3:  # background plus two locally separate contacts
            repaired |= component > 0
    return repaired


def _color_seeds(rough_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(rough_bgr, cv2.COLOR_BGR2LAB)
    smooth = cv2.bilateralFilter(lab, 7, 18, 5)
    edge_a = cv2.Canny(smooth[:, :, 1], 10, 32, L2gradient=True) > 0
    edge_b = cv2.Canny(smooth[:, :, 2], 10, 32, L2gradient=True) > 0
    gray = cv2.cvtColor(rough_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(rough_bgr.astype(np.int16))
    saturation = np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r])
    red_ink = (r - np.maximum(g, b) > 22) & (saturation > 30) & (r < 250)
    cyan_ink = (np.minimum(b, g) - r > 20) & (saturation > 30)
    green_ink = (g - np.maximum(b, r) > 42) & (saturation > 45) & (g > 105)
    red = red_ink | (edge_a & (r > b + 10))
    blue = cyan_ink | (edge_b & (b > r + 10))
    purple = (r > g + 8) & (b > g + 8) & (gray > 70)
    return red, blue, green_ink, purple


def restore_production_green(strict_bgr: np.ndarray, rough_bgr: np.ndarray) -> np.ndarray:
    """Restore the production green centreline from the rough drawing.

    The written prompt lists four colours, while all three supplied clean
    references contain a fifth green semantic line around the eye/face.  The
    saturated source-green stroke is authoritative.  Unlike the black cleanup
    geometry, the production reference uses this short semantic stroke at a
    mixed one/two-pixel width.  Skeletonising it unconditionally made only
    51--60% as many green pixels as the clean line.  Keep the high-confidence
    source core instead; the stronger dominance threshold removes JPEG fringe
    pixels without consulting the finished frame.
    """
    values = rough_bgr.astype(np.int16)
    blue, green, red = cv2.split(values)
    dominance = green - np.maximum(blue, red)
    saturation = values.max(axis=2) - values.min(axis=2)
    gray = cv2.cvtColor(rough_bgr, cv2.COLOR_BGR2GRAY)
    green_centreline = (dominance > 70) & (saturation > 35) & (gray < 220)
    out = strict_bgr[:, :, :3].copy()
    out[green_centreline] = GREEN_BGR
    return out


def clean_neural(rough_bgr: np.ndarray, white_probability: np.ndarray,
                 threshold: float = 0.30, *, preserve_green: bool = False) -> np.ndarray:
    """Learned stroke normalisation plus explicit production-colour semantics."""
    line = white_probability < threshold
    red_seed, blue_seed, green_seed, purple = _color_seeds(rough_bgr)
    line &= ~(cv2.dilate(purple.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    # Registration marks and handwritten production metadata are not character art.
    line[:120] = False
    line[100:, int(0.82 * line.shape[1]):] = False
    line = _thin(line)
    repaired = _repair_short_gaps(line)

    red_near = cv2.dilate(red_seed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    blue_near = cv2.dilate(blue_seed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    green_near = cv2.dilate(green_seed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    green = repaired & green_near & ~red_near & ~blue_near & preserve_green
    red = repaired & red_near & ~blue_near & ~green_near
    blue = repaired & blue_near & ~red_near & ~green_near
    black = repaired & ~red & ~blue & ~green
    out = np.full_like(rough_bgr, 255)
    out[black] = (0, 0, 0)
    out[blue] = (255, 0, 0)
    out[red] = (0, 0, 255)
    out[green] = (0, 255, 0)
    # Cropped strokes at the lower frame boundary lack two-sided context for
    # the neural detector.  Preserve the high-recall rule result there so the
    # cut-off clothing regions remain fillable.
    boundary_start = int(0.86 * out.shape[0])
    rule = clean_rough(rough_bgr, preserve_green=preserve_green)
    out[boundary_start:] = rule[boundary_start:]
    return out


def clean_rough(rough_bgr: np.ndarray, *, preserve_green: bool = False) -> np.ndarray:
    """Rule baseline: chroma-aware edges, semantic line recovery, thinning."""
    lab = cv2.cvtColor(rough_bgr, cv2.COLOR_BGR2LAB)
    smooth = cv2.bilateralFilter(lab, 7, 18, 5)
    # Recover both dark drawn strokes and boundaries between flat rough-color regions.
    gray = cv2.cvtColor(rough_bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < 150
    edge_l = cv2.Canny(smooth[:, :, 0], 12, 38, L2gradient=True) > 0
    edge_a = cv2.Canny(smooth[:, :, 1], 10, 32, L2gradient=True) > 0
    edge_b = cv2.Canny(smooth[:, :, 2], 10, 32, L2gradient=True) > 0
    # Chroma edges are handled below by dedicated red/blue masks.  Mixing them
    # into black structure creates double contours around coloured guide fills.
    structure = _thin(dark | edge_l)

    b, g, r = cv2.split(rough_bgr.astype(np.int16))
    saturation = np.maximum.reduce([b, g, r]) - np.minimum.reduce([b, g, r])
    red_ink = (r - np.maximum(g, b) > 22) & (saturation > 30) & (r < 250)
    cyan_ink = (np.minimum(b, g) - r > 20) & (saturation > 30)
    green_ink = (g - np.maximum(b, r) > 42) & (saturation > 45) & (g > 105)
    purple_aux = (r > g + 8) & (b > g + 8) & (gray > 70)
    # Only keep thin boundaries for pastel guide fills; saturated strokes survive whole.
    red = _thin(red_ink | (edge_a & (r > b + 10)))
    blue = _thin(cyan_ink | (edge_b & (b > r + 10)))
    green = _thin(green_ink) if preserve_green else np.zeros_like(green_ink)
    structure &= ~(cv2.dilate((red | blue | green).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    structure &= ~(cv2.dilate(purple_aux.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    # Handwritten rough note at frame right is production metadata, not character geometry.
    structure[100:, int(0.82 * structure.shape[1]):] = False
    # Repair short JPEG/rough-line gaps after thinning; connector pixels are black.
    combined = (structure | blue | red | green).astype(np.uint8)
    repaired = _repair_short_gaps(combined)
    structure |= repaired & ~blue & ~red

    out = np.full_like(rough_bgr, 255)
    out[structure] = (0, 0, 0)
    out[blue] = (255, 0, 0)
    out[red] = (0, 0, 255)
    out[green] = (0, 255, 0)
    return out


def run(data_root: Path, output_dir: Path, *, preserve_green: bool = True,
        use_noncommercial_weight: bool = False) -> list[Path]:
    source = data_root / "KTK_04_246B" / "源文件" / "描原"
    outputs = []
    mapping = {"A1.jpg": "A0001.tga", "A2.jpg": "A0006.tga", "A3.jpg": "A0009.tga"}
    weights = Path(__file__).resolve().parents[1] / "models" / "line_thinning_siggraph2018.pth"
    for src_name, out_name in mapping.items():
        rough = imread(source / src_name)[:, :, :3]
        try:
            if not use_noncommercial_weight:
                raise FileNotFoundError("non-commercial research weight disabled")
            from .neural_thinning import predict_white_probability
            probability = predict_white_probability(rough, weights)
            out = clean_neural(rough, probability, preserve_green=preserve_green)
        except (FileNotFoundError, RuntimeError):
            out = clean_rough(rough, preserve_green=preserve_green)
        path = output_dir / out_name
        imwrite(path, out)
        outputs.append(path)
    return outputs
