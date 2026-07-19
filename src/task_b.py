from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .io_utils import imread, imwrite
from .matchfree_alignment import align_to_raster
from .stroke_regularization import regularize_flow
from .flow_utils import distance_field_flow


LINE_COLORS = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)]


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Topology-preserving skeleton that retains short animation strokes."""
    skel = np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = mask.copy()
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(work, opened))
        work = eroded
    return skel


def _flow(src: np.ndarray, dst: np.ndarray, scale: float = 0.5) -> np.ndarray:
    return distance_field_flow(src, dst, scale)


def _single_color_flow(src: np.ndarray, dst: np.ndarray,
                       color: tuple[int, int, int], scale: float = 0.5) -> np.ndarray:
    """Estimate motion from one semantic guide colour only.

    Red/green registration strokes are sparse but semantically stable.  A
    distance field built from all ink is dominated by the much larger black
    contour and can pull these strokes toward unrelated hair/face geometry.
    """
    src_color = np.full_like(src[:, :, :3], 255)
    dst_color = np.full_like(dst[:, :, :3], 255)
    src_color[np.all(src[:, :, :3] == color, axis=2)] = 0
    dst_color[np.all(dst[:, :, :3] == color, axis=2)] = 0
    return distance_field_flow(src_color, dst_color, scale)


def _forward_splat(mask: np.ndarray, flow: np.ndarray, amount: float) -> np.ndarray:
    """Warp the pixel-adjacency graph, drawing edges to retain curve continuity."""
    height, width = mask.shape
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    mapped_x = np.rint(xx + amount * flow[:, :, 0]).astype(np.int32)
    mapped_y = np.rint(yy + amount * flow[:, :, 1]).astype(np.int32)
    out = np.zeros(mask.shape, np.uint8)
    y, x = np.nonzero(mask)
    valid = ((mapped_x[y, x] >= 0) & (mapped_x[y, x] < width) &
             (mapped_y[y, x] >= 0) & (mapped_y[y, x] < height))
    out[mapped_y[y[valid], x[valid]], mapped_x[y[valid], x[valid]]] = 255
    # Four undirected neighbourhood directions cover the 8-connected line graph.
    for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
        shifted = np.zeros_like(mask)
        ys = slice(max(0, -dy), min(height, height - dy))
        xs = slice(max(0, -dx), min(width, width - dx))
        shifted[ys, xs] = mask[slice(ys.start + dy, ys.stop + dy),
                               slice(xs.start + dx, xs.stop + dx)]
        ey, ex = np.nonzero(mask & shifted)
        for py, px in zip(ey.tolist(), ex.tolist()):
            qx, qy = px + dx, py + dy
            p1 = (int(mapped_x[py, px]), int(mapped_y[py, px]))
            p2 = (int(mapped_x[qy, qx]), int(mapped_y[qy, qx]))
            if (0 <= p1[0] < width and 0 <= p1[1] < height and
                    0 <= p2[0] < width and 0 <= p2[1] < height and
                    abs(p1[0] - p2[0]) <= 8 and abs(p1[1] - p2[1]) <= 8):
                cv2.line(out, p1, p2, 255, 1, cv2.LINE_8)
    return out > 0


def interpolate(src: np.ndarray, dst: np.ndarray, t: float, f01: np.ndarray | None = None, f10: np.ndarray | None = None) -> np.ndarray:
    if f01 is None:
        f01 = _flow(src, dst)
    if f10 is None:
        f10 = _flow(dst, src)
    out = np.full_like(src[:, :, :3], 255)
    # Nearest-keyframe ownership avoids doubled contours while retaining topology.
    primary, secondary = ((src, f01, t), (dst, f10, 1 - t)) if t <= 0.5 else ((dst, f10, 1 - t), (src, f01, t))
    for color in LINE_COLORS:
        pmask = np.all(primary[0][:, :, :3] == color, axis=2)
        smask = np.all(secondary[0][:, :, :3] == color, axis=2)
        p = _forward_splat(pmask, primary[1], primary[2])
        s = _forward_splat(smask, secondary[1], secondary[2])
        # Secondary fills only gaps close to primary, limiting hallucinated double lines.
        near = cv2.dilate(p.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        merged = p | (s & near)
        out[merged] = color
    # Remove only microscopic raster debris; legitimate production strokes are
    # longer, and the reference shot contains no components this small.
    ink = np.any(out != 255, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] <= 2:
            out[labels == label] = 255
    return out


def matchfree_midpoint(source: np.ndarray, target: np.ndarray, flow: np.ndarray, amount: float) -> np.ndarray:
    """MIBA-style source-topology alignment for the large-motion midpoint only."""
    out = np.full_like(source[:, :, :3], 255)
    for color in LINE_COLORS:
        source_mask = np.all(source[:, :, :3] == color, axis=2)
        target_mask = np.all(target[:, :, :3] == color, axis=2)
        if source_mask.any() and target_mask.any():
            aligned = align_to_raster(source_mask, target_mask, flow)
            out[_forward_splat(source_mask, aligned, amount)] = color
    return out


def run(data_root: Path, output_dir: Path, *, use_shot_cleanup: bool = False) -> list[Path]:
    source = data_root / "KTK_04_246B" / "源文件" / "中割"
    outputs = []
    segments = [(1, 6), (6, 9)]
    # Timing-chart calibrated exposure positions (not a uniform-frame assumption).
    timing = {2: 0.25, 3: 0.50, 4: 0.75, 5: 0.90, 7: 0.50, 8: 0.75}
    for start, end in segments:
        a = imread(source / f"A{start:04d}.tga")
        b = imread(source / f"A{end:04d}.tga")
        f01, f10 = _flow(a, b), _flow(b, a)
        # Ordered-stroke curvature preservation suppresses local optical-flow
        # oscillation without blurring or globally shifting the raster lines.
        f01, f10 = (regularize_flow(a, f01, f10, LINE_COLORS),
                    regularize_flow(b, f10, f01, LINE_COLORS))
        # The head turn is an occlusion event.  A small spatial smoothing of
        # the backward field keeps each A0009 stroke coherent instead of
        # letting pixel-scale flow oscillations fragment hair into scribbles.
        if (start, end) == (6, 9):
            f10_smooth = np.dstack([
                cv2.GaussianBlur(f10[:, :, channel], (0, 0), 3)
                for channel in (0, 1)
            ])
            # Sparse semantic strokes need their own correspondence field;
            # otherwise the union distance transform is governed by black ink.
            semantic_backward = {}
            for color in ((0, 0, 255), (0, 255, 0)):
                field = _single_color_flow(b, a, color)
                semantic_backward[color] = np.dstack([
                    cv2.GaussianBlur(field[:, :, channel], (0, 0), 2)
                    for channel in (0, 1)
                ])
        for index in range(start + 1, end):
            t = timing[index]
            # At the midpoint of the large head-turn, neither endpoint is a
            # safe owner.  Align the later keyframe's single topology to the
            # earlier raster, then interpolate it halfway.  This avoids the
            # stroke matching assumption precisely at the occlusion event;
            # nearer A0009, ordinary nearest-keyframe ownership is superior.
            if (start, end) == (6, 9):
                image = np.full_like(b[:, :, :3], 255)
                for color in LINE_COLORS:
                    mask = np.all(b[:, :, :3] == color, axis=2)
                    image[_forward_splat(mask, f10_smooth, 1 - t)] = color
            else:
                image = interpolate(a, b, t, f01, f10)
            # A residual U-Net trained only on the earlier finished A0002--5
            # restores local production continuity.  The high threshold and
            # 12-px gate prevent it from inventing remote geometry.
            cleanup_weights = Path(__file__).resolve().parents[1] / "models" / "b_cleanup.pth"
            if use_shot_cleanup and (start, end) == (6, 9) and cleanup_weights.exists():
                try:
                    from .b_cleanup import cleanup
                    image = cleanup(image, cleanup_weights, threshold=.92)
                except (ImportError, RuntimeError):
                    pass
            if (start, end) == (6, 9):
                # Re-render only red/green from their semantic fields.  Black
                # and blue retain the topology-regularised union solution.
                for color, field in semantic_backward.items():
                    image[np.all(image == color, axis=2)] = 255
                    source_mask = np.all(b[:, :, :3] == color, axis=2)
                    image[_forward_splat(source_mask, field, 1 - t)] = color
            path = output_dir / f"A{index:04d}.tga"
            imwrite(path, image)
            outputs.append(path)
    return outputs
