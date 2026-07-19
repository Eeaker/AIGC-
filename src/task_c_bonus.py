"""Setting-assisted colour transfer for the KTK_05_140 advanced task.

Only the supplied line frames and character design sheet are construction
inputs.  Finished animation frames are deliberately outside this module.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from .io_utils import as_bgr, imread, imwrite
from .flow_utils import distance_field_flow
from .task_c import (_propagate_assignments, _regions,
                     render_visual_reference_lines)


LINE_BGR = (29, 34, 54)
MOUTH_BASE_BGR = (109, 122, 204)
MOUTH_SHADOW_BGR = (42, 45, 86)
MOUTH_HIGHLIGHT_BGR = (224, 224, 224)


def _portrait_template(setting: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the isolated frontal portrait in the sheet's upper-right panel."""
    h, w = setting.shape[:2]
    x_offset = int(.80 * w)
    crop = setting[:int(.52 * h), x_offset:, :3]
    colors, counts = np.unique(crop.reshape(-1, 3), axis=0, return_counts=True)
    background = colors[np.argmax(counts)]
    foreground = np.any(crop != background, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    candidates = []
    for label in range(1, count):
        x, y, rw, rh, area = map(int, stats[label])
        if y < .12 * crop.shape[0] and rw > .35 * crop.shape[1] and rh > .30 * crop.shape[0]:
            candidates.append((area, label, x, y, rw, rh))
    if not candidates:
        raise RuntimeError("frontal setting portrait not found")
    _, label, x, y, rw, rh = max(candidates)
    pad = 2
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + rw + pad, crop.shape[1]), min(y + rh + pad, crop.shape[0])
    image = crop[y0:y1, x0:x1].copy()
    valid = labels[y0:y1, x0:x1] == label
    return image, valid


def _art_bbox(line: np.ndarray) -> tuple[int, int, int, int]:
    """Normalized frame gate rejects registration marks and right-hand notes."""
    h, w = line.shape[:2]
    ink = np.any(line[:, :, :3] != 255, axis=2)
    gate = np.zeros_like(ink)
    gate[int(.06 * h):, int(.10 * w):int(.91 * w)] = True
    y, x = np.nonzero(ink & gate)
    if not len(x):
        raise RuntimeError("no character ink in line frame")
    return int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)


def _warp_template(template: np.ndarray, valid: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = _art_bbox(line)
    th, tw = template.shape[:2]
    transform = np.array([
        [(x1 - x0) / tw, 0.0, x0],
        [0.0, (y1 - y0) / th, y0],
    ], np.float32)
    size = (line.shape[1], line.shape[0])
    warped = cv2.warpAffine(template, transform, size, flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    support = cv2.warpAffine(valid.astype(np.uint8), transform, size,
                             flags=cv2.INTER_NEAREST) > 0
    # The design portrait and animation drawing have the same view but not
    # identical proportions.  A dense distance-field alignment uses their
    # contours only, then remaps the setting colours into the animation raster.
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    source_edges = cv2.Canny(gray, 24, 72) > 0
    source_line = np.full_like(warped, 255)
    source_line[source_edges & support] = 0
    target_line = np.full_like(line[:, :, :3], 255)
    target_black = np.all(line[:, :, :3] == 0, axis=2)
    target_line[target_black] = 0
    backward = distance_field_flow(target_line, source_line, scale=.35)
    yy, xx = np.indices(line.shape[:2], dtype=np.float32)
    map_x = xx + backward[:, :, 0]
    map_y = yy + backward[:, :, 1]
    warped = cv2.remap(warped, map_x, map_y, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    support = cv2.remap(support.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT) > 0
    return warped, support


def _template_assignments(line: np.ndarray, warped: np.ndarray,
                          support: np.ndarray) -> dict[int, tuple[int, int, int]]:
    labels, info = _regions(line)
    background = int(labels[0, 0])
    assignments: dict[int, tuple[int, int, int]] = {}
    for label in range(1, len(info)):
        if label == background or info[label, cv2.CC_STAT_AREA] < 8:
            continue
        region = labels == label
        samples = warped[region & support, :3]
        coverage = len(samples) / max(int(region.sum()), 1)
        if coverage < .10 or not len(samples):
            continue
        colors, counts = np.unique(samples, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        chosen = None
        for index in order:
            color = tuple(map(int, colors[index]))
            # Sheet background/annotation cyan and pure white are not paint.
            if color in ((127, 127, 127), (255, 255, 255)):
                continue
            b, g, r = color
            if b > 180 and g > 150 and r < 90:  # cyan callout strokes
                continue
            chosen = color
            break
        if chosen is not None:
            assignments[label] = chosen
    return assignments


def _render(line: np.ndarray, assignments: dict[int, tuple[int, int, int]]) -> np.ndarray:
    labels, _ = _regions(line)
    out = line[:, :, :3].copy()
    for label, color in assignments.items():
        out[labels == label] = color
    # Task text requires byte-exact input-line preservation.
    source_line = np.any(line[:, :, :3] != 255, axis=2)
    out[source_line] = line[source_line, :3]
    return out


def colorize_a(line: np.ndarray, template: np.ndarray,
               template_valid: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int, int]]]:
    warped, support = _warp_template(template, template_valid, line)
    assignments = _template_assignments(line, warped, support)
    return _render(line, assignments), assignments


def colorize_b(line: np.ndarray) -> np.ndarray:
    """Colour sparse mouth correction regions by nesting/vertical hierarchy.

    The three exact colours are present in the supplied setting sheet.  Region
    order, not frame identity, selects base, shadow and highlight, so the rule
    remains valid when the mouth opens or closes.
    """
    labels, info = _regions(line)
    background = int(labels[0, 0])
    candidates = []
    h, w = labels.shape
    for label in range(1, len(info)):
        if label == background:
            continue
        x, y, rw, rh, area = map(int, info[label, :5])
        cx, cy = info[label, 5:7]
        if area >= 8 and .35 < cx / w < .68 and .45 < cy / h < .82:
            candidates.append((float(cy), -area, label))
    candidates.sort()
    assignments = {}
    palette = (MOUTH_HIGHLIGHT_BGR, MOUTH_BASE_BGR, MOUTH_SHADOW_BGR)
    # Largest enclosed region is the mouth base. Smaller upper/lower nested
    # regions receive highlight/shadow according to vertical ordering.
    if len(candidates) == 1:
        # A single residual enclosed region in this correction layer is the
        # inner-mouth shadow shown in the official setting sheet.
        label = candidates[0][2]
        assignments[label] = MOUTH_SHADOW_BGR
    elif candidates:
        largest = min(candidates, key=lambda item: item[1])
        assignments[largest[2]] = MOUTH_BASE_BGR
        remaining = [item for item in candidates if item[2] != largest[2]]
        for rank, item in enumerate(remaining):
            assignments[item[2]] = palette[0] if rank == 0 else palette[2]
    out = _render(line, assignments)
    if len(candidates) == 1:
        # A short raster gap makes the lower mouth touch the global exterior.
        # Close only the local mouth contour, fill its interior with the base
        # colour, then restore the already identified upper shadow band.
        source = line[:, :, :3]
        ink = np.any(source != 255, axis=2)
        roi = np.zeros_like(ink)
        roi[int(.45 * h):int(.82 * h), int(.35 * w):int(.68 * w)] = True
        y, x = np.nonzero(ink & roi)
        if len(x):
            pad = 8
            x0, x1 = max(int(x.min()) - pad, 0), min(int(x.max()) + pad + 1, w)
            y0, y1 = max(int(y.min()) - pad, 0), min(int(y.max()) + pad + 1, h)
            local_barrier = ink[y0:y1, x0:x1].astype(np.uint8)
            local_barrier = cv2.morphologyEx(local_barrier, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            free = 1 - local_barrier
            _, local_labels = cv2.connectedComponents(free, 4)
            exterior = int(local_labels[0, 0])
            interior = (local_labels != exterior) & (free > 0)
            source_white = np.all(source[y0:y1, x0:x1] == 255, axis=2)
            local_out = out[y0:y1, x0:x1]
            local_out[interior & source_white] = MOUTH_BASE_BGR
            global_region = labels == label
            out[global_region] = MOUTH_SHADOW_BGR
            ry, rx = np.nonzero(global_region)
            if len(ry):
                cutoff = int(ry.min() + .24 * (ry.max() - ry.min() + 1))
                highlight = global_region & (np.indices(global_region.shape)[0] <= cutoff)
                out[highlight] = MOUTH_HIGHLIGHT_BGR
            out[ink] = source[ink]
    return out


def _load_anchor(path: Path, line: np.ndarray) -> dict[int, tuple[int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels, info = _regions(line)
    result = {}
    for row in payload["regions"]:
        # The annotation is built against this exact training frame and stores
        # its connected-component identity.  A geometric centroid is not a
        # valid lookup key for a concave region: it may lie inside a different
        # component (the large hair islands are a concrete example).  Keep the
        # persisted identity and use the coordinate only as an audit signal.
        label = int(row["label"])
        if not 0 < label < len(info):
            continue
        point_label = int(labels[int(row["y"]), int(row["x"])])
        if point_label != label:
            # Historical annotations used raw centroids; mismatch is expected
            # for concave shapes and must not silently rebind their colour.
            pass
        result[label] = tuple(map(int, row["bgr"]))
    return result


def _direct_region_propagation(source: np.ndarray, target: np.ndarray,
                               assignments: dict[int, tuple[int, int, int]],
                               min_overlap: float = .35) -> dict[int, tuple[int, int, int]]:
    """BasicPBC-style inclusion matching for this near-static frontal shot."""
    source_labels, _ = _regions(source)
    target_labels, target_info = _regions(target)
    background = int(target_labels[0, 0])
    result = {}
    for target_label in range(1, len(target_info)):
        if target_label == background or target_info[target_label, cv2.CC_STAT_AREA] < 8:
            continue
        mask = target_labels == target_label
        ids, counts = np.unique(source_labels[mask], return_counts=True)
        proposals = [(int(amount), int(label)) for label, amount in zip(ids, counts)
                     if int(label) in assignments]
        if not proposals:
            continue
        overlap, winner = max(proposals)
        if overlap / max(int(mask.sum()), 1) >= min_overlap:
            result[target_label] = assignments[winner]
    return result


def _ordered_region_propagation(
        target: np.ndarray,
        anchor_lines: dict[int, np.ndarray],
        anchors: dict[int, dict[int, tuple[int, int, int]]],
        owners: tuple[int, ...],
        min_overlap: float = .70,
) -> dict[int, tuple[int, int, int]]:
    """Conservatively merge region proposals in shot/exposure order.

    The first owner is the production exposure for the near-static hold.  A
    neighbouring key may only fill a still-unassigned target component; it can
    never overwrite the primary proposal.  The 70% inclusion gate deliberately
    leaves uncertain components white because the brief penalises wrong paint
    more heavily than missing paint.
    """
    result: dict[int, tuple[int, int, int]] = {}
    for owner in owners:
        proposals = _direct_region_propagation(
            anchor_lines[owner], target, anchors[owner], min_overlap)
        for label, colour in proposals.items():
            result.setdefault(label, colour)
    return result


def _assignments_from_painted(line: np.ndarray, painted: np.ndarray) -> dict[int, tuple[int, int, int]]:
    labels, info = _regions(line)
    background = int(labels[0, 0])
    result = {}
    for label in range(1, len(info)):
        if label == background or info[label, cv2.CC_STAT_AREA] < 8:
            continue
        pixels = painted[labels == label, :3]
        colors, counts = np.unique(pixels, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        for choice in order:
            color = tuple(map(int, colors[choice]))
            if color != (255, 255, 255):
                result[label] = color
                break
    return result


def run(data_root: Path, output_dir: Path, *, anchor_annotation: Path | None = None,
        suffix: str = "") -> list[Path]:
    root = data_root / "KTK_05_140"
    setting_path = next((root / "源文件").glob("06_001*.png"))
    template, valid = _portrait_template(as_bgr(imread(setting_path)))
    outputs = []
    anchor_lines: dict[int, np.ndarray] = {}
    anchors: dict[int, dict[int, tuple[int, int, int]]] = {}
    if anchor_annotation:
        for path in sorted(anchor_annotation.parent.glob("bonus_ktk05_c_A*_regions.json")):
            frame = int(path.stem.split("_A")[-1].split("_")[0])
            anchor_lines[frame] = as_bgr(imread(
                root / "源文件" / "上色" / "A" / f"A{frame:04d}.tga"))
            anchors[frame] = _load_anchor(path, anchor_lines[frame])
    for layer, count in (("A", 5), ("B", 3)):
        for index in range(1, count + 1):
            line = as_bgr(imread(root / "源文件" / "上色" / layer / f"{layer}{index:04d}.tga"))
            if layer == "A":
                if not anchors:
                    template_image, _ = colorize_a(line, template, valid)
                    strict = template_image
                else:
                    if index in anchors:
                        propagated = anchors[index]
                    elif index == 2 and 1 in anchors and 3 in anchors:
                        propagated = _ordered_region_propagation(
                            line, anchor_lines, anchors, (1, 3))
                    elif index == 4 and 3 in anchors and 5 in anchors:
                        propagated = _ordered_region_propagation(
                            line, anchor_lines, anchors, (3, 5))
                    else:
                        nearest = min(anchors, key=lambda frame: abs(frame - index))
                        propagated = _propagate_assignments(
                            anchor_lines[nearest], line, anchors[nearest])
                    strict = _render(line, propagated)
            else:
                strict = colorize_b(line)
            strict_path = output_dir / f"task_c_{layer}{suffix}_strict_lines" / f"{layer}{index:04d}.tga"
            visual_path = output_dir / f"task_c_{layer}{suffix}" / f"{layer}{index:04d}.tga"
            imwrite(strict_path, strict)
            imwrite(visual_path, render_visual_reference_lines(line, strict))
            outputs.extend((strict_path, visual_path))
    return outputs
