"""Geometry-aware flow regularisation on ordered animation strokes."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _skeleton(mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    work = mask.copy()
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, kernel)
        opened = cv2.dilate(eroded, kernel)
        result = cv2.bitwise_or(result, cv2.subtract(work, opened))
        work = eroded
    return result > 0


def _neighbours(mask: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    result = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yy, xx = y + dy, x + dx
            if not (0 <= yy < h and 0 <= xx < w and mask[yy, xx]):
                continue
            if dx and dy and (mask[y, xx] or mask[yy, x]):
                continue
            result.append((yy, xx))
    return result


def _trace_paths(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    pixels = list(zip(*np.nonzero(mask)))
    graph = {p: _neighbours(mask, *p) for p in pixels}
    visited: set[frozenset[tuple[int, int]]] = set()
    paths: list[list[tuple[int, int]]] = []

    def follow(start: tuple[int, int], nxt: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start]
        previous, current = start, nxt
        while True:
            visited.add(frozenset((previous, current)))
            path.append(current)
            candidates = [point for point in graph[current]
                          if point != previous and
                          frozenset((current, point)) not in visited]
            if len(graph[current]) != 2 or not candidates:
                break
            previous, current = current, candidates[0]
        return path

    for point in pixels:
        if len(graph[point]) == 2:
            continue
        for neighbour in graph[point]:
            if frozenset((point, neighbour)) not in visited:
                paths.append(follow(point, neighbour))
    for point in pixels:  # closed loops
        for neighbour in graph[point]:
            if frozenset((point, neighbour)) not in visited:
                paths.append(follow(point, neighbour))
    return paths


def _sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(field, x[None].astype(np.float32), y[None].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).ravel()


def _curvature_solve(displacement: np.ndarray, confidence: np.ndarray,
                     points: np.ndarray, strength: float) -> np.ndarray:
    """Solve (W + lambda D2' D2)x = Wd with the validated sparse system."""
    n = len(points)
    spacing = np.maximum(np.linalg.norm(np.diff(points, axis=0), axis=1), 1e-6)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for index in range(1, n - 1):
        left, right = spacing[index - 1], spacing[index]
        scale = 2.0 / (left + right)
        for column, value in (
            (index - 1, scale / left),
            (index, -scale * (1.0 / left + 1.0 / right)),
            (index + 1, scale / right),
        ):
            rows.append(index - 1)
            columns.append(column)
            values.append(float(value))
    second_difference = sparse.coo_matrix(
        (values, (rows, columns)), shape=(n - 2, n)
    ).tocsr()
    system = sparse.diags(confidence) + strength * (
        second_difference.T @ second_difference
    )
    return np.column_stack([
        spsolve(system, confidence * displacement[:, channel])
        for channel in range(2)
    ])


def regularize_flow(image: np.ndarray, flow: np.ndarray, reverse: np.ndarray,
                    colors: list[tuple[int, int, int]],
                    strength: float = 0.03) -> np.ndarray:
    """Suppress high-frequency flow bending while preserving stroke shape."""
    result = flow.copy()
    accumulated = np.zeros_like(flow, np.float64)
    weights = np.zeros(flow.shape[:2], np.float64)
    for color in colors:
        exact = np.all(image[:, :, :3] == color, axis=2)
        for path in _trace_paths(_skeleton(exact.astype(np.uint8) * 255)):
            if len(path) < 7:
                continue
            yy = np.asarray([point[0] for point in path], np.int32)
            xx = np.asarray([point[1] for point in path], np.int32)
            displacement = flow[yy, xx].astype(np.float64)
            mapped_x = xx + displacement[:, 0]
            mapped_y = yy + displacement[:, 1]
            back_x = _sample(reverse[:, :, 0], mapped_x, mapped_y)
            back_y = _sample(reverse[:, :, 1], mapped_x, mapped_y)
            error = np.hypot(displacement[:, 0] + back_x,
                             displacement[:, 1] + back_y)
            confidence = np.maximum(np.exp(-(error / 3.0) ** 2), 0.08)
            points = np.column_stack((xx, yy)).astype(np.float64)
            smooth = _curvature_solve(displacement, confidence, points, strength)
            for channel in range(2):
                np.add.at(accumulated[:, :, channel], (yy, xx),
                          confidence * smooth[:, channel])
            np.add.at(weights, (yy, xx), confidence)

    used = weights > 0
    result[used] = (accumulated[used] / weights[used, None]).astype(np.float32)
    # Extend skeleton motion across the same colour's 2--3 px line width.
    for color in colors:
        exact = np.all(image[:, :, :3] == color, axis=2)
        seeds = used & exact
        if not np.any(seeds):
            continue
        distance, labels = cv2.distanceTransformWithLabels(
            (~seeds).astype(np.uint8), cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros((int(labels.max()) + 1, 2), np.float32)
        sy, sx = np.nonzero(seeds)
        lookup[labels[sy, sx]] = result[sy, sx]
        fill = exact & ~seeds & (distance <= 3.0)
        result[fill] = lookup[labels[fill]]
        used |= fill
    return result
