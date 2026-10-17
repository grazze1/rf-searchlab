"""Finite optical coverage and persistent queue, independent of RF orientation."""

from functools import lru_cache
from math import floor

import numpy as np

from rf_searchlab.algorithms.route.tour import open_tour
from rf_searchlab.core.belief import wire_point
from rf_searchlab.core.domain import GeometryError, Point
from rf_searchlab.core.geometry import distance, max_distance


@lru_cache(maxsize=8192)
def grid(poly):
    if not poly:
        raise GeometryError("Cannot cover an empty localization set")
    xs, ys = [p.x_m for p in poly], [p.y_m for p in poly]
    ix, iy = np.meshgrid(
        np.arange(floor(min(xs) / 25), floor(max(xs) / 25) + 1),
        np.arange(floor(min(ys) / 25), floor(max(ys) / 25) + 1),
        indexing="ij",
    )
    centers = np.column_stack((ix.ravel() * 25 + 12.5, iy.ravel() * 25 + 12.5))
    vertices = np.column_stack((xs, ys))
    keep = np.ones(len(centers), dtype=bool)
    # Conservative separating-axis test: only strictly separated cells are removed.
    # Axis projections use an outward length-scaled cushion; uncertain cells are retained.
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        axis = np.array([a[1] - b[1], b[0] - a[0]])
        norm = np.linalg.norm(axis)
        if norm == 0:
            continue
        pp = vertices @ axis
        cp = centers @ axis
        spread = 12.5 * np.sum(np.abs(axis))
        pad = norm * 1e-6
        keep &= (cp + spread >= pp.min() - pad) & (cp - spread <= pp.max() + pad)
    cells = []
    for x, y in centers[keep]:
        p = wire_point(Point(float(x), float(y)))
        vv = tuple(
            Point(float(x + dx), float(y + dy))
            for dx, dy in ((-12.5, -12.5), (12.5, -12.5), (12.5, 12.5), (-12.5, 12.5))
        )
        if max_distance(p, vv) > 19.95:
            raise GeometryError("Optical cell not covered by one serialized clear circle")
        cells.append(p)
    if not cells:
        raise GeometryError("Finite optical cover is empty")
    return tuple(cells)


def ordered_cover(poly, current):
    points = grid(poly)
    # Large optical covers use deterministic nearest-neighbour, avoiding unbounded local search.
    # The complete queue, not an optimistically truncated prefix, is priced/executed.
    if len(points) <= 9:
        length, order = open_tour((current.x_m, current.y_m), tuple((p.x_m, p.y_m) for p in points))
        return tuple(points[i] for i in order), length
    remaining = set(points)
    result, length, prev = [], 0.0, current
    while remaining:
        p = min(remaining, key=lambda p: (distance(prev, p), p.x_m, p.y_m))
        length += distance(prev, p)
        result.append(p)
        remaining.remove(p)
        prev = p
    return tuple(result), length


@lru_cache(maxsize=16384)
def service_upper(poly, current):
    points, length = ordered_cover(poly, current)
    return length / 5 + 3 * (len(points) - 1) + 5, points


def remaining_upper(b, current):
    if b.optical:
        points = b.optical[b.optical_index :]
        if not points:
            raise GeometryError("Optical queue exhausted without successful clear")
        prev, length = current, 0.0
        for p in points:
            length += distance(prev, p)
            prev = p
        return length / 5 + 3 * (len(points) - 1) + 5, points
    return service_upper(b.polygon, current)
