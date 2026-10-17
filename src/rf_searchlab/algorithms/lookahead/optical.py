"""Individually certified rectangular cells; no union-of-disks exclusion shortcut."""

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from math import floor

import numpy as np

from rf_searchlab.algorithms.directional.optical import grid
from rf_searchlab.algorithms.route.tour import open_tour
from rf_searchlab.core.belief import wire_point
from rf_searchlab.core.domain import GeometryError, Point
from rf_searchlab.core.geometry import diameter, max_distance


@dataclass(frozen=True)
class Cell:
    center: Point
    corners: tuple[Point, ...]


@dataclass(frozen=True)
class Cover:
    channel: int
    queue_id: str
    cells: tuple[Cell, ...]
    cursor: int = 0


def disjoint(cell, poly):
    for vertices in (cell.corners, poly):
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            ux, uy = a.y_m - b.y_m, b.x_m - a.x_m
            norm = (ux * ux + uy * uy) ** 0.5
            if norm == 0:
                continue
            aa = [p.x_m * ux + p.y_m * uy for p in cell.corners]
            bb = [p.x_m * ux + p.y_m * uy for p in poly]
            if max(aa) < min(bb) - norm * 1e-6 or max(bb) < min(aa) - norm * 1e-6:
                return True
    return False


def excluded(cell, belief):
    if any(max_distance(e.position, cell.corners) <= 19.95 for e in belief.exclusions):
        return True
    return disjoint(cell, belief.polygon)


@lru_cache(maxsize=4096)
def cells_for(poly, stripes=False):
    if not stripes:
        return tuple(
            Cell(
                p,
                tuple(
                    Point(p.x_m + x, p.y_m + y)
                    for x, y in ((-12.5, -12.5), (12.5, -12.5), (12.5, 12.5), (-12.5, 12.5))
                ),
            )
            for p in grid(poly)
        )
    d, a, b = diameter(poly)
    if d < 1e-8:
        return cells_for(poly, False)
    u = np.array([(b.x_m - a.x_m) / d, (b.y_m - a.y_m) / d])
    v = np.array([-u[1], u[0]])
    vv = np.array([(p.x_m, p.y_m) for p in poly])
    x, y = vv @ u, vv @ v
    width, height = 34.4, 20.0  # half diagonal <19.95 with ample serialization margin
    xx, yy = np.meshgrid(
        np.arange(floor(x.min() / width), floor(x.max() / width) + 1),
        np.arange(floor(y.min() / height), floor(y.max() / height) + 1),
        indexing="ij",
    )
    mid = np.outer(xx.ravel() * width + width / 2, u) + np.outer(yy.ravel() * height + height / 2, v)
    keep = np.ones(len(mid), dtype=bool)
    for p, q in zip(vv, np.roll(vv, -1, axis=0)):
        axis = np.array([p[1] - q[1], q[0] - p[0]])
        norm = np.linalg.norm(axis)
        if norm == 0:
            continue
        pp, cc = vv @ axis, mid @ axis
        spread = width / 2 * abs(u @ axis) + height / 2 * abs(v @ axis)
        keep &= (cc + spread >= pp.min() - norm * 1e-6) & (cc - spread <= pp.max() + norm * 1e-6)
    result = []
    for center in mid[keep]:
        corners = tuple(
            Point(*map(float, center + dx * u + dy * v))
            for dx, dy in (
                (-width / 2, -height / 2),
                (width / 2, -height / 2),
                (width / 2, height / 2),
                (-width / 2, height / 2),
            )
        )
        point = wire_point(Point(*map(float, center)))
        if max_distance(point, corners) > 19.95:
            raise GeometryError("Serialized stripe cell lacks single-circle coverage")
        result.append(Cell(point, corners))
    if not result:
        raise GeometryError("Empty stripe cover")
    return tuple(result)


def order_cells(cells, current):
    if len(cells) <= 9:
        _, order = open_tour((current.x_m, current.y_m), tuple((c.center.x_m, c.center.y_m) for c in cells))
        return tuple(cells[i] for i in order)
    # Nearest-neighbour with vectorized distances, deterministic stable ties.
    coords = np.array([(c.center.x_m, c.center.y_m) for c in cells])
    active = np.ones(len(cells), dtype=bool)
    prev = np.array([current.x_m, current.y_m])
    result = []
    while active.any():
        dd = np.sum((coords - prev) ** 2, axis=1)
        dd[~active] = np.inf
        i = int(np.argmin(dd))
        active[i] = False
        result.append(cells[i])
        prev = coords[i]
    return tuple(result)


def make_cover(b, current, stripes):
    cells = tuple(c for c in cells_for(b.polygon, stripes) if not excluded(c, b))
    if not cells:
        raise GeometryError("Optical cover exhausted without clear success")
    cells = order_cells(cells, current)
    digest = sha256(repr((b.channel, b.version, cells)).encode()).hexdigest()[:24]
    return Cover(b.channel, digest, cells)


def remaining(cover, b):
    return tuple((i, c) for i, c in enumerate(cover.cells) if i >= cover.cursor and not excluded(c, b))


def commit_progress(state, plan):
    """Only called after accepted actions; check skipped cells from real failures/P."""
    beliefs = list(state.beliefs)
    for cover in plan.covers:
        b = beliefs[cover.channel - 1]
        if not b.optical:
            continue
        if tuple(c.center for c in cover.cells) != b.optical or not 0 <= cover.cursor <= len(cover.cells):
            raise GeometryError("Optical cell ledger disagrees with persistent wire queue")
        if any(max_distance(c.center, c.corners) > 19.95 for c in cover.cells):
            raise GeometryError("Optical cell lost its single-circle guarantee")
        if any(e.request_id not in state.applied for e in b.exclusions):
            raise GeometryError("Optical exclusion does not reference an accepted response")
        if cover.cursor > b.optical_index:
            if not all(excluded(c, b) for c in cover.cells[b.optical_index : cover.cursor]):
                raise GeometryError("Optical cursor skipped a cell without real exclusion")
            beliefs[b.channel - 1] = replace(b, optical_index=cover.cursor)
    return replace(state, beliefs=tuple(beliefs))


def cover_from_dict(value):
    return Cover(
        value["channel"],
        value["queue_id"],
        tuple(Cell(Point(**c["center"]), tuple(Point(**p) for p in c["corners"])) for c in value["cells"]),
        value["cursor"],
    )
