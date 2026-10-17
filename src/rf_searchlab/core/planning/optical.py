"""Finite rotated 25m grid coverage, persistent cell identity and actual route costs."""

from math import cos, floor, sin

from rf_searchlab.core.belief import wire_point
from rf_searchlab.core.domain import GeometryError, OpticalCell, OpticalPlan, Point, Polygon
from rf_searchlab.core.geometry import HalfPlane, clip, distance, max_distance, mec


def transform(p: Point, origin: Point, angle: float, inverse: bool = False) -> Point:
    """Rigid world/grid transform; inverse maps grid coordinates back to world."""
    c, s = cos(angle), sin(angle)
    if inverse:
        return Point(origin.x_m + c * p.x_m - s * p.y_m, origin.y_m + s * p.x_m + c * p.y_m)
    dx, dy = p.x_m - origin.x_m, p.y_m - origin.y_m
    return Point(c * dx + s * dy, -s * dx + c * dy)


def optical_plan(poly: Polygon, current: Point, angle: float, version: int) -> OpticalPlan:
    """Construct all intersecting grid cells, including boundary-only intersections.

    The fixed transform, cell IDs, snake ordering and accepted progress are persisted.
    """
    if not poly:
        raise GeometryError("Cannot cover empty actual set")
    origin = Point(0, 0)
    local = tuple(transform(p, origin, angle) for p in poly)
    xmin, xmax = min(p.x_m for p in local), max(p.x_m for p in local)
    ymin, ymax = min(p.y_m for p in local), max(p.y_m for p in local)
    cells: list[OpticalCell] = []
    for j in range(floor((ymin - 1e-6) / 25), floor((ymax + 1e-6) / 25) + 1):
        row = []
        for i in range(floor((xmin - 1e-6) / 25), floor((xmax + 1e-6) / 25) + 1):
            piece = local
            for hp in (
                HalfPlane(1, 0, (i + 1) * 25),
                HalfPlane(-1, 0, -i * 25),
                HalfPlane(0, 1, (j + 1) * 25),
                HalfPlane(0, -1, -j * 25),
            ):
                piece = clip(piece, hp)
                if not piece:
                    break
            if piece:
                center = wire_point(transform(Point((i + 0.5) * 25, (j + 0.5) * 25), origin, angle, True))
                row.append(OpticalCell(i, j, center))
        cells.extend(row if j % 2 == 0 else reversed(row))
    if not cells:
        raise GeometryError("Nonempty set yielded no optical cells")
    if distance(current, cells[-1].center) < distance(current, cells[0].center):
        cells.reverse()
    return OpticalPlan(tuple(cells), angle, origin, version)


def route_upper(plan: OpticalPlan, current: Point) -> float:
    """Cost L/5+3K+2 of this concrete remaining route, assuming source still present."""
    remaining = plan.cells[plan.index :]
    if not remaining:
        raise GeometryError("Optical route exhausted without success")
    points = (current,) + tuple(c.center for c in remaining)
    length = sum(distance(a, b) for a, b in zip(points, points[1:]))
    return length / 5 + 3 * len(remaining) + 2


def clear_position(poly: Polygon, current: Point, margin: float) -> Point | None:
    """Prefer current position, otherwise verified MEC center; no nearest-point claim."""
    if max_distance(current, poly) <= 20 - margin:
        return current
    center = wire_point(mec(poly).center)
    return center if max_distance(center, poly) <= 20 - margin else None


def completion_cost(poly: Polygon, current: Point, angle: float, margin: float, version: int) -> float:
    """Numerically verified cost of an explicitly constructed single-source tail."""
    p = clear_position(poly, current, margin)
    return (
        distance(p, current) / 5 + 5
        if p is not None
        else route_upper(optical_plan(poly, current, angle, version), current)
    )
