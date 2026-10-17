"""Pure convex geometry with explicit numerical outward padding, in metres/radians."""

from dataclasses import dataclass
from itertools import combinations
from math import atan2, cos, hypot, inf, pi, sin
from random import Random

from rf_searchlab.core.domain import CircleBound, GeometryError, Point, Polygon


def distance(a: Point, b: Point) -> float:
    """Euclidean distance in metres."""
    return hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def cross(a: Point, b: Point, c: Point) -> float:
    """Signed doubled triangle area; positive for CCW."""
    return (b.x_m - a.x_m) * (c.y_m - a.y_m) - (b.y_m - a.y_m) * (c.x_m - a.x_m)


@dataclass(frozen=True)
class HalfPlane:
    nx: float
    ny: float
    b: float
    source: str = ""


def clip(poly: Polygon, hp: HalfPlane, padding: float = 1e-7) -> Polygon:
    """Clip a CCW convex polygon, segment or point against n.p <= b+padding*|n|.

    Empty input stays empty. No inward vertex simplification is performed.
    """
    if not poly:
        return ()
    norm = hypot(hp.nx, hp.ny)
    if norm == 0:
        return poly if hp.b >= 0 else ()
    bound = hp.b + padding * norm
    result: list[Point] = []
    previous = poly[-1]
    dp = hp.nx * previous.x_m + hp.ny * previous.y_m - bound
    for current in poly:
        dc = hp.nx * current.x_m + hp.ny * current.y_m - bound
        if (dc <= 0) != (dp <= 0):
            t = dp / (dp - dc)
            result.append(
                Point(
                    previous.x_m + t * (current.x_m - previous.x_m),
                    previous.y_m + t * (current.y_m - previous.y_m),
                )
            )
        if dc <= 0:
            result.append(current)
        previous, dp = current, dc
    return tuple(p for i, p in enumerate(result) if i == 0 or p != result[i - 1])


def wedge(s: Point, theta_rad: float, epsilon_rad: float, source: str = "") -> tuple[HalfPlane, HalfPlane]:
    """Return the two halfplanes of a forward wedge, including wrap across zero."""
    if not 0 < epsilon_rad < pi / 2:
        raise ValueError("Wedge half width must be in (0, pi/2)")
    lo, hi = theta_rad - epsilon_rad, theta_rad + epsilon_rad
    a, b = (sin(lo), -cos(lo)), (-sin(hi), cos(hi))
    return (
        HalfPlane(*a, a[0] * s.x_m + a[1] * s.y_m, source),
        HalfPlane(*b, b[0] * s.x_m + b[1] * s.y_m, source),
    )


def clip_wedge(poly: Polygon, s: Point, theta: float, epsilon: float, padding: float = 1e-7) -> Polygon:
    """Intersect with a wedge; empty hypothetical branches return ()."""
    for hp in wedge(s, theta, epsilon):
        poly = clip(poly, hp, padding)
    return poly


def disk_outer(center: Point, radius: float, sides: int = 256, padding: float = 1e-7) -> Polygon:
    """Circumscribed regular polygon, not a chordal/inscribed approximation."""
    if radius <= 0 or sides < 3:
        raise ValueError("Positive radius and at least three supports required")
    r = (radius + padding) / cos(pi / sides)
    return tuple(
        Point(
            center.x_m + r * cos((2 * j + 1) * pi / sides),
            center.y_m + r * sin((2 * j + 1) * pi / sides),
        )
        for j in range(sides)
    )


def clip_disk(
    poly: Polygon, center: Point, radius: float, sides: int = 256, padding: float = 1e-7
) -> Polygon:
    """Intersect with all outward supporting halfplanes of a disk."""
    for j in range(sides):
        nx, ny = cos(2 * pi * j / sides), sin(2 * pi * j / sides)
        poly = clip(poly, HalfPlane(nx, ny, radius + nx * center.x_m + ny * center.y_m), padding)
    return poly


def segment_distance(p: Point, a: Point, b: Point) -> float:
    """Distance to a closed segment; coincident endpoints are allowed."""
    dx, dy = b.x_m - a.x_m, b.y_m - a.y_m
    norm = dx * dx + dy * dy
    t = min(1.0, max(0.0, ((p.x_m - a.x_m) * dx + (p.y_m - a.y_m) * dy) / norm)) if norm else 0
    return distance(p, Point(a.x_m + t * dx, a.y_m + t * dy))


def contains(poly: Polygon, p: Point, tolerance: float = 1e-6) -> bool:
    """Numerical membership of a CCW convex set, including degenerate sets."""
    if not poly:
        return False
    if len(poly) == 1:
        return distance(p, poly[0]) <= tolerance
    area = sum(cross(poly[0], a, b) for a, b in zip(poly, poly[1:]))
    if abs(area) <= 1e-12:
        return min(segment_distance(p, a, b) for a, b in zip(poly, poly[1:] + poly[:1])) <= tolerance
    return all(
        cross(a, b, p) >= -tolerance * max(1.0, distance(a, b)) for a, b in zip(poly, poly[1:] + poly[:1])
    )


def polygon_distance(p: Point, poly: Polygon) -> float:
    """True point-to-edge distance, zero inside; infinity for empty sets."""
    if not poly:
        return inf
    if contains(poly, p, 0):
        return 0.0
    return min(segment_distance(p, a, b) for a, b in zip(poly, poly[1:] + poly[:1]))


def max_distance(p: Point, poly: Polygon) -> float:
    """Maximum vertex distance; empty sets must never produce clearance certificates."""
    if not poly:
        raise GeometryError("Empty set has no enclosing-circle certificate")
    return max(distance(p, v) for v in poly)


def diameter(poly: Polygon) -> tuple[float, Point, Point]:
    """Brute-force convex diameter and endpoints; reject empty input."""
    if not poly:
        raise GeometryError("Empty diameter")
    a, b = max(((a, b) for a in poly for b in poly), key=lambda pair: distance(*pair))
    return distance(a, b), a, b


def same_diameter_circle(poly: Polygon) -> bool:
    """Check the unique midpoint circle of a farthest pair against all vertices."""
    d, a, b = diameter(poly)
    return max_distance(Point((a.x_m + b.x_m) / 2, (a.y_m + b.y_m) / 2), poly) <= d / 2 + 1e-8


def pair_circle(a: Point, b: Point) -> CircleBound:
    """Circle supported by two endpoints."""
    return CircleBound(Point((a.x_m + b.x_m) / 2, (a.y_m + b.y_m) / 2), distance(a, b) / 2)


def triple_circle(a: Point, b: Point, c: Point) -> CircleBound | None:
    """Circumcircle using translated coordinates; None for numerical collinearity."""
    bx, by, cx, cy = b.x_m - a.x_m, b.y_m - a.y_m, c.x_m - a.x_m, c.y_m - a.y_m
    det = 2 * (bx * cy - by * cx)
    if abs(det) <= 1e-13 * max(1.0, hypot(bx, by) * hypot(cx, cy)):
        return None
    bb, cc = bx * bx + by * by, cx * cx + cy * cy
    center = Point(a.x_m + (cy * bb - by * cc) / det, a.y_m + (bx * cc - cx * bb) / det)
    return CircleBound(center, distance(center, a))


def mec(poly: Polygon, padding: float = 1e-7) -> CircleBound:
    """Reproducible incremental 2D MEC, followed by full vertex rechecking.

    Output is a checked floating-point upper bound, not an exact optimum certificate.
    """
    if not poly:
        raise GeometryError("Empty MEC")
    points = list(dict.fromkeys(poly))
    Random(1729).shuffle(points)
    circle = CircleBound(points[0], 0)
    for i, a in enumerate(points):
        if distance(circle.center, a) <= circle.radius_m + 1e-9:
            continue
        circle = CircleBound(a, 0)
        for j, b in enumerate(points[:i]):
            if distance(circle.center, b) <= circle.radius_m + 1e-9:
                continue
            circle = pair_circle(a, b)
            for c in points[:j]:
                if distance(circle.center, c) <= circle.radius_m + 1e-9:
                    continue
                trial = triple_circle(a, b, c)
                if trial is None:
                    trial = max(
                        (pair_circle(a, b), pair_circle(a, c), pair_circle(b, c)),
                        key=lambda q: q.radius_m,
                    )
                circle = trial
    return CircleBound(circle.center, max_distance(circle.center, poly) + padding)


def reference_mec(poly: Polygon) -> CircleBound:
    """Independent exhaustive 1/2/3-support verification for small point sets."""
    if not poly:
        raise GeometryError("Empty reference MEC")
    circles = [CircleBound(p, 0) for p in poly]
    circles.extend(pair_circle(a, b) for a, b in combinations(poly, 2))
    circles.extend(c for a, b, d in combinations(poly, 3) if (c := triple_circle(a, b, d)) is not None)
    valid = [c for c in circles if max_distance(c.center, poly) <= c.radius_m + 1e-7]
    return min(valid, key=lambda c: c.radius_m)


def angle_envelope(s: Point, poly: Polygon) -> tuple[float, float]:
    """Unwrapped vertex bearing interval for an external point; span must be <pi."""
    if polygon_distance(s, poly) <= 5:
        raise GeometryError("Angle envelope requires distance >5m")
    angles = sorted(atan2(p.y_m - s.y_m, p.x_m - s.x_m) % (2 * pi) for p in poly)
    gaps = [(angles[(i + 1) % len(angles)] - a) % (2 * pi) for i, a in enumerate(angles)]
    if len(angles) == 1:
        return angles[0], angles[0]
    cut = max(range(len(gaps)), key=gaps.__getitem__)
    start = angles[(cut + 1) % len(angles)]
    values = [start + (a - start) % (2 * pi) for a in angles]
    if max(values) - start >= pi:
        raise GeometryError("No external continuous angular branch")
    return start, max(values)


def boundary_interval(planes: tuple[HalfPlane, ...], index: int) -> tuple[float, float] | None:
    """Parameter interval on a constraint line; distinguish point, edge and unbounded.

    No artificial large bounding box. Empty returns None, unbounded endpoints infinity.
    """
    h = planes[index]
    norm = hypot(h.nx, h.ny)
    if norm == 0:
        raise ValueError("Zero boundary normal")
    px, py = h.nx * h.b / norm**2, h.ny * h.b / norm**2
    tx, ty = -h.ny / norm, h.nx / norm
    lo, hi = -inf, inf
    for k in planes:
        a = k.nx * tx + k.ny * ty
        b = k.b - k.nx * px - k.ny * py
        if abs(a) < 1e-12:
            if b < -1e-8:
                return None
        elif a > 0:
            hi = min(hi, b / a)
        else:
            lo = max(lo, b / a)
    return None if lo > hi + 1e-8 else (lo, hi)


@dataclass(frozen=True)
class Intersection:
    """Region angular-set classification; unbounded sets never receive a fake finite box."""

    kind: str
    vertices: Polygon
    effective_edges: tuple[int, ...]
    diameter_m: float | None


def intersect_halfplanes(planes: tuple[HalfPlane, ...]) -> Intersection:
    """Recover boundary endpoints and classify empty/unbounded/point/segment/polygon.

    Intended for modest Region sets. Omnidirectional uses incremental bounded polygon clipping instead.
    """
    if any(h.nx == h.ny == 0 and h.b < 0 for h in planes):
        return Intersection("empty", (), (), None)
    indices = [i for i, h in enumerate(planes) if hypot(h.nx, h.ny) > 0]
    if not indices:
        return Intersection("unbounded", (), (), inf)
    intervals = [(i, boundary_interval(planes, i)) for i in indices]
    valid = [(i, bounds) for i, bounds in intervals if bounds is not None]
    if not valid:
        return Intersection("empty", (), (), None)
    if any(lo == -inf or hi == inf for _, (lo, hi) in valid):
        return Intersection("unbounded", (), (), inf)
    vertices: list[Point] = []
    edge_indices = []
    for i, (lo, hi) in valid:
        h = planes[i]
        norm = hypot(h.nx, h.ny)
        for value in (lo, hi):
            p = Point(
                h.nx * h.b / norm**2 - value * h.ny / norm,
                h.ny * h.b / norm**2 + value * h.nx / norm,
            )
            if not any(distance(p, q) < 1e-7 for q in vertices):
                vertices.append(p)
        if hi - lo > 1e-7:
            edge_indices.append(i)
    center = Point(sum(p.x_m for p in vertices) / len(vertices), sum(p.y_m for p in vertices) / len(vertices))
    poly = tuple(sorted(vertices, key=lambda p: atan2(p.y_m - center.y_m, p.x_m - center.x_m)))
    area = abs(sum(cross(center, a, b) for a, b in zip(poly, poly[1:] + poly[:1])))
    kind = "point" if len(poly) == 1 else "segment" if area < 1e-8 else "polygon"
    # Collinear endpoint order is immaterial for diameter; only 2D positive-length faces are edges.
    unique_edges: list[int] = []
    if kind == "polygon":
        for i in edge_indices:
            h = planes[i]
            n = hypot(h.nx, h.ny)
            if not any(
                abs(h.nx / n - planes[j].nx / hypot(planes[j].nx, planes[j].ny)) < 1e-10
                and abs(h.ny / n - planes[j].ny / hypot(planes[j].nx, planes[j].ny)) < 1e-10
                for j in unique_edges
            ):
                unique_edges.append(i)
    return Intersection(kind, poly, tuple(unique_edges), diameter(poly)[0])
