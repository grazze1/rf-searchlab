"""Exact bearing-region geometry and linear-time convex diameter analysis.

The diameter pass is O(n) for ordered convex vertices. Half-plane boundary
recovery is a separate O(m²) step (plus endpoint deduplication).
"""

import itertools
import math
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key

from rf_searchlab.core import geometry as geometry_helpers
from rf_searchlab.core.domain import Point


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _vertices(vertices):
    """Linear normalization: CCW, no duplicate or straight-edge inner vertices.

    Input must describe a convex boundary in cyclic order, or a collinear set.
    This does not compute a convex hull or silently reorder an arbitrary point cloud.
    """
    p = []
    for v in vertices:
        v = tuple(x if isinstance(x, Fraction) else float(x) for x in v)
        if not all(map(math.isfinite, v)):
            raise ValueError("Vertices must be finite")
        if not p or v != p[-1]:
            p.append(v)
    if len(p) > 1 and p[0] == p[-1]:
        p.pop()
    if not p:
        raise ValueError("Empty region has no diameter endpoints")
    if len(p) < 3:
        return p
    if all(_cross(p[0], p[1], v) == 0 for v in p[2:]):
        return [min(p), max(p)]
    terms = [_cross(p[0], a, b) for a, b in itertools.pairwise(p)]
    signed_area = sum(terms) if isinstance(p[0][0], Fraction) else math.fsum(terms)
    if signed_area < 0:
        p.reverse()
    # Remove collinear intermediate points before the strict convex '>' sweep.
    # Each vertex is pushed/popped at most once; wrap cleanup uses deque ends.
    stack = []
    for v in p:
        while len(stack) > 1 and _cross(stack[-2], stack[-1], v) == 0:
            stack.pop()
        stack.append(v)
    ring = deque(stack)
    while len(ring) > 2:
        if _cross(ring[-1], ring[0], ring[1]) == 0:
            ring.popleft()
        elif _cross(ring[-2], ring[-1], ring[0]) == 0:
            ring.pop()
        else:
            break
    p = list(ring)
    if any(_cross(p[i - 1], p[i], p[(i + 1) % len(p)]) <= 0 for i in range(len(p))):
        raise ValueError("Expected a cyclically ordered convex boundary")
    return p


@dataclass(frozen=True)
class Diameter:
    squared_m2: float
    diameter_m: float
    pD: tuple[float, float]
    qD: tuple[float, float]
    vertex_count: int
    support_advances: int
    candidate_pairs: int
    parallel_ties: int


@dataclass(frozen=True)
class Intersection(geometry_helpers.Intersection):
    diameter_result: Diameter | None = None
    exact_vertices: tuple = ()


def rotating_calipers(vertices):
    """Algorithm 1, lines 2–9. Keep j across edges; sqrt only after the sweep."""
    p = _vertices(vertices)
    n = len(p)

    def squared(a, b):
        return (p[a][0] - p[b][0]) ** 2 + (p[a][1] - p[b][1]) ** 2

    if n <= 2:
        d2 = squared(0, n - 1)
        return Diameter(d2, math.sqrt(d2), p[0], p[-1], n, 0, 1, 0)
    dmax, pair, j = 0.0, (0, 0), 1  # Paper j=2, converted to zero-based indexing.
    advances = candidates = ties = 0
    for i in range(n):
        successor = (i + 1) % n

        def gamma(k, i=i, successor=successor):
            return _cross(p[i], p[successor], p[k])

        while gamma((j + 1) % n) > gamma(j):
            j = (j + 1) % n
            advances += 1
            if advances > 2 * n:
                raise ArithmeticError("Support sweep exceeded convex linear bound")
        opposite = [j]
        if gamma((j + 1) % n) == gamma(j):
            opposite.append((j + 1) % n)
            ties += 1
        for a in (i, successor):
            for b in opposite:
                candidates += 1
                value = squared(a, b)
                if value > dmax:
                    dmax, pair = value, (a, b)
    return Diameter(dmax, math.sqrt(dmax), p[pair[0]], p[pair[1]], n, advances, candidates, ties)


def same_diameter_circle(vertices, diameter=None):
    """Algorithm 1, lines 10–13; squared criterion (5.16), no radius padding.

    Evaluate ||2v-pD-qD||² <= ||pD-qD||² to avoid rounding the midpoint.
    Near equality, use exact rational arithmetic on the supplied float coordinates;
    the roundoff threshold only selects arithmetic, never enlarges the circle.
    """
    p = [tuple(x if isinstance(x, Fraction) else float(x) for x in v) for v in vertices]
    d = diameter if diameter is not None else rotating_calipers(p)
    center = tuple(a + (b - a) / 2 for a, b in zip(d.pD, d.qD))
    dx, dy = d.qD[0] - d.pD[0], d.qD[1] - d.pD[1]
    maximum, covered, exact_checks = 0.0, True, 0
    for x, y in p:
        ax, ay = x - d.pD[0], y - d.pD[1]
        ux, uy = 2 * ax - dx, 2 * ay - dy
        value = ux * ux + uy * uy
        maximum = max(maximum, value / 4)
        scale = (2 * abs(ax) + abs(dx)) ** 2 + (2 * abs(ay) + abs(dy)) ** 2 + d.squared_m2
        if abs(value - d.squared_m2) <= 64 * math.ulp(scale):
            exact_checks += 1
            px, py, qx, qy, vx, vy = map(Fraction, (*d.pD, *d.qD, x, y))
            inside = (2 * vx - px - qx) ** 2 + (2 * vy - py - qy) ** 2 <= (px - qx) ** 2 + (py - qy) ** 2
        else:
            inside = value <= d.squared_m2
        covered = covered and inside
    return {
        "center_m": list(map(float, center)),
        "radius_m": d.diameter_m / 2,
        "max_vertex_distance_squared_m2": float(maximum),
        "radius_squared_m2": float(d.squared_m2 / 4),
        "exact_boundary_checks": exact_checks,
        "cover_flag": covered,
    }


def intersect_halfplanes(planes):
    """Exact predicates on supplied coefficients, followed by exact boundary intersection.

    No fixed length, area, parallelism or feasibility threshold changes the
    represented region. Only final vertices are rounded to float for calipers.
    """
    planes = tuple(planes)
    if any(not math.isfinite(v) for h in planes for v in (h.nx, h.ny, h.b)):
        raise ValueError("Half-plane coefficients must be finite")
    exact = [tuple(map(Fraction, (h.nx, h.ny, h.b))) for h in planes]
    if any(nx == ny == 0 and b < 0 for nx, ny, b in exact):
        return Intersection("empty", (), (), None)
    indices = [i for i, (nx, ny, _) in enumerate(exact) if nx or ny]
    if not indices:
        return Intersection("unbounded", (), (), math.inf)
    valid = []
    for i in indices:
        nx, ny, bound = exact[i]
        px, py = (bound / nx, Fraction(0)) if nx else (Fraction(0), bound / ny)
        tx, ty = -ny, nx
        lo = hi = None
        feasible = True
        for kx, ky, kb in exact:
            coefficient = kx * tx + ky * ty
            slack = kb - kx * px - ky * py
            if coefficient == 0:
                if slack < 0:
                    feasible = False
                    break
            elif coefficient > 0:
                limit = slack / coefficient
                hi = limit if hi is None else min(hi, limit)
            else:
                limit = slack / coefficient
                lo = limit if lo is None else max(lo, limit)
        if feasible and (lo is None or hi is None or lo <= hi):
            valid.append((i, lo, hi, px, py, tx, ty))
    if not valid:
        return Intersection("empty", (), (), None)
    if any(lo is None or hi is None for _, lo, hi, *_ in valid):
        return Intersection("unbounded", (), (), math.inf)
    endpoints, edges, face_keys = set(), [], set()
    for i, lo, hi, px, py, tx, ty in valid:
        for value in (lo, hi):
            endpoints.add((px + value * tx, py + value * ty))
        nx, ny, bound = exact[i]
        scale = abs(nx) if nx else abs(ny)
        key = (nx / scale, ny / scale, bound / scale)
        if hi > lo and key not in face_keys:
            face_keys.add(key)
            edges.append(i)
    endpoints = sorted(endpoints)
    collinear = len(endpoints) < 3 or all(_cross(endpoints[0], endpoints[1], p) == 0 for p in endpoints[2:])
    kind = "point" if len(endpoints) == 1 else "segment" if collinear else "polygon"
    if kind == "polygon":

        def compare_normals(i, j):
            ax, ay, _ = exact[i]
            bx, by, _ = exact[j]
            # Same circular order as atan2, without rounding close directions
            # to equal angles. Positive x belongs after the negative y half.
            ah = 0 if ay < 0 or (ay == 0 and ax >= 0) else 1
            bh = 0 if by < 0 or (by == 0 and bx >= 0) else 1
            if ah != bh:
                return -1 if ah < bh else 1
            cross = ax * by - ay * bx
            return -1 if cross > 0 else 1 if cross < 0 else 0

        edges.sort(key=cmp_to_key(compare_normals))
        vertices = []
        for i, j in zip(edges, edges[1:] + edges[:1]):
            ax, ay, ab = exact[i]
            bx, by, bb = exact[j]
            det = ax * by - ay * bx
            if det == 0:
                raise ArithmeticError("Adjacent effective boundaries are parallel")
            vertices.append(((ab * by - ay * bb) / det, (ax * bb - ab * bx) / det))
    else:
        vertices, edges = ([endpoints[0], endpoints[-1]] if kind == "segment" else endpoints), []
    # Keep exact vertices through the sweep: rounding tiny faces to float first
    # can collapse distinct points or create an artificial concave turn.
    diameter = rotating_calipers(vertices)
    poly = tuple(Point(*map(float, v)) for v in vertices)
    return Intersection(kind, poly, tuple(edges), diameter.diameter_m, diameter, tuple(vertices))


def bearing_halfplanes(sensors_m, bearings_deg, epsilon_deg=1.0):
    """Construct wedges with outward binary64 roundoff guards, in radians.

    The guard covers degree/radian/trigonometric and dot-product evaluation;
    it is not a change to the physical ±epsilon measurement-error assumption.
    Exact half-plane input to intersect_halfplanes receives no such guard.
    """
    if len(sensors_m) != len(bearings_deg) or len(sensors_m) == 0:
        raise ValueError("Supply equally many sensors and bearings, at least one")
    if not math.isfinite(epsilon_deg) or not 0 < epsilon_deg < 90:
        raise ValueError("epsilon_deg must be in (0, 90)")
    result, maximum_guard = [], 0.0
    for sensor, angle in zip(sensors_m, bearings_deg):
        if len(sensor) != 2 or not all(math.isfinite(float(x)) for x in (*sensor, angle)):
            raise ValueError("Sensor coordinates and bearings must be finite")
        x, y = map(float, sensor)
        theta = math.radians(float(angle) % 360.0)
        eps = math.radians(epsilon_deg)
        guard = 64 * max(math.ulp(theta), math.ulp(eps), math.ulp(math.pi))
        maximum_guard = max(maximum_guard, guard)
        for nx, ny in (
            (math.sin(theta - eps - guard), -math.cos(theta - eps - guard)),
            (-math.sin(theta + eps + guard), math.cos(theta + eps + guard)),
        ):
            bound = nx * x + ny * y
            padding = 8 * (math.ulp(nx * x) + math.ulp(ny * y) + math.ulp(bound))
            result.append(geometry_helpers.HalfPlane(nx, ny, math.nextafter(bound + padding, math.inf)))
    return tuple(result), maximum_guard


def solve(sensors_m, bearings_deg, epsilon_deg=1.0):
    """Algorithm 1: angle wedges -> P -> rotating calipers -> circle coverage."""
    if len(sensors_m) != len(bearings_deg) or len(sensors_m) == 0:
        raise ValueError("Supply equally many sensors and bearings, at least one")
    if not 0 < epsilon_deg < 90:
        raise ValueError("epsilon_deg must be in (0, 90)")
    planes, guard = bearing_halfplanes(sensors_m, bearings_deg, epsilon_deg)
    region = intersect_halfplanes(planes)
    p = [(v.x_m, v.y_m) for v in region.vertices]
    result = {
        "algorithm": "rotating_calipers",
        "kind": region.kind,
        "vertices_m": p,
        "effective_edges": list(region.effective_edges),
        "diameter_m": region.diameter_m,
        "farthest_pair_m": None,
        "cover_flag": None,
        "circle": None,
        "epsilon_deg": epsilon_deg,
        "numerical_angle_guard_rad": guard,
        "geometry_predicates": "exact_on_guarded_binary64_halfplanes",
    }
    if p:
        d = region.diameter_result
        circle = same_diameter_circle(region.exact_vertices, d)
        result.update(
            farthest_pair_m=[list(map(float, d.pD)), list(map(float, d.qD))],
            circle=circle,
            cover_flag=circle["cover_flag"],
        )
    return result
