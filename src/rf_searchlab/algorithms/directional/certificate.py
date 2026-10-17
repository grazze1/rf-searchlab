"""Exact continuous-position/continuous-direction certificates, never sampling proofs."""

import csv
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256

import numpy as np

from rf_searchlab.algorithms.directional.belief import Status
from rf_searchlab.algorithms.directional.config import RESOURCE
from rf_searchlab.core.domain import Certificate, GeometryError, Point

SCALE = 1_000_000
DOMAIN = 1800 * SCALE
LIMIT = 1000 * SCALE - 10
STEP = 25 * SCALE


def integer_point(p):
    # No silent rounding here: every proof is about the actual canonical wire coordinate.
    x, y = Decimal(str(p.x_m)) * SCALE, Decimal(str(p.y_m)) * SCALE
    if x != int(x) or y != int(y):
        raise GeometryError("Certificate coordinate is not a six-decimal wire point")
    return int(x), int(y)


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull(points):
    p = sorted(set(points))
    if len(p) < 2:
        return tuple(p)
    lo, hi = [], []
    for q in p:
        while len(lo) > 1 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(hi) > 1 and cross(hi[-2], hi[-1], q) <= 0:
            hi.pop()
        hi.append(q)
    return tuple(lo[:-1] + hi[:-1])


def inside(p, h):
    if len(h) == 1:
        return p == h[0]
    if len(h) == 2:
        return (
            cross(h[0], h[1], p) == 0
            and min(h[0][0], h[1][0]) <= p[0] <= max(h[0][0], h[1][0])
            and min(h[0][1], h[1][1]) <= p[1] <= max(h[0][1], h[1][1])
        )
    return bool(h) and all(cross(a, b, p) >= 0 for a, b in zip(h, h[1:] + h[:1]))


def corners(b):
    x, y, xx, yy = b
    return ((x, y), (xx, y), (xx, yy), (x, yy))


def touches(b):
    x, y, xx, yy = b
    a, c = min(max(0, x), xx), min(max(0, y), yy)
    return a * a + c * c <= DOMAIN * DOMAIN


def split(b):
    x, y, xx, yy = b
    if (x + xx) % 2 or (y + yy) % 2:
        raise GeometryError("Cell cannot be subdivided on integer grid")
    mx, my = (x + xx) // 2, (y + yy) // 2
    return ((x, y, mx, my), (mx, y, xx, my), (x, my, mx, yy), (mx, my, xx, yy))


@lru_cache(maxsize=1)
def roots():
    return tuple(
        b
        for i in range(-73, 73)
        for j in range(-73, 73)
        if touches(b := (i * STEP, j * STEP, (i + 1) * STEP, (j + 1) * STEP))
    )


def cell_witness(b, points):
    vv = corners(b)
    indices = tuple(
        i for i, s in enumerate(points) if all((s[0] - x) ** 2 + (s[1] - y) ** 2 <= LIMIT**2 for x, y in vv)
    )
    h = hull(tuple(points[i] for i in indices))
    return indices if len(h) >= 3 and all(inside(v, h) for v in vv) else None


def check_witness(b, indices, points):
    if (
        len(indices) < 3
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or not 0 <= i < len(points) for i in indices)
    ):
        raise GeometryError("Invalid witness indices")
    if len(b) != 4 or any(type(v) is not int for v in b) or b[0] >= b[2] or b[1] >= b[3]:
        raise GeometryError("Invalid closed cell")
    vv = corners(b)
    for i in indices:
        s = points[i]
        if any((s[0] - x) ** 2 + (s[1] - y) ** 2 > LIMIT**2 for x, y in vv):
            raise GeometryError("Witness violates full-cell reception distance")
    h = hull(tuple(points[i] for i in indices))
    if len(h) < 3 or not all(inside(v, h) for v in vv):
        raise GeometryError("Corners are not in one witness hull")


def verify_cover(leaves, points):
    """Reconstruct every disk-intersecting root and child, including tangent cells."""
    remaining = dict(leaves)
    for b, w in remaining.items():
        check_witness(b, w, points)

    def visit(b, depth=0):
        if not touches(b):
            return
        if b in remaining:
            remaining.pop(b)
            return
        if depth >= 4:
            raise GeometryError(f"Missing certificate leaf {b}")
        for child in split(b):
            visit(child, depth + 1)

    for b in roots():
        visit(b)
    if remaining:
        raise GeometryError("Extraneous or overlapping certificate leaves")


def read_points(path):
    with path.open(encoding="utf-8") as f:
        points = []
        for r in csv.DictReader(f):
            x, y = Decimal(r["x"]) * SCALE, Decimal(r["y"]) * SCALE
            if not x.is_finite() or not y.is_finite() or x != int(x) or y != int(y):
                raise GeometryError("CSV coordinates must be exact finite six-decimal values")
            points.append((int(x), int(y)))
        return tuple(points)


@dataclass(frozen=True)
class Layout:
    name: str
    points: tuple
    leaves: tuple
    witnesses: tuple
    digest: str

    @property
    def stops(self):
        return tuple(Point(x / SCALE, y / SCALE) for x, y in self.points)


def load_ring(directory=RESOURCE):
    data = json.loads((directory / "ring_cover_certificate.json").read_text(encoding="utf-8"))
    points = read_points(directory / "ring_stations_25.csv")
    if (
        data["scale"] != SCALE
        or data["stop_count"] != 25
        or data["min_rx_radius_m"] != 1000
        or len(set(points)) != 25
        or points != tuple(map(tuple, data["stops_scaled"]))
        or data["failed"]
        or data["uncertified_leaf_count"] != 0
        or data["source_domain_radius_m"] != 1800
        or Decimal(str(data["certified_range_m"])) * SCALE != LIMIT
    ):
        raise GeometryError("Ring metadata/CSV mismatch")
    leaves = {tuple(x["box_scaled"]): tuple(x["stop_indices"]) for x in data["leaves"]}
    if len(leaves) != len(data["leaves"]) or len(leaves) != data["certified_leaf_count"]:
        raise GeometryError("Duplicated leaves or incorrect leaf count")
    if data["root_count_including_tangent_only_cells"] != len(roots()):
        raise GeometryError("Root count mismatch")
    verify_cover(leaves, points)
    payload = json.dumps([points, list(leaves.items())], separators=(",", ":")).encode()
    return Layout("ring25", points, tuple(leaves), tuple(leaves.values()), sha256(payload).hexdigest())


@lru_cache(maxsize=2)
def load_triangle():
    points = read_points(RESOURCE / "triangular_stations_31.csv")
    if len(set(points)) != 31:
        raise GeometryError("Triangular fallback must contain 31 points")
    # Check the serialized lattice rather than trusting the filename.
    from math import sqrt

    generated = tuple(
        (round(990 * (i + j / 2) * SCALE), round(990 * sqrt(3) * j / 2 * SCALE))
        for i in range(-5, 6)
        for j in range(-5, 6)
        if 990**2 * (i * i + i * j + j * j) <= 2790**2
    )
    if points != generated:
        raise GeometryError("Triangular CSV disagrees with the declared lattice")
    queue = [(b, 0) for b in roots()]
    leaves = {}
    while queue:
        b, d = queue.pop()
        w = cell_witness(b, points)
        if w is not None:
            leaves[b] = w
        elif d < 4:
            queue.extend((c, d + 1) for c in split(b) if touches(c))
        else:
            raise GeometryError("Triangular fallback certification failed")
    verify_cover(leaves, points)
    payload = json.dumps([points, list(leaves.items())], separators=(",", ":")).encode()
    return Layout("triangle31", points, tuple(leaves), tuple(leaves.values()), sha256(payload).hexdigest())


@lru_cache(maxsize=2)
def layout_for(variant):
    fallback = load_triangle()
    if variant == "A":
        return fallback, None
    try:
        return load_ring(), None
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        ArithmeticError,
        GeometryError,
    ) as exc:
        return fallback, f"ring25_rejected: {type(exc).__name__}: {exc}"


class Engine:
    """Cached geometry returns proof indices; only accepted history binds request IDs."""

    def __init__(self, layout, dynamic=False):
        self.layout, self.dynamic = layout, dynamic
        self.boxes = np.array([corners(b) for b in layout.leaves], dtype=float) / SCALE
        self.proofs = {}
        self.seconds = 0.0
        self.progress = {}

    @lru_cache(maxsize=512)
    def eligible(self, p):
        # Floating point is ONLY a broad phase. Exact predicates validate every witness.
        s = np.array(p, dtype=float) / SCALE
        return np.max(np.sum((self.boxes - s) ** 2, axis=2), axis=1) <= (LIMIT / SCALE + 1e-6) ** 2

    @lru_cache(maxsize=256)
    def coverage(self, points):
        if len(points) < 3:
            return ()
        # Fast fixed-witness path, followed by arbitrary measured-stop hulls for residuals.
        lookup = {p: i for i, p in enumerate(points)}
        result = {}
        for k, w in enumerate(self.layout.witnesses):
            if all(self.layout.points[i] in lookup for i in w):
                result[k] = tuple(lookup[self.layout.points[i]] for i in w)
        residual = np.array([i for i in range(len(self.layout.leaves)) if i not in result], dtype=int)
        if not len(residual):
            return tuple(result.items())
        matrix = np.array([self.eligible(p)[residual] for p in points]).T
        packed = np.packbits(matrix, axis=1)
        unique, inverse = np.unique(packed, axis=0, return_inverse=True)
        for group, row in enumerate(unique):
            selected = tuple(np.flatnonzero(np.unpackbits(row)[: len(points)]).tolist())
            if len(selected) < 3:
                continue
            h = hull(tuple(points[i] for i in selected))
            if len(h) < 3:
                continue
            indices = residual[inverse == group]
            vv = self.boxes[indices]
            mask = np.ones(len(indices), dtype=bool)
            for a, b in zip(h, h[1:] + h[:1]):
                ax, ay, bx, by = a[0] / SCALE, a[1] / SCALE, b[0] / SCALE, b[1] / SCALE
                det = (bx - ax) * (vv[:, :, 1] - ay) - (by - ay) * (vv[:, :, 0] - ax)
                mask &= np.all(det >= -1e-5, axis=1)
            for k in indices[mask].tolist():
                box = self.layout.leaves[k]
                try:
                    check_witness(box, selected, points)
                except GeometryError:
                    continue  # uncertain broad-phase results never certify
                result[k] = selected
        return tuple(result.items())

    def condition_complete(self, real_points, planned_points):
        """Geometry-only future-all-negative check. This method cannot mutate evidence."""
        points = tuple(sorted(set(map(integer_point, real_points + planned_points))))
        return len(self.coverage(points)) == len(self.layout.leaves)

    def update(self, state, channel):
        from time import perf_counter

        start = perf_counter()
        b = state.beliefs[channel - 1]
        if b.status != Status.UNKNOWN or b.ever_detected:
            return state
        history = {}
        for o in b.observations:
            if o.result == "no_signal" and o.request_id in state.applied:
                history.setdefault(integer_point(o.position), o.request_id)
        if not self.dynamic:
            if not all(p in history for p in self.layout.points):
                self.progress[channel] = 0
                return state
            points = self.layout.points
            proof = tuple(enumerate(self.layout.witnesses))
        else:
            points = tuple(sorted(history))
            proof = self.coverage(points)
        self.progress[channel] = len(proof)
        self.seconds += perf_counter() - start
        if len(proof) != len(self.layout.leaves):
            return state
        ids = tuple(history[p] for p in points)
        cert = Certificate("directional_negative_hull", channel, ids, b.version)
        beliefs = list(state.beliefs)
        beliefs[channel - 1] = replace(b, status=Status.CERTIFIED_ABSENT, certificate=cert)
        self.proofs[channel] = dict(
            layout=self.layout.name,
            layout_hash=self.layout.digest,
            channel=channel,
            points_scaled=points,
            request_ids=ids,
            leaves=[dict(box_scaled=self.layout.leaves[k], witness_indices=w) for k, w in proof],
        )
        return replace(state, beliefs=tuple(beliefs))


def verify_actual_proof(proof, state):
    """Independent binding check: no other channel, predicted or unaccepted request can pass."""
    c = proof["channel"]
    b = state.beliefs[c - 1]
    if b.ever_detected:
        raise GeometryError("Previously discovered channel cannot be absent")
    negatives = {
        o.request_id: integer_point(o.position)
        for o in b.observations
        if o.result == "no_signal" and o.request_id in state.applied
    }
    points = tuple(map(tuple, proof["points_scaled"]))
    ids = proof["request_ids"]
    if len(ids) != len(points) or any(negatives.get(rid) != p for rid, p in zip(ids, points)):
        raise GeometryError("Absence witness is not accepted same-channel negative history")
    leaves = {tuple(x["box_scaled"]): tuple(x["witness_indices"]) for x in proof["leaves"]}
    if len(leaves) != len(proof["leaves"]):
        raise GeometryError("Duplicate runtime certificate leaf")
    verify_cover(leaves, points)
