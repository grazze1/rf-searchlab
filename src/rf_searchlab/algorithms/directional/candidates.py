"""Finite RF/clear candidates and history-compatible hypotheses used ONLY for ranking."""

from dataclasses import dataclass
from functools import lru_cache
from math import atan2, cos, degrees, pi, sin

from rf_searchlab.algorithms.directional.belief import bound, safe_clear
from rf_searchlab.algorithms.directional.visibility import angular_visible, positive_hull_contains, visible
from rf_searchlab.core.belief import wire_point
from rf_searchlab.core.domain import Point
from rf_searchlab.core.geometry import contains, diameter, distance, max_distance


def clear_points(b, current, successor=None):
    circle = bound(b.polygon)
    r = 19.95 - circle.radius_m - 2e-6
    points = [wire_point(current), wire_point(circle.center)]
    if r >= 0:
        targets = [current] + ([successor] if successor is not None else [])
        for p in targets:
            d = distance(p, circle.center)
            t = min(1, r / max(d, 1e-12))
            points.append(
                wire_point(
                    Point(
                        circle.center.x_m + t * (p.x_m - circle.center.x_m),
                        circle.center.y_m + t * (p.y_m - circle.center.y_m),
                    )
                )
            )
    return tuple(p for p in dict.fromkeys(points) if safe_clear(b, p))


def rf_points(b, current, config):
    poly = b.polygon
    d, a, z = diameter(poly)
    if d < 1e-8:
        return ()
    if distance(current, a) > distance(current, z):
        a, z = z, a
    ux, uy = (z.x_m - a.x_m) / d, (z.y_m - a.y_m) / d
    step = min(d, max(config.approach_fraction * d, d - 990))
    widths = (0, 30, 60, 100) if config.directional else (config.approach_lateral_m,)
    raw = []
    if d >= config.approach_min_diameter_m or config.directional:
        raw.extend(
            Point(a.x_m + step * ux - side * w * uy, a.y_m + step * uy + side * w * ux)
            for w in widths
            for side in (-1, 1)
        )
    m = bound(poly).center
    positives = [o.position for o in b.observations if o.result in ("direction", "near")]
    if positives:
        anchor = positives[0]
        obs = next(o for o in b.observations if o.result in ("direction", "near"))
        theta = (obs.svd_deg or 0) * pi / 180
        raw.extend(
            Point(
                anchor.x_m + 750 * cos(theta) - side * 100 * sin(theta),
                anchor.y_m + 750 * sin(theta) + side * 100 * cos(theta),
            )
            for side in (-1, 1)
        )
    raw.extend(
        Point(m.x_m + r * cos(k * pi / 2), m.y_m + r * sin(k * pi / 2)) for r in (30, 100) for k in range(4)
    )
    raw.append(current)
    if config.directional and len(positives) > 1:
        proposed = [
            Point(
                sum(p.x_m for p in positives) / len(positives),
                sum(p.y_m for p in positives) / len(positives),
            )
        ]
        proposed.extend(
            Point((a.x_m + b.x_m) / 2, (a.y_m + b.y_m) / 2) for a, b in zip(positives, positives[1:])
        )
        raw.extend(p for p in proposed if positive_hull_contains(b, p))
    history = {o.position for o in b.observations}
    result = []
    for p in dict.fromkeys(wire_point(p) for p in raw):
        if p not in history and max_distance(p, poly) <= 1500:
            result.append(p)
    # Prioritise inherited approach points; no distance-based guaranteed_receive flag.
    return tuple(result[: config.candidate_limit])


@dataclass(frozen=True)
class PredictedObservation:
    """Scoring-only history, with a distinct type and no accepted request ID."""

    position: Point
    result: str
    svd_deg: float | None


@dataclass(frozen=True)
class Hypothesis:
    position: Point
    phi: float | None
    radius_low: float
    radius_high: float
    high_open: bool

    @property
    def radius(self):
        return (self.radius_low + self.radius_high) / 2


def representatives(poly):
    m = Point(sum(p.x_m for p in poly) / len(poly), sum(p.y_m for p in poly) / len(poly))
    d, a, b = diameter(poly)
    raw = [m]
    # Interior positions along the uncertain long axis; no claim of exhaustive support.
    for t in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        p = Point(a.x_m + t * (b.x_m - a.x_m), a.y_m + t * (b.y_m - a.y_m))
        raw.append(Point(0.98 * p.x_m + 0.02 * m.x_m, 0.98 * p.y_m + 0.02 * m.y_m))
    raw.extend(poly[:: max(1, len(poly) // 4)])
    return tuple(p for p in dict.fromkeys(raw) if contains(poly, p, 0))


@lru_cache(maxsize=2048)
def hypotheses(b, limit=64):
    result = []
    for g in representatives(b.polygon):
        if distance(g, Point(0, 0)) > 1800:
            continue
        if any(distance(g, e.position) <= e.radius_m for e in b.exclusions):
            continue
        for phi in (None,) + tuple(k * pi / 12 for k in range(24)):
            low, high, high_open, valid = 1000.0, 1500.0, False, True
            for o in b.observations:
                d = distance(g, o.position)
                facing = angular_visible(o.position, g, phi)
                if o.result in ("direction", "near"):
                    if not facing or (o.result == "near" and d > 5) or (o.result == "direction" and d <= 5):
                        valid = False
                        break
                    if o.result == "direction":
                        theta = degrees(atan2(g.y_m - o.position.y_m, g.x_m - o.position.x_m)) % 360
                        if abs((theta - o.svd_deg + 180) % 360 - 180) > 1.0050001:
                            valid = False
                            break
                    low = max(low, d)
                elif facing and d <= high:
                    high, high_open = d, True
            if valid and (low < high or (low == high and not high_open)):
                result.append(Hypothesis(g, phi, low, high, high_open))
    if len(result) <= limit:
        return tuple(result)
    return tuple(result[int(i * len(result) / limit)] for i in range(limit))


def predict(h, p, error=0.0):
    if not visible(p, h.position, h.radius, h.phi):
        return "no_signal", None
    if distance(p, h.position) <= 5:
        return "near", None
    angle = degrees(atan2(h.position.y_m - p.y_m, h.position.x_m - p.x_m)) + error
    return "direction", round(angle % 360, 2) % 360
