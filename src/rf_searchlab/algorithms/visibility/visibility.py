"""Ranking-only reception probability with a fine orientation grid.

Each accepted observation constrains a directional source's facing angle to an arc
(positive: candidate stop inside the closed half-disk; near-range negative: outside).
The probability that a new stop receives is the surviving-arc fraction that also
covers the stop, averaged over finite source positions inside the location set and
both physical types. This is used to rank actions only; it never certifies anything.
"""

from functools import lru_cache
from math import atan2, degrees

import numpy as np

from rf_searchlab.algorithms.directional.candidates import representatives
from rf_searchlab.algorithms.directional.visibility import positive_hull_contains
from rf_searchlab.core.domain import Point
from rf_searchlab.core.geometry import contains, diameter, distance

BINS = 720
PHI = np.arange(BINS) * (360.0 / BINS)


def arc(alpha_deg):
    d = (PHI - alpha_deg + 180.0) % 360.0 - 180.0
    return np.abs(d) <= 90.0 + 1e-9


def bearing(frm, to):
    return degrees(atan2(to.y_m - frm.y_m, to.x_m - frm.x_m))


@lru_cache(maxsize=4096)
def positions(poly):
    raw = list(representatives(poly))
    d, a, z = diameter(poly)
    m = Point(sum(p.x_m for p in poly) / len(poly), sum(p.y_m for p in poly) / len(poly))
    if d > 1e-8:
        # Dense along the long axis and denser at both tips, where the location set is
        # narrowest and reception is most sensitive to the true source position.
        fractions = [k / 48 for k in range(1, 48)]
        fractions += [0.004, 0.01, 0.996, 0.99]
        for t in fractions:
            p = Point(a.x_m + t * (z.x_m - a.x_m), a.y_m + t * (z.y_m - a.y_m))
            raw.append(Point(0.99 * p.x_m + 0.01 * m.x_m, 0.99 * p.y_m + 0.01 * m.y_m))
    return tuple(p for p in dict.fromkeys(raw) if contains(poly, p, 0))


@lru_cache(maxsize=4096)
def orientation_sets(b):
    """Per source position: (position, directional arc mask or None, omni consistent)."""
    result = []
    for g in positions(b.polygon):
        if distance(g, Point(0, 0)) > 1800:
            continue
        if any(distance(g, e.position) <= e.radius_m for e in b.exclusions):
            continue
        allowed = np.ones(BINS, dtype=bool)
        omni = True
        valid = True
        for o in b.observations:
            d = distance(o.position, g)
            if o.result in ("direction", "near"):
                if d > 1500 or (o.result == "near" and d > 5):
                    valid = False
                    break
                allowed &= arc(bearing(g, o.position))
            elif o.result == "no_signal" and d <= 1000:
                allowed &= ~arc(bearing(g, o.position))
                omni = False
        if not valid or (not allowed.any() and not omni):
            continue
        result.append((g, allowed if allowed.any() else None, omni))
    return tuple(result)


def reception_probability(b, p):
    if positive_hull_contains(b, p):
        return 1.0
    sets = orientation_sets(b)
    if not sets:
        return 0.0
    total = 0.0
    for g, allowed, omni in sets:
        d = distance(p, g)
        if d > 1500:
            continue
        in_range = 1.0 if d <= 1000 else (1500 - d) / 500
        values = []
        if allowed is not None:
            values.append(in_range * (allowed & arc(bearing(g, p))).sum() / allowed.sum())
        if omni:
            values.append(in_range)
        total += sum(values) / len(values)
    return total / len(sets)
