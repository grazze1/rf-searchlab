"""Candidate families reserve slots; only exact submitted coordinates are deduplicated."""

from dataclasses import replace
from math import cos, pi, sin

from rf_searchlab.algorithms.directional.candidates import rf_points as original_points
from rf_searchlab.algorithms.directional.visibility import positive_hull_contains
from rf_searchlab.core.belief import wire_point
from rf_searchlab.core.domain import Point
from rf_searchlab.core.geometry import max_distance


def candidates(b, current, config):
    history = {o.position for o in b.observations}
    positive = [o for o in b.observations if o.result in ("direction", "near")]
    result = []

    def add(points, limit):
        n = 0
        for raw in points:
            p = wire_point(raw)
            if p not in history and p not in result and max_distance(p, b.polygon) <= 1500:
                result.append(p)
                n += 1
                if n >= limit:
                    break

    add((current,), 1)
    if positive:
        ps = [o.position for o in positive]
        proposed = [Point(sum(p.x_m for p in ps) / len(ps), sum(p.y_m for p in ps) / len(ps))]
        proposed += [Point((a.x_m + z.x_m) / 2, (a.y_m + z.y_m) / 2) for a, z in zip(ps, ps[1:])]
        add((wire_point(p) for p in proposed if positive_hull_contains(b, wire_point(p))), 2)
        anchor = positive[-1]
        theta = (anchor.svd_deg or 0) * pi / 180
        recovery = [
            Point(
                anchor.position.x_m + side * w * (-sin(theta)),
                anchor.position.y_m + side * w * cos(theta),
            )
            for w in (30, 60, 100)
            for side in (-1, 1)
        ]
        add(recovery, 3)
    classical = original_points(b, current, replace(config, variant="C"))
    add(classical, 3)
    lateral = original_points(b, current, replace(config, variant="D"))
    add(lateral, 3)
    add(classical + lateral, max(1, 12 - len(result)))
    return tuple(result[:12])
