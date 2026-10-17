"""Direction-aware ranking activates only after a real near-range negative."""

from dataclasses import replace
from math import cos, nextafter, radians, sin
from time import perf_counter

from rf_searchlab.algorithms.directional.candidates import hypotheses, predict, rf_points
from rf_searchlab.algorithms.directional.optical import remaining_upper
from rf_searchlab.algorithms.directional.policy import inherited_rank
from rf_searchlab.algorithms.directional.visibility import positive_hull_contains
from rf_searchlab.algorithms.lookahead.policy import Decision
from rf_searchlab.core.belief import wire_point
from rf_searchlab.core.domain import Intent, Point


def radii(h):
    high = nextafter(h.radius_high, float("-inf")) if h.high_open else h.radius_high
    return tuple(
        r
        for r in dict.fromkeys((h.radius_low, h.radius, high))
        if h.radius_low <= r <= h.radius_high and (not h.high_open or r < h.radius_high)
    )


def candidates(b, current, config):
    positive = [o for o in b.observations if o.result in ("direction", "near")]
    if not positive:
        return ()
    anchor = positive[-1]
    p = anchor.position
    theta = radians(anchor.svd_deg or 0.0)
    history = {o.position for o in b.observations}
    result = []

    def add(points, quota):
        used = 0
        for raw in points:
            q = wire_point(raw)
            if q not in history and q not in result:
                result.append(q)
                used += 1
                if used >= quota:
                    break

    add((current,), 1)
    hull = [
        Point(
            sum(o.position.x_m for o in positive) / len(positive),
            sum(o.position.y_m for o in positive) / len(positive),
        )
    ]
    hull += [
        Point((a.position.x_m + z.position.x_m) / 2, (a.position.y_m + z.position.y_m) / 2)
        for a, z in zip(positive, positive[1:])
    ]
    add((q for q in hull if positive_hull_contains(b, wire_point(q))), 2)
    add(
        (
            Point(current.x_m + t * (p.x_m - current.x_m), current.y_m + t * (p.y_m - current.y_m))
            for t in (0.5, 0.8)
        ),
        2,
    )
    add(
        (
            Point(
                p.x_m + 30 * cos(theta) - side * w * sin(theta),
                p.y_m + 30 * sin(theta) + side * w * cos(theta),
            )
            for w in (30, 60, 100)
            for side in (-1, 1)
        ),
        4,
    )
    add(rf_points(b, current, config), max(1, 12 - len(result)))
    return tuple(result[:12])


def recover(state, decision, config, deadline):
    c = decision.intent.channel
    b = state.beliefs[c - 1]
    if (
        decision.intent.guaranteed
        or decision.intent.path != "/measure"
        or b.optical
        or b.rf_used >= config.rf_budget
    ):
        return decision
    latest = b.observations[-1] if b.observations else None
    witness = next(
        (
            w
            for w in reversed(state.direction_witnesses)
            if latest and w.channel == c and w.request_id == latest.request_id
        ),
        None,
    )
    if witness is None or witness.request_id not in state.applied:
        return decision
    worlds = tuple(h for h in hypotheses(b, config.scenario_limit) if h.phi is not None)
    if not worlds:
        decision.detail["direction_fallback"] = "hypotheses_empty_not_absence"
        return decision
    points = candidates(b, state.position, config)
    if not points:
        return decision
    upper, _ = remaining_upper(b, state.position)
    penalty = min(600.0, max(100.0, upper * 0.5))
    scored = []
    for p in points:
        if perf_counter() >= deadline:
            decision.detail["direction_fallback"] = "deadline"
            return decision
        probabilities = []
        for h in worlds:
            values = [
                predict(replace(h, radius_low=r, radius_high=r, high_open=False), p)[0] != "no_signal"
                for r in radii(h)
            ]
            probabilities.append(sum(values) / len(values))
        probability = sum(probabilities) / len(probabilities)
        score = inherited_rank(state, b, p) + (1 - probability) * penalty
        scored.append((score, p, probability))
    score, p, probability = min(scored, key=lambda x: (x[0], x[1].x_m, x[1].y_m))
    detail = dict(
        decision.detail,
        category="direction_recovery_rf",
        witness_request_id=witness.request_id,
        witness_polygon_version=witness.polygon_version,
        witness_max_distance_m=witness.max_distance_m,
        predicted_receive=probability,
        hypotheses=len(worlds),
        candidate_count=len(points),
        direction_rank=score,
        rank_kind="Geometry heuristic plus reception risk; not expected seconds",
        guaranteed_receive=positive_hull_contains(b, p),
    )
    return Decision(
        Intent("/measure", p, c, "real_negative_direction_recovery", b.version),
        decision.plan,
        detail,
    )
