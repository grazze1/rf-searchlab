"""Compare RF against the same persistent optical cells and common route boundary."""

from dataclasses import replace
from math import radians
from time import perf_counter

from rf_searchlab.algorithms.directional.candidates import clear_points, hypotheses, predict
from rf_searchlab.algorithms.directional.policy import inherited_rank, rf_cost
from rf_searchlab.algorithms.lookahead.optical import remaining
from rf_searchlab.algorithms.lookahead.policy import Decision
from rf_searchlab.algorithms.lookahead.service import common_tail, optical_cost, score, tail_point, terminal
from rf_searchlab.algorithms.resilient.recovery import candidates, radii
from rf_searchlab.core.domain import Intent
from rf_searchlab.core.geometry import clip_disk, clip_wedge, distance


def compare(state, b, plan, cover, rest, optical, config, deadline):
    worlds = hypotheses(b, config.scenario_limit)
    if not worlds:
        optical.detail["fallback_reason"] = "hypotheses_empty_not_absence"
        return optical
    scenarios = []
    selected = tuple(worlds[int(i * len(worlds) / min(4, len(worlds)))] for i in range(min(4, len(worlds))))
    for h in selected:
        for radius in radii(h):
            fixed = replace(h, radius_low=radius, radius_high=radius, high_open=False)
            scenarios.extend((fixed, e) for e in (-1.0, 0.0, 1.0))
    scenarios = scenarios[: config.scenario_limit]
    tail = common_tail(state, b, plan)
    optical_values = [optical_cost(h, state.position, rest, tail, state.channel) for h, _ in scenarios]
    if not scenarios or any(v is None for v in optical_values):
        return optical

    def upper(p):
        ps = tuple(cell.center for _, cell in rest)
        return sum(distance(a, z) / 5 for a, z in zip((p,) + ps, ps)) + 3 * (len(ps) - 1) + 5

    optical_values.append(upper(state.position) + terminal(rest[-1][1].center, tail, state.channel))
    best = score(optical_values, config.risk_weight)
    chosen = optical
    evaluations = []
    for p in sorted(candidates(b, state.position, config), key=lambda p: inherited_rank(state, b, p))[:4]:
        values = []
        branches = set()
        for h, error in scenarios:
            if perf_counter() >= deadline:
                return optical
            result, angle = predict(h, p, error)
            branches.add(result)
            if result == "near":
                future = 5 + terminal(p, tail, b.channel)
            else:
                poly = b.polygon
                if result == "direction":
                    poly = clip_wedge(poly, p, radians(angle), config.epsilon_rad, config.geometry_padding_m)
                    poly = clip_disk(poly, p, 1500, config.disk_sides, config.geometry_padding_m)
                if not poly:
                    break
                hypothetical = replace(b, polygon=poly)
                safe = clear_points(hypothetical, p, tail_point(tail))
                if safe:
                    future = min(distance(p, q) / 5 + 5 + terminal(q, tail, b.channel) for q in safe)
                else:
                    # Cell identities/order survive the hypothetical RF exactly as in execution.
                    future = optical_cost(h, p, remaining(cover, hypothetical), tail, b.channel)
            if future is None:
                break
            values.append(rf_cost(state, b.channel, p) + future)
        if len(values) != len(scenarios):
            continue
        guard = rf_cost(state, b.channel, p) + upper(p) + terminal(rest[-1][1].center, tail, b.channel)
        values.append(guard)
        value = score(values, config.risk_weight)
        evaluations.append(dict(point=(p.x_m, p.y_m), score_s=value, branches=sorted(branches)))
        if value < best:
            best = value
            chosen = Decision(
                Intent("/measure", p, b.channel, "persistent_optical_recovery", b.version),
                plan,
                dict(category="optical_recovery_rf", guaranteed_receive=False),
            )
    chosen.detail.update(
        score_kind="Matched persistent cells and early-stop scenarios",
        score_s=best,
        optical_score_s=score(optical_values, config.risk_weight),
        rf_scores=evaluations,
        scenario_count=len(scenarios),
    )
    return chosen
