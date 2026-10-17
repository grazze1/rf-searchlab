"""Matched finite-scenario RF/optical continuation and bounded recovery service."""

from dataclasses import dataclass, replace
from math import radians
from time import perf_counter

from rf_searchlab.algorithms.directional.belief import RESOLVED, bound
from rf_searchlab.algorithms.directional.candidates import clear_points, hypotheses, predict, rf_points
from rf_searchlab.algorithms.directional.policy import Decision, inherited_rank, rf_cost
from rf_searchlab.algorithms.directional.visibility import positive_hull_contains
from rf_searchlab.algorithms.lookahead.candidates import candidates
from rf_searchlab.algorithms.lookahead.optical import Cover, cells_for, make_cover, remaining
from rf_searchlab.algorithms.lookahead.planning import station_channels
from rf_searchlab.core.domain import GeometryError, Intent, Point
from rf_searchlab.core.geometry import clip_disk, clip_wedge, distance


def put_cover(plan, cover):
    covers = tuple(c for c in plan.covers if c.channel != cover.channel) + (cover,)
    return replace(plan, covers=covers)


def optical_choice(state, b, plan, config):
    cover = next((c for c in plan.covers if c.channel == b.channel), None)
    if cover is None:
        if b.optical:
            lookup = {c.center: c for c in cells_for(b.polygon, False)}
            if not all(p in lookup for p in b.optical):
                raise GeometryError("Cannot reconstruct pre-existing cover from its current polygon")
            cover = Cover(
                b.channel,
                f"legacy-{b.channel}-{b.version}",
                tuple(lookup[p] for p in b.optical),
                b.optical_index,
            )
        else:
            cover = make_cover(b, state.position, config.stripes)
    rest = remaining(cover, b)
    if not rest:
        raise GeometryError("Optical cells exhausted while source remains unresolved")
    i, cell = rest[0]
    length = sum(
        distance(a, z)
        for a, z in zip(
            (state.position,) + tuple(c.center for _, c in rest), tuple(c.center for _, c in rest)
        )
    )
    proposal = put_cover(plan, replace(cover, cursor=i + 1))
    queue = () if b.optical else tuple(c.center for c in cover.cells)
    d = Decision(
        Intent("/clear", cell.center, b.channel, "service_grid", b.version),
        dict(
            category="optical",
            queue_id=cover.queue_id,
            optical_remaining=len(rest),
            optical_upper_s=length / 5 + 3 * (len(rest) - 1) + 5,
            cell_index=i,
            skipped_cells=i - cover.cursor,
        ),
        queue,
    )
    return d, proposal, cover, rest


@dataclass(frozen=True)
class Tail:
    point: Point
    rf_channel: int | None = None


def common_tail(state, b, plan):
    if plan.route:
        point = plan.route[0]
        channels = station_channels(state, plan, point)
        return Tail(point, channels[0] if channels else None)
    others = [
        k for k in state.beliefs if k.channel != b.channel and k.ever_detected and k.status not in RESOLVED
    ]
    if others:
        # Same arrival boundary for both actions, not a promise that this point receives RF.
        target = min(
            others,
            key=lambda k: (distance(bound(b.polygon).center, bound(k.polygon).center), k.channel),
        )
        return Tail(bound(target.polygon).center)
    return None


def tail_point(successor):
    return successor.point if isinstance(successor, Tail) else successor


def terminal(p, successor, incoming_channel=None):
    if successor is None:
        return 0
    cost = distance(p, tail_point(successor)) / 5
    if isinstance(successor, Tail) and successor.rf_channel is not None:
        cost += 5 + int(incoming_channel != successor.rf_channel)
    return cost


def optical_cost(world, start, rest, successor, incoming_channel=None):
    cost, prev = 0.0, start
    for _, cell in rest:
        q = cell.center
        cost += distance(prev, q) / 5
        if distance(world.position, q) <= 20:
            return cost + 5 + terminal(q, successor, incoming_channel)
        cost += 3
        prev = q
    return None


def branch_cost(b, p, result, angle, h, config, successor):
    if result == "near":
        return 5 + terminal(p, successor, b.channel)
    poly = b.polygon
    if result == "direction":
        poly = clip_wedge(poly, p, radians(angle), config.epsilon_rad, config.geometry_padding_m)
        poly = clip_disk(poly, p, 1500, config.disk_sides, config.geometry_padding_m)
        if not poly:
            return None
    shadow = replace(b, polygon=poly, ever_detected=True)
    safe = clear_points(shadow, p, tail_point(successor))
    if safe:
        return min(distance(p, q) / 5 + 5 + terminal(q, successor, b.channel) for q in safe)
    cover = make_cover(shadow, p, config.stripes)
    return optical_cost(h, p, tuple(enumerate(cover.cells)), successor, b.channel)


def score(values, risk):
    return (1 - risk) * sum(values) / len(values) + risk * max(values)


def compare(state, b, points, worlds, config, deadline, optical, rest, successor):
    # Identical (position,type,direction,radius,error) work scenarios price BOTH choices.
    selected = tuple(worlds[int(i * len(worlds) / min(4, len(worlds)))] for i in range(min(4, len(worlds))))
    scenarios = []
    from math import nextafter

    for h in selected:
        high = nextafter(h.radius_high, float("-inf")) if h.high_open else h.radius_high
        for r in dict.fromkeys((h.radius_low, h.radius, high)):
            if r < h.radius_low:
                continue
            fixed = replace(h, radius_low=r, radius_high=r, high_open=False)
            scenarios.extend((fixed, e) for e in (-1.0, 0.0, 1.0))
    scenarios = scenarios[: config.scenario_limit]
    optical_values = [optical_cost(h, state.position, rest, successor, state.channel) for h, _ in scenarios]
    if not scenarios or any(v is None for v in optical_values):
        return optical
    # One identically weighted conservative continuation scenario accompanies BOTH actions.
    # It is a work guard, not a sampled probability or an local-distribution claim.
    optical_guard = optical.detail["optical_upper_s"] + terminal(rest[-1][1].center, successor, state.channel)
    optical_values.append(optical_guard)
    best = score(optical_values, config.risk_weight)
    chosen = optical
    evaluations = []
    for p in sorted(points, key=lambda p: inherited_rank(state, b, p))[: config.rf_refine_limit]:
        if perf_counter() >= deadline:
            break
        values = []
        branches = set()
        for h, error in scenarios:
            if perf_counter() >= deadline:
                break
            result, angle = predict(h, p, error)
            branches.add(result)
            after = branch_cost(b, p, result, angle, h, config, successor)
            if after is None:
                break
            values.append(rf_cost(state, b.channel, p) + after)
        if len(values) != len(scenarios):
            break
        cov = make_cover(b, p, config.stripes)
        length = sum(
            distance(a, z)
            for a, z in zip((p,) + tuple(c.center for c in cov.cells), tuple(c.center for c in cov.cells))
        )
        guard = (
            rf_cost(state, b.channel, p)
            + length / 5
            + 3 * (len(cov.cells) - 1)
            + 5
            + terminal(cov.cells[-1].center, successor, b.channel)
        )
        values.append(guard)
        value = score(values, config.risk_weight)
        evaluations.append(
            dict(
                point=(p.x_m, p.y_m),
                score_s=value,
                branches=sorted(branches),
                no_signal_guard_s=guard,
            )
        )
        if value < best:
            best = value
            chosen = Decision(
                Intent("/measure", p, b.channel, "matched_continuation_rf", b.version),
                dict(category="time_rf", guaranteed_receive=positive_hull_contains(b, p)),
            )
    chosen.detail.update(
        score_s=best,
        rf_scores=evaluations,
        scenario_count=len(scenarios),
        score_kind="matched_optical_early_stop_and_RF_continuation",
        optical_score_s=score(optical_values, config.risk_weight),
        successor=(tail_point(successor).x_m, tail_point(successor).y_m) if successor else None,
        successor_rf_channel=successor.rf_channel if isinstance(successor, Tail) else None,
    )
    return chosen


def service_decision(state, b, plan, config, deadline, fallback):
    if fallback.intent.guaranteed:
        return fallback, plan
    points = (
        candidates(b, state.position, config)
        if config.candidate_families
        else rf_points(b, state.position, config)
    )
    was_optical = bool(b.optical)
    recoveries = dict(plan.recoveries)
    recovery_at = dict(plan.recovery_at)
    failures = len(b.exclusions)
    recovery = bool(
        was_optical
        and config.recovery_batch
        and recoveries.get(b.channel, 0) < 2
        and failures - recovery_at.get(b.channel, 0) >= config.recovery_batch
    )
    can_rf = bool(points) and (b.rf_used < config.rf_budget and not was_optical or recovery)
    if recovery:
        # A rejected RF comparison also consumes this short-batch review opportunity;
        # otherwise every subsequent optical failure would re-run the expensive lookahead.
        recovery_at[b.channel] = failures
        plan = replace(plan, recovery_at=tuple(sorted(recovery_at.items())))
    optical, new_plan, cover, rest = optical_choice(state, b, plan, config)
    if not can_rf:
        return optical, new_plan
    if not config.continuation and not recovery:
        if fallback.intent.path == "/clear":
            return optical, new_plan
        if config.candidate_families:
            p = min(points, key=lambda p: inherited_rank(state, b, p))
            fallback = Decision(
                Intent("/measure", p, b.channel, "family_rf", b.version),
                dict(category="rf_families", guaranteed_receive=positive_hull_contains(b, p)),
            )
        return fallback, plan
    if perf_counter() >= deadline:
        optical.detail["fallback_reason"] = "remaining_or_soft_budget"
        fallback.detail["fallback_reason"] = "remaining_or_soft_budget"
        return (optical, new_plan) if was_optical else (fallback, plan)
    hypothesis_start = perf_counter()
    worlds = hypotheses(b, config.scenario_limit)
    hypothesis_s = perf_counter() - hypothesis_start
    if not worlds:
        optical.detail["fallback_reason"] = "hypotheses_empty_not_absence"
        return optical, new_plan
    successor = common_tail(state, b, plan)
    if config.selective_planning:
        recent = [o.result for o in b.observations[-2:]]
        cheap = sorted(inherited_rank(state, b, p) for p in points)
        close = len(cheap) > 1 and cheap[1] - cheap[0] <= 30
        if not (len(rest) >= 16 or recent == ["no_signal", "no_signal"] or close or recovery):
            fallback.detail["planning_gate"] = "small_cover_clear_candidate_margin"
            return fallback, plan
    continuation_start = perf_counter()
    chosen = compare(state, b, points, worlds, config, deadline, optical, rest, successor)
    continuation_s = perf_counter() - continuation_start
    chosen.detail.update(hypothesis_s=hypothesis_s, continuation_s=continuation_s)
    if perf_counter() > deadline:
        chosen = optical if was_optical else fallback
        chosen.detail["fallback_reason"] = "planning_soft_budget"
        chosen.detail.update(hypothesis_s=hypothesis_s, continuation_s=continuation_s)
    if chosen.intent.path == "/clear":
        return optical, new_plan
    if recovery:
        recoveries[b.channel] = recoveries.get(b.channel, 0) + 1
        recovery_at[b.channel] = failures
        plan = replace(
            plan,
            recoveries=tuple(sorted(recoveries.items())),
            recovery_at=tuple(sorted(recovery_at.items())),
        )
        chosen.detail["recovery_rf_number"] = recoveries[b.channel]
    return chosen, plan
