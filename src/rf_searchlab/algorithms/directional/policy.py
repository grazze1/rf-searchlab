"""One-action Mixed policies A--F. Predictions rank; only observed evidence certifies."""

from dataclasses import dataclass, replace
from math import atan2, radians, sin
from time import perf_counter

from rf_searchlab.algorithms.directional.belief import RESOLVED, bound, safe_clear
from rf_searchlab.algorithms.directional.candidates import clear_points, hypotheses, predict, rf_points
from rf_searchlab.algorithms.directional.optical import remaining_upper, service_upper
from rf_searchlab.algorithms.directional.planner import build_route, fixed_tasks, shared_scan_cost
from rf_searchlab.algorithms.directional.visibility import positive_hull_contains
from rf_searchlab.core.domain import GeometryError, Intent, Point
from rf_searchlab.core.geometry import clip_disk, clip_wedge, distance, polygon_distance


@dataclass(frozen=True)
class Decision:
    intent: Intent
    detail: dict
    optical_queue: tuple = ()


def immediate_clear(state):
    for b in state.beliefs:
        if b.status in RESOLVED:
            continue
        if b.near_position is not None:
            return Intent("/clear", b.near_position, b.channel, "near", b.version, guaranteed=True)
    for b in state.beliefs:
        if b.status not in RESOLVED and safe_clear(b, state.position):
            return Intent("/clear", state.position, b.channel, "safe_here", b.version, guaranteed=True)
    return None


def rf_cost(state, c, p):
    return distance(state.position, p) / 5 + 5 + int(state.channel != c)


def inherited_rank(state, b, p):
    """Omnidirectional-style cheap geometric ranking, without a reception guarantee."""
    m = bound(b.polygon).center
    theta = atan2(m.y_m - p.y_m, m.x_m - p.x_m)
    angles = [
        atan2(m.y_m - o.position.y_m, m.x_m - o.position.x_m)
        for o in b.observations
        if o.result in ("direction", "near")
    ]
    quality = max((abs(sin(theta - a)) for a in angles), default=0)
    return rf_cost(state, b.channel, p) + distance(p, m) / 5 + 20 / (0.05 + quality)


def optical_decision(state, b, reason):
    upper, points = remaining_upper(b, state.position)
    return Decision(
        Intent("/clear", points[0], b.channel, reason, b.version),
        dict(category="optical", optical_upper_s=upper, optical_remaining=len(points)),
        () if b.optical else points,
    )


def baseline_service(state, b, config):
    safe = clear_points(b, state.position)
    if safe:
        p = min(safe, key=lambda p: distance(state.position, p))
        return Decision(
            Intent("/clear", p, b.channel, "safe_clear", b.version, guaranteed=True),
            dict(category="safe_clear"),
        )
    if b.optical or b.rf_used >= config.rf_budget:
        return optical_decision(state, b, "optical_queue")
    points = rf_points(b, state.position, config)
    if not points:
        return optical_decision(state, b, "no_new_rf")
    # Preserve the inherited elongated-set approach before changing its ranking in D.
    from rf_searchlab.core.geometry import diameter

    inherited = points[:2] if diameter(b.polygon)[0] >= config.approach_min_diameter_m else points
    p = min(inherited, key=lambda p: inherited_rank(state, b, p))
    return Decision(
        Intent("/measure", p, b.channel, "inherited_rf", b.version),
        dict(category="rf", guaranteed_receive=False),
    )


def price_after(b, p, result, angle, config):
    if result == "near":
        return 5.0, p
    poly = b.polygon
    if result == "direction":
        poly = clip_wedge(poly, p, radians(angle), config.epsilon_rad, config.geometry_padding_m)
        poly = clip_disk(poly, p, 1500, config.disk_sides, config.geometry_padding_m)
        if not poly:
            return None
    # no_signal keeps the same location set. This is an executable finite continuation,
    # not a false claim that the source must become visible at the next stop.
    hypothetical = replace(b, polygon=poly, ever_detected=True)
    safe = clear_points(hypothetical, p)
    if safe:
        q = min(safe, key=lambda q: distance(p, q))
        return distance(p, q) / 5 + 5, q
    upper, queue = service_upper(poly, p)
    return upper, queue[-1]


class Policy:
    def __init__(self, config, engine):
        self.config, self.engine = config, engine
        self.optimization_actions = 0
        self.detail = {}

    def decide(self, state):
        start = perf_counter()
        config = self.config
        deadline = start + config.planning_budget_s
        urgent = immediate_clear(state)
        if urgent:
            return self.finish(Decision(urgent, dict(category="immediate_clear")), start)
        known = [b for b in state.beliefs if b.ever_detected and b.status not in RESOLVED]
        obligations = fixed_tasks(state, self.engine.layout)
        forced = bool(obligations) and self.optimization_actions >= config.fairness_actions
        if not known or forced:
            route = build_route(state, self.engine, replace(config, variant="B"), deadline)
            for p in route.stops:
                if route.channels[p]:
                    c = route.channels[p][0]
                    self.optimization_actions = 0
                    return self.finish(
                        Decision(
                            Intent("/measure", p, c, "fixed_obligation", state.beliefs[c - 1].version),
                            dict(
                                category="coverage",
                                fairness_forced=forced,
                                route_length_m=route.length,
                            ),
                        ),
                        start,
                    )
            raise GeometryError("No known source and no unmeasured certificate obligation")
        b = min(known, key=lambda b: (distance(state.position, bound(b.polygon).center), b.channel))
        fallback = baseline_service(state, b, replace(config, variant="B"))
        self.optimization_actions += 1
        decision = fallback
        if config.directional and fallback.intent.path == "/measure":
            points = rf_points(b, state.position, config)
            worlds = hypotheses(b, config.scenario_limit)
            if not worlds:
                decision = optical_decision(state, b, "empty_scoring_hypotheses")
                decision.detail["fallback_reason"] = "hypotheses_empty_not_absence"
            elif config.time_compare:
                decision = self.compare_time(state, b, points, worlds, deadline, fallback)
            else:
                scored = []
                for p in points:
                    results = [predict(h, p)[0] for h in worlds]
                    prob = sum(r != "no_signal" for r in results) / len(results)
                    # D deliberately remains a geometry heuristic; E introduces seconds-only
                    # executable continuation comparison, measured as its own ablation.
                    value = inherited_rank(state, b, p) + (1 - prob) * 100
                    scored.append((value, p, prob))
                if scored:
                    value, p, prob = min(scored, key=lambda x: (x[0], x[1].x_m, x[1].y_m))
                    decision = Decision(
                        Intent("/measure", p, b.channel, "direction_aware_rf", b.version),
                        dict(
                            category="directional_rf",
                            predicted_receive=prob,
                            guaranteed_receive=positive_hull_contains(b, p),
                            heuristic_score=value,
                            hypotheses=len(worlds),
                        ),
                    )
        if config.shared and perf_counter() < deadline:
            route = build_route(state, self.engine, config, deadline)
            if perf_counter() < deadline:
                for p in route.stops[:2]:
                    cs = route.channels[p]
                    if not cs:
                        continue
                    scan = shared_scan_cost(state, route, p, cs)
                    c = cs[0]
                    # Compare actual first-action and successor route cost; all channels still pay.
                    service = distance(state.position, decision.intent.position) / 5
                    service += 5 + int(decision.intent.path == "/measure" and state.channel != b.channel)
                    via = rf_cost(state, c, p) + distance(p, decision.intent.position) / 5
                    gain = self.partial_gain(state, c, p)
                    threshold = 0.65 if config.strict65 else 0.0
                    if (
                        p == state.position
                        and gain > threshold
                        and scan <= 6.0
                        or via < service + scan
                        and gain > threshold
                    ):
                        decision = Decision(
                            Intent("/measure", p, c, "shared_scan", state.beliefs[c - 1].version),
                            dict(
                                category="shared_scan",
                                scan_increment_s=scan,
                                conditional_deleted=route.deleted_tasks,
                                partial_direction_gain=gain,
                                route_length_m=route.length,
                            ),
                        )
                        break
        if perf_counter() > deadline:
            fallback.detail["fallback_reason"] = "planning_soft_budget"
            fallback.detail["discarded_category"] = decision.detail.get("category")
            decision = fallback
        return self.finish(decision, start)

    def partial_gain(self, state, c, p):
        # Coarse position/direction samples ONLY reward partially closed surrounds.
        # They have no connection to the exact certificate update API.
        from math import cos, pi, sin

        from rf_searchlab.algorithms.directional.visibility import visible

        negatives = [o.position for o in state.beliefs[c - 1].observations if o.result == "no_signal"]
        remaining, gain = 0, 0
        for radius in (450, 1200, 1800):
            for k in range(12):
                g = Point(radius * cos(k * pi / 6), radius * sin(k * pi / 6))
                for j in range(8):
                    phi = j * pi / 4
                    if not any(visible(s, g, 1000, phi) for s in negatives):
                        remaining += 1
                        gain += visible(p, g, 1000, phi)
        return gain / max(1, remaining)

    def compare_time(self, state, b, points, worlds, deadline, fallback):
        optical = optical_decision(state, b, "time_selected_optical")
        subsequent = [
            bound(other.polygon).center
            for other in state.beliefs
            if other.channel != b.channel and other.ever_detected and other.status not in RESOLVED
        ]
        subsequent.extend(p for pending in fixed_tasks(state, self.engine.layout).values() for p in pending)
        successor = (
            min(subsequent, key=lambda p: distance(bound(b.polygon).center, p)) if subsequent else None
        )

        def departure(p):
            return distance(p, successor) / 5 if successor is not None else 0.0

        _, optical_points = remaining_upper(b, state.position)
        best = optical.detail["optical_upper_s"] + departure(optical_points[-1])
        chosen = optical
        evaluations = []
        # Endpoint errors are included within the scenario budget, not in addition to it.
        # Shared source-radius intervals are sampled at both feasible endpoints and midpoint.
        from math import nextafter

        expanded = []
        count = min(
            len(worlds),
            self.config.hypothesis_refine_limit,
            max(1, self.config.scenario_limit // 9),
        )
        selected = tuple(worlds[int(i * len(worlds) / count)] for i in range(count))
        for h in selected:
            high = nextafter(h.radius_high, float("-inf")) if h.high_open else h.radius_high
            for r in dict.fromkeys((h.radius_low, h.radius, high)):
                if r >= h.radius_low:
                    expanded.append(replace(h, radius_low=r, radius_high=r, high_open=False))
        worlds = tuple(expanded)
        ranked = sorted(
            points,
            key=lambda p: (
                inherited_rank(state, b, p)
                + 100 * (1 - sum(predict(h, p)[0] != "no_signal" for h in worlds) / len(worlds))
            ),
        )
        for p in ranked[: self.config.rf_refine_limit]:
            if perf_counter() >= deadline:
                break
            values, branches = [], set()
            for h in worlds:
                for error in (-1.0, 0.0, 1.0):
                    result, angle = predict(h, p, error)
                    branches.add(result)
                    after = price_after(b, p, result, angle, self.config)
                    if after is None:
                        continue
                    continuation = after[0] + departure(after[1])
                    if (
                        self.config.lookahead_depth == 2
                        and result == "direction"
                        and perf_counter() < deadline
                    ):
                        continuation = min(
                            continuation, self.second_rf_cost(b, p, angle, h, successor, deadline)
                        )
                    values.append(rf_cost(state, b.channel, p) + continuation)
                    if perf_counter() >= deadline:
                        break
                if perf_counter() >= deadline:
                    break
            if len(values) != len(worlds) * 3:
                break  # no optimistic score from an interrupted subset of branches
            # Finite hypotheses can miss a legal visibility branch. Include a conservative
            # no-signal continuation unless real-positive convexity proves reception.
            if "no_signal" not in branches and not positive_hull_contains(b, p):
                after = price_after(b, p, "no_signal", None, self.config)
                values.append(rf_cost(state, b.channel, p) + after[0] + departure(after[1]))
                branches.add("no_signal")
            if "near" not in branches and polygon_distance(p, b.polygon) <= 5:
                values.append(rf_cost(state, b.channel, p) + 5 + departure(p))
                branches.add("near")
            score = (1 - self.config.risk_weight) * sum(values) / len(values) + self.config.risk_weight * max(
                values
            )
            evaluations.append(dict(point=(p.x_m, p.y_m), score_s=score, branches=sorted(branches)))
            if score < best:
                best = score
                prob = sum(predict(h, p)[0] != "no_signal" for h in worlds) / len(worlds)
                chosen = Decision(
                    Intent("/measure", p, b.channel, "time_selected_rf", b.version),
                    dict(
                        category="time_rf",
                        predicted_receive=prob,
                        guaranteed_receive=positive_hull_contains(b, p),
                    ),
                )
        chosen.detail.update(
            score_s=best,
            rf_scores=evaluations,
            optical_upper_s=optical.detail["optical_upper_s"],
            successor=(successor.x_m, successor.y_m) if successor else None,
            score_kind="finite_scenario_mean_and_max_seconds",
        )
        return chosen

    def second_rf_cost(self, b, first, angle, h, successor, deadline):
        """Optional bounded second RF; hypothetical history never enters the real reducer."""
        from rf_searchlab.algorithms.directional.candidates import PredictedObservation

        poly = clip_wedge(b.polygon, first, radians(angle), self.config.epsilon_rad)
        poly = clip_disk(poly, first, 1500, self.config.disk_sides)
        if not poly:
            return float("inf")
        shadow = replace(
            b,
            polygon=poly,
            observations=b.observations + (PredictedObservation(first, "direction", angle),),
        )
        values = []
        for q in rf_points(shadow, first, self.config)[:2]:
            if perf_counter() >= deadline:
                break
            worst = []
            for error in (-1.0, 0.0, 1.0):
                result, bearing = predict(h, q, error)
                after = price_after(shadow, q, result, bearing, self.config)
                if after is not None:
                    tail = distance(after[1], successor) / 5 if successor is not None else 0
                    worst.append(distance(first, q) / 5 + 5 + after[0] + tail)
            if len(worst) == 3:
                values.append(max(worst))
        return min(values, default=float("inf"))

    def finish(self, decision, start):
        decision.detail.update(
            planning_s=perf_counter() - start,
            variant=self.config.variant,
            optimization_actions=self.optimization_actions,
            certificate_progress=dict(self.engine.progress),
        )
        self.detail = decision.detail
        return decision


def install_queue(state, decision):
    if not decision.optical_queue:
        return state
    c = decision.intent.channel
    beliefs = list(state.beliefs)
    if not beliefs[c - 1].optical:
        beliefs[c - 1] = replace(beliefs[c - 1], optical=decision.optical_queue, optical_index=0)
    return replace(state, beliefs=tuple(beliefs))
