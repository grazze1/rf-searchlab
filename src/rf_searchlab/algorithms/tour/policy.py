"""Adaptive search: block-level service tour, orientation-safe near-range RF,
capped en-route detours, and near-station third bearings.

Everything here only chooses among legal actions; reducers, certificates and the
executor use the shared observation and certificate components. Probabilities are ranking aids, never proofs.
"""

from dataclasses import replace
from math import radians, sin
from time import perf_counter

from rf_searchlab.algorithms.directional.belief import RESOLVED, bound
from rf_searchlab.algorithms.directional.candidates import clear_points, rf_points
from rf_searchlab.algorithms.directional.policy import baseline_service, inherited_rank
from rf_searchlab.algorithms.directional.visibility import positive_hull_contains
from rf_searchlab.algorithms.lookahead.planning import reconcile, station_channels
from rf_searchlab.algorithms.lookahead.policy import Decision
from rf_searchlab.algorithms.resilient.route import optimize
from rf_searchlab.algorithms.revisit.policy import Policy as RevisitPolicy
from rf_searchlab.algorithms.visibility.policy import Policy as VisibilityPolicy
from rf_searchlab.algorithms.visibility.policy import bearing_novelty, bump_rebearing, rebearing_count
from rf_searchlab.algorithms.visibility.visibility import reception_probability
from rf_searchlab.core.belief import wire_point
from rf_searchlab.core.domain import Intent, Point
from rf_searchlab.core.geometry import distance, max_distance


def backoff_points(b, config):
    """Stops short of the location set, inside the cone spanned by receiving stations.

    If stations s1, s2 received a source at g, the closed half-disk is convex, so the
    whole triangle (g, s1, s2) is received; points a few hundred metres from g along the
    station directions or their bisectors are therefore received whenever the true
    position is near the estimate. Bisectors add bearing diversity. Ranking only.
    """
    from math import atan2, cos, pi, sin

    m = bound(b.polygon).center
    history = {o.position for o in b.observations}
    angles = sorted(
        {
            atan2(o.position.y_m - m.y_m, o.position.x_m - m.x_m) % (2 * pi)
            for o in b.observations
            if o.result in ("direction", "near") and distance(o.position, m) > 1e-6
        }
    )
    directions = list(angles)
    for a, z in zip(angles, angles[1:] + [angles[0] + 2 * pi] if angles else []):
        gap = z - a
        if 1e-9 < gap < pi:
            directions.append(a + gap / 2)
    raw = [
        Point(m.x_m + back * cos(t), m.y_m + back * sin(t))
        for t in directions
        for back in config.safe_rf_backoffs_m
    ]
    out = []
    for p in dict.fromkeys(wire_point(p) for p in raw):
        if p not in history and max_distance(p, b.polygon) <= 1500:
            out.append(p)
    return tuple(out)


class ServicePolicy(VisibilityPolicy):
    """component with a capped en-route detour and near-station third bearings."""

    def enroute(self, state, plan, current, known):
        d = super().enroute(state, plan, current, known)
        if d is None or d.detail.get("category") != "enroute_clear":
            return d
        if d.detail["detour_m"] > self.config.enroute_max_detour_m:
            return None
        return d

    def rebearing(self, state, plan, known):
        d = super().rebearing(state, plan, known)
        if d is not None or not self.config.near_rebearing:
            return d
        cfg = self.config
        p = state.position
        best = None
        for b in known:
            if b.optical or b.near_position is not None:
                continue
            if rebearing_count(plan, b.channel) >= cfg.rebearing_limit:
                continue
            if any(o.position == p for o in b.observations):
                continue
            positives = sum(o.result in ("direction", "near") for o in b.observations)
            if positives < 2:
                continue
            circle = bound(b.polygon)
            if not cfg.rebearing_min_radius_m < circle.radius_m <= cfg.near_rebearing_max_radius_m:
                continue
            if distance(p, circle.center) > cfg.near_rebearing_station_m:
                continue
            novelty = bearing_novelty(b, p)
            if novelty < sin(radians(cfg.near_rebearing_min_angle_deg)):
                continue
            prob = reception_probability(b, p)
            if prob < cfg.rebearing_min_prob:
                continue
            key = (b.channel != state.channel, -prob, b.channel)
            if best is None or key < best[0]:
                best = (key, b, novelty, prob, circle.radius_m)
        if best is None:
            return None
        _, b, novelty, prob, radius = best
        return Decision(
            Intent("/measure", p, b.channel, "near_station_rebearing", b.version),
            bump_rebearing(plan, b.channel),
            dict(
                category="station_rebearing",
                macro="station",
                near_station=True,
                predicted_receive=prob,
                bearing_novelty=novelty,
                mec_radius_m=radius,
                rebearing_index=rebearing_count(plan, b.channel) + 1,
                guaranteed_receive=positive_hull_contains(b, p),
            ),
        )


class Policy(RevisitPolicy):
    def __init__(self, config, engine):
        super().__init__(config, engine)
        self.reference = ServicePolicy(config, engine)

    def decide(self, state, plan, remaining_real_s):
        start = perf_counter()
        cfg = self.config
        deadline = start + min(cfg.planning_budget_s, max(0, remaining_real_s - 0.05))
        tour_detail = {}
        if cfg.service_tour:
            plan, tour_detail = self.plan_tour(state, plan, deadline)
        decision = super().decide(state, plan, remaining_real_s)
        if (
            cfg.safe_rf
            and decision.intent.path == "/measure"
            and not decision.intent.guaranteed
            and decision.detail.get("macro") == "service"
            and decision.detail.get("category") in ("rf", "direction_recovery_rf")
        ):
            decision = self.safe_rf(state, decision)
        if tour_detail:
            decision.detail.update(tour_detail)
        decision.detail.update(planning_s=perf_counter() - start, profile=cfg.name)
        return decision

    # ---- P0-a: block-level service tour -------------------------------------------

    def plan_tour(self, state, plan, deadline):
        cfg = self.config
        if plan.service is not None and state.beliefs[plan.service - 1].status not in RESOLVED:
            return plan, {}
        if plan.pickup_channel is not None:
            return plan, {}
        current = reconcile(state, plan, self.engine)
        if current.route or station_channels(state, current, state.position):
            return plan, {}
        known = [b for b in state.beliefs if b.ever_detected and b.status not in RESOLVED]
        if len(known) < 2:
            return plan, {}
        shadow = self.reference.shadow(state, plan)
        try:
            order, value, method, first = self.block_tour(shadow, known, deadline)
        except TimeoutError as exc:
            return plan, dict(tour_fallback=str(exc))
        kept = tuple(c for c in plan.tour_order if c in {b.channel for b in known})
        if kept and kept != order:
            old = self.tour_length(shadow, known, kept, deadline)
            if old is not None and value > old - cfg.service_tour_margin_m / 5:
                order, value, method = kept, old, "kept_previous_order"
        new = replace(
            plan,
            service=order[0],
            service_actions=0,
            station=None,
            station_due=False,
            tour_order=order,
            tour_versions=tuple((b.channel, b.version) for b in known),
            tour_count=plan.tour_count + 1,
            macro_id=plan.macro_id + 1,
        )
        return new, dict(
            tour_order=order,
            tour_estimate_s=value,
            tour_method=method,
            tour_first_action_s=first,
            tour_kind="Block entry/exit estimate from public geometry; not a guaranteed time",
        )

    def block(self, shadow, b, position, channel):
        """(entry, exit, internal_s) of one service block from a given start."""
        cfg = self.config
        probe = replace(shadow, position=position, channel=channel)
        d = baseline_service(probe, b, cfg)
        if d.intent.path == "/clear" and d.intent.guaranteed:
            return d.intent.position, d.intent.position, 5.0
        entry = d.intent.position
        circle = bound(b.polygon)
        safe = clear_points(b, circle.center)
        exit_ = safe[0] if safe else wire_point(circle.center)
        internal = 5.0 + int(channel != b.channel) + distance(entry, exit_) / 5 + 5.0
        if d.intent.path == "/clear":  # optical queue head
            internal += d.detail.get("optical_upper_s", 0.0)
        return entry, exit_, internal

    def block_tour(self, shadow, known, deadline):
        n = len(known)
        blocks = [self.block(shadow, b, shadow.position, shadow.channel) for b in known]
        start = [distance(shadow.position, e) / 5 + c for e, _, c in blocks]
        edges = [[0.0] * n for _ in range(n)]
        for i in range(n):
            _, exit_i, _ = blocks[i]
            for j in range(n):
                if i == j:
                    continue
                if perf_counter() >= deadline:
                    raise TimeoutError("block_tour_budget")
                entry_j, exit_j, internal_j = self.block(shadow, known[j], exit_i, known[i].channel)
                edges[i][j] = distance(exit_i, entry_j) / 5 + internal_j
        order, value, method = optimize(start, edges, deadline)
        return tuple(known[i].channel for i in order), value, method, start[order[0]]

    def tour_length(self, shadow, known, order, deadline):
        index = {b.channel: i for i, b in enumerate(known)}
        pos, ch, total = shadow.position, shadow.channel, 0.0
        for c in order:
            if perf_counter() >= deadline:
                return None
            b = known[index[c]]
            entry, exit_, internal = self.block(shadow, b, pos, ch)
            total += distance(pos, entry) / 5 + internal
            pos, ch = exit_, b.channel
        return total

    # ---- P0-b: orientation-safe near-range RF -------------------------------------

    def safe_rf(self, state, decision):
        from rf_searchlab.algorithms.resilient.recovery import candidates

        cfg = self.config
        c = decision.intent.channel
        b = state.beliefs[c - 1]
        original = decision.intent.position
        p0 = reception_probability(b, original)
        if p0 >= cfg.safe_rf_min_prob:
            decision.detail.update(safe_rf="kept", fine_receive=p0)
            return decision
        shadow = self.reference.shadow(state, decision.plan)
        sb = shadow.beliefs[c - 1]
        pool = backoff_points(b, cfg) + candidates(sb, state.position, cfg)
        pool += rf_points(sb, state.position, cfg)
        scored = []
        for p in dict.fromkeys(pool):
            prob = reception_probability(b, p)
            scored.append((inherited_rank(state, b, p), prob, p))
        safe = [s for s in scored if s[1] >= cfg.safe_rf_min_prob]
        if safe:
            rank, prob, p = min(safe, key=lambda s: (s[0], s[2].x_m, s[2].y_m))
            tag = "safe_rf"
        else:
            eligible = [s for s in scored if s[1] >= cfg.service_guard_min_prob]
            if not eligible:
                decision.detail.update(safe_rf="no_candidate", fine_receive=p0)
                return decision
            rank, prob, p = min(eligible, key=lambda s: (s[0] + (1 - s[1]) * 300, s[2].x_m, s[2].y_m))
            tag = "service_guard"
        if p == original or prob <= p0:
            decision.detail.update(safe_rf="kept", fine_receive=p0)
            return decision
        detail = dict(
            decision.detail,
            safe_rf=tag,
            original_point=(original.x_m, original.y_m),
            original_fine_receive=p0,
            predicted_receive=prob,
            safe_rf_rank_s=rank,
            guaranteed_receive=positive_hull_contains(b, p),
        )
        new = Decision(
            Intent("/measure", p, c, tag + "_" + decision.intent.reason, b.version),
            replace(decision.plan, safe_rf_count=decision.plan.safe_rf_count + 1),
            detail,
        )
        return new
