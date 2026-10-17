"""Free triangulation at fixed stops plus cheap en-route guaranteed clears.

Both additions only add accepted physical actions; the frozen component policy still makes
every other decision, and no prediction enters the certificate or belief reducers.
"""

from dataclasses import replace
from math import atan2, radians, sin
from time import perf_counter

from rf_searchlab.algorithms.directional.belief import RESOLVED, bound
from rf_searchlab.algorithms.directional.candidates import clear_points
from rf_searchlab.algorithms.directional.policy import immediate_clear
from rf_searchlab.algorithms.directional.visibility import positive_hull_contains
from rf_searchlab.algorithms.lookahead.planning import reconcile, station_channels
from rf_searchlab.algorithms.lookahead.policy import Decision
from rf_searchlab.algorithms.resilient.policy import Policy as Previous
from rf_searchlab.algorithms.visibility.visibility import reception_probability
from rf_searchlab.core.domain import Intent
from rf_searchlab.core.geometry import distance, polygon_distance


def rebearing_count(plan, channel):
    return dict(plan.rebearings).get(channel, 0)


def bump_rebearing(plan, channel):
    counts = dict(plan.rebearings)
    counts[channel] = counts.get(channel, 0) + 1
    return replace(plan, rebearings=tuple(sorted(counts.items())))


def bearing_novelty(b, p):
    """Smallest |sin| between the new bearing and every existing positive bearing."""
    m = bound(b.polygon).center
    theta = atan2(m.y_m - p.y_m, m.x_m - p.x_m)
    novelty = 1.0
    for o in b.observations:
        if o.result in ("direction", "near"):
            a = atan2(m.y_m - o.position.y_m, m.x_m - o.position.x_m)
            novelty = min(novelty, abs(sin(theta - a)))
    return novelty


class Policy:
    def __init__(self, config, engine):
        self.config, self.engine = config, engine
        self.reference = Previous(config, engine)
        self.stops = frozenset(engine.layout.stops)

    def decide(self, state, plan, remaining_real_s):
        start = perf_counter()
        cfg = self.config
        urgent = immediate_clear(state)
        if urgent:
            return self.finish(Decision(urgent, plan, dict(category="immediate_clear")), start)
        current = reconcile(state, plan, self.engine)
        known = [b for b in state.beliefs if b.ever_detected and b.status not in RESOLVED]
        committed = plan.service is not None and state.beliefs[plan.service - 1].status not in (RESOLVED)
        pending_here = bool(station_channels(state, current, state.position))
        mid_batch = plan.station is not None and bool(station_channels(state, current, plan.station))
        if current.route and known and not committed and not pending_here and not mid_batch:
            if cfg.rebearing and state.position in self.stops:
                d = self.rebearing(state, plan, known)
                if d is not None:
                    return self.finish(d, start)
            if cfg.enroute_clear:
                d = self.enroute(state, plan, current, known)
                if d is not None:
                    return self.finish(d, start)
        shadow = self.shadow(state, plan)
        decision = self.reference.decide(shadow, plan, remaining_real_s)
        category = decision.detail.get("category")
        if (
            decision.intent.path == "/measure"
            and not decision.intent.guaranteed
            and (
                (cfg.recovery_gate and category == "direction_recovery_rf")
                or (cfg.blind_guard and category == "rf")
            )
        ):
            decision = self.guard(shadow, decision)
        decision.detail.update(planning_s=perf_counter() - start, profile=cfg.name)
        return decision

    def guard(self, state, decision):
        """Only when the frozen choice looks blind under the fine orientation model,
        re-rank the frozen candidate pools with the frozen penalty formula and the
        finer reception probability. Normal decisions are left untouched."""
        from rf_searchlab.algorithms.directional.candidates import rf_points
        from rf_searchlab.algorithms.directional.optical import remaining_upper
        from rf_searchlab.algorithms.directional.policy import inherited_rank
        from rf_searchlab.algorithms.resilient.recovery import candidates

        cfg = self.config
        c = decision.intent.channel
        b = state.beliefs[c - 1]
        original = decision.intent.position
        prob0 = reception_probability(b, original)
        if prob0 >= cfg.guard_min_prob:
            decision.detail.update(guard="kept", fine_receive=prob0)
            return decision
        pool = candidates(b, state.position, cfg) + rf_points(b, state.position, cfg)
        points = list(dict.fromkeys(pool))
        if not points:
            return decision
        upper, _ = remaining_upper(b, state.position)
        penalty = min(600.0, max(100.0, upper * 0.5))
        scored = []
        for p in points:
            prob = reception_probability(b, p)
            scored.append((inherited_rank(state, b, p) + (1 - prob) * penalty, prob, p))
        eligible = [s for s in scored if s[1] >= cfg.guard_min_prob] or scored
        value, prob, p = min(eligible, key=lambda s: (s[0], s[2].x_m, s[2].y_m))
        if p == original or prob <= prob0:
            decision.detail.update(guard="kept", fine_receive=prob0)
            return decision
        detail = dict(
            decision.detail,
            category=decision.detail["category"],
            guard="reranked",
            original_point=(original.x_m, original.y_m),
            original_fine_receive=prob0,
            predicted_receive=prob,
            guard_score_s=value,
            guaranteed_receive=positive_hull_contains(b, p),
        )
        return Decision(
            Intent("/measure", p, c, "guarded_" + decision.intent.reason, b.version),
            decision.plan,
            detail,
        )

    def shadow(self, state, plan):
        """Re-bearings never consume the classical RF budget of the frozen service, and a
        bounded extension keeps RF available while the optical fallback would be huge."""
        from rf_searchlab.algorithms.directional.optical import remaining_upper

        cfg = self.config
        counts = dict(plan.rebearings)
        beliefs = list(state.beliefs)
        changed = False
        for i, b in enumerate(beliefs):
            if b.status in RESOLVED or not b.ever_detected or b.optical:
                continue
            used = max(0, b.rf_used - counts.get(b.channel, 0))
            if cfg.rf_extension and cfg.rf_budget <= used < cfg.rf_budget + cfg.rf_extension:
                upper, _ = remaining_upper(b, state.position)
                if upper > cfg.rf_extension_upper_s:
                    used = cfg.rf_budget - 1
            if used != b.rf_used:
                beliefs[i] = replace(b, rf_used=used)
                changed = True
        return replace(state, beliefs=tuple(beliefs)) if changed else state

    def rebearing(self, state, plan, known):
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
            if polygon_distance(p, b.polygon) > cfg.rebearing_max_range_m:
                continue
            circle = bound(b.polygon)
            if circle.radius_m <= cfg.rebearing_min_radius_m:
                continue
            novelty = bearing_novelty(b, p)
            if novelty < sin(radians(cfg.rebearing_min_angle_deg)):
                continue
            prob = reception_probability(b, p)
            if prob < cfg.rebearing_min_prob:
                continue
            key = (b.channel != state.channel, -prob * novelty, b.channel)
            if best is None or key < best[0]:
                best = (key, b, novelty, prob, circle.radius_m)
        if best is None:
            return None
        _, b, novelty, prob, radius = best
        return Decision(
            Intent("/measure", p, b.channel, "station_rebearing", b.version),
            bump_rebearing(plan, b.channel),
            dict(
                category="station_rebearing",
                macro="station",
                predicted_receive=prob,
                bearing_novelty=novelty,
                mec_radius_m=radius,
                rebearing_index=rebearing_count(plan, b.channel) + 1,
                guaranteed_receive=positive_hull_contains(b, p),
            ),
        )

    def enroute(self, state, plan, current, known):
        cfg = self.config
        nxt = current.route[0]
        base = distance(state.position, nxt)
        best = None
        for b in known:
            if b.optical:
                continue
            safe = clear_points(b, state.position, nxt)
            if not safe:
                continue
            q = min(safe, key=lambda s: (distance(state.position, s) + distance(s, nxt), s.x_m))
            detour = distance(state.position, q) + distance(q, nxt) - base
            later = min(distance(s, q) for s in current.route)
            if detour <= cfg.enroute_ratio * later + cfg.enroute_slack_m:
                if best is None or detour < best[0]:
                    best = (detour, b, q, later)
        if best is None:
            return self.enroute_rf(state, plan, current, known) if cfg.enroute_rf else None
        detour, b, q, later = best
        return Decision(
            Intent("/clear", q, b.channel, "enroute_safe_clear", b.version, guaranteed=True),
            replace(plan, enroute_clears=plan.enroute_clears + 1),
            dict(
                category="enroute_clear",
                macro="enroute",
                detour_m=detour,
                later_one_way_m=later,
                successor=(nxt.x_m, nxt.y_m),
            ),
        )

    def enroute_rf(self, state, plan, current, known):
        """One cheap bearing for an almost-localised source lying near the scan path."""
        from rf_searchlab.algorithms.directional.candidates import rf_points
        from rf_searchlab.algorithms.resilient.recovery import candidates

        cfg = self.config
        nxt = current.route[0]
        base = distance(state.position, nxt)
        best = None
        for b in known:
            if b.optical or b.near_position is not None:
                continue
            exhausted = rebearing_count(plan, b.channel) >= cfg.rebearing_limit
            if b.rf_used >= cfg.rf_budget or exhausted:
                continue
            circle = bound(b.polygon)
            if circle.radius_m > cfg.enroute_rf_radius_m:
                continue
            pool = candidates(b, state.position, cfg) + rf_points(b, state.position, cfg)
            for q in dict.fromkeys(pool):
                detour = distance(state.position, q) + distance(q, nxt) - base
                later = min(distance(s, q) for s in current.route)
                if detour > cfg.enroute_ratio * later + cfg.enroute_slack_m:
                    continue
                novelty = bearing_novelty(b, q)
                if novelty < sin(radians(cfg.rebearing_min_angle_deg)):
                    continue
                prob = reception_probability(b, q)
                if prob < cfg.enroute_rf_min_prob:
                    continue
                score = detour / 5 + (1 - prob) * 100 + 20 / (0.05 + novelty)
                if best is None or score < best[0]:
                    best = (score, b, q, detour, prob, novelty)
        if best is None:
            return None
        score, b, q, detour, prob, novelty = best
        return Decision(
            Intent("/measure", q, b.channel, "enroute_rf", b.version),
            bump_rebearing(plan, b.channel),
            dict(
                category="enroute_rf",
                macro="enroute",
                detour_m=detour,
                predicted_receive=prob,
                bearing_novelty=novelty,
                mec_radius_m=bound(b.polygon).radius_m,
                guaranteed_receive=positive_hull_contains(b, q),
                successor=(nxt.x_m, nxt.y_m),
            ),
        )

    def finish(self, decision, start):
        decision.detail.update(
            planning_s=perf_counter() - start,
            profile=self.config.name,
            macro_id=decision.plan.macro_id,
            service_channel=decision.plan.service,
            service_actions=decision.plan.service_actions,
        )
        return decision
