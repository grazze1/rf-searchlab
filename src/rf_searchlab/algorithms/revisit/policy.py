"""Shorter remaining scan tours and bounded pickups based on public geometry."""

from dataclasses import replace
from time import perf_counter

from rf_searchlab.algorithms.directional.belief import RESOLVED, bound
from rf_searchlab.algorithms.directional.candidates import clear_points
from rf_searchlab.algorithms.directional.policy import baseline_service
from rf_searchlab.algorithms.lookahead.planning import reconcile, route_length, station_channels
from rf_searchlab.algorithms.route.tour import open_tour
from rf_searchlab.algorithms.visibility.policy import Policy as Previous
from rf_searchlab.algorithms.visibility.visibility import reception_probability
from rf_searchlab.core.geometry import distance


def repair_route(state, plan):
    """Permute outstanding scan stops only; never delete any coverage obligation."""
    nodes = plan.route
    length, order = open_tour((state.position.x_m, state.position.y_m), tuple((p.x_m, p.y_m) for p in nodes))
    old = route_length(state.position, nodes)
    if length < old - 1e-6:
        return replace(
            plan, route=tuple(nodes[i] for i in order), scan_reroutes=plan.scan_reroutes + 1
        ), old - length
    return plan, 0.0


class Policy:
    def __init__(self, config, engine):
        self.config, self.engine = config, engine
        self.reference = Previous(config, engine)

    def decide(self, state, plan, remaining_real_s):
        start = perf_counter()
        cfg = self.config
        if not (cfg.reroute_scan or cfg.local_pickup):
            return self.reference.decide(state, plan, remaining_real_s)
        saving = 0.0
        abort = None
        if plan.pickup_channel is not None:
            b = state.beliefs[plan.pickup_channel - 1]
            failed_probe = plan.pickup_steps and b.observations[-1].result == "no_signal"
            exhausted = plan.pickup_steps >= cfg.pickup_rf_limit and not clear_points(b, state.position)
            if b.status in RESOLVED or failed_probe or exhausted or b.optical:
                abort = "resolved" if b.status in RESOLVED else "bounded_pickup_resume_scan"
                plan = replace(
                    plan,
                    pickup_channel=None,
                    pickup_steps=0,
                    service=None,
                    service_actions=0,
                    station=None,
                    station_due=True,
                )
        current = reconcile(state, plan, self.engine)
        idle = plan.service is None or state.beliefs[plan.service - 1].status in RESOLVED
        boundary = (
            idle
            and not station_channels(state, current, state.position)
            and not (plan.station is not None and station_channels(state, current, plan.station))
        )
        if cfg.reroute_scan and boundary and current.route:
            plan, saving = repair_route(state, current)
            current = plan
        fallback = self.reference.decide(state, plan, remaining_real_s)
        decision = fallback
        if (
            cfg.local_pickup
            and abort is None
            and boundary
            and current.route
            and state.position in self.reference.stops
            and fallback.detail.get("category") == "coverage"
        ):
            offer = self.pickup(state, current)
            if offer is not None:
                channel, detail = offer
                proposal = replace(
                    current,
                    station=None,
                    service=channel,
                    station_due=False,
                    service_actions=0,
                    pickup_channel=channel,
                    pickup_steps=0,
                    pickup_attempted=current.pickup_attempted + (channel,),
                    pickup_count=current.pickup_count + 1,
                )
                decision = self.reference.decide(state, proposal, remaining_real_s)
                decision.detail.update(detail)
        if decision.plan.pickup_channel is not None:
            c = decision.plan.pickup_channel
            if decision.intent.path == "/measure" and decision.intent.channel == c:
                decision = replace(
                    decision,
                    plan=replace(decision.plan, pickup_steps=decision.plan.pickup_steps + 1),
                )
            decision.detail["local_pickup_channel"] = c
        decision.detail.update(
            planning_s=perf_counter() - start, scan_route_saving_m=saving, pickup_stop=abort
        )
        return decision

    def pickup(self, state, plan):
        """Take a nearby source before its closest scan stop is left behind.

        This geometric estimate ranks an optional service only. It does not certify
        reception, clearability or absence; accepted responses govern all progress.
        """
        cfg = self.config
        shadow = self.reference.shadow(state, plan)
        best = None
        for b in shadow.beliefs:
            if (
                not b.ever_detected
                or b.status in RESOLVED
                or b.optical
                or b.channel in plan.pickup_attempted
                or b.near_position is not None
                or b.rf_used >= cfg.rf_budget
            ):
                continue
            circle = bound(b.polygon)
            here = distance(state.position, circle.center)
            later = min(distance(p, circle.center) for p in plan.route)
            if (
                circle.radius_m > cfg.pickup_max_radius_m
                or here > cfg.pickup_max_distance_m
                or later - here < cfg.pickup_min_future_gap_m
            ):
                continue
            offer = baseline_service(shadow, b, cfg)
            if offer.intent.path != "/measure" or offer.optical_queue:
                continue
            q = offer.intent.position
            detour = (
                distance(state.position, q)
                + distance(q, plan.route[0])
                - distance(state.position, plan.route[0])
            )
            if detour > cfg.pickup_max_detour_m:
                continue
            probability = reception_probability(b, q)
            if probability < cfg.pickup_min_probability:
                continue
            score = detour / 5 + 6 - (later - here) / 5
            item = (
                score,
                b.channel,
                dict(
                    pickup_estimated_score_s=score,
                    pickup_detour_m=detour,
                    pickup_receive_probability=probability,
                    pickup_future_gap_m=later - here,
                ),
            )
            if best is None or item[:2] < best[:2]:
                best = item
        return (best[1], best[2]) if best is not None else None
