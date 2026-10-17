"""One action per call, persistent station batches and finite committed source service."""

from dataclasses import dataclass, replace
from time import perf_counter

from rf_searchlab.algorithms.directional.belief import RESOLVED
from rf_searchlab.algorithms.directional.candidates import clear_points
from rf_searchlab.algorithms.directional.optical import remaining_upper
from rf_searchlab.algorithms.directional.policy import Policy as LegacyPolicy
from rf_searchlab.algorithms.directional.policy import baseline_service, immediate_clear
from rf_searchlab.algorithms.lookahead.planning import Plan, reconcile, segment_delta, station_channels
from rf_searchlab.core.domain import GeometryError, Intent
from rf_searchlab.core.geometry import distance


@dataclass(frozen=True)
class Decision:
    intent: Intent
    plan: Plan
    detail: dict
    optical_queue: tuple = ()


class Policy:
    def __init__(self, config, engine):
        self.config, self.engine = config, engine
        self.reference = LegacyPolicy(config, engine)

    def decide(self, state, plan, remaining_real_s):
        start = perf_counter()
        cfg = self.config
        if plan.covers:
            plan = replace(
                plan,
                covers=tuple(c for c in plan.covers if state.beliefs[c.channel - 1].status not in RESOLVED),
            )
        deadline = start + min(cfg.planning_budget_s, max(0, remaining_real_s - 0.05))
        urgent = immediate_clear(state)
        if urgent:
            return self.finish(urgent, plan, dict(category="immediate_clear"), start)
        if not (cfg.station_batch or cfg.service_commit or cfg.two_phase):
            self.reference.optimization_actions = plan.optimization_actions
            d = self.reference.decide(state)
            plan = replace(plan, optimization_actions=self.reference.optimization_actions)
            return self.finish(d.intent, plan, d.detail, start, d.optical_queue)
        plan = reconcile(state, plan, self.engine)
        if plan.service is not None and state.beliefs[plan.service - 1].status in RESOLVED:
            plan = replace(plan, service=None, station_due=True, service_actions=0)
        if plan.station is not None and not station_channels(state, plan, plan.station):
            plan = replace(plan, station=None, station_due=False)
        conditional_s, deleted = 0.0, 0
        if cfg.conditional_tasks and plan.station is None and plan.service is None:
            from rf_searchlab.algorithms.lookahead.conditional import improve

            t = perf_counter()
            plan, deleted = improve(state, plan, self.engine, deadline)
            plan = reconcile(state, plan, self.engine)
            if station_channels(state, plan, state.position):
                plan = replace(
                    plan,
                    route=(state.position,) + tuple(p for p in plan.route if p != state.position),
                    station_due=True,
                )
            conditional_s = perf_counter() - t
        known = [b for b in state.beliefs if b.ever_detected and b.status not in RESOLVED]
        if (
            plan.station is None
            and plan.service is None
            and plan.route
            and (plan.station_due or not known or cfg.two_phase)
        ):
            plan = replace(plan, station=plan.route[0], macro_id=plan.macro_id + 1)
        if plan.station is not None:
            point = plan.station
            channels = station_channels(state, plan, point)
            if not channels:
                raise GeometryError("Persistent station without any valid task")
            c = channels[0]
            next_plan = plan if cfg.station_batch else replace(plan, station=None, station_due=False)
            return self.finish(
                Intent("/measure", point, c, "station_batch", state.beliefs[c - 1].version),
                next_plan,
                dict(
                    category="coverage",
                    macro="station",
                    station_remaining=len(channels),
                    conditional_certificate_s=conditional_s,
                    conditional_deleted=deleted,
                ),
                start,
            )
        if not known:
            raise GeometryError("No known source and no valid persistent station task")
        if plan.service is None:
            successor = plan.route[0] if plan.route else None
            choices = []
            for b in known:
                safe = clear_points(b, state.position, successor)
                if safe:
                    q = min(safe, key=lambda p: segment_delta(state.position, (p,), successor))
                    points, actions = (q,), 5
                else:
                    upper, points = remaining_upper(b, state.position)
                    actions = upper - sum(
                        distance(a, z) / 5 for a, z in zip((state.position,) + tuple(points), points)
                    )
                score = segment_delta(state.position, points, successor) / 5 + actions
                choices.append((score, b.channel))
            _, c = min(choices)
            plan = replace(plan, service=c, service_actions=0, macro_id=plan.macro_id + 1)
        b = state.beliefs[plan.service - 1]
        d = baseline_service(state, b, cfg)
        plan = replace(plan, service_actions=plan.service_actions + 1)
        if not cfg.service_commit and plan.service_actions >= cfg.fairness_actions:
            plan = replace(plan, service=None, station_due=True, service_actions=0)
        # Phase-two service modules replace this baseline only when explicitly enabled.
        if any((cfg.candidate_families, cfg.stripes, cfg.continuation, cfg.recovery_batch)):
            from rf_searchlab.algorithms.lookahead.service import service_decision

            d, plan = service_decision(state, b, plan, cfg, deadline, d)
        return self.finish(
            d.intent,
            plan,
            dict(
                d.detail,
                macro="service",
                conditional_certificate_s=conditional_s,
                conditional_deleted=deleted,
            ),
            start,
            d.optical_queue,
        )

    def finish(self, intent, plan, detail, start, optical_queue=()):
        detail = dict(
            detail,
            planning_s=perf_counter() - start,
            profile=self.config.name,
            macro_id=plan.macro_id,
            service_channel=plan.service,
            service_actions=plan.service_actions,
        )
        return Decision(intent, plan, detail, optical_queue)
