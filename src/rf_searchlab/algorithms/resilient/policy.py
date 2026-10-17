"""component scan/service commitment reused; only reviewed source selection changes."""

from dataclasses import replace
from time import perf_counter

from rf_searchlab.algorithms.directional.belief import RESOLVED
from rf_searchlab.algorithms.lookahead.planning import reconcile
from rf_searchlab.algorithms.lookahead.policy import Policy as PreviousPolicy
from rf_searchlab.algorithms.resilient.route import choose


class Policy:
    def __init__(self, config, engine):
        self.config, self.engine = config, engine
        self.reference = PreviousPolicy(config, engine)

    def decide(self, state, plan, remaining_real_s):
        start = perf_counter()
        deadline = start + min(self.config.planning_budget_s, max(0, remaining_real_s - 0.05))
        fallback = self.reference.decide(state, plan, remaining_real_s)
        decision = fallback
        idle = plan.service is None or state.beliefs[plan.service - 1].status in RESOLVED
        known = [b for b in state.beliefs if b.ever_detected and b.status not in RESOLVED]
        if (
            self.config.route_mode != "legacy"
            and idle
            and known
            and fallback.detail.get("macro") == "service"
            and not reconcile(state, plan, self.engine).route
        ):
            try:
                order, detail = choose(state, known, self.config, deadline - 0.01)
                proposal = replace(
                    fallback.plan,
                    service=order[0],
                    service_actions=0,
                    source_route=order,
                    source_versions=tuple((b.channel, b.version) for b in known),
                )
                decision = self.reference.decide(
                    state, proposal, max(0, remaining_real_s - (perf_counter() - start))
                )
                decision.detail.update(detail, source_route=order)
            except TimeoutError as exc:
                decision.detail["fallback_reason"] = str(exc)
        if self.config.direction_recovery and decision.detail.get("macro") == "service":
            from rf_searchlab.algorithms.resilient.recovery import recover

            decision = recover(state, decision, self.config, deadline)
        if self.config.long_cover_threshold and decision.detail.get("macro") == "service":
            from rf_searchlab.algorithms.resilient.optical import gated_optical

            decision = gated_optical(state, decision, self.config, deadline)
        if perf_counter() > deadline:
            decision = fallback
            decision.detail["fallback_reason"] = "planning_soft_budget"
        decision.detail.update(planning_s=perf_counter() - start, profile=self.config.name)
        return decision
