"""Adapters accept public observations only; no simulator/scenario imports."""

from dataclasses import asdict

from rf_searchlab.algorithms.route.certificate import update_observation_fine
from rf_searchlab.algorithms.route.policy import PlanPolicy
from rf_searchlab.algorithms.shared.policy import JointPolicy
from rf_searchlab.algorithms.shared.state import CONFIG, is_task_certified_complete
from rf_searchlab.core.belief import initial_state
from rf_searchlab.core.domain import Intent

REGISTRY = {
    "coverage_route": "omni",
    "shared_search": "omni",
    "adaptive_tour": "mixed",
    "adaptive_tour_no_pickup": "mixed",
}


class PublicPolicy:
    def __init__(self, name):
        if name not in REGISTRY:
            raise ValueError("Unknown strategy")
        self.name = name
        self.environment = REGISTRY[name]
        self.detail = {}
        self.pending = None
        if self.environment == "omni":
            self.state = initial_state("PUBLIC", CONFIG)
            self.policy = PlanPolicy() if name == "coverage_route" else JointPolicy(max_learning=200)
        else:
            # Load the packaged, tested adaptive service profile.
            import json
            from pathlib import Path

            from rf_searchlab.algorithms.directional.certificate import Engine, layout_for
            from rf_searchlab.algorithms.resilient.belief import initial_state as initial
            from rf_searchlab.algorithms.tour.config import Config
            from rf_searchlab.algorithms.tour.planning import Plan
            from rf_searchlab.algorithms.tour.policy import Policy

            values = json.loads(
                (Path(__file__).parent / "data/adaptive_tour.json").read_text(encoding="utf-8")
            )
            values["safe_rf_backoffs_m"] = tuple(values["safe_rf_backoffs_m"])
            if name == "adaptive_tour_no_pickup":
                values["local_pickup"] = False
            self.config = Config(**values)
            layout, error = layout_for("C")
            if error:
                raise ValueError(f"Coverage certificate unavailable: {error}")
            self.engine = Engine(layout, True)
            self.policy = Policy(self.config, self.engine)
            self.plan = Plan()
            self.state = initial(self.config)

    def complete(self):
        if self.environment == "omni":
            return is_task_certified_complete(self.state).complete
        from rf_searchlab.algorithms.resilient.belief import complete

        return complete(self.state)

    def next_action(self, remaining=1200):
        if not self.state.entered:
            return Intent("/enter")
        if self.complete():
            return Intent("/exit", reason="complete")
        if self.environment == "omni":
            action = self.policy.decide(self.state)
            self.detail = dict(self.policy.detail)
            return action.intent
        from rf_searchlab.algorithms.directional.policy import install_queue

        self.pending = self.policy.decide(self.state, self.plan, remaining)
        self.state = install_queue(self.state, self.pending)
        self.detail = dict(self.pending.detail)
        return self.pending.intent

    def observe(self, envelope, response):
        if not response.accepted:
            return
        if self.environment == "omni":
            self.state = update_observation_fine(self.state, envelope, response)
        else:
            from rf_searchlab.algorithms.resilient.observation import apply_event
            from rf_searchlab.algorithms.resilient.optical import commit_progress

            self.state = apply_event(self.state, envelope, response, self.config)
            if self.pending:
                self.plan = self.pending.plan
                if self.plan.covers:
                    self.state = commit_progress(self.state, self.plan)
                self.pending = None
            if response.result == "no_signal":
                self.state = self.engine.update(self.state, envelope.intent.channel)

    def public_summary(self):
        return {
            "position": asdict(self.state.position),
            "channel": self.state.channel,
            "virtual_time_s": self.state.virtual_time_s,
            "channels": [
                {
                    "channel": b.channel,
                    "status": str(b.status),
                    "vertices": [[p.x_m, p.y_m] for p in b.polygon] if b.ever_detected else [],
                    "certificate": asdict(b.certificate) if b.certificate else None,
                }
                for b in self.state.beliefs
            ],
        }
