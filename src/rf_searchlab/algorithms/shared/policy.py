"""Atomic policy choices and persistent fairness/finite-fallback controller."""

from typing import Any

from rf_searchlab.algorithms.shared.planning import (
    CandidateAction,
    choose_reference,
    immediate_clear,
    make_candidates,
    next_fallback_action,
)
from rf_searchlab.algorithms.shared.state import RESOLVED, bound, unknown_cells
from rf_searchlab.core.domain import State, Status


class JointPolicy:
    def __init__(self, name: str = "classical_joint", model: Any = None, max_learning: int = 200):
        if name not in (
            "classical_joint",
            "find_then_clear",
            "clear_on_discovery",
            "learned_joint",
        ):
            raise ValueError("Unknown atomic policy")
        self.name, self.model, self.max_learning = name, model, max_learning
        self.steps = 0
        self.committed_fallback = False
        self.stagnant = 0
        self.progress: tuple[int, int, float] | None = None
        self.owner = None
        self.model_fallbacks = 0
        self.model_overrides = 0
        self.detail: dict[str, Any] = {}

    def decide(self, state: State) -> CandidateAction:
        urgent = immediate_clear(state)
        if urgent:
            self.detail = {"mode": "hard_immediate_clear"}
            return urgent
        progress = (
            state.successes,
            sum(len(unknown_cells(state, b.channel)) for b in state.beliefs if b.status == Status.UNKNOWN),
            round(
                sum(
                    bound(b.polygon).radius_m
                    for b in state.beliefs
                    if b.ever_detected and b.status not in RESOLVED
                ),
                2,
            ),
        )
        self.stagnant = self.stagnant + 1 if progress == self.progress else 0
        self.progress = progress
        self.steps = len(state.applied)
        if self.steps > self.max_learning or self.stagnant >= 40:
            self.committed_fallback = True
        if self.committed_fallback:
            self.detail = {"mode": "committed_fallback"}
            return next_fallback_action(state)
        candidates = make_candidates(state)
        if self.name == "find_then_clear":
            pending = [a for a in candidates if a.operation == "measure"]
            if pending:
                candidates = pending
        elif self.name == "clear_on_discovery":
            known = [b for b in state.beliefs if b.ever_detected and b.status not in RESOLVED]
            if known:
                if self.owner not in [b.channel for b in known]:
                    self.owner = min(
                        known,
                        key=lambda b: min(
                            (
                                i
                                for i, rid in enumerate(state.applied)
                                if any(
                                    o.request_id == rid and o.result in ("direction", "near")
                                    for o in b.observations
                                )
                            ),
                            default=0,
                        ),
                    ).channel
                candidates = [a for a in candidates if a.channel == self.owner]
        reference = choose_reference(state, candidates)
        self.detail = {"mode": self.name, "candidates": len(candidates), "model_override": False}
        if self.name == "learned_joint":
            if self.model is None:
                self.model_fallbacks += 1
                self.detail["fallback_reason"] = "model_missing"
            else:
                action, reason = self.model.choose(state, candidates, reference)
                self.detail["fallback_reason"] = reason
                if action != reference:
                    self.model_overrides += 1
                    self.detail["model_override"] = True
                else:
                    self.model_fallbacks += 1
                return action
        return reference
