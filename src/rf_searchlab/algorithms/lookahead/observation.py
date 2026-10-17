"""Shared physical reducer and accepted observation indexing."""

from dataclasses import replace

from rf_searchlab.algorithms.directional.observation import action_cost
from rf_searchlab.algorithms.directional.observation import apply_event as base_apply
from rf_searchlab.algorithms.lookahead.belief import PresentWitness, apply_count

__all__ = ["action_cost", "apply_event"]


def apply_event(state, envelope, response, config):
    if not response.accepted or envelope.request_id in state.applied:
        return state
    updated = base_apply(state, envelope, response, config)
    c = envelope.intent.channel
    if response.result in ("direction", "near", "success") and c not in {w.channel for w in updated.present}:
        updated = replace(updated, present=updated.present + (PresentWitness(c, envelope.request_id),))
    return apply_count(updated, config.count_bound)
