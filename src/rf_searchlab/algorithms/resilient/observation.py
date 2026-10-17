"""Accepted near-range negatives constrain direction; never remove any location."""

from dataclasses import replace

from rf_searchlab.algorithms.lookahead.observation import action_cost
from rf_searchlab.algorithms.lookahead.observation import apply_event as previous_apply
from rf_searchlab.algorithms.resilient.belief import DirectionWitness
from rf_searchlab.core.geometry import max_distance

__all__ = ["apply_event", "action_cost"]


def apply_event(state, envelope, response, config):
    if not response.accepted or envelope.request_id in state.applied:
        return state
    updated = previous_apply(state, envelope, response, config)
    if envelope.path == "/measure" and response.result == "no_signal":
        b = state.beliefs[envelope.intent.channel - 1]
        if b.ever_detected:
            farthest = max_distance(envelope.intent.position, b.polygon)
            if farthest <= 1000 - 1e-6:
                witness = DirectionWitness(b.channel, envelope.request_id, b.version, farthest)
                updated = replace(updated, direction_witnesses=updated.direction_witnesses + (witness,))
    return updated
