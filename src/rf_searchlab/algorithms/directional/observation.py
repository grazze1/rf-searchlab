"""One accepted-event clock/reducer for every Mixed strategy and replay."""

from dataclasses import replace

from rf_searchlab.algorithms.directional.belief import update
from rf_searchlab.core.domain import ProtocolError
from rf_searchlab.core.geometry import distance


def action_cost(state, intent, result):
    if intent.path not in ("/measure", "/clear"):
        return dict(movement_m=0.0, movement_s=0.0, switch_s=0, measure_s=0, clear_s=0)
    moved = distance(state.position, intent.position)
    return dict(
        movement_m=moved,
        movement_s=round(moved / 5 * 1e6) / 1e6,
        switch_s=int(intent.path == "/measure" and state.channel != intent.channel),
        measure_s=5 if intent.path == "/measure" else 0,
        clear_s=(5 if result == "success" else 3) if intent.path == "/clear" else 0,
    )


def apply_event(state, envelope, response, config):
    if envelope.request_id in state.applied or not response.accepted:
        return state
    intent = envelope.intent
    if (
        state.exited
        or (intent.path != "/enter" and not state.entered)
        or (intent.path == "/enter" and state.entered)
    ):
        raise ProtocolError("Illegal Mixed lifecycle transition")
    if response.timestamp_ms < state.timestamp_ms:
        raise ProtocolError("Server timestamp regressed")
    cost = action_cost(state, intent, response.result)
    expected = state.virtual_time_s + sum(v for k, v in cost.items() if k.endswith("_s"))
    if abs(expected - response.virtual_time_s) > 2e-6:
        raise ProtocolError(f"Virtual time mismatch: {expected} != {response.virtual_time_s}")
    new = state
    if intent.path in ("/measure", "/clear"):
        beliefs = list(state.beliefs)
        beliefs[intent.channel - 1] = update(
            beliefs[intent.channel - 1],
            intent,
            response,
            envelope.request_id,
            config,
            len(state.applied),
        )
        new = replace(
            state,
            beliefs=tuple(beliefs),
            position=intent.position,
            channel=intent.channel if intent.path == "/measure" else state.channel,
            movement_m=state.movement_m + cost["movement_m"],
            switches=state.switches + cost["switch_s"],
            measures=state.measures + int(intent.path == "/measure"),
            successes=state.successes + int(response.result == "success"),
            failures=state.failures + int(response.result == "no_target_in_range"),
            blind_measures=state.blind_measures + int(response.result == "no_signal"),
        )
    elif intent.path == "/enter":
        new = replace(state, entered=True)
    elif intent.path == "/exit":
        new = replace(state, exited=True)
    return replace(
        new,
        virtual_time_s=response.virtual_time_s,
        timestamp_ms=response.timestamp_ms,
        applied=state.applied + (envelope.request_id,),
    )
