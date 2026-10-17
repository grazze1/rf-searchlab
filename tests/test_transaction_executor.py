"""Fault injection into the reusable serial transaction executor."""

from dataclasses import replace

import pytest

from rf_searchlab.core.application import Executor, apply_event
from rf_searchlab.core.belief import initial_state
from rf_searchlab.core.config import Config
from rf_searchlab.core.domain import (
    BudgetExhausted,
    CommunicationUnresolved,
    GeometryError,
    Intent,
    Point,
    ProtocolError,
    RawResponse,
    Response,
)
from rf_searchlab.core.protocol import dumps, make_envelope
from rf_searchlab.simulator import Simulator, Source, WorldTruth


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now


class Sink:
    def __init__(self):
        self.events = []
        self.states = []

    def append(self, kind, value):
        self.events.append((kind, value))

    def snapshot(self, state):
        self.states.append(state)


def executor(transport):
    config = Config()
    sink, clock = Sink(), Clock()
    return Executor(initial_state("PUBLIC", config), transport, sink, config, clock=clock), sink, clock


def test_response_loss_retries_identical_id_without_double_charge():
    simulator = Simulator(WorldTruth((Source(1, 0, 0, 1000),), 7))

    class LostResponse:
        calls = []

        def execute(self, envelope):
            self.calls.append(envelope)
            response = simulator.execute(envelope)
            if len(self.calls) == 1:
                raise TimeoutError("response lost after execution")
            return response

    transport = LostResponse()
    engine, sink, clock = executor(transport)
    entered = engine.perform(Intent("/enter"))
    assert transport.calls[0] == transport.calls[1]
    assert len(entered.applied) == len(simulator._cache) == 1
    assert entered.virtual_time_s == 0 and engine.pending is None
    near = engine.perform(Intent("/measure", Point(0, 0), 1))
    assert near.virtual_time_s == 5
    with pytest.raises(GeometryError, match="near"):
        engine.perform(Intent("/measure", Point(1, 0), 2))
    cleared = engine.perform(Intent("/clear", Point(0, 0), 1))
    assert cleared.virtual_time_s == 10 and cleared.successes == 1
    assert len(sink.states) == 3 and engine.last_action.after == cleared
    assert any(kind == "transport_error" for kind, _ in sink.events)
    clock.now = 1201
    with pytest.raises(BudgetExhausted):
        engine.perform(Intent("/exit"))


def test_unresolved_and_rejected_actions_cannot_advance_public_state():
    class Timeout:
        def execute(self, envelope):
            raise TimeoutError("lost")

    engine, sink, _ = executor(Timeout())
    before = engine.state
    with pytest.raises(CommunicationUnresolved):
        engine.perform(Intent("/enter"))
    assert engine.state == before and engine.pending is not None
    assert len([kind for kind, _ in sink.events if kind == "transport_error"]) == 2
    with pytest.raises(CommunicationUnresolved, match="Previous transaction"):
        engine.perform(Intent("/enter"))

    class Rejected:
        def execute(self, envelope):
            return RawResponse(200, dumps({"accepted": False, "real_timestamp_ms": 1, "virtual_time_s": 0}))

    rejected, _, _ = executor(Rejected())
    with pytest.raises(ProtocolError, match="accepted=false"):
        rejected.perform(Intent("/enter"))
    assert rejected.state == before and rejected.pending is not None


def test_preflight_and_reducer_reject_stale_or_impossible_evidence():
    simulator = Simulator(WorldTruth((Source(1, 100, 0, 1000),), 7))
    engine, _, _ = executor(simulator)
    with pytest.raises(GeometryError, match="completion"):
        engine.perform(Intent("/exit", reason="complete"))
    engine.perform(Intent("/enter"))
    with pytest.raises(GeometryError, match="Stale"):
        engine.perform(Intent("/measure", Point(0, 0), 1, set_version=99))
    with pytest.raises(GeometryError, match="containment"):
        engine.perform(Intent("/clear", Point(0, 0), 1, guaranteed=True))
    engine.max_virtual_s = 0
    with pytest.raises(BudgetExhausted, match="Virtual"):
        engine.perform(Intent("/measure", Point(0, 0), 1))

    state = engine.state
    event = make_envelope(Intent("/measure", Point(0, 0), 1), "PUBLIC", "m", 0)
    with pytest.raises(ProtocolError, match="regressed"):
        apply_event(replace(state, timestamp_ms=10), event, Response(True, 9, 5, "direction", 0), Config())
    with pytest.raises(ProtocolError, match="time mismatch"):
        apply_event(state, event, Response(True, 2, 100, "direction", 0), Config())
