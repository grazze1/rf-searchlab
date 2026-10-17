"""The sole serial executor and pure reducer; planners cannot mutate actual state."""

import time
from dataclasses import asdict, replace
from uuid import uuid4

from rf_searchlab.core.belief import complete, coverage_points, number_upper_bound, update
from rf_searchlab.core.config import Config
from rf_searchlab.core.domain import (
    AcceptedAction,
    BudgetExhausted,
    CommunicationUnresolved,
    Envelope,
    EventSink,
    GeometryError,
    Intent,
    ProtocolError,
    Response,
    SolverError,
    State,
    Transport,
)
from rf_searchlab.core.geometry import distance, max_distance
from rf_searchlab.core.protocol import make_envelope, parse_response


def apply_event(state: State, envelope: Envelope, response: Response, config: Config) -> State:
    """Pure idempotent reducer; rejected actions leave state byte-for-byte unchanged."""
    if envelope.request_id in state.applied or not response.accepted:
        return state
    intent = envelope.intent
    if (
        state.exited
        or (intent.path != "/enter" and not state.entered)
        or (intent.path == "/enter" and state.entered)
    ):
        raise ProtocolError("Illegal lifecycle transition")
    if response.timestamp_ms < state.timestamp_ms:
        raise ProtocolError("Server timestamp regressed")
    expected = state.virtual_time_s
    new = state
    if intent.path in ("/measure", "/clear"):
        if intent.position is None or intent.channel is None:
            raise ProtocolError("Incomplete action intent")
        moved = distance(state.position, intent.position)
        switch = int(intent.path == "/measure" and state.channel != intent.channel)
        cost = 5 if intent.path == "/measure" or response.result == "success" else 3
        expected += round(moved / 5 * 1e6) / 1e6 + switch + cost
        beliefs = list(state.beliefs)
        beliefs[intent.channel - 1] = update(
            beliefs[intent.channel - 1],
            intent,
            response,
            envelope.request_id,
            config,
            len(state.applied),
        )
        points = coverage_points()
        if intent.path == "/measure" and intent.position in points:
            streak = 0
        else:
            streak = state.noncoverage_stations + int(moved > 0 or state.noncoverage_stations == 0)
        new = replace(
            state,
            beliefs=number_upper_bound(tuple(beliefs)),
            position=intent.position,
            channel=intent.channel if intent.path == "/measure" else state.channel,
            noncoverage_stations=streak,
            movement_m=state.movement_m + moved,
            measures=state.measures + int(intent.path == "/measure"),
            switches=state.switches + switch,
            successes=state.successes + int(response.result == "success"),
            failures=state.failures + int(response.result == "no_target_in_range"),
        )
    elif intent.path == "/enter":
        new = replace(state, entered=True)
    elif intent.path == "/exit":
        new = replace(state, exited=True)
    if abs(expected - response.virtual_time_s) > 2e-6:
        raise ProtocolError(f"Virtual time mismatch: expected {expected}, received {response.virtual_time_s}")
    return replace(
        new,
        virtual_time_s=response.virtual_time_s,
        timestamp_ms=response.timestamp_ms,
        applied=state.applied + (envelope.request_id,),
    )


class Executor:
    """One pending transaction at a time, including after ambiguous network failure."""

    def __init__(
        self,
        state: State,
        transport: Transport,
        sink: EventSink,
        config: Config,
        robot_id: str = "OFFLINE",
        clock: object = time,
        retries: int = 1,
    ) -> None:
        from typing import cast

        from rf_searchlab.core.domain import Clock

        self.state = state
        self.transport = transport
        self.sink = sink
        self.config = config
        self.robot_id = robot_id
        self.clock = cast(Clock, clock)
        self.retries = retries
        self.pending: Envelope | None = None
        self.deadline: float | None = None
        self.max_virtual_s = 360000.0
        self.last_action: AcceptedAction | None = None

    def perform(self, intent: Intent) -> State:
        """Preflight -> pending log -> serial send -> raw log -> validate -> reduce -> snapshot."""
        self.last_action = None
        if self.pending is not None:
            raise CommunicationUnresolved("Previous transaction unresolved; no new ID allowed")
        now = self.clock.monotonic()
        if self.deadline is not None and now >= self.deadline:
            raise BudgetExhausted("Real deadline reached")
        if self.state.virtual_time_s >= self.max_virtual_s:
            raise BudgetExhausted("Virtual deadline reached")
        if intent.path == "/exit" and intent.reason == "complete" and not complete(self.state):
            raise GeometryError("No global completion certificate")
        envelope = make_envelope(intent, self.robot_id, uuid4().hex, now)
        intent = envelope.intent
        if intent.channel is not None:
            b = self.state.beliefs[intent.channel - 1]
            if intent.set_version is not None and intent.set_version != b.version:
                raise GeometryError("Stale planning set version")
            if b.near_position is not None and (
                intent.path != "/clear" or intent.position != b.near_position
            ):
                raise GeometryError("near requires same-position immediate clear")
            if (
                intent.guaranteed
                and intent.position is not None
                and max_distance(intent.position, b.polygon) > 20 - self.config.safety_margin_m
            ):
                raise GeometryError("Serialized clearance failed containment check")
        near = next((b for b in self.state.beliefs if b.near_position is not None), None)
        if near is not None and intent.channel != near.channel:
            raise GeometryError("Another channel cannot interrupt near clearance")
        self.pending = envelope
        self.sink.append(
            "pending",
            {
                "path": envelope.path,
                "raw_hex": envelope.raw.hex(),
                "request_id": envelope.request_id,
                "first_sent_s": now,
                "intent": asdict(intent),
            },
        )
        for attempt in range(self.retries + 1):
            if self.deadline is not None and self.clock.monotonic() >= self.deadline:
                raise CommunicationUnresolved("Deadline while transaction pending")
            try:
                raw = self.transport.execute(envelope)
            except (OSError, TimeoutError) as exc:
                self.sink.append(
                    "transport_error",
                    {"request_id": envelope.request_id, "attempt": attempt, "error": str(exc)},
                )
                if attempt == self.retries:
                    raise CommunicationUnresolved("Response missing after bounded identical retry") from exc
                continue
            self.sink.append(
                "raw",
                {
                    "request_id": envelope.request_id,
                    "status": raw.status,
                    "body_hex": raw.body.hex(),
                    "received_s": self.clock.monotonic(),
                },
            )
            response = parse_response(envelope.path, raw)
            if not response.accepted:
                raise ProtocolError("accepted=false; retain last accepted state")
            before = self.state
            new = apply_event(before, envelope, response, self.config)
            if intent.path == "/enter":
                if response.remaining_s is None or response.max_virtual_s is None:
                    raise ProtocolError("Missing enter budgets")
                self.deadline = envelope.first_sent_s + response.remaining_s
                self.max_virtual_s = response.max_virtual_s
            self.state = new
            self.sink.append(
                "applied", {"request_id": envelope.request_id, "virtual_time_s": new.virtual_time_s}
            )
            self.sink.snapshot(new)
            self.last_action = AcceptedAction(envelope, response, before, new)
            self.pending = None
            return new
        raise SolverError("Unreachable retry state")
