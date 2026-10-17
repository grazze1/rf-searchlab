"""Immutable domain values. Coordinates/distances use metres, clocks use seconds."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Protocol


class SolverError(RuntimeError):
    """A recorded run must stop, without an automatic exit request."""


class GeometryError(SolverError):
    """An actual feasible set or numerical containment is inconsistent."""


class ProtocolError(SolverError):
    """A response does not establish a valid state transition."""


class CommunicationUnresolved(SolverError):
    """The server may have executed the pending command."""


class BudgetExhausted(SolverError):
    """No new action is allowed after the known deadline."""


class HumanRequired(SolverError):
    """Fresh human practice authorization is missing."""


@dataclass(frozen=True, slots=True)
class Point:
    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        if any(type(v) not in (int, float) or not isfinite(v) for v in (self.x_m, self.y_m)):
            raise ValueError("Coordinates must be finite numbers")


Polygon = tuple[Point, ...]


class Status(StrEnum):
    UNKNOWN = "UNKNOWN"
    LOCALIZING = "LOCALIZING"
    READY = "READY"
    CLEARED = "CLEARED"
    ABSENT = "ABSENT"


@dataclass(frozen=True)
class CircleBound:
    center: Point
    radius_m: float
    level: str = "checked_numeric_upper"
    target_set: str = "P_outer"


@dataclass(frozen=True)
class Observation:
    request_id: str
    position: Point
    result: str
    svd_deg: float | None


@dataclass(frozen=True)
class Exclusion:
    position: Point
    radius_m: float
    request_id: str
    conditional_on_existence: bool = True


@dataclass(frozen=True)
class OpticalCell:
    i: int
    j: int
    center: Point


@dataclass(frozen=True)
class OpticalPlan:
    cells: tuple[OpticalCell, ...]
    angle_rad: float
    origin: Point
    set_version: int
    index: int = 0


@dataclass(frozen=True)
class Certificate:
    kind: str
    channel: int | None
    request_ids: tuple[str, ...]
    set_version: int


@dataclass(frozen=True)
class Belief:
    channel: int
    polygon: Polygon
    status: Status = Status.UNKNOWN
    ever_detected: bool = False
    observations: tuple[Observation, ...] = ()
    exclusions: tuple[Exclusion, ...] = ()
    coverage: tuple[int, ...] = ()
    version: int = 0
    rf_used: int = 0
    optical: OpticalPlan | None = None
    near_position: Point | None = None
    certificate: Certificate | None = None
    last_served: int = 0


@dataclass(frozen=True)
class State:
    run_id: str
    beliefs: tuple[Belief, ...]
    position: Point = Point(0, 0)
    channel: int = 1
    virtual_time_s: float = 0
    timestamp_ms: float = 0
    entered: bool = False
    exited: bool = False
    applied: tuple[str, ...] = ()
    noncoverage_stations: int = 0
    movement_m: float = 0
    measures: int = 0
    switches: int = 0
    successes: int = 0
    failures: int = 0


@dataclass(frozen=True)
class Intent:
    path: str
    position: Point | None = None
    channel: int | None = None
    reason: str = ""
    set_version: int | None = None
    optical_plan: OpticalPlan | None = None
    guaranteed: bool = False


@dataclass(frozen=True)
class Envelope:
    path: str
    raw: bytes
    request_id: str
    first_sent_s: float
    intent: Intent


@dataclass(frozen=True)
class RawResponse:
    status: int
    body: bytes


@dataclass(frozen=True)
class Response:
    accepted: bool
    timestamp_ms: float
    virtual_time_s: float
    result: str | None = None
    svd_deg: float | None = None
    remaining_s: int | None = None
    max_virtual_s: float | None = None
    max_real_s: float | None = None


@dataclass(frozen=True)
class AcceptedAction:
    """An actual validated transition, exposed without changing perform's State result."""

    envelope: Envelope
    response: Response
    before: State
    after: State


class Transport(Protocol):
    def execute(self, envelope: Envelope) -> RawResponse:
        """Execute one fixed command, returning the entire raw response."""


class Clock(Protocol):
    def monotonic(self) -> float:
        """Return monotonic time in seconds."""


class EventSink(Protocol):
    def append(self, kind: str, record: object) -> None:
        """Persist a transaction record before returning."""

    def snapshot(self, state: State) -> None:
        """Atomically replace the latest state snapshot."""
