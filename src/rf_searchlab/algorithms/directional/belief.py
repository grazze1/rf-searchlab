"""Mixed public state and positive-only conservative location reduction."""

from dataclasses import dataclass, replace
from enum import StrEnum
from functools import lru_cache
from math import radians

from rf_searchlab.algorithms.directional.config import Config
from rf_searchlab.core.belief import wire_point
from rf_searchlab.core.domain import Certificate, Exclusion, GeometryError, Observation, Point
from rf_searchlab.core.geometry import clip_disk, clip_wedge, disk_outer, max_distance, mec


class Status(StrEnum):
    UNKNOWN = "UNKNOWN"
    DISCOVERED = "DISCOVERED"
    READY_TO_CLEAR = "READY_TO_CLEAR"
    CLEARED = "CLEARED"
    CERTIFIED_ABSENT = "CERTIFIED_ABSENT"


RESOLVED = (Status.CLEARED, Status.CERTIFIED_ABSENT)


@dataclass(frozen=True)
class Belief:
    channel: int
    polygon: tuple[Point, ...]
    status: Status = Status.UNKNOWN
    ever_detected: bool = False
    observations: tuple[Observation, ...] = ()
    exclusions: tuple[Exclusion, ...] = ()
    version: int = 0
    rf_used: int = 0
    near_position: Point | None = None
    certificate: Certificate | None = None
    # Serialised persistent queue; cursor advances only on accepted clear responses.
    optical: tuple[Point, ...] = ()
    optical_index: int = 0
    last_served: int = 0


@dataclass(frozen=True)
class State:
    beliefs: tuple[Belief, ...]
    run_id: str = "PUBLIC"
    position: Point = Point(0, 0)
    channel: int = 1
    virtual_time_s: float = 0
    timestamp_ms: float = 0
    entered: bool = False
    exited: bool = False
    applied: tuple[str, ...] = ()
    movement_m: float = 0
    measures: int = 0
    switches: int = 0
    successes: int = 0
    failures: int = 0
    blind_measures: int = 0


def initial_state(config: Config = Config()) -> State:
    poly = disk_outer(Point(0, 0), 1800, config.disk_sides, config.geometry_padding_m)
    return State(tuple(Belief(c, poly) for c in range(1, 21)))


@lru_cache(maxsize=8192)
def bound(poly):
    return mec(poly)


def safe_clear(b: Belief, p: Point) -> bool:
    return bool(b.ever_detected and b.polygon and max_distance(wire_point(p), b.polygon) <= 19.95)


def complete(state: State) -> bool:
    cleared = [b for b in state.beliefs if b.status == Status.CLEARED]
    # Count distinct channels with accepted clear evidence, never a raw response count.
    if len(cleared) == 16 and all(
        b.certificate
        and b.certificate.kind == "clear_success"
        and all(r in state.applied for r in b.certificate.request_ids)
        for b in cleared
    ):
        return True
    return all(
        b.status in RESOLVED
        and b.certificate
        and b.certificate.request_ids
        and b.certificate.kind
        == ("clear_success" if b.status == Status.CLEARED else "directional_negative_hull")
        and all(r in state.applied for r in b.certificate.request_ids)
        and (b.status != Status.CERTIFIED_ABSENT or not b.ever_detected)
        for b in state.beliefs
    )


def update(b, intent, response, rid, config, step):
    if b.status in RESOLVED:
        raise GeometryError("Action on resolved Mixed channel")
    p = intent.position
    if p is None:
        raise GeometryError("Missing accepted coordinate")
    old_detected = b.ever_detected
    b = replace(b, version=b.version + 1, last_served=step)
    if intent.path == "/clear":
        if response.result == "success":
            return replace(
                b,
                status=Status.CLEARED,
                ever_detected=True,
                near_position=None,
                optical=(),
                optical_index=0,
                certificate=Certificate("clear_success", b.channel, (rid,), b.version),
            )
        if b.near_position is not None or intent.guaranteed:
            raise GeometryError("Mixed guaranteed/near clear failed; stop and retain raw history")
        index = b.optical_index
        if b.optical and index < len(b.optical) and p == b.optical[index]:
            index += 1
        return replace(b, exclusions=b.exclusions + (Exclusion(p, 20, rid),), optical_index=index)
    obs = Observation(rid, p, response.result, response.svd_deg)
    b = replace(b, observations=b.observations + (obs,), rf_used=b.rf_used + int(old_detected))
    if response.result == "no_signal":
        return b  # Neither disk exclusion nor positive/negative bisector is legal for Mixed.
    poly = b.polygon
    if response.result == "near":
        poly = clip_disk(poly, p, 5, config.disk_sides, config.geometry_padding_m)
        b = replace(b, near_position=p)
    elif response.result == "direction":
        poly = clip_wedge(poly, p, radians(response.svd_deg), config.epsilon_rad, config.geometry_padding_m)
        poly = clip_disk(poly, p, 1500, config.disk_sides, config.geometry_padding_m)
    else:
        raise GeometryError("Unknown measurement result")
    if not poly:
        raise GeometryError("Mixed positive history contradicts conservative location set")
    ready = max_distance(wire_point(bound(poly).center), poly) <= 19.95
    return replace(
        b,
        polygon=poly,
        ever_detected=True,
        status=Status.READY_TO_CLEAR if ready else Status.DISCOVERED,
    )
