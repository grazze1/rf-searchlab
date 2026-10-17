"""Observation-driven belief updates and auditable completion certificates."""

from dataclasses import replace
from math import cos, pi, radians, sin, sqrt

from rf_searchlab.core.config import Config
from rf_searchlab.core.domain import (
    Belief,
    Certificate,
    Exclusion,
    GeometryError,
    Intent,
    Observation,
    Point,
    Response,
    State,
    Status,
)
from rf_searchlab.core.geometry import HalfPlane, clip, clip_disk, clip_wedge, disk_outer, mec


def wire_point(point: Point) -> Point:
    """Canonical six-decimal command coordinates; certificates use this exact value."""
    return Point(round(point.x_m, 6), round(point.y_m, 6))


def coverage_points() -> tuple[Point, ...]:
    """Center plus radius-1200 hexagon, frozen at the serialization precision."""
    return (Point(0, 0),) + tuple(
        wire_point(Point(1200 * cos(k * pi / 3), 1200 * sin(k * pi / 3))) for k in range(6)
    )


def coverage_bound() -> float:
    """Analytical covering radius plus maximum coordinate rounding displacement."""
    return max(1200 / sqrt(3), sqrt(1800**2 + 1200**2 - sqrt(3) * 1800 * 1200)) + sqrt(2) * 0.5e-6


def initial_state(run_id: str, config: Config) -> State:
    """Initialize all twenty unknown channels without scene information."""
    poly = disk_outer(Point(0, 0), 1800, config.disk_sides, config.geometry_padding_m)
    return State(run_id, tuple(Belief(c, poly) for c in range(1, 21)))


def complete(state: State) -> bool:
    """Only certified ABSENT and confirmed CLEARED channels terminate the task."""
    return all(b.status in (Status.CLEARED, Status.ABSENT) for b in state.beliefs)


def update(b: Belief, intent: Intent, response: Response, rid: str, config: Config, step: int) -> Belief:
    """Apply one accepted action to one channel, retaining every observation."""
    p = intent.position
    if p is None:
        raise ValueError("Belief action requires position")
    if b.status in (Status.CLEARED, Status.ABSENT):
        raise GeometryError("Action attempted on resolved channel")
    optical = b.optical or intent.optical_plan
    if optical is not None and intent.reason == "optical":
        if optical.index >= len(optical.cells) or optical.cells[optical.index].center != p:
            raise GeometryError("Optical queue mismatch")
        optical = replace(optical, index=optical.index + 1)
    b = replace(b, version=b.version + 1, optical=optical, last_served=step)
    if intent.path == "/clear":
        if response.result == "success":
            return replace(
                b,
                status=Status.CLEARED,
                ever_detected=True,
                near_position=None,
                certificate=Certificate("clear_success", b.channel, (rid,), b.version),
            )
        if b.near_position is not None or intent.guaranteed:
            raise GeometryError("Guaranteed/near clearance failed; retain raw evidence")
        return replace(b, exclusions=b.exclusions + (Exclusion(p, 20, rid),))
    obs = Observation(rid, p, response.result or "", response.svd_deg)
    coverage = b.coverage
    points = coverage_points()
    if p in points:
        coverage = tuple(sorted(set(coverage) | {points.index(p)}))
    rf = b.rf_used + int(b.ever_detected)
    b = replace(b, observations=b.observations + (obs,), coverage=coverage, rf_used=rf)
    if response.result == "near":
        poly = clip_disk(b.polygon, p, 5, config.disk_sides, config.geometry_padding_m)
        b = replace(b, polygon=poly, ever_detected=True, status=Status.READY, near_position=p)
    elif response.result == "direction":
        if response.svd_deg is None:
            raise ValueError("Missing direction")
        poly = clip_wedge(
            b.polygon, p, radians(response.svd_deg), config.epsilon_rad, config.geometry_padding_m
        )
        poly = clip_disk(poly, p, 1500, config.disk_sides, config.geometry_padding_m)
        b = replace(b, polygon=poly, ever_detected=True, status=Status.LOCALIZING)
    elif response.result == "no_signal":
        b = replace(b, exclusions=b.exclusions + (Exclusion(p, 1000, rid),))
    else:
        raise ValueError("Unknown observation")
    poly = b.polygon
    positives = [o for o in b.observations if o.result in ("direction", "near")]
    negatives = [o for o in b.observations if o.result == "no_signal"]
    # Only newly introduced pairs: unchanged old constraints are already represented.
    pairs = ((a, n) for a in positives for n in negatives if rid in (a.request_id, n.request_id))
    for a, n in pairs:
        s, t = a.position, n.position
        hp = HalfPlane(
            2 * (t.x_m - s.x_m),
            2 * (t.y_m - s.y_m),
            t.x_m**2 + t.y_m**2 - s.x_m**2 - s.y_m**2,
            f"{a.request_id}/{n.request_id}",
        )
        poly = clip(poly, hp, config.geometry_padding_m)
    if b.ever_detected and not poly:
        raise GeometryError(f"Channel {b.channel} actual observations produced empty set")
    b = replace(b, polygon=poly)
    if b.ever_detected and mec(poly, config.geometry_padding_m).radius_m <= 20 - config.safety_margin_m:
        b = replace(b, status=Status.READY)
    if not b.ever_detected and len(coverage) == 7:
        evidence = tuple(
            o.request_id for o in b.observations if o.position in points and o.result == "no_signal"
        )
        b = replace(
            b,
            status=Status.ABSENT,
            certificate=Certificate("seven_point_no_signal", b.channel, evidence, b.version),
        )
    return b


def number_upper_bound(beliefs: tuple[Belief, ...]) -> tuple[Belief, ...]:
    """Sixteen distinct ever-detected channels certify the remaining channels absent."""
    found = [b for b in beliefs if b.ever_detected]
    if len(found) > 16:
        raise GeometryError("More than 16 distinct channels detected")
    if len(found) != 16:
        return beliefs
    ids = tuple(
        next(
            (o.request_id for o in b.observations if o.result in ("direction", "near")),
            b.certificate.request_ids[0] if b.certificate else "",
        )
        for b in found
    )
    return tuple(
        replace(
            b,
            status=Status.ABSENT,
            certificate=Certificate("source_count_upper_16", b.channel, ids, b.version),
        )
        if not b.ever_detected
        else b
        for b in beliefs
    )
