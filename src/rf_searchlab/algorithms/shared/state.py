"""Observable reducer, conservative full-cell coverage and auditable certificates."""

from dataclasses import dataclass, replace
from functools import lru_cache
from math import radians

import numpy as np

from rf_searchlab.algorithms.config import BASELINE
from rf_searchlab.core.application import apply_event
from rf_searchlab.core.belief import initial_state
from rf_searchlab.core.domain import (
    Certificate,
    CircleBound,
    Envelope,
    GeometryError,
    Point,
    Response,
    State,
    Status,
)
from rf_searchlab.core.geometry import clip_disk, clip_wedge, mec

CONFIG = replace(BASELINE, disk_sides=64)
VERSION = "atomic-joint-v1"
RESOLVED = (Status.CLEARED, Status.ABSENT)


@lru_cache(maxsize=1)
def cell_corners() -> np.ndarray:
    """100m closed squares intersecting the source disk; include boundary cells.

    Keeping the entire square (even outside the disk) is conservative. A cell is
    excluded only when ALL four corners fit within one received no-signal disk.
    Seven radius-1200 anchors from the frozen, analytically proved cover remain
    the finite fallback if this coarse representation cannot certify the edges.
    """
    cells = []
    for x in range(-1800, 1800, 100):
        for y in range(-1800, 1800, 100):
            dx = max(x, 0, -x - 100)
            dy = max(y, 0, -y - 100)
            if dx * dx + dy * dy <= 1800**2:
                cells.append([(x, y), (x + 100, y), (x, y + 100), (x + 100, y + 100)])
    result = np.array(cells, dtype=float)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=8192)
def covered_cells(p: Point) -> frozenset[int]:
    squared = np.sum((cell_corners() - (p.x_m, p.y_m)) ** 2, axis=2)
    return frozenset(np.flatnonzero(np.max(squared, axis=1) <= (1000 - 1e-5) ** 2).tolist())


@lru_cache(maxsize=4096)
def remaining_cells(negatives: tuple[Point, ...]) -> frozenset[int]:
    covered: set[int] = set()
    for p in negatives:
        covered.update(covered_cells(p))
    return frozenset(range(len(cell_corners()))) - covered


def unknown_cells(state: State, channel: int) -> frozenset[int]:
    b = state.beliefs[channel - 1]
    return remaining_cells(tuple(o.position for o in b.observations if o.result == "no_signal"))


@dataclass(frozen=True)
class CertificateResult:
    complete: bool
    reason: str
    observation_ids: tuple[str, ...]


def is_task_certified_complete(state: State) -> CertificateResult:
    cleared = [b for b in state.beliefs if b.status == Status.CLEARED]
    applied = set(state.applied)

    def valid_evidence(b):
        cert = b.certificate
        negative_kinds = (
            "full_cell_negative_cover",
            "full_fine_cell_negative_cover",
            "seven_point_no_signal",
        )
        negatives = {o.request_id for o in b.observations if o.result == "no_signal"}
        return bool(
            cert
            and cert.channel == b.channel
            and cert.request_ids
            and all(rid in applied for rid in cert.request_ids)
            and (
                (b.status == Status.CLEARED and cert.kind == "clear_success")
                or (
                    b.status == Status.ABSENT
                    and (
                        (cert.kind in negative_kinds and set(cert.request_ids) <= negatives)
                        or (cert.kind == "source_count_upper_16" and len(cleared) == 16)
                    )
                )
            )
        )

    cleared_complete = len(cleared) == 16 and all(valid_evidence(b) for b in cleared)
    all_resolved = all(b.status in RESOLVED and valid_evidence(b) for b in state.beliefs)
    evidence_ids = (
        [rid for b in cleared for rid in b.certificate.request_ids]
        if all(valid_evidence(b) for b in cleared)
        else []
    )
    unique_evidence = len(evidence_ids) == len(set(evidence_ids))
    cleared_complete = cleared_complete and unique_evidence
    all_resolved = all_resolved and unique_evidence
    done = cleared_complete or all_resolved
    evidence = tuple(
        dict.fromkeys(rid for b in state.beliefs if b.certificate for rid in b.certificate.request_ids)
    )
    return CertificateResult(
        done,
        "sixteen_clear_successes" if cleared_complete else "all_channels" if done else "unresolved_channels",
        evidence,
    )


def update_observation(state: State, envelope: Envelope, response: Response) -> State:
    """Rejected/duplicate actions never mutate state; recovery never declares absence."""
    if not response.accepted or envelope.request_id in state.applied:
        return state
    try:
        new = apply_event(state, envelope, response, CONFIG)
    except GeometryError as exc:
        if "empty set" not in str(exc):
            raise
        # Rebuild from positive observations only, dropping optional paired negative
        # halfplanes. This is an outer recovery, not a relaxed bearing error bound.
        c = envelope.intent.channel
        if c is None:
            raise
        b = state.beliefs[c - 1]
        poly = initial_state("PUBLIC", CONFIG).beliefs[0].polygon
        for obs in b.observations:
            if obs.result == "direction":
                poly = clip_wedge(poly, obs.position, radians(obs.svd_deg or 0), CONFIG.epsilon_rad)
                poly = clip_disk(poly, obs.position, 1500, CONFIG.disk_sides)
            elif obs.result == "near":
                poly = clip_disk(poly, obs.position, 5, CONFIG.disk_sides)
        if not poly:
            raise GeometryError("Positive observation history is inconsistent") from exc
        # Preserve all raw history. Applying the new event may still fail, in which
        # case this run is explicitly incomplete rather than silently losing a source.
        beliefs = list(state.beliefs)
        beliefs[c - 1] = replace(b, polygon=poly, status=Status.LOCALIZING)
        try:
            new = apply_event(replace(state, beliefs=tuple(beliefs)), envelope, response, CONFIG)
        except GeometryError as retry_exc:
            raise GeometryError("Positive observation history is inconsistent") from retry_exc
    beliefs = list(new.beliefs)
    c = envelope.intent.channel
    if c is not None:
        b = beliefs[c - 1]
        if b.status == Status.UNKNOWN and not unknown_cells(new, c):
            evidence = tuple(o.request_id for o in b.observations if o.result == "no_signal")
            beliefs[c - 1] = replace(
                b,
                status=Status.ABSENT,
                certificate=Certificate("full_cell_negative_cover", c, evidence, b.version),
            )
    return replace(new, beliefs=tuple(beliefs))


@lru_cache(maxsize=8192)
def bound(poly: tuple[Point, ...]) -> CircleBound:
    return mec(poly)
