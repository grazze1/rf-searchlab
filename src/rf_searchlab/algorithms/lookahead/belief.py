"""Count upper bound uses independently verifiable accepted positive evidence only."""

from dataclasses import dataclass, replace

from rf_searchlab.algorithms.directional.belief import RESOLVED, Status
from rf_searchlab.algorithms.directional.belief import State as BaseState
from rf_searchlab.algorithms.directional.belief import initial_state as base_initial
from rf_searchlab.algorithms.lookahead.config import Config
from rf_searchlab.core.domain import Certificate, GeometryError, Observation


@dataclass(frozen=True)
class PresentWitness:
    channel: int
    request_id: str


@dataclass(frozen=True)
class State(BaseState):
    present: tuple[PresentWitness, ...] = ()


def initial_state(config=Config()):
    return State(**vars(base_initial(config)))


def verified_positive(state, channel, rid):
    if not 1 <= channel <= 20 or rid not in state.applied:
        return False
    b = state.beliefs[channel - 1]
    if any(
        type(o) is Observation and o.request_id == rid and o.result in ("near", "direction")
        for o in b.observations
    ):
        return True
    return bool(
        b.status == Status.CLEARED
        and b.certificate
        and b.certificate.kind == "clear_success"
        and b.certificate.channel == channel
        and rid in b.certificate.request_ids
    )


def verify_count(state, channel, ids):
    witnesses = state.present
    if (
        type(channel) is not int
        or not 1 <= channel <= 20
        or len(witnesses) != 16
        or len({w.channel for w in witnesses}) != 16
        or len(set(ids)) != 16
        or set(ids) != {w.request_id for w in witnesses}
        or channel in {w.channel for w in witnesses}
        or state.beliefs[channel - 1].ever_detected
        or not all(verified_positive(state, w.channel, w.request_id) for w in witnesses)
    ):
        raise GeometryError("COUNT_UPPER_BOUND requires 16 distinct accepted positive channels")


def apply_count(state, enabled=True):
    if len(state.present) > 16:
        raise GeometryError("More than 16 real positive channels contradict the source bound")
    if not enabled or len(state.present) != 16:
        return state
    ids = tuple(w.request_id for w in state.present)
    beliefs = list(state.beliefs)
    for b in beliefs:
        if b.status == Status.UNKNOWN:
            verify_count(state, b.channel, ids)
            beliefs[b.channel - 1] = replace(
                b,
                status=Status.CERTIFIED_ABSENT,
                certificate=Certificate("COUNT_UPPER_BOUND", b.channel, ids, b.version),
            )
    return replace(state, beliefs=tuple(beliefs))


def complete(state):
    cleared = [b for b in state.beliefs if b.status == Status.CLEARED]

    def good_clear(b):
        return bool(
            b.certificate
            and b.certificate.kind == "clear_success"
            and b.certificate.channel == b.channel
            and b.certificate.request_ids
            and all(verified_positive(state, b.channel, rid) for rid in b.certificate.request_ids)
        )

    if len(cleared) == 16 and all(good_clear(b) for b in cleared):
        return True
    for b in state.beliefs:
        cert = b.certificate
        if b.status not in RESOLVED or not cert or cert.channel != b.channel or not cert.request_ids:
            return False
        if b.status == Status.CLEARED:
            if not good_clear(b):
                return False
        elif cert.kind == "COUNT_UPPER_BOUND":
            verify_count(state, b.channel, cert.request_ids)
        elif (
            cert.kind != "directional_negative_hull"
            or b.ever_detected
            or not all(r in state.applied for r in cert.request_ids)
        ):
            return False
    return True


def reject_duplicate(state, intent):
    if intent.channel is None:
        return
    b = state.beliefs[intent.channel - 1]
    if b.status in RESOLVED:
        raise GeometryError("Action on resolved channel")
    if intent.path == "/measure" and any(o.position == intent.position for o in b.observations):
        raise GeometryError("Duplicate same-channel serialized RF coordinate")
    if intent.path == "/clear" and any(e.position == intent.position for e in b.exclusions):
        raise GeometryError("Duplicate failed optical coordinate")
