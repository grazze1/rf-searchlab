from dataclasses import dataclass

from rf_searchlab.algorithms.lookahead.belief import State as PreviousState
from rf_searchlab.algorithms.lookahead.belief import complete, reject_duplicate, verify_count
from rf_searchlab.algorithms.lookahead.belief import initial_state as previous_initial

__all__ = [
    "State",
    "DirectionWitness",
    "initial_state",
    "complete",
    "verify_count",
    "reject_duplicate",
]


@dataclass(frozen=True)
class DirectionWitness:
    channel: int
    request_id: str
    polygon_version: int
    max_distance_m: float


@dataclass(frozen=True)
class State(PreviousState):
    direction_witnesses: tuple[DirectionWitness, ...] = ()


def initial_state(config):
    return State(**vars(previous_initial(config)))
