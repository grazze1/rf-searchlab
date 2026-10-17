from dataclasses import dataclass

from rf_searchlab.algorithms.lookahead.planning import Plan as PreviousPlan


@dataclass(frozen=True)
class Plan(PreviousPlan):
    source_route: tuple[int, ...] = ()
    source_versions: tuple = ()
