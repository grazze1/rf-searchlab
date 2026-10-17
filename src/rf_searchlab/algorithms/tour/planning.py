from dataclasses import dataclass

from rf_searchlab.algorithms.revisit.planning import Plan as Previous


@dataclass(frozen=True)
class Plan(Previous):
    # Committed service order (channels) and the belief versions it was computed for.
    tour_order: tuple[int, ...] = ()
    tour_versions: tuple = ()
    tour_count: int = 0
    safe_rf_count: int = 0
