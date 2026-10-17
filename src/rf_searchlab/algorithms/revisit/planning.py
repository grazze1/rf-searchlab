from dataclasses import dataclass

from rf_searchlab.algorithms.visibility.planning import Plan as Previous


@dataclass(frozen=True)
class Plan(Previous):
    pickup_channel: int | None = None
    pickup_steps: int = 0
    pickup_attempted: tuple = ()
    pickup_count: int = 0
    scan_reroutes: int = 0
