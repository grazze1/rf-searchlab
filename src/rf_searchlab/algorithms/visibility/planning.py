from dataclasses import dataclass

from rf_searchlab.algorithms.resilient.planning import Plan as Previous


@dataclass(frozen=True)
class Plan(Previous):
    # (channel, count) pairs of accepted re-bearings taken at fixed stops.
    rebearings: tuple = ()
    enroute_clears: int = 0
