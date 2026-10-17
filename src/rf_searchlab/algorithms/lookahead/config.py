"""Explicit feature switches, avoiding overloaded alphabetic variants."""

from dataclasses import dataclass

from rf_searchlab.algorithms.directional.config import Config as BaseConfig


@dataclass(frozen=True)
class Config(BaseConfig):
    variant: str = "C"
    name: str = "count"
    count_bound: bool = True
    station_batch: bool = False
    service_commit: bool = False
    two_phase: bool = False
    candidate_families: bool = False
    stripes: bool = False
    continuation: bool = False
    recovery_batch: int = 0
    recovery_limit: int = 2
    conditional_tasks: bool = False
    selective_planning: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.recovery_batch not in (0, 4, 8, 16) or self.recovery_limit != 2:
            raise ValueError("Recovery work must be bounded by reviewed settings")


def phase_one():
    return {
        "count": Config(),
        "station": Config(name="station", count_bound=False, station_batch=True),
        "commit": Config(name="commit", count_bound=False, service_commit=True),
        "schedule": Config(name="schedule", count_bound=False, station_batch=True, service_commit=True),
        "core": Config(name="core", station_batch=True, service_commit=True),
        "two_phase": Config(name="two_phase", station_batch=True, service_commit=True, two_phase=True),
    }
