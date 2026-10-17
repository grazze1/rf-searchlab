from dataclasses import dataclass

from rf_searchlab.algorithms.lookahead.config import Config as Previous


@dataclass(frozen=True)
class Config(Previous):
    name: str = "tour"
    station_batch: bool = True
    service_commit: bool = True
    two_phase: bool = True
    route_mode: str = "tour"
    direction_recovery: bool = False
    long_cover_threshold: int = 0
    long_cover_recovery: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.route_mode not in ("legacy", "nearest", "tour"):
            raise ValueError("Unknown source route mode")
        if self.long_cover_threshold not in (0, 32, 64, 128):
            raise ValueError("Unreviewed optical gate")
        if self.long_cover_recovery not in (0, 4, 8, 16):
            raise ValueError("Unbounded optical recovery")
