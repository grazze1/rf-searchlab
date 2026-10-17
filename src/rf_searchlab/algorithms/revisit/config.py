from dataclasses import dataclass

from rf_searchlab.algorithms.visibility.config import Config as Previous


@dataclass(frozen=True)
class Config(Previous):
    reroute_scan: bool = False
    local_pickup: bool = False
    pickup_max_radius_m: float = 150.0
    pickup_max_distance_m: float = 400.0
    pickup_min_future_gap_m: float = 100.0
    pickup_max_detour_m: float = 250.0
    pickup_min_probability: float = 0.65
    pickup_rf_limit: int = 2

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.pickup_rf_limit <= 2:
            raise ValueError("Local pickup must stop within two RF attempts")
        if not 0 <= self.pickup_min_probability <= 1:
            raise ValueError("Invalid reception probability")
        if not 0 < self.pickup_max_radius_m <= 300:
            raise ValueError("Invalid local pickup radius")
        if not 0 < self.pickup_max_distance_m <= 600:
            raise ValueError("Invalid local pickup distance")
        if self.pickup_min_future_gap_m < 0 or self.pickup_max_detour_m < 0:
            raise ValueError("Invalid local pickup cost gate")
