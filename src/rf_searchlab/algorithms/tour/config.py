from dataclasses import dataclass

from rf_searchlab.algorithms.revisit.config import Config as Previous


@dataclass(frozen=True)
class Config(Previous):
    name: str = "adaptive_tour"
    # P0-a: block-level open tour over remaining known sources once the scan is done.
    service_tour: bool = True
    service_tour_margin_m: float = 20.0
    # P0-b: orientation-safe near-range RF for service decisions.
    safe_rf: bool = True
    safe_rf_min_prob: float = 0.9
    safe_rf_backoffs_m: tuple = (100.0, 200.0, 300.0, 400.0)
    service_guard_min_prob: float = 0.6
    # P1: en-route clear detour cap.
    enroute_max_detour_m: float = 500.0
    # P2: near-station third bearing for almost-localised sources.
    near_rebearing: bool = True
    near_rebearing_station_m: float = 600.0
    near_rebearing_max_radius_m: float = 60.0
    near_rebearing_min_angle_deg: float = 6.0

    def __post_init__(self):
        super().__post_init__()
        if self.service_tour_margin_m < 0 or self.enroute_max_detour_m < 0:
            raise ValueError("Invalid tour/en-route thresholds")
        if not 0 <= self.safe_rf_min_prob <= 1 or not 0 <= self.service_guard_min_prob <= 1:
            raise ValueError("Invalid safe-RF probabilities")
        if not all(0 < x <= 990 for x in self.safe_rf_backoffs_m):
            raise ValueError("Back-off must stay inside the guaranteed reception band")
        if not 0 < self.near_rebearing_station_m <= 1500 or self.near_rebearing_max_radius_m < 0:
            raise ValueError("Invalid near re-bearing thresholds")
        if not 0 < self.near_rebearing_min_angle_deg < 90:
            raise ValueError("Invalid near re-bearing angle")
