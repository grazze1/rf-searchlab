from dataclasses import dataclass

from rf_searchlab.algorithms.resilient.config import Config as Previous


@dataclass(frozen=True)
class Config(Previous):
    name: str = "rebearing"
    route_mode: str = "nearest"
    direction_recovery: bool = True
    rebearing: bool = True
    rebearing_limit: int = 4
    rebearing_min_radius_m: float = 25.0
    rebearing_min_angle_deg: float = 12.0
    rebearing_min_prob: float = 0.5
    rebearing_max_range_m: float = 1500.0
    enroute_clear: bool = True
    enroute_ratio: float = 1.0
    enroute_slack_m: float = 0.0
    enroute_rf: bool = True
    enroute_rf_radius_m: float = 150.0
    enroute_rf_min_prob: float = 0.6
    recovery_gate: bool = True
    blind_guard: bool = True
    guard_min_prob: float = 0.35
    rf_extension: int = 0
    rf_extension_upper_s: float = 150.0

    def __post_init__(self):
        super().__post_init__()
        if not (self.station_batch and self.service_commit and self.two_phase):
            raise ValueError("component requires the frozen two-phase station-batch base")
        if not 0 <= self.rebearing_limit <= 8 or not 0 <= self.rebearing_min_prob <= 1:
            raise ValueError("Unreviewed re-bearing budget")
        if not 0 < self.rebearing_min_angle_deg < 90 or self.rebearing_min_radius_m < 0:
            raise ValueError("Invalid re-bearing novelty thresholds")
        if not 1000 <= self.rebearing_max_range_m <= 1500:
            raise ValueError("Re-bearing range must stay inside the physical reception band")
        if self.enroute_ratio < 0 or self.enroute_slack_m < 0:
            raise ValueError("Invalid en-route clear thresholds")
        if not 0 <= self.guard_min_prob <= 1 or not 0 <= self.rf_extension <= 8:
            raise ValueError("Invalid blind guard or RF extension budget")
        if self.enroute_rf_radius_m < 0 or not 0 <= self.enroute_rf_min_prob <= 1:
            raise ValueError("Invalid en-route RF thresholds")
        if self.rf_extension_upper_s < 0:
            raise ValueError("Invalid RF extension threshold")
