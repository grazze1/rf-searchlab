"""Strict algorithm configuration; physical constants are not configuration knobs."""

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

SOURCE_RADIUS_M = 1800.0
RECEPTION_MIN_M = 1000.0
RECEPTION_MAX_M = 1500.0
NEAR_M = 5.0
CLEAR_M = 20.0
SPEED_M_S = 5.0
PHYSICAL_ERROR_DEG = 1.0
STRATEGIES = ("cwm", "rasm", "rasm_batch", "rasm_route", "rasm_shared", "coverage_first_route")
EPISODE_STRATEGIES = STRATEGIES[2:]


@dataclass(frozen=True)
class Config:
    strategy: str = "cwm"
    disk_sides: int = 256
    geometry_padding_m: float = 1e-7
    bearing_padding_deg: float = 0.01
    safety_margin_m: float = 0.05
    rf_budget: int = 6
    candidate_limit: int = 96
    top_m: int = 12
    angle_intervals: int = 12
    planning_budget_s: float = 0.5
    audit: bool = False
    max_actions: int = 20000
    optical_batch_max_cells: int = 16
    episode_internal_budget_s: float = 120.0
    episode_rf_max_actions: int = 2
    shared_max_rf_actions: int = 2
    shared_gain_margin_s: float = 1.0
    route_eval_limit: int = 512
    route_passes: int = 2
    budget_mode: str = "work"
    planning_work_limit: int = 20000
    clearance_refinement: bool = False
    adaptive_angles: bool = False
    optical_refinement: bool = False
    max_angle_intervals: int = 96
    shared_legacy_threshold: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES or type(self.audit) is not bool:
            raise ValueError("Unknown strategy or invalid audit mode")
        if self.budget_mode not in ("work", "time"):
            raise ValueError("budget_mode must be work or time")
        for name in (
            "clearance_refinement",
            "adaptive_angles",
            "optical_refinement",
            "shared_legacy_threshold",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Invalid {name}")
        for name in (
            "disk_sides",
            "rf_budget",
            "candidate_limit",
            "top_m",
            "angle_intervals",
            "max_actions",
            "optical_batch_max_cells",
            "episode_rf_max_actions",
            "shared_max_rf_actions",
            "route_eval_limit",
            "route_passes",
            "planning_work_limit",
            "max_angle_intervals",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("rf_budget", "shared_max_rf_actions") else 1):
                raise ValueError(f"Invalid {name}")
        if self.disk_sides < 32:
            raise ValueError("disk_sides must be >=32")
        for name in (
            "geometry_padding_m",
            "bearing_padding_deg",
            "safety_margin_m",
            "planning_budget_s",
            "episode_internal_budget_s",
            "shared_gain_margin_s",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name}")
        if self.bearing_padding_deg < 0.01 or self.bearing_padding_deg > 0.1:
            raise ValueError("Transport padding must remain conservative, 0.01..0.1 degrees")
        if not 1e-10 <= self.geometry_padding_m <= 1e-4 or not 0.01 <= self.safety_margin_m <= 0.5:
            raise ValueError("Numerical safety settings outside reviewed bounds")
        if self.episode_internal_budget_s < 11 or self.max_angle_intervals < self.angle_intervals:
            raise ValueError("Episode budget must reserve near; angle refinement cannot coarsen")

    @property
    def epsilon_rad(self) -> float:
        return math.radians(PHYSICAL_ERROR_DEG + self.bearing_padding_deg)


def load_config(path: Path | None = None, strategy: str | None = None) -> Config:
    """Read TOML algorithm keys; reject all unknown keys and physical overrides."""
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path else {}
    unknown = set(data) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    if strategy is not None:
        data["strategy"] = strategy
    return Config(**data)
