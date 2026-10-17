"""Algorithm switches, separate from the immutable public physical rules."""

import tomllib
from dataclasses import dataclass
from math import radians
from pathlib import Path

RESOURCE = Path(__file__).resolve().parents[2] / "data" / "coverage"


@dataclass(frozen=True)
class Config:
    variant: str = "A"
    rf_budget: int = 6
    fairness_actions: int = 8
    candidate_limit: int = 12
    scenario_limit: int = 64
    rf_refine_limit: int = 4
    hypothesis_refine_limit: int = 4
    planning_budget_s: float = 0.5
    risk_weight: float = 0.25
    approach_fraction: float = 0.35
    approach_lateral_m: float = 100.0
    approach_min_diameter_m: float = 400.0
    strict65: bool = False
    lookahead_depth: int = 1
    safety_margin_m: float = 0.05
    geometry_padding_m: float = 1e-7
    disk_sides: int = 64

    def __post_init__(self):
        if self.variant not in "ABCDEF" or len(self.variant) != 1:
            raise ValueError("Unknown Mixed variant")
        if not 1 <= self.candidate_limit <= 12 or not 1 <= self.scenario_limit <= 64:
            raise ValueError("Candidate/scenario budget outside reviewed limits")
        if (
            not 1 <= self.rf_refine_limit <= self.candidate_limit
            or not 1 <= self.hypothesis_refine_limit <= 7
        ):
            raise ValueError("Invalid fine-ranking work limit")
        if self.lookahead_depth not in (1, 2) or self.rf_budget < 0:
            raise ValueError("Invalid lookahead/RF budget")
        if self.fairness_actions < 1 or self.planning_budget_s <= 0:
            raise ValueError("Invalid progress/work budget")
        if not 0 <= self.risk_weight <= 1:
            raise ValueError("Invalid risk weight")

    @property
    def epsilon_rad(self):
        return radians(1.0 + 0.005 + 1e-7)

    @property
    def dynamic(self):
        return self.variant >= "C"

    @property
    def directional(self):
        return self.variant >= "D"

    @property
    def time_compare(self):
        return self.variant >= "E"

    @property
    def shared(self):
        return self.variant >= "F"


def load_config(path: Path) -> Config:
    return Config(**tomllib.loads(path.read_text(encoding="utf-8")))
