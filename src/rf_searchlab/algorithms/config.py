"""Finite, serializable experiment controls; physics remains in the frozen package."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from rf_searchlab.core.config import Config

BASELINE = Config(strategy="rasm_shared", budget_mode="work", clearance_refinement=True)


@dataclass(frozen=True)
class SearchConfig:
    depth: int = 2
    beam_width: int = 4
    max_candidates: int = 12
    hypotheses: int = 4
    node_budget: int = 128
    cpu_soft_s: float = 0.5
    hypothesis_attempts: int = 1024
    seed: int = 1729
    rollout_actions: int = 20000

    def __post_init__(self) -> None:
        for name in (
            "depth",
            "beam_width",
            "max_candidates",
            "hypotheses",
            "node_budget",
            "hypothesis_attempts",
            "rollout_actions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid search {name}")
        if self.cpu_soft_s < 0:
            raise ValueError("Negative soft budget")


TEACHER = SearchConfig(depth=3, beam_width=8, hypotheses=8, node_budget=512, cpu_soft_s=0)


def load_search(path: Path | None, default: SearchConfig = TEACHER) -> SearchConfig:
    """Read only known search keys; missing keys inherit the explicit profile."""
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path else {}
    data = data.get("search", data)
    allowed = {f.name for f in fields(SearchConfig)}
    if set(data) - allowed:
        raise ValueError(f"Unknown search keys: {set(data) - allowed}")
    return SearchConfig(**{f.name: data.get(f.name, getattr(default, f.name)) for f in fields(default)})
