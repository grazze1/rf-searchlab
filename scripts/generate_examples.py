"""Regenerate the small, synthetic scenario examples shipped with the product."""

import json
from dataclasses import asdict
from pathlib import Path

from rf_searchlab.simulator import generate

folder = Path(__file__).resolve().parents[1] / "examples"
for environment, seed in (("omni", 5101), ("mixed", 5201)):
    scene = generate(seed, environment, count=5)
    (folder / f"{environment}_scene.json").write_text(
        json.dumps(asdict(scene), ensure_ascii=False, indent=2), encoding="utf-8"
    )
