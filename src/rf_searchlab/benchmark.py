"""Frozen paired synthetic suites; failed rows are always retained."""

import csv
import hashlib
import json
from datetime import datetime
from uuid import uuid4

import numpy as np

from rf_searchlab.application import Run
from rf_searchlab.simulator import generate
from rf_searchlab.storage import atomic_json, root


def scenario_plan(suite, count):
    """Keep development, validation and held-out world seeds disjoint."""
    sizes = {"smoke": 3, "regression": 30, "stress": 200}
    if suite not in sizes or type(count) is not int or not 1 <= count <= sizes[suite]:
        raise ValueError("Invalid benchmark suite or size")
    specifications = []
    offsets = {
        "smoke": (("omni", 1000), ("mixed", 2000)),
        "regression": (("omni", 10000), ("mixed", 20000)),
        "stress": (("omni", 30000), ("mixed", 40000)),
    }
    for environment, offset in offsets[suite]:
        for i in range(count):
            specifications.append(
                {
                    "seed": offset + i,
                    "environment": environment,
                    "count": [10, 13, 16][i % 3],
                    "family": ["uniform", "cluster", "boundary", "radius1000", "outward"][i % 5],
                }
            )
    return specifications


def benchmark(suite="smoke", limit=None):
    sizes = {"smoke": 3, "regression": 30, "stress": 200}
    if suite not in sizes:
        raise ValueError("Unknown suite")
    count = sizes[suite] if limit is None else limit
    specifications = scenario_plan(suite, count)
    folder = root() / "benchmarks" / f"{suite}-{uuid4().hex[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(
        folder / "frozen.json",
        {
            "schema_version": 1,
            "suite": suite,
            "split": {"smoke": "development", "regression": "validation", "stress": "heldout"}[suite],
            "specifications": specifications,
            "sha256": hashlib.sha256(json.dumps(specifications, sort_keys=True).encode()).hexdigest(),
            "max_actions": 1500,
            "wall_seconds_per_run": 60,
            "created_at": datetime.now().astimezone().isoformat(),
        },
    )
    rows = []
    for spec in specifications:
        strategies = (
            ["shared_search", "coverage_route"]
            if spec["environment"] == "omni"
            else ["adaptive_tour", "adaptive_tour_no_pickup"]
        )
        for strategy in strategies:
            run = Run(generate(**spec), strategy, max_actions=1500, wall_seconds=60)
            result = run.execute()
            rows.append(
                dict(**result, scenario_seed=spec["seed"], family=spec["family"], source_count=spec["count"])
            )
            with (folder / "cases.csv").open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(
                f"{suite}: {len(rows)}/{2 * len(specifications)} {strategy} {result['status']} {run.id}",
                flush=True,
            )
    summary = []
    for strategy in sorted({r["strategy"] for r in rows}):
        group = [r for r in rows if r["strategy"] == strategy]
        values = [r["total_s"] for r in group if r["status"] == "completed"]
        summary.append(
            {
                "strategy": strategy,
                "n": len(group),
                "success_rate": sum(r["status"] == "completed" for r in group) / len(group),
                "failures": [r["run_id"] for r in group if r["status"] != "completed"],
                "success_conditional_mean_s": float(np.mean(values)) if values else None,
                "success_conditional_p50_s": float(np.quantile(values, 0.5)) if values else None,
                "success_conditional_p95_s": float(np.quantile(values, 0.95)) if values else None,
                "maximum_success_s": max(values) if values else None,
            }
        )
    atomic_json(
        folder / "summary.json",
        {
            "schema_version": 1,
            "suite": suite,
            "results": summary,
            "caveat": "Synthetic paired samples; no universal performance guarantee.",
        },
    )
    (folder / "report.md").write_text(
        "# 配对合成基准\n\n失败局保留于 cases.csv。Mixed no_pickup 为关闭顺路拾取的消融。\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```",
        encoding="utf-8",
    )
    return folder
