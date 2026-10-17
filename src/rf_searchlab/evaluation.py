"""Independent timing audit and read-only replay: never calls a policy or world."""

import json
from math import hypot, isclose

from rf_searchlab.storage import read_json, run_path


def ledger(events):
    pos = (0.0, 0.0)
    channel = 1
    total = 0.0
    seen = set()
    costs = {"movement_s": 0.0, "measurement_s": 0.0, "switch_s": 0.0, "optical_s": 0.0, "clear_s": 0.0}
    for i, e in enumerate(events):
        if e["sequence"] != i:
            raise ValueError("Non-contiguous event sequence")
        rid = e["request_id"]
        if rid in seen:
            raise ValueError("Repeated applied request")
        seen.add(rid)
        a = e["action"]
        r = e["response"]
        if not r["accepted"]:
            continue
        if a["path"] in ("/measure", "/clear"):
            p = (a["position"]["x_m"], a["position"]["y_m"])
            costs["movement_s"] += hypot(p[0] - pos[0], p[1] - pos[1]) / 5
            pos = p
            if a["path"] == "/measure":
                costs["measurement_s"] += 5
                costs["switch_s"] += a["channel"] != channel
                channel = a["channel"]
            else:
                costs["optical_s"] += 3
                costs["clear_s"] += 2 * (r["result"] == "success")
        total = sum(costs.values())
        if not isclose(total, r["virtual_time_s"], rel_tol=0, abs_tol=(i + 1) * 0.5001e-6 + 1e-8):
            raise ValueError(f"Timing mismatch at event {i}")
    return dict(**costs, total_s=total, movement_m=costs["movement_s"] * 5, actions=len(events))


def replay(run_id):
    p = run_path(run_id)
    events = []
    errors = []
    with (p / "events.jsonl").open(encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                ledger(events + [e])
                events.append(e)
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
                break
    manifest = read_json(p / "manifest.json")
    status = (
        "partial"
        if errors or manifest["status"] not in ("completed", "failed", "cancelled")
        else manifest["status"]
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "events": events,
        "ledger": ledger(events),
        "errors": errors,
    }
