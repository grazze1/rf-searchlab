"""Deterministic, offline reconstruction of a committed run prefix."""

import json
import os
from dataclasses import asdict

from rf_searchlab.core.protocol import intent_from_dict, loads, make_envelope, parse_response
from rf_searchlab.evaluation import replay
from rf_searchlab.policies import PublicPolicy
from rf_searchlab.simulator import Simulator
from rf_searchlab.storage import read_json


def reconstruct(run):
    """Audit every prior decision/response before making another action."""
    result = replay(run.id)
    if result["errors"]:
        raise ValueError(f"Cannot restore corrupt events: {result['errors'][0]}")
    events = result["events"]
    decisions = []
    with (run.path / "decisions.jsonl").open("rb") as file:
        for line in file:
            if len(line) > 4_000_000:
                raise ValueError("Decision too large")
            decisions.append(loads(line))
    if len(decisions) < len(events) or len(decisions) > len(events) + 1:
        raise ValueError("Decision/event count mismatch")
    env = Simulator(run.truth, run.error_mode, run.observation_model)
    policy = PublicPolicy(run.strategy)
    public_states = []
    planning_s = 0.0
    for i, event in enumerate(events):
        decision = decisions[i]
        if event["sequence"] != i or decision["sequence"] != i:
            raise ValueError("Recovery sequence mismatch")
        intent = policy.next_action(decision.get("remaining_wall_s", run.wall_seconds))
        if asdict(intent) != event["action"] or decision["action"] != event["action"]:
            raise ValueError(f"Policy decision mismatch at event {i}")
        envelope = make_envelope(intent_from_dict(event["action"]), "LOCAL_ONLY", event["request_id"], 0)
        response = parse_response(envelope.path, env.execute(envelope))
        if asdict(response) != event["response"]:
            raise ValueError(f"Simulator response mismatch at event {i}")
        policy.observe(envelope, response)
        public_states.append(dict(sequence=i, **policy.public_summary()))
        planning_s += decision.get("planning_s", 0.0)

    pending = read_json(run.path / "pending.json") if (run.path / "pending.json").exists() else None
    if pending is not None and pending["sequence"] == len(events) - 1:
        if pending["action"] != events[-1]["action"] or pending["request_id"] != events[-1]["request_id"]:
            raise ValueError("Stale pending action does not match committed event")
        pending = None
    if len(decisions) == len(events) + 1:
        decision = decisions[-1]
        if decision["sequence"] != len(events):
            raise ValueError("Pending decision sequence mismatch")
        expected = {
            "sequence": len(events),
            "request_id": f"action-{len(events)}",
            "action": decision["action"],
            "remaining_wall_s": decision.get("remaining_wall_s", run.wall_seconds),
        }
        if pending is not None and pending != expected:
            raise ValueError("Pending action/decision mismatch")
        pending = expected
        planning_s += decision.get("planning_s", 0.0)
    elif pending is not None:
        raise ValueError("Pending action lacks durable decision")

    # A crash may have committed an event before writing its public state.
    states_path = run.path / "public_states.jsonl"
    temp = states_path.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as file:
        for state in public_states:
            file.write(json.dumps(state, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    temp.replace(states_path)
    return events, env, policy, pending, planning_s
