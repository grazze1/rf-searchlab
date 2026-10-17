"""Crash-boundary tests against persisted, real offline actions."""

import json

import pytest

from rf_searchlab import application
from rf_searchlab.application import Run
from rf_searchlab.evaluation import replay
from rf_searchlab.policies import PublicPolicy
from rf_searchlab.simulator import Simulator, generate
from rf_searchlab.storage import exclusive_run


@pytest.mark.parametrize("environment,strategy", [("omni", "coverage_route"), ("mixed", "adaptive_tour")])
def test_resume_pending_action_reuses_accepted_request_id(monkeypatch, environment, strategy):
    run = Run(generate(101, environment), strategy, wall_seconds=60)
    original = Simulator.execute
    calls = 0

    def crash_before_execution(simulator, envelope):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt("simulated process loss before response")
        return original(simulator, envelope)

    with monkeypatch.context() as patch:
        patch.setattr(Simulator, "execute", crash_before_execution)
        with pytest.raises(KeyboardInterrupt):
            run.execute()
    pending = json.loads((run.path / "pending.json").read_text())
    assert pending["sequence"] == 2 and pending["request_id"] == "action-2"
    assert replay(run.id)["status"] == "partial"

    restored = Run.restore(run.id)
    metrics = restored.execute()
    assert metrics["status"] == "completed" and metrics["all_cleared"]
    events = replay(run.id)["events"]
    assert [e["request_id"] for e in events] == [f"action-{i}" for i in range(len(events))]
    decisions = (run.path / "decisions.jsonl").read_text().splitlines()
    assert len(decisions) == len(events)
    assert json.loads((run.path / "pending.json").read_text()) is None


def test_restore_repairs_state_log_after_committed_event(monkeypatch):
    run = Run(generate(101, "omni"), "coverage_route", wall_seconds=60)
    original = PublicPolicy.observe
    calls = 0

    def crash_before_reducer(policy, envelope, response):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt("simulated process loss after durable event")
        return original(policy, envelope, response)

    with monkeypatch.context() as patch:
        patch.setattr(PublicPolicy, "observe", crash_before_reducer)
        with pytest.raises(KeyboardInterrupt):
            run.execute()
    assert len((run.path / "events.jsonl").read_text().splitlines()) == 3
    assert len((run.path / "public_states.jsonl").read_text().splitlines()) == 2
    metrics = Run.restore(run.id).execute()
    assert metrics["status"] == "completed"
    assert len((run.path / "public_states.jsonl").read_text().splitlines()) == metrics["actions"]


def test_restore_accepts_stale_pending_after_committed_event(monkeypatch):
    run = Run(generate(101), "coverage_route", wall_seconds=60)
    original = application.append_event
    committed = 0

    def crash_after_event_write(path, data):
        nonlocal committed
        original(path, data)
        if path.name == "events.jsonl":
            committed += 1
            if committed == 3:
                raise KeyboardInterrupt("event durable, pending marker not cleared")

    with monkeypatch.context() as patch:
        patch.setattr(application, "append_event", crash_after_event_write)
        with pytest.raises(KeyboardInterrupt):
            run.execute()
    pending = json.loads((run.path / "pending.json").read_text())
    assert pending["sequence"] == 2
    assert len((run.path / "events.jsonl").read_text().splitlines()) == 3
    assert Run.restore(run.id).execute()["status"] == "completed"
    events = replay(run.id)["events"]
    assert len({e["request_id"] for e in events}) == len(events)


def test_restore_rejects_modified_input_and_unfinished_legacy_record():
    run = Run(generate(101), "coverage_route")
    private = run.path / "evaluation_private/world.json"
    world = json.loads(private.read_text())
    world["noise_seed"] += 1
    private.write_text(json.dumps(world))
    with pytest.raises(ValueError, match="input hash"):
        Run.restore(run.id)
    private.write_text(json.dumps({**world, "noise_seed": world["noise_seed"] - 1}))
    run.manifest.pop("recovery_schema")
    (run.path / "manifest.json").write_text(json.dumps(run.manifest))
    with pytest.raises(ValueError, match="no supported recovery"):
        Run.restore(run.id)


def test_restore_refuses_concurrent_execution():
    run = Run(generate(101), "coverage_route")
    with exclusive_run(run.path), pytest.raises(RuntimeError, match="another process"):
        Run.restore(run.id).execute()


def test_restore_committed_exit_without_duplicate_action():
    run = Run(generate(101, "mixed"), "adaptive_tour", wall_seconds=60)
    original = run.execute()
    assert original["status"] == "completed"
    run.status("running")  # Model a crash after the exit event, before completion metadata.
    before = (run.path / "events.jsonl").read_bytes()
    recovered = Run.restore(run.id).execute()
    assert recovered["status"] == "completed"
    assert recovered["actions"] == original["actions"]
    assert (run.path / "events.jsonl").read_bytes() == before
    manifest = json.loads((run.path / "manifest.json").read_text())
    assert manifest["recovery_count"] == 1
