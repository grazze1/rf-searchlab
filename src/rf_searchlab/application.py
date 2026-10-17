"""Serial run service shared by CLI and UI; pause at committed action boundaries."""

import csv
import hashlib
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

from rf_searchlab import __version__
from rf_searchlab.core.protocol import make_envelope, parse_response
from rf_searchlab.evaluation import ledger
from rf_searchlab.policies import REGISTRY, PublicPolicy
from rf_searchlab.simulator import Simulator, from_dict
from rf_searchlab.storage import append_event, atomic_json, exclusive_run, read_json, run_path


class Run:
    def __init__(
        self,
        truth,
        strategy,
        max_actions=1500,
        wall_seconds=120,
        error_mode="spatial",
        observation_model="bounded_round2",
    ):
        if strategy not in REGISTRY or REGISTRY[strategy] != truth.environment:
            raise ValueError("Strategy/environment mismatch")
        if type(max_actions) is not int or not 1 <= max_actions <= 20000 or not 1 <= wall_seconds <= 1200:
            raise ValueError("Budget outside bounds")
        self.id = uuid4().hex
        self.path = run_path(self.id)
        self.path.mkdir(parents=True, exist_ok=False)
        self.strategy = strategy
        self.truth = truth
        self.max_actions = max_actions
        self.wall_seconds = wall_seconds
        self.error_mode = error_mode
        self.observation_model = observation_model
        self.events = []
        self.metrics = None
        self._lock = threading.Lock()
        self._pause = threading.Event()
        self._cancel = threading.Event()
        self._started = False
        self._restored = False
        self.manifest = {
            "schema_version": 1,
            "run_id": self.id,
            "status": "created",
            "version": __version__,
            "strategy": strategy,
            "environment": truth.environment,
            "mode": truth.mode,
            "origin": "synthetic_simulation",
            "observation_model": observation_model,
            "clock_model": "microsecond_per_action",
            "input_sha256": hashlib.sha256(json.dumps(asdict(truth), sort_keys=True).encode()).hexdigest(),
            "created_at": datetime.now().astimezone().isoformat(),
            "failure": None,
            "recovery_schema": 1,
        }
        config = {
            "strategy": strategy,
            "max_actions": max_actions,
            "wall_seconds": wall_seconds,
            "error_mode": error_mode,
            "observation_model": observation_model,
        }
        self.manifest["config_sha256"] = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        atomic_json(self.path / "manifest.json", self.manifest)
        atomic_json(self.path / "config.resolved.json", config)
        (self.path / "evaluation_private").mkdir()
        atomic_json(self.path / "evaluation_private/world.json", asdict(truth))
        (self.path / "events.jsonl").touch()
        (self.path / "decisions.jsonl").touch()

    @classmethod
    def restore(cls, run_id):
        """Open an interrupted offline run; execute() audits its history under a lock."""
        path = run_path(run_id)
        manifest = read_json(path / "manifest.json")
        if manifest.get("run_id") != run_id or manifest.get("recovery_schema") != 1:
            raise ValueError("Run has no supported recovery record")
        if manifest.get("version") != __version__:
            raise ValueError("Run version differs from installed code")
        if manifest["status"] not in ("created", "running", "paused"):
            raise ValueError("Only interrupted runs can be restored")
        config = read_json(path / "config.resolved.json")
        if (
            hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            != manifest["config_sha256"]
        ):
            raise ValueError("Run configuration hash mismatch")
        truth_data = read_json(path / "evaluation_private/world.json")
        if (
            hashlib.sha256(json.dumps(truth_data, sort_keys=True).encode()).hexdigest()
            != manifest["input_sha256"]
        ):
            raise ValueError("Run input hash mismatch")
        truth = from_dict(truth_data)
        if config["strategy"] != manifest["strategy"] or truth.environment != manifest["environment"]:
            raise ValueError("Run identity mismatch")
        obj = cls.__new__(cls)
        obj.id, obj.path, obj.manifest = run_id, path, manifest
        obj.strategy, obj.truth = config["strategy"], truth
        obj.max_actions, obj.wall_seconds = config["max_actions"], config["wall_seconds"]
        obj.error_mode, obj.observation_model = config["error_mode"], config["observation_model"]
        obj.events, obj.metrics = [], None
        obj._lock = threading.Lock()
        obj._pause = threading.Event()
        obj._cancel = threading.Event()
        obj._started = False
        obj._restored = True
        return obj

    def status(self, value, failure=None):
        with self._lock:
            self.manifest.update(status=value, failure=failure)
            atomic_json(self.path / "manifest.json", self.manifest)

    def pause(self):
        self._pause.set()

    def resume(self):
        self._pause.clear()

    def cancel(self):
        self._cancel.set()
        self._pause.clear()

    def execute(self):
        with exclusive_run(self.path):
            return self._execute_locked()

    def _execute_locked(self):
        with self._lock:
            if self._started:
                raise RuntimeError("Run already started")
            self._started = True
        if self._restored:
            from rf_searchlab.recovery import reconstruct

            self.events, env, policy, pending, planning = reconstruct(self)
            self.manifest["recovery_count"] = self.manifest.get("recovery_count", 0) + 1
            self.manifest["resumed_at"] = datetime.now().astimezone().isoformat()
        else:
            env = Simulator(self.truth, self.error_mode, self.observation_model)
            policy = PublicPolicy(self.strategy)
            pending = None
            planning = 0.0
        self.status("running")
        started = time.perf_counter()
        try:
            if self.events and self.events[-1]["action"]["path"] == "/exit":
                if not policy.complete() or not env.score()["all_cleared"]:
                    raise ValueError("Committed exit lacks completed task evidence")
                self.status("completed")
            for i in range(len(self.events), self.max_actions + 3):
                if self.manifest["status"] == "completed":
                    break
                if self._cancel.is_set():
                    self.status("cancelled")
                    break
                if self._pause.is_set():
                    self.status("paused")
                    pause_start = time.perf_counter()
                    while self._pause.is_set() and not self._cancel.wait(0.05):
                        pass
                    started += time.perf_counter() - pause_start
                    if self._cancel.is_set():
                        self.status("cancelled")
                        break
                    self.status("running")
                remaining = self.wall_seconds - (time.perf_counter() - started)
                if env.virtual_time >= 360000:
                    raise TimeoutError("Virtual time budget exhausted")
                if remaining <= 0:
                    raise TimeoutError("Wall budget exhausted")
                if pending is not None and pending["sequence"] == i:
                    intent = policy.next_action(pending["remaining_wall_s"])
                    if asdict(intent) != pending["action"]:
                        raise ValueError("Pending policy action changed during restore")
                    decision_remaining = pending["remaining_wall_s"]
                    reuse_decision = True
                    pending = None
                else:
                    t = time.perf_counter()
                    intent = policy.next_action(remaining)
                    duration = time.perf_counter() - t
                    planning += duration
                    decision_remaining = remaining
                    reuse_decision = False
                if i >= self.max_actions and intent.path != "/exit":
                    raise TimeoutError("Action budget exhausted")
                if time.perf_counter() - started > self.wall_seconds:
                    raise TimeoutError("Planning exceeded wall budget")
                envelope = make_envelope(intent, "LOCAL_ONLY", f"action-{i}", 0)
                if not reuse_decision:
                    append_event(
                        self.path / "decisions.jsonl",
                        {
                            "schema_version": 1,
                            "sequence": i,
                            "reason": policy.detail,
                            "action": asdict(envelope.intent),
                            "remaining_wall_s": decision_remaining,
                            "planning_s": duration,
                        },
                    )
                atomic_json(
                    self.path / "pending.json",
                    {
                        "sequence": i,
                        "request_id": envelope.request_id,
                        "action": asdict(envelope.intent),
                        "remaining_wall_s": decision_remaining,
                    },
                )
                response = parse_response(envelope.path, env.execute(envelope))
                if not response.accepted:
                    raise ValueError("Action rejected; public state preserved")
                # Persist the executed action before reducer/planner code can fail.
                event = {
                    "schema_version": 1,
                    "sequence": i,
                    "request_id": envelope.request_id,
                    "action": asdict(envelope.intent),
                    "response": asdict(response),
                }
                append_event(self.path / "events.jsonl", event)
                self.events.append(event)
                atomic_json(self.path / "pending.json", None)
                policy.observe(envelope, response)
                append_event(self.path / "public_states.jsonl", dict(sequence=i, **policy.public_summary()))
                if intent.path == "/exit":
                    if not policy.complete():
                        raise ValueError("Exit lacks public completion certificate")
                    if not env.score()["all_cleared"]:
                        raise ValueError("Scoring contradicts completion")
                    self.status("completed")
                    break
            else:
                raise TimeoutError("Action budget exhausted")
        except Exception as exc:  # noqa: BLE001 -- persist arbitrary policy failures at the run boundary
            self.status("failed", f"{type(exc).__name__}: {exc}")
        costs = ledger(self.events)
        score = env.score()
        self.metrics = dict(
            schema_version=1,
            run_id=self.id,
            strategy=self.strategy,
            environment=self.truth.environment,
            status=self.manifest["status"],
            failure=self.manifest["failure"],
            **score,
            cleared_fraction=score["cleared"] / score["true_count"] if score["true_count"] else None,
            seconds_per_cleared=costs["total_s"] / score["cleared"] if score["cleared"] else None,
            reported_virtual_time_s=env.virtual_time,
            planning_s=planning,
            runtime_s=time.perf_counter() - started,
            **costs,
        )
        atomic_json(self.path / "metrics.json", self.metrics)
        with (self.path / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.metrics))
            w.writeheader()
            w.writerow(self.metrics)
        (self.path / "report.md").write_text(
            "# 合成仿真报告\n\n此结果来自自建离线环境。\n\n```json\n"
            + json.dumps(self.metrics, ensure_ascii=False, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
        return self.metrics


class TaskManager:
    """One interactive worker; interrupted runs can be restored through the CLI."""

    def __init__(self):
        self.runs = {}
        self.lock = threading.Lock()
        self.worker = None

    def start(self, truth, strategy, **kwargs):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise RuntimeError("A task is already running")
            run = Run(truth, strategy, **kwargs)
            self.runs[run.id] = run
            self.worker = threading.Thread(target=run.execute, name="rf-offline-run", daemon=True)
            self.worker.start()
            return run.id
