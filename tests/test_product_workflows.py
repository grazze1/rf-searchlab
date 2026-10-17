"""Exercise published commands and saved experiment data through user workflows."""

import csv
import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

import rf_searchlab
from rf_searchlab.benchmark import benchmark
from rf_searchlab.cli import main
from rf_searchlab.evaluation import replay


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["rf-searchlab", *map(str, args)])
    return main()


def test_published_localization_commands(tmp_path, monkeypatch, capsys):
    bearings = tmp_path / "bearings.json"
    bearings.write_text(json.dumps({"sensors_m": [[0, 0]], "bearings_deg": [0]}))
    output = tmp_path / "analysis"
    assert invoke(monkeypatch, "locate", "--input", bearings, "--out", output) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["kind"] == "unbounded" and result["diameter_m"] is None
    assert (output / "figure.html").is_file()
    assert invoke(monkeypatch, "locate", "--input", bearings, "--out", output) == 2
    capsys.readouterr()

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"sensor_m": [0, 0], "bearing_deg": 25, "precision": "fast"}))
    target = tmp_path / "next-measurement"
    assert invoke(monkeypatch, "plan", "--input", plan, "--out", target) == 0
    data = json.loads((target / "result.json").read_text())
    assert data["recommended"]["reception_bound_m"] < 1000


def test_search_replay_and_report_commands(tmp_path, monkeypatch, capsys):
    assert invoke(monkeypatch, "run", "--environment", "omni", "--seed", 101, "--count", 3) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["status"] == "completed" and metrics["cleared"] == 3
    rid = metrics["run_id"]
    assert invoke(monkeypatch, "replay", "--run-id", rid) == 0
    logged = json.loads(capsys.readouterr().out)
    assert logged["event_count"] == metrics["actions"]
    assert logged["ledger"]["total_s"] == metrics["total_s"]
    assert invoke(monkeypatch, "report", "--run-id", rid) == 0
    assert rid in capsys.readouterr().out
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_workspace_can_move_without_a_machine_path(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "new-workspace"
    assert invoke(monkeypatch, "init", "--directory", workspace) == 0
    capsys.readouterr()
    marker = json.loads((workspace / "workspace.json").read_text())
    assert marker == {"schema_version": 1, "project_id": "RF-SearchLab", "offline": True}
    relocated = tmp_path / "relocated-workspace"
    workspace.rename(relocated)
    monkeypatch.setenv("RF_SEARCHLAB_HOME", str(relocated))
    assert invoke(monkeypatch, "doctor") == 0
    assert json.loads(capsys.readouterr().out)["writable"]


def test_real_comparison_and_workbench_exports(tmp_path):
    folder = benchmark("smoke", limit=1)
    with (folder / "cases.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4 and all(row["status"] == "completed" for row in rows)
    assert {row["strategy"] for row in rows} == {
        "coverage_route",
        "shared_search",
        "adaptive_tour",
        "adaptive_tour_no_pickup",
    }
    for row in rows:
        assert replay(row["run_id"])["status"] == "completed"

    app = AppTest.from_file(str(Path(rf_searchlab.__file__).with_name("ui.py")), default_timeout=30).run()
    assert not app.exception and app.dataframe
    app.sidebar.radio[0].set_value("基准比较与导出").run()
    assert not app.exception and len(app.dataframe) == 2
    app.checkbox[0].check().run()
    assert app.dataframe[1].value.empty

    app.sidebar.radio[0].set_value("回放与诊断").run()
    assert not app.exception
    rid = app.selectbox[0].value
    events = replay(rid)["events"]
    direction = next(i for i, event in enumerate(events) if event["response"]["result"] == "direction")
    app.slider[0].set_value(direction).run()
    app.checkbox[0].check().run()
    assert not app.exception and app.warning
    assert "真值视图" in app.warning[0].value
