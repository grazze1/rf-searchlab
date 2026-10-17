from pathlib import Path

from streamlit.testing.v1 import AppTest

import rf_searchlab
from rf_searchlab.application import Run
from rf_searchlab.evaluation import replay
from rf_searchlab.simulator import generate


def test_chinese_ui_empty_state_and_localization():
    app = AppTest.from_file(str(Path(rf_searchlab.__file__).with_name("ui.py")), default_timeout=30).run()
    assert not app.exception
    assert "无线电源定位" in app.title[0].value
    app.sidebar.radio[0].set_value("定位分析").run()
    app.button[0].click().run()
    assert not app.exception and app.json
    app.radio[0].set_value("主动测点规划").run()
    app.button[0].click().run()
    assert not app.exception and app.json
    app.sidebar.radio[0].set_value("搜索任务").run()
    assert not app.exception and len(app.selectbox) >= 2
    app.sidebar.radio[0].set_value("回放与诊断").run()
    assert not app.exception and app.info


def test_replay_can_play_partial_crash_log_without_report():
    run = Run(generate(9), "shared_search", max_actions=3)
    assert run.execute()["status"] == "failed"
    run.status("running")  # Simulate a process ending before finalization.
    (run.path / "report.md").unlink()
    (run.path / "metrics.json").unlink()
    app = AppTest.from_file(str(Path(rf_searchlab.__file__).with_name("ui.py")), default_timeout=30).run()
    app.sidebar.radio[0].set_value("回放与诊断").run()
    assert not app.exception
    assert replay(run.id)["status"] == "partial"
    assert app.slider[0].value == len(run.events) - 1
    app.button[2].click().run()
    assert not app.exception
    assert app.session_state[f"playing-{run.id}"]
    assert app.slider[0].value == 1
    app.run()
    assert app.slider[0].value == len(run.events) - 1
    assert not app.exception
    app.sidebar.radio[0].set_value("基准比较与导出").run()
    assert not app.exception and app.info
