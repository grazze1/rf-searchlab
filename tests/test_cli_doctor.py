import json
import sys

from rf_searchlab.cli import main


def test_doctor_checks_offline_marker_and_local_write(tmp_path, monkeypatch, capsys):
    marker = {"project_id": "RF-SearchLab", "actual_root": str(tmp_path), "offline": True}
    (tmp_path / "workspace.json").write_text(json.dumps(marker))
    (tmp_path / ".work").mkdir()
    monkeypatch.setattr(sys, "argv", ["rf-searchlab", "doctor"])
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["offline"] is True and report["writable"] is True
    assert report["free_bytes"] >= 100_000_000
    assert not list((tmp_path / ".work").iterdir())

    marker["offline"] = False
    (tmp_path / "workspace.json").write_text(json.dumps(marker))
    assert main() == 2
    assert "Invalid offline project manifest" in capsys.readouterr().err
