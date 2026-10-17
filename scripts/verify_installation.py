"""Verify an installed wheel from a moved workspace containing no source code."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

parser = argparse.ArgumentParser()
parser.add_argument("--python", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
python = Path(args.python).resolve()
work = root / ".work"
workspace = work / f"install-test-{uuid4().hex[:8]}"
environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def command(*arguments, cwd=None, expected=0):
    result = subprocess.run(
        [str(python), "-X", "utf8", "-I", *arguments],
        cwd=cwd or root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != expected:
        raise RuntimeError(f"{arguments}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result.stdout


command("-m", "rf_searchlab.cli", "init", "--directory", str(workspace))
relocated = workspace.with_name(workspace.name + " moved 空间")
assert workspace.resolve().is_relative_to(work.resolve())
workspace.rename(relocated)
environment["RF_SEARCHLAB_HOME"] = str(relocated)
location = json.loads(
    command(
        "-c",
        'import json,rf_searchlab,sys;print(json.dumps({"module":rf_searchlab.__file__,"prefix":sys.prefix}))',
        cwd=relocated,
    )
)
assert Path(location["module"]).resolve().is_relative_to(Path(location["prefix"]).resolve())
assert not Path(location["module"]).resolve().is_relative_to((root / "src").resolve())
doctor = json.loads(command("-m", "rf_searchlab.cli", "doctor", cwd=relocated))
assert doctor["writable"] and doctor["offline"] and doctor["version"] == "1.0.0"
for name in ("bearings.json", "measurement_plan.json"):
    shutil.copyfile(root / "examples" / name, relocated / "examples" / name)
command(
    "-m",
    "rf_searchlab.cli",
    "locate",
    "--input",
    "examples/bearings.json",
    "--out",
    "runs/analysis",
    cwd=relocated,
)
command(
    "-m",
    "rf_searchlab.cli",
    "plan",
    "--input",
    "examples/measurement_plan.json",
    "--out",
    "runs/measurement",
    cwd=relocated,
)
metrics = []
for kind in ("omni", "mixed"):
    result = json.loads(
        command(
            "-m",
            "rf_searchlab.cli",
            "run",
            "--environment",
            kind,
            "--seed",
            "101",
            "--count",
            "3",
            cwd=relocated,
        )
    )
    assert result["status"] == "completed" and result["cleared"] == result["true_count"] == 3
    recorded = json.loads(
        command("-m", "rf_searchlab.cli", "replay", "--run-id", result["run_id"], cwd=relocated)
    )
    assert recorded["ledger"]["total_s"] == result["total_s"]
    assert result["run_id"] in command(
        "-m", "rf_searchlab.cli", "report", "--run-id", result["run_id"], cwd=relocated
    )
    metrics.append(result)

prepare = """
from rf_searchlab.application import Run
from rf_searchlab.simulator import Simulator, generate
run=Run(generate(101,count=3),'coverage_route')
original=Simulator.execute
counter=[0]
def interrupt(env,envelope):
    counter[0]+=1
    if counter[0]==3: raise KeyboardInterrupt()
    return original(env,envelope)
Simulator.execute=interrupt
try: run.execute()
except KeyboardInterrupt: print(run.id)
"""
rid = command("-c", prepare, cwd=relocated).strip()
resumed = json.loads(command("-m", "rf_searchlab.cli", "resume-run", "--run-id", rid, cwd=relocated))
assert resumed["status"] == "completed" and resumed["cleared"] == 3
output = {
    "schema_version": 1,
    "status": "PASS",
    "workspace": str(relocated.relative_to(root)),
    "version": doctor["version"],
    "package_from_environment": True,
    "localization": True,
    "measurement_planning": True,
    "search": metrics,
    "recovery": resumed,
}
(work / "installation-verification.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(
    json.dumps(
        {
            "status": "PASS",
            "version": doctor["version"],
            "runs": [row["run_id"] for row in metrics],
            "resumed_run": rid,
        },
        indent=2,
    )
)
