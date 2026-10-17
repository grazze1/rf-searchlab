"""Run the local release gates without concealing a failed check."""

import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
work = root / ".work"
work.mkdir(exist_ok=True)
os.environ.update(RF_SEARCHLAB_HOME=str(root), TEMP=str(work), TMP=str(work), PYTHONDONTWRITEBYTECODE="1")
commands = [
    ["-m", "ruff", "check", "src", "tests", "scripts"],
    [
        "-m",
        "mypy",
        "--follow-imports=silent",
        "--disable-error-code=import-untyped",
        *map(str, sorted((root / "src/rf_searchlab").glob("*.py"))),
    ],
    [
        "-m",
        "pytest",
        "tests",
        "-q",
        "--basetemp=.work/check-tests",
        "--cov=src/rf_searchlab",
        "--cov-report=json:.work/coverage.json",
        "--cov-report=term",
        "--cov-fail-under=80",
    ],
]
results = []
for command in commands:
    result = subprocess.run([sys.executable, *command], check=False)
    results.append({"command": [Path(sys.executable).name, *command], "exit_code": result.returncode})

critical = [
    "geometry.py",
    "simulator.py",
    "evaluation.py",
    "algorithms/shared/state.py",
    "algorithms/route/certificate.py",
    "algorithms/directional/certificate.py",
    "algorithms/resilient/observation.py",
]
coverage_path = work / "coverage.json"
core = {}
if coverage_path.exists():
    report = json.loads(coverage_path.read_text())
    for suffix in critical:
        matches = [
            value["summary"]["percent_covered"]
            for name, value in report["files"].items()
            if name.replace("\\", "/").endswith("/rf_searchlab/" + suffix)
        ]
        core[suffix] = matches[0] if len(matches) == 1 else None
print("Critical coverage:", json.dumps(core, indent=2))
core_passed = len(core) == len(critical) and all(value is not None and value >= 90 for value in core.values())
output = {"checks": results, "critical_coverage": core, "critical_passed": core_passed}
(work / "checks.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
raise SystemExit(0 if all(row["exit_code"] == 0 for row in results) and core_passed else 1)
