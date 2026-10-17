"""Local commands. No online backend is accepted."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from rf_searchlab import __version__
from rf_searchlab.storage import atomic_json, read_json, root, run_path


def main():
    parser = argparse.ArgumentParser(prog="rf-searchlab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("ui")
    p.add_argument("--port", type=int, default=8531)
    p = sub.add_parser("init")
    p.add_argument("--directory", required=True)
    for command in ["locate", "plan"]:
        p = sub.add_parser(command)
        p.add_argument("--input", required=True)
        p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--environment", choices=["omni", "mixed"], default="omni")
    p.add_argument("--strategy")
    p.add_argument("--scenario")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--count", type=int, default=10)
    p.add_argument(
        "--family", choices=["uniform", "cluster", "boundary", "radius1000", "outward"], default="uniform"
    )
    p.add_argument("--max-actions", type=int, default=1500)
    p.add_argument("--wall-seconds", type=float, default=120)
    for name in ["replay", "report", "resume-run"]:
        p = sub.add_parser(name)
        p.add_argument("--run-id", required=True)
    p = sub.add_parser("benchmark")
    p.add_argument("--suite", choices=["smoke", "regression", "stress"], default="smoke")
    p.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        if args.command == "init":
            home = Path(args.directory).resolve()
            if home.exists() and any(home.iterdir()):
                raise ValueError("Initialization requires a new or empty directory")
            home.mkdir(parents=True, exist_ok=True)
            for name in ["runs", "benchmarks", "examples", ".work"]:
                (home / name).mkdir()
            atomic_json(
                home / "workspace.json",
                {"schema_version": 1, "project_id": "RF-SearchLab", "offline": True},
            )
            print(home)
            return 0
        home = root()
        if args.command == "doctor":
            import numpy
            import plotly
            import scipy
            import streamlit

            marker = read_json(home / "workspace.json")
            if not isinstance(marker, dict):
                raise ValueError("Invalid project manifest shape")
            if marker.get("project_id") != "RF-SearchLab" or marker.get("offline") is not True:
                raise ValueError("Invalid offline project manifest")
            free_bytes = shutil.disk_usage(home).free
            if free_bytes < 100_000_000:
                raise OSError("Less than 100 MB free in project filesystem")
            work = home / ".work"
            if not work.is_dir():
                raise ValueError("Missing project work directory")
            with tempfile.NamedTemporaryFile(dir=work) as probe:
                probe.write(b"RF-SearchLab local write probe")
                probe.flush()
            print(
                json.dumps(
                    {
                        "version": __version__,
                        "root": str(home),
                        "python": sys.version,
                        "free_bytes": free_bytes,
                        "writable": True,
                        "offline": True,
                        "dependencies": {
                            m.__name__: m.__version__ for m in [numpy, scipy, streamlit, plotly]
                        },
                    },
                    indent=2,
                )
            )
        elif args.command in ("locate", "plan"):
            from rf_searchlab.localization import figure, locate, plan_measurement

            out = Path(args.out).resolve()
            if not out.is_relative_to(home):
                raise ValueError("Output must stay inside project")
            out.mkdir(parents=True, exist_ok=False)
            result = (locate if args.command == "locate" else plan_measurement)(read_json(args.input))
            atomic_json(out / "result.json", result)
            figure(result).write_html(out / "figure.html", include_plotlyjs=True)
            print(out)
        elif args.command == "run":
            from rf_searchlab.application import Run
            from rf_searchlab.simulator import from_dict, generate

            truth = (
                from_dict(read_json(args.scenario))
                if args.scenario
                else generate(args.seed, args.environment, args.count, args.family)
            )
            if truth.environment != args.environment:
                raise ValueError("Scenario/environment mismatch")
            strategy = args.strategy or ("coverage_route" if args.environment == "omni" else "adaptive_tour")
            run = Run(truth, strategy, args.max_actions, args.wall_seconds)
            metrics = run.execute()
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0 if metrics["status"] == "completed" else 1
        elif args.command == "replay":
            from rf_searchlab.evaluation import replay

            result = replay(args.run_id)
            result["event_count"] = len(result.pop("events"))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result["status"] == "partial" else 0
        elif args.command == "resume-run":
            from rf_searchlab.application import Run

            metrics = Run.restore(args.run_id).execute()
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0 if metrics["status"] == "completed" else 1
        elif args.command == "report":
            print((run_path(args.run_id) / "report.md").read_text(encoding="utf-8"))
        elif args.command == "benchmark":
            from rf_searchlab.benchmark import benchmark

            print(benchmark(args.suite, args.limit))
        elif args.command == "ui":
            if not 1024 <= args.port <= 65535:
                raise ValueError("UI port must be 1024..65535")
            env = dict(os.environ, RF_SEARCHLAB_HOME=str(home), STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(Path(__file__).with_name("ui.py")),
                    "--server.address=127.0.0.1",
                    f"--server.port={args.port}",
                    "--browser.gatherUsageStats=false",
                ],
                env=env,
                cwd=home,
            )
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
