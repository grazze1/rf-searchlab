"""Build a source ZIP and Python distributions from a reviewed allowlist."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
work = root / ".work"
work.mkdir(exist_ok=True)
os.environ.update(TEMP=str(work), TMP=str(work))
metadata = tomllib.loads((root / "pyproject.toml").read_text())["project"]
version = metadata["version"]
subprocess.run([sys.executable, "scripts/dependency_inventory.py"], check=True)
subprocess.run([sys.executable, "-m", "build", "--no-isolation"], check=True)
dist = root / "dist"
prefix = f"rf-searchlab-{version}"
top_files = [
    "README.md",
    "LICENSE",
    "NOTICE",
    "CHANGELOG.md",
    "pyproject.toml",
    "MANIFEST.in",
    "requirements-lock.txt",
    "workspace.json",
    ".gitignore",
    "Start.cmd",
]
directories = ["src", "tests", "examples", "docs", "scripts", ".streamlit", ".github"]
files = [root / name for name in top_files]
for directory in directories:
    if (root / directory).exists():
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and not any(part == "__pycache__" or part.endswith(".egg-info") for part in path.parts)
            and path.suffix not in (".pyc", ".pyo")
        )
bundle = dist / f"{prefix}-source.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Release path escapes source tree")
        archive.write(path, f"{prefix}/{path.relative_to(root).as_posix()}")
    for name in (".work", "runs", "benchmarks"):
        archive.writestr(f"{prefix}/{name}/", "")

audit = []
for path in sorted(dist.iterdir()):
    if not (path.name.endswith((".whl", ".tar.gz", "-source.zip"))):
        continue
    if path.suffix in (".whl", ".zip"):
        with zipfile.ZipFile(path) as archive:
            entries = [(n, archive.read(n)) for n in archive.namelist() if not n.endswith("/")]
    else:
        with tarfile.open(path) as archive:
            entries = [
                (member.name, archive.extractfile(member).read())
                for member in archive.getmembers()
                if member.isfile()
            ]
    findings = []
    for name, data in entries:
        if any(part in name.split("/") for part in (".venv", ".work", "__pycache__", "runs", "benchmarks")):
            findings.append(name)
        if re.search(rb"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?!/)", data) or re.search(
            rb"(?i)\bq[1-4](?:ml|solver|r\d+)?\b", data
        ):
            findings.append(name)
    audit.append(
        {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "entries": len(entries),
            "findings": sorted(set(findings)),
        }
    )
if any(row["findings"] for row in audit):
    raise ValueError(json.dumps(audit, indent=2))
(dist / "SHA256SUMS.txt").write_text(
    "".join(f"{row['sha256']}  {row['file']}\n" for row in audit), encoding="utf-8"
)
(work / "package-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
