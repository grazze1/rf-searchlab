"""Versioned files, atomic metadata and bounded input. No executable serialization."""

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path


def root():
    p = Path(os.environ.get("RF_SEARCHLAB_HOME", Path.cwd())).resolve()
    if not (p / "workspace.json").is_file():
        raise ValueError("Set RF_SEARCHLAB_HOME to the project directory")
    return p


def run_path(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Invalid run ID")
    base = (root() / "runs").resolve()
    p = base / run_id
    if not p.resolve().is_relative_to(base):
        raise ValueError("Run path escapes project")
    return p


def read_json(path, limit=4_000_000):
    p = Path(path)
    if p.stat().st_size > limit:
        raise ValueError("Input too large")

    def invalid(x):
        raise ValueError(f"Invalid JSON number: {x}")

    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError("Duplicate JSON key")
            d[k] = v
        return d

    return json.loads(p.read_text(encoding="utf-8-sig"), parse_constant=invalid, object_pairs_hook=pairs)


def atomic_json(path, data):
    path = Path(path)
    raw = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_event(path, data):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def exclusive_run(path):
    """An OS-released per-run lock; a crashed process cannot strand it."""
    lock_path = Path(path) / "run.lock"
    with lock_path.open("a+b") as file:
        file.seek(0, os.SEEK_END)
        if file.tell() == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Run is active in another process") from exc
            try:
                yield
            finally:
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Run is active in another process") from exc
            try:
                yield
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)
