#!/usr/bin/env python
"""Make Julia's environment match the laptop's `uv.lock` before a build is queued (login node, stdlib only).

    uv run --no-project --python 3.14 infrastructure/extension/cluster/env_sync.py \\
        --repo ~/projects/arena-extended/repo --state ~/projects/arena-extended/env-synced-lock.sha256 \\
        --queue ~/projects/arena-extended/queue --expected <laptop uv.lock sha256> \\
        --uv ~/projects/arena-extended/tools/uv --uv-version <laptop uv version>

remote_build.sh and submit_batch.sh run this over ssh after copying the sources. It compares the
laptop's lock digest with the digest of the lock that last synced successfully, recorded in the
state file. Only a successful `uv sync --locked` writes that file, so copying a new lock alone never
counts as a sync. A stale environment is synced on the login node, with no GPU allocation:

1. Julia's uv is brought to the laptop's uv version (`uv self update <version>`).
2. The Python minor in `.python-version` is installed at its latest patch.
3. `uv sync --locked`, without the optional `cuda-kernels` extra (it is laptop-only).

Any failure exits nonzero, so the caller queues or submits nothing. A sync is refused while a
build is running, since it would change packages under that build.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


class SyncError(RuntimeError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    if subprocess.run(command, cwd=cwd, check=False).returncode != 0:
        raise SyncError(f"`{' '.join(command)}` failed")


def uv_version(uv: str) -> str:
    out = subprocess.run([uv, "--version"], capture_output=True, text=True, check=True).stdout
    return out.split()[1]


def sync(repo: Path, state: Path, queue: Path, expected: str, uv: str, wanted_uv: str) -> str:
    """Bring Julia's environment to the lock with digest `expected`; return what was done."""
    copied = digest(repo / "uv.lock")
    if copied != expected:
        raise SyncError(
            f"Julia's uv.lock ({copied[:12]}) differs from the laptop's ({expected[:12]}); "
            "copying the sources did not bring the lock across"
        )
    synced = state.read_text().strip() if state.exists() else None
    if synced == expected and (repo / ".venv" / "bin" / "python").exists():
        return f"Julia's environment matches lock {expected[:12]}"
    running = sorted(path.stem for path in queue.glob("*.running"))
    if running:
        raise SyncError(
            f"builds are running on the worker ({', '.join(running)}); sync after they finish"
        )

    if uv_version(uv) != wanted_uv:
        run([uv, "self", "update", wanted_uv], cwd=repo)
    python = (repo / ".python-version").read_text().strip()
    run([uv, "python", "install", "--upgrade", python], cwd=repo)
    run([uv, "sync", "--locked", "--python", python], cwd=repo)

    partial = state.with_name(state.name + ".tmp")
    partial.write_text(expected + "\n")
    partial.replace(state)
    return f"synced Julia's environment to lock {expected[:12]} (was {(synced or 'never synced')[:12]})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", type=Path, required=True, help="the shared checkout on Julia")
    parser.add_argument("--state", type=Path, required=True, help="digest of the last synced lock")
    parser.add_argument("--queue", type=Path, required=True, help="the worker's request queue")
    parser.add_argument("--expected", required=True, help="sha256 of the laptop's uv.lock")
    parser.add_argument("--uv", default="uv", help="Julia's uv executable")
    parser.add_argument("--uv-version", required=True, help="the laptop's uv version")
    args = parser.parse_args()
    try:
        print(
            sync(args.repo, args.state, args.queue, args.expected, args.uv, args.uv_version),
            flush=True,
        )
    except SyncError as error:
        print(
            f"env sync failed: {error}; stopping before anything is queued or submitted",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
