#!/usr/bin/env python
"""Queue one section build for the Julia worker job and wait for its outcome (login node, stdlib only).

    python infrastructure/extension/cluster/remote_request.py run 8.4 --queue ~/projects/arena-extended/queue

remote_build.sh calls this over ssh. It finds the running (or pending) `arena-worker` job,
writes `<queue>/<id>.req` addressed to that job, and waits until worker.py renames the request
to `.done` or `.failed`, then prints the end of the build log.

While waiting it checks that the worker job still exists. If the job is gone without an outcome,
the request becomes `.abandoned` (so a later worker never runs it) and the caller fails at once
with what diagnostics there are. An outcome written just before the worker exits still counts.

Exit status: 0 build done, 1 build failed, 2 worker gone, 3 no worker job to address.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKER_NAME = "arena-worker"
# Slurm states in which the worker job may still run builds.
ALIVE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "SUSPENDED", "REQUEUED", "RESIZING"}
DONE, FAILED, WORKER_GONE, NO_WORKER = 0, 1, 2, 3
LOG_NOISE = re.compile(r"W1008|Warning|warn\(")


def squeue(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["squeue", *args], capture_output=True, text=True, check=False)


def find_worker() -> tuple[str, str] | None:
    """The job id and state of this user's worker job; a running one wins over a pending one."""
    result = squeue("--me", f"--name={WORKER_NAME}", "-h", "-o", "%i %T")
    if result.returncode != 0:
        print(f"squeue failed: {result.stderr.strip()}", flush=True)
        return None
    jobs = [line.split() for line in result.stdout.splitlines() if line.strip()]
    jobs = [(job, state) for job, state in jobs if state in ALIVE_STATES]
    jobs.sort(key=lambda job_state: job_state[1] != "RUNNING")
    return jobs[0] if jobs else None


def worker_alive(job: str) -> bool | None:
    """True or False when Slurm answers, None when it cannot be asked right now."""
    result = squeue("-h", "-j", job, "-o", "%T")
    if result.returncode != 0:
        return False if "Invalid job id" in result.stderr else None
    states = result.stdout.split()
    return bool(states) and states[0] in ALIVE_STATES


def outcome(queue: Path, request_id: str) -> int | None:
    if (queue / f"{request_id}.done").exists():
        return DONE
    if (queue / f"{request_id}.failed").exists():
        return FAILED
    if (queue / f"{request_id}.abandoned").exists():  # e.g. by a replacement worker
        return WORKER_GONE
    return None


def abandon(queue: Path, request_id: str) -> None:
    for suffix in (".req", ".running"):
        try:
            (queue / f"{request_id}{suffix}").rename(queue / f"{request_id}.abandoned")
            return
        except FileNotFoundError:
            continue


def tail(path: Path, lines: int = 25) -> str:
    if not path.exists():
        return f"({path.name} does not exist)"
    kept = [
        line for line in path.read_text(errors="replace").splitlines() if not LOG_NOISE.search(line)
    ]
    return "\n".join(kept[-lines:])


def sacct_state(job: str) -> str:
    try:
        result = subprocess.run(
            ["sacct", "-j", job, "-X", "-n", "-o", "State,ExitCode,Elapsed"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except OSError, subprocess.TimeoutExpired:
        return "unknown (sacct unavailable)"
    return " ".join(result.stdout.split()) or "unknown"


def report_lost_worker(queue: Path, runs: Path, request_id: str, job: str) -> None:
    print(f"worker job {job} is gone before {request_id} finished; sacct: {sacct_state(job)}")
    print(f"--- build log {request_id}.log ---")
    print(tail(queue / f"{request_id}.log"))
    print(f"--- worker output worker-{job}.out ---")
    print(tail(runs / f"worker-{job}.out", lines=15), flush=True)


def run(section: str, queue: Path, runs: Path, poll: float) -> int:
    worker = find_worker()
    if worker is None:
        print(
            f"no {WORKER_NAME} job is running or pending; submit one first with "
            "`sbatch --qos=normal infrastructure/extension/cluster/worker.sbatch`",
            flush=True,
        )
        return NO_WORKER
    job, state = worker
    queue.mkdir(parents=True, exist_ok=True)
    request_id = f"{section}-{int(time.time())}-{os.getpid()}"
    request = queue / f"{request_id}.req"
    # Write under another name first, so the worker never reads a half-written request.
    partial = queue / f"{request_id}.partial"
    partial.write_text(f"{section} {job}\n")
    partial.rename(request)
    print(f"queued {request_id} for worker job {job} ({state})", flush=True)

    finished = False
    warned = False
    try:
        while True:
            result = outcome(queue, request_id)
            alive = None if result is not None else worker_alive(job)
            if alive is False:
                result = outcome(
                    queue, request_id
                )  # it may have finished just before the worker exited
                if result is None:
                    abandon(queue, request_id)
                    finished = True
                    report_lost_worker(queue, runs, request_id, job)
                    return WORKER_GONE
            if result == WORKER_GONE:
                finished = True
                print(f"{request_id} was abandoned before it finished (a new worker started?)")
                print(tail(queue / f"{request_id}.log"), flush=True)
                return result
            if result is not None:
                finished = True
                word = "done" if result == DONE else "failed"
                print(f"build {word}: {section} ({request_id})")
                print(tail(queue / f"{request_id}.log"), flush=True)
                return result
            if alive is None and not warned:
                print("squeue did not answer; still waiting", flush=True)
                warned = True
            time.sleep(poll)
    finally:
        if not finished:
            # The caller went away (Ctrl-C or a dropped ssh): never leave a queued build for a later worker.
            try:
                request.rename(request.with_suffix(".abandoned"))
            except FileNotFoundError:
                pass


def hang_up(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="queue a section build and wait for its outcome")
    run_parser.add_argument("section")
    run_parser.add_argument("--queue", type=Path, required=True)
    run_parser.add_argument(
        "--runs", type=Path, help="where worker-<job>.out is written (default: <queue>/../runs)"
    )
    run_parser.add_argument("--poll", type=float, default=10.0, help="seconds between checks")
    args = parser.parse_args()
    for signum in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(signum, hang_up)
    runs = args.runs or args.queue.parent / "runs"
    return run(args.section, args.queue, runs, args.poll)


if __name__ == "__main__":
    sys.exit(main())
