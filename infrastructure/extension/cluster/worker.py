#!/usr/bin/env python
"""A build worker for a single cluster job: one thread per GPU executes queued section builds.

    python infrastructure/extension/cluster/worker.py --queue ~/projects/arena-extended/queue

Clients (see remote_request.py) write `<queue>/<id>.req` containing a section number and the
Slurm job id of the worker meant to run it. A free GPU thread claims a request addressed to its
own job (or to no job) by renaming it to `<id>.running`, runs `build.py <section> --execute`
with that GPU under the 30-minute lesson limit, writes the output to `<id>.log`, and finally
renames the request to `<id>.done` or `<id>.failed`. Two requests for the same section never
run at once.

A request for another job is never run: its worker is gone, and its caller has given up on it.
Builds left `.running` by an earlier worker become `.abandoned`; they are never rerun. The worker
exits after 30 minutes with no queued or active build, or once `<queue>/STOP` exists and its
running builds have finished.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_batch import (
    LESSON_TIME_LIMIT_S,
    TIMEOUT_STATUS,
    run_lesson,
    stop_builds_on_termination,
)

IDLE_EXIT_S = 1800


def parse_request(text: str) -> tuple[str, str | None]:
    """A request holds `<section> [<worker job id>]`."""
    section, *job = text.split()
    return section, job[0] if job else None


def queued_at(request: Path) -> float:
    """When a request was queued; one its caller just withdrew sorts last and is skipped."""
    try:
        return request.stat().st_mtime
    except FileNotFoundError:
        return float("inf")


class Worker:
    def __init__(self, queue: Path, job: str | None, idle_exit: float, poll: float) -> None:
        self.queue, self.job, self.idle_exit, self.poll = queue, job, idle_exit, poll
        self.lock = threading.Lock()
        self.running: set[str] = set()
        self.last_active = time.monotonic()
        self.stopped = threading.Event()

    def abandon_stale_builds(self) -> None:
        """Builds an earlier worker left running have lost their caller; never run them again."""
        for stale in self.queue.glob("*.running"):
            with stale.with_suffix(".log").open("a") as log:
                log.write(f"\nabandoned: its worker stopped mid-build; seen by job {self.job}\n")
            stale.rename(stale.with_suffix(".abandoned"))

    def claim(self) -> tuple[Path, str] | None:
        with self.lock:
            for req in sorted(self.queue.glob("*.req"), key=queued_at):
                try:
                    section, job = parse_request(req.read_text())
                except FileNotFoundError, ValueError:
                    continue
                if (job is not None and job != self.job) or section in self.running:
                    continue
                running = req.with_suffix(".running")
                try:
                    req.rename(running)
                except FileNotFoundError:
                    continue
                self.running.add(section)
                return running, section
        return None

    def idle_too_long(self) -> bool:
        with self.lock:
            if self.running:
                return False
            return time.monotonic() - self.last_active >= self.idle_exit

    def build(self, gpu: str, running: Path, section: str) -> None:
        log = running.with_suffix(".log")
        start = time.time()
        try:
            try:
                status = run_lesson(section, gpu=gpu, log=log)
            except Exception as exc:  # noqa: BLE001 -- any launch failure is that lesson's failure
                status = f"FAILED({type(exc).__name__})"
            minutes = (time.time() - start) / 60
            with log.open("a") as f:
                note = " (over the lesson time limit)" if status == TIMEOUT_STATUS else ""
                f.write(
                    f"\nsection {section} on GPU {gpu}: {status}{note} after {minutes:.1f} min\n"
                )
            try:
                running.rename(running.with_suffix(".done" if status == "ok" else ".failed"))
            except FileNotFoundError:
                status += ", but its request had been moved away (abandoned)"
            print(
                f"[{time.strftime('%H:%M:%S')}] {section} on GPU {gpu}: {status}, {minutes:.1f} min",
                flush=True,
            )
        finally:
            with self.lock:
                self.running.discard(section)
                self.last_active = time.monotonic()

    def gpu_loop(self, gpu: str) -> None:
        while not self.stopped.is_set():
            if (self.queue / "STOP").exists():
                self.stopped.set()
                break
            job = self.claim()
            if job is not None:
                self.build(gpu, *job)
                continue
            if self.idle_too_long():
                print(f"idle for {self.idle_exit:.0f}s with an empty queue; exiting", flush=True)
                self.stopped.set()
                break
            time.sleep(self.poll)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument(
        "--idle-exit",
        type=float,
        default=IDLE_EXIT_S,
        help=f"exit after this many seconds with no queued or active build (default: {IDLE_EXIT_S})",
    )
    parser.add_argument("--poll", type=float, default=5.0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    stop_builds_on_termination()
    args.queue.mkdir(parents=True, exist_ok=True)
    (args.queue / "STOP").unlink(missing_ok=True)
    worker = Worker(args.queue, os.environ.get("SLURM_JOB_ID"), args.idle_exit, args.poll)
    worker.abandon_stale_builds()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpus = [g.strip() for g in visible.split(",") if g.strip()]
    print(
        f"worker job {worker.job} on {os.uname().nodename}, GPUs {gpus}, queue {args.queue}, "
        f"lesson limit {LESSON_TIME_LIMIT_S}s",
        flush=True,
    )
    threads = [threading.Thread(target=worker.gpu_loop, args=(g,)) for g in gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("worker stopped", flush=True)


if __name__ == "__main__":
    main()
