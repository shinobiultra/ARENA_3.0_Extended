#!/usr/bin/env python
"""A long-running build worker for a single cluster job: one thread per GPU executes queued section builds.

    python infrastructure/extension/cluster/worker.py --queue ~/projects/arena-extended/queue

Clients (see remote_build.sh) sync sources, then write `<queue>/<id>.req` containing a section number. A free
GPU thread claims it by renaming it to `<id>.running`, runs `build.py <section> --execute` with that GPU, writes
the output to `<id>.log`, and finally renames the request to `<id>.done` or `<id>.failed`. Two requests for the
same section never run at once. Create `<queue>/STOP` to make the worker exit once the running builds finish.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "infrastructure" / "extension" / "build.py"
running_sections: set[str] = set()
lock = threading.Lock()


def claim(queue: Path) -> tuple[Path, str] | None:
    with lock:
        for req in sorted(queue.glob("*.req"), key=lambda p: p.stat().st_mtime):
            section = req.read_text().strip()
            if section in running_sections:
                continue
            running = req.with_suffix(".running")
            try:
                req.rename(running)
            except FileNotFoundError:
                continue
            running_sections.add(section)
            return running, section
    return None


def gpu_loop(gpu: str, queue: Path, timeout: int) -> None:
    while not (queue / "STOP").exists():
        job = claim(queue)
        if job is None:
            time.sleep(5)
            continue
        running, section = job
        log = running.with_suffix(".log")
        start = time.time()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
        with log.open("w") as f:
            f.write(f"section {section} on GPU {gpu}, host {os.uname().nodename}\n")
            f.flush()
            proc = subprocess.run([sys.executable, str(BUILD), section, "--execute", "--timeout", str(timeout)],
                                  cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            f.write(f"\nexit {proc.returncode} after {(time.time() - start) / 60:.1f} min\n")
        running.rename(running.with_suffix(".done" if proc.returncode == 0 else ".failed"))
        with lock:
            running_sections.discard(section)
        print(f"[{time.strftime('%H:%M:%S')}] {section} on GPU {gpu}: exit {proc.returncode}, "
              f"{(time.time() - start) / 60:.1f} min", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
    args = parser.parse_args()
    args.queue.mkdir(parents=True, exist_ok=True)
    (args.queue / "STOP").unlink(missing_ok=True)
    for stale in args.queue.glob("*.running"):  # left over from a previous worker
        stale.rename(stale.with_suffix(".req"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpus = [g.strip() for g in visible.split(",") if g.strip()]
    print(f"worker on {os.uname().nodename}, GPUs {gpus}, queue {args.queue}", flush=True)
    threads = [threading.Thread(target=gpu_loop, args=(g, args.queue, args.timeout)) for g in gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("worker stopped", flush=True)


if __name__ == "__main__":
    main()
