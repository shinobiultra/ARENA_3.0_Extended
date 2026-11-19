#!/usr/bin/env python
"""Execute many extension sections in parallel, one per GPU, inside a single cluster job.

    python infrastructure/extension/cluster/run_batch.py 8.4 9.1 12.1 ... --out runs/<name>

Each GPU gets a worker that repeatedly takes the next section from the queue and runs
`build.py <section> --execute` with `CUDA_VISIBLE_DEVICES` set to that GPU. Logs go to
`<out>/<section>.log`, and `<out>/summary.tsv` records status and wall time per section.
Sections are started longest-first when `--order` gives expected minutes (e.g. `8.4=12`).

Every lesson gets the same 30-minute wall-clock limit (Spec #39, RT9). A lesson that runs past
it is stopped, its processes are killed and it is recorded as `FAILED(timeout)`; the batch
goes on with the other lessons. The CLI exits nonzero if any requested section fails, times
out or has no recorded result.
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "infrastructure" / "extension" / "build.py"
# The same limit as build.py's default; build.py enforces it and stops itself with TIMEOUT_EXIT_CODE.
LESSON_TIME_LIMIT_S = 1800
TIMEOUT_EXIT_CODE = 124
# Extra time before the runner kills a build that did not stop itself at its limit.
KILL_GRACE_S = 60
TIMEOUT_STATUS = "FAILED(timeout)"


def kill_process_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_lesson(
    section: str,
    *,
    gpu: str,
    log: Path,
    time_limit: int = LESSON_TIME_LIMIT_S,
    kill_grace: float = KILL_GRACE_S,
) -> str:
    """Run `build.py <section> --execute` on one GPU within the lesson limit; return its status.

    The status is `ok`, `FAILED(timeout)` or `FAILED(<exit code>)`. The build runs in its own
    process group, which is killed when the build ends, so nothing it started outlives it.
    Exceptions from launching the build are written to the log and re-raised.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
    command = [
        sys.executable,
        str(BUILD),
        section,
        "--execute",
        "--time-limit",
        str(time_limit),
    ]
    with log.open("w") as f:
        try:
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            traceback.print_exc(file=f)
            raise
        try:
            returncode = proc.wait(timeout=time_limit + kill_grace)
        except subprocess.TimeoutExpired:
            kill_process_group(proc.pid)
            proc.wait()
            f.write(f"\nTIMEOUT: killed after the {time_limit}s lesson time limit\n")
            return TIMEOUT_STATUS
        finally:
            kill_process_group(proc.pid)
    if returncode == 0:
        return "ok"
    if returncode == TIMEOUT_EXIT_CODE:
        return TIMEOUT_STATUS
    return f"FAILED({returncode})"


def n_gpus() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([v for v in visible.split(",") if v.strip()])
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    return max(1, len(out.stdout.strip().splitlines()))


def worker(
    gpu: str,
    todo: queue.Queue,
    out: Path,
    results: list,
    lock: threading.Lock,
    time_limit: int,
    kill_grace: float,
) -> None:
    while True:
        try:
            section = todo.get_nowait()
        except queue.Empty:
            return
        log = out / f"{section}.log"
        start = time.time()
        try:
            status = run_lesson(
                section, gpu=gpu, log=log, time_limit=time_limit, kill_grace=kill_grace
            )
        except Exception as exc:  # noqa: BLE001 -- any launch failure is that lesson's failure
            status = f"FAILED({type(exc).__name__})"
            print(f"{section}: {status}: {exc} (log: {log})", file=sys.stderr, flush=True)
        minutes = (time.time() - start) / 60
        with lock:
            with (out / "summary.tsv").open("a") as f:
                f.write(f"{section}\t{status}\t{minutes:.1f}\tgpu{gpu}\n")
            results.append((section, status, minutes, gpu))
            print(
                f"[{time.strftime('%H:%M:%S')}] {section}: {status} in {minutes:.1f} min on GPU {gpu}",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sections", nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--order", nargs="*", default=[], help="expected minutes per section, e.g. 8.4=12"
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=LESSON_TIME_LIMIT_S,
        help="wall-clock seconds for each lesson (default: 30 minutes)",
    )
    parser.add_argument("--kill-grace", type=float, default=KILL_GRACE_S, help=argparse.SUPPRESS)
    args = parser.parse_args()

    expected = {k: float(v) for k, v in (item.split("=") for item in args.order)}
    sections = sorted(args.sections, key=lambda s: -expected.get(s, 5.0))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.tsv").write_text("section\tstatus\tminutes\tgpu\n")

    todo: queue.Queue = queue.Queue()
    for s in sections:
        todo.put(s)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = [g.strip() for g in visible.split(",")] if visible else [str(i) for i in range(n_gpus())]
    print(f"{len(sections)} sections on {len(gpus)} GPU(s): {sections}", flush=True)

    results: list = []
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(g, todo, args.out, results, lock, args.time_limit, args.kill_grace),
        )
        for g in gpus
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    missing = Counter(sections) - Counter(s for s, _, _, _ in results)
    for section in missing.elements():
        status = "FAILED(missing result)"
        with (args.out / "summary.tsv").open("a") as f:
            f.write(f"{section}\t{status}\t-\t-\n")
        results.append((section, status, 0.0, "-"))
        print(f"{section}: {status}; worker did not record completion", file=sys.stderr, flush=True)
    failed = [s for s, status, _, _ in results if status != "ok"]
    print(f"Done: {len(results) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
