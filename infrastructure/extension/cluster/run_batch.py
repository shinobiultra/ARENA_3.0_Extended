#!/usr/bin/env python
"""Execute many extension sections in parallel, one per GPU, inside a single cluster job.

    python infrastructure/extension/cluster/run_batch.py 8.4 9.1 12.1 ... --out runs/<name>

Each GPU gets a worker that repeatedly takes the next section from the queue and runs
`build.py <section> --execute` with `CUDA_VISIBLE_DEVICES` set to that GPU. Logs go to
`<out>/<section>.log`, and `<out>/summary.tsv` records status and wall time per section.
Sections are started longest-first when `--order` gives expected minutes (e.g. `8.4=12`).
The CLI exits nonzero if any requested section fails or has no recorded result.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "infrastructure" / "extension" / "build.py"


def n_gpus() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([v for v in visible.split(",") if v.strip()])
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    return max(1, len(out.stdout.strip().splitlines()))


def worker(gpu: str, todo: queue.Queue, out: Path, results: list, lock: threading.Lock, timeout: int) -> None:
    while True:
        try:
            section = todo.get_nowait()
        except queue.Empty:
            return
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
        log = out / f"{section}.log"
        start = time.time()
        try:
            with log.open("w") as f:
                try:
                    proc = subprocess.run(
                        [sys.executable, str(BUILD), section, "--execute", "--timeout", str(timeout)],
                        cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                    )
                except Exception:
                    traceback.print_exc(file=f)
                    raise
            status = "ok" if proc.returncode == 0 else f"FAILED({proc.returncode})"
        except Exception as exc:
            status = f"FAILED({type(exc).__name__})"
            print(f"{section}: {status}: {exc} (log: {log})", file=sys.stderr, flush=True)
        minutes = (time.time() - start) / 60
        with lock:
            with (out / "summary.tsv").open("a") as f:
                f.write(f"{section}\t{status}\t{minutes:.1f}\tgpu{gpu}\n")
            results.append((section, status, minutes, gpu))
            print(f"[{time.strftime('%H:%M:%S')}] {section}: {status} in {minutes:.1f} min on GPU {gpu}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sections", nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--order", nargs="*", default=[], help="expected minutes per section, e.g. 8.4=12")
    parser.add_argument("--timeout", type=int, default=3600, help="per-cell timeout in seconds")
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
    threads = [threading.Thread(target=worker, args=(g, todo, args.out, results, lock, args.timeout)) for g in gpus]
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
