"""The Julia build worker (`cluster/worker.py`) on CPU fixtures: no Slurm, GPU or model.

The worker runs as a real subprocess over a temporary queue; a fixture build.py stands in for
the lesson build, and `SLURM_JOB_ID` names the worker's job.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

CLUSTER = Path(__file__).resolve().parents[1] / "infrastructure/extension/cluster"
pytestmark = pytest.mark.cpu
JOB = "4242"

FIXTURE_BUILD = """\
import sys
print('fixture build', sys.argv[1:], flush=True)
if sys.argv[1] == 'bad':
    raise SystemExit(7)
if sys.argv[1] == 'overrun':
    print('TIMEOUT: overrun ran past its time limit', flush=True)
    raise SystemExit(124)
"""


def start_worker(tmp_path: Path, *, idle_exit: float = 3) -> tuple[subprocess.Popen, Path]:
    cluster = tmp_path / "infrastructure/extension/cluster"
    cluster.mkdir(parents=True, exist_ok=True)
    for name in ("worker.py", "run_batch.py"):
        shutil.copyfile(CLUSTER / name, cluster / name)
    (cluster.parent / "build.py").write_text(FIXTURE_BUILD)
    queue = tmp_path / "queue"
    queue.mkdir(exist_ok=True)
    proc = subprocess.Popen(
        [
            sys.executable,
            str(cluster / "worker.py"),
            "--queue",
            str(queue),
            "--idle-exit",
            str(idle_exit),
            "--poll",
            "0.1",
        ],
        env={**os.environ, "SLURM_JOB_ID": JOB, "CUDA_VISIBLE_DEVICES": "g0"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, queue


def request(queue: Path, name: str, section: str, job: str | None = JOB) -> None:
    (queue / f"{name}.req").write_text(f"{section} {job}\n" if job else f"{section}\n")


def wait_for(path: Path, seconds: float = 20) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def finish(proc: subprocess.Popen, seconds: float = 20) -> str:
    try:
        out, _ = proc.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        pytest.fail(f"worker did not exit:\n{out}")
    return out


def test_requests_finish_as_done_or_failed_with_the_lesson_limit(tmp_path):
    proc, queue = start_worker(tmp_path)
    request(queue, "a", "good")
    request(queue, "b", "bad")
    request(queue, "c", "overrun")
    for name, suffix in (("a", ".done"), ("b", ".failed"), ("c", ".failed")):
        assert wait_for(queue / f"{name}{suffix}"), finish(proc)
    finish(proc)
    assert "'--execute', '--time-limit', '1800'" in (queue / "a.log").read_text()
    assert "timeout" in (queue / "c.log").read_text().lower()
    assert "FAILED(7)" in (queue / "b.log").read_text()


def test_an_unaddressed_request_is_still_served(tmp_path):
    proc, queue = start_worker(tmp_path)
    request(queue, "manual", "good", job=None)
    assert wait_for(queue / "manual.done"), finish(proc)
    finish(proc)


def test_a_request_for_another_worker_job_is_never_run(tmp_path):
    proc, queue = start_worker(tmp_path, idle_exit=1.5)
    request(queue, "orphan", "good", job="9999")
    out = finish(proc)
    assert (queue / "orphan.req").exists(), out
    assert not (queue / "orphan.log").exists()


def test_a_build_left_running_by_a_lost_worker_is_abandoned_not_rerun(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "stale.running").write_text(f"good {JOB}\n")
    proc, queue = start_worker(tmp_path, idle_exit=1.5)
    out = finish(proc)
    assert (queue / "stale.abandoned").exists(), out
    assert not (queue / "stale.req").exists()
    assert not (queue / "stale.done").exists()
    assert "abandoned" in (queue / "stale.log").read_text()


def test_the_worker_exits_after_its_idle_time_with_an_empty_queue(tmp_path):
    start = time.monotonic()
    proc, _ = start_worker(tmp_path, idle_exit=1.5)
    out = finish(proc)
    assert proc.returncode == 0, out
    assert 1.5 <= time.monotonic() - start < 15
    assert "idle" in out


def test_the_idle_clock_restarts_after_each_build(tmp_path):
    proc, queue = start_worker(tmp_path, idle_exit=2)
    time.sleep(1.2)
    request(queue, "late", "good")
    assert wait_for(queue / "late.done", seconds=5), finish(proc)
    finished = time.monotonic()
    finish(proc)
    assert time.monotonic() - finished >= 1.5


def test_the_default_idle_exit_is_thirty_minutes():
    out = subprocess.run(
        [sys.executable, str(CLUSTER / "worker.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "1800" in out
