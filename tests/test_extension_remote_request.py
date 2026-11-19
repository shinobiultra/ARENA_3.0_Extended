"""The login-node side of a remote build (`cluster/remote_request.py`) on CPU fixtures.

A fake `squeue` on PATH reports the worker job's state from a file; the test plays the worker
by renaming the request files the way worker.py does.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "infrastructure/extension/cluster/remote_request.py"
pytestmark = pytest.mark.cpu
JOB = "777"

FAKE_SQUEUE = """\
#!/bin/sh
state=$(cat "$FAKE_SLURM_STATE" 2>/dev/null)
if [ -e "$FAKE_SLURM_STATE.error" ]; then echo "slurm_load_jobs error: Socket timed out" >&2; exit 1; fi
case "$*" in
  *-j*)
    if [ -z "$state" ]; then echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1; fi
    echo "$state" ;;
  *)
    if [ -n "$state" ]; then echo "$FAKE_SLURM_JOB $state"; fi ;;
esac
"""


class Cluster:
    def __init__(self, tmp_path: Path) -> None:
        self.base = tmp_path / "arena-extended"
        self.queue = self.base / "queue"
        self.runs = self.base / "runs"
        self.queue.mkdir(parents=True)
        self.runs.mkdir()
        self.state = tmp_path / "slurm_state"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        squeue = bin_dir / "squeue"
        squeue.write_text(FAKE_SQUEUE)
        squeue.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_SLURM_STATE": str(self.state),
            "FAKE_SLURM_JOB": JOB,
        }

    def worker(self, state: str | None) -> None:
        self.state.write_text(state or "")

    def start(self, section: str = "8.4", **env: str) -> subprocess.Popen:
        return subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "run",
                section,
                "--queue",
                str(self.queue),
                "--poll",
                "0.1",
            ],
            env={**self.env, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def wait_for_request(self, suffix: str = ".req", seconds: float = 10) -> Path:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            found = list(self.queue.glob(f"*{suffix}"))
            if found:
                return found[0]
            time.sleep(0.05)
        raise AssertionError(f"no {suffix} request appeared")


def finish(proc: subprocess.Popen, seconds: float = 20) -> str:
    try:
        out, _ = proc.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        pytest.fail(f"the caller kept waiting:\n{out}")
    return out


def play_worker(cluster: Cluster, outcome: str, log: str) -> None:
    req = cluster.wait_for_request()
    running = req.with_suffix(".running")
    req.rename(running)
    running.with_suffix(".log").write_text(log)
    running.rename(running.with_suffix(outcome))


@pytest.fixture
def cluster(tmp_path):
    return Cluster(tmp_path)


def test_a_request_is_addressed_to_the_running_worker_and_succeeds(cluster):
    cluster.worker("RUNNING")
    proc = cluster.start("8.4")
    req = cluster.wait_for_request()
    assert req.read_text().split() == ["8.4", JOB]
    threading.Thread(target=play_worker, args=(cluster, ".done", "built 8.4\n")).start()
    out = finish(proc)
    assert proc.returncode == 0, out
    assert "built 8.4" in out


def test_a_failed_build_exits_nonzero_with_its_log(cluster):
    cluster.worker("RUNNING")
    proc = cluster.start()
    threading.Thread(target=play_worker, args=(cluster, ".failed", "Traceback: boom\n")).start()
    out = finish(proc)
    assert proc.returncode == 1, out
    assert "boom" in out


def test_no_worker_job_fails_at_once_without_queueing(cluster):
    cluster.worker(None)
    proc = cluster.start()
    out = finish(proc, seconds=10)
    assert proc.returncode == 3, out
    assert "no arena-worker job" in out
    assert list(cluster.queue.iterdir()) == []


def test_a_pending_worker_is_waited_for(cluster):
    cluster.worker("PENDING")
    proc = cluster.start()
    cluster.wait_for_request()
    time.sleep(0.5)
    assert proc.poll() is None
    cluster.worker("RUNNING")
    threading.Thread(target=play_worker, args=(cluster, ".done", "ok\n")).start()
    out = finish(proc)
    assert proc.returncode == 0, out


def test_worker_lost_during_a_build_fails_promptly_and_abandons_the_request(cluster):
    cluster.worker("RUNNING")
    (cluster.runs / f"worker-{JOB}.out").write_text(
        "slurmstepd: error: *** JOB 777 CANCELLED ***\n"
    )
    proc = cluster.start()
    req = cluster.wait_for_request()
    running = req.with_suffix(".running")
    req.rename(running)
    running.with_suffix(".log").write_text("halfway through 8.4\n")
    cluster.worker(None)
    out = finish(proc, seconds=10)
    assert proc.returncode == 2, out
    assert f"worker job {JOB} is gone" in out
    assert "halfway through 8.4" in out
    assert "CANCELLED" in out
    assert running.with_suffix(".abandoned").exists()
    assert not running.exists()


def test_worker_lost_before_claiming_abandons_the_queued_request(cluster):
    cluster.worker("RUNNING")
    proc = cluster.start()
    req = cluster.wait_for_request()
    cluster.worker(None)
    out = finish(proc, seconds=10)
    assert proc.returncode == 2, out
    assert req.with_suffix(".abandoned").exists()
    assert not req.exists()


def test_a_completion_that_races_worker_exit_keeps_its_outcome(cluster):
    cluster.worker("RUNNING")
    proc = cluster.start()
    play_worker(cluster, ".done", "finished just before exit\n")
    cluster.worker(None)
    out = finish(proc)
    assert proc.returncode == 0, out
    assert "finished just before exit" in out
    assert not list(cluster.queue.glob("*.abandoned"))


def test_a_slurm_hiccup_is_not_mistaken_for_a_lost_worker(cluster):
    cluster.worker("RUNNING")
    proc = cluster.start()
    cluster.wait_for_request()
    hiccup = Path(f"{cluster.state}.error")
    hiccup.touch()
    time.sleep(0.6)
    assert proc.poll() is None, "the caller gave up while Slurm was briefly unreachable"
    hiccup.unlink()
    threading.Thread(target=play_worker, args=(cluster, ".done", "ok\n")).start()
    out = finish(proc)
    assert proc.returncode == 0, out
