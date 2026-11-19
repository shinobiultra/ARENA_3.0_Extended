"""The two Julia entry points, remote_build.sh and submit_batch.sh, end to end on CPU fixtures.

They run from a small copy of the repo against a "remote" home in a temporary directory. A fake
ssh runs remote commands locally (rsync goes through it too), and fake uv, squeue and sbatch stand
in for the cluster's tools. The test plays the worker by renaming request files like worker.py.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CLUSTER = "infrastructure/extension/cluster"
SECTION, CHAPTER, EXDIR = "8.4", "chapter8_automated_circuits", "part4_acdc_circuit_metrics"
JOB = "777"
pytestmark = pytest.mark.cpu

FAKES = {
    "ssh": """\
#!/bin/bash
while [ $# -gt 0 ]; do case $1 in -o|-p|-l|-i) shift 2 ;; -*) shift ;; *) break ;; esac; done
shift
export HOME="$FAKE_REMOTE_HOME"
cd "$HOME"
exec bash -c "$*"
""",
    "uv": """\
#!/bin/bash
echo "uv $*" >> "$FAKE_CALLS"
case $1 in
  --version) echo "uv 0.13.9" ;;
  run) shift; while [ $# -gt 0 ]; do case $1 in --no-project) shift ;; --python) shift 2 ;; *) break ;; esac; done
       exec "$FAKE_PYTHON" "$@" ;;
  sync) [ -z "$FAKE_UV_SYNC_FAIL" ] || { echo "error: the lock is out of date" >&2; exit 2; }
        mkdir -p .venv/bin && ln -sf "$FAKE_PYTHON" .venv/bin/python ;;
esac
""",
    "squeue": """\
#!/bin/sh
state=$(cat "$FAKE_SLURM_STATE" 2>/dev/null)
case "$*" in
  *-j*) [ -n "$state" ] || { echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1; }
        echo "$state" ;;
  *) [ -z "$state" ] || echo "$FAKE_SLURM_JOB $state" ;;
esac
""",
    "sbatch": '#!/bin/sh\necho "sbatch $*" >> "$FAKE_CALLS"\necho "Submitted batch job 4343"\n',
}


class Setup:
    def __init__(self, tmp_path: Path) -> None:
        self.local = tmp_path / "laptop"
        self.home = tmp_path / "julia_home"
        self.base = self.home / "projects/arena-extended"
        self.remote_repo = self.base / "repo"
        self.queue = self.base / "queue"
        for path in (self.remote_repo, self.queue, self.base / "runs", self.base / "tools"):
            path.mkdir(parents=True)
        shutil.copytree(REPO / CLUSTER, self.local / CLUSTER)
        # Model downloads are not part of these tests.
        (self.local / CLUSTER / "prefetch.py").write_text(
            "import sys\nprint('prefetched', sys.argv[1:])\n"
        )
        (self.local / "infrastructure/core").mkdir(parents=True)
        shutil.copy(REPO / "infrastructure/core/config.yaml", self.local / "infrastructure/core")
        (self.local / "uv.lock").write_text("version = 1\n")
        (self.local / ".python-version").write_text("3.14\n")
        (self.local / ".venv").symlink_to(REPO / ".venv")
        (self.local / CHAPTER / "exercises" / EXDIR).mkdir(parents=True)

        self.calls = tmp_path / "calls.log"
        self.calls.touch()
        self.state = tmp_path / "slurm_state"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name, text in FAKES.items():
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        shutil.copy(bin_dir / "uv", self.base / "tools/uv")
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "ARENA_JULIA_HOST": "julia.test",
            "ARENA_JULIA_SSH": str(bin_dir / "ssh"),
            "FAKE_REMOTE_HOME": str(self.home),
            "FAKE_PYTHON": sys.executable,
            "FAKE_CALLS": str(self.calls),
            "FAKE_SLURM_STATE": str(self.state),
            "FAKE_SLURM_JOB": JOB,
        }

    def worker(self, state: str | None) -> None:
        self.state.write_text(state or "")

    def run(self, script: str, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.local / CLUSTER / script), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def remote_notebook(self) -> Path:
        path = self.remote_repo / CHAPTER / "exercises" / EXDIR / "8.4_solutions.ipynb"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def local_notebook(self) -> Path:
        return self.local / CHAPTER / "exercises" / EXDIR / "8.4_solutions.ipynb"

    def play_worker(self, outcome: str | None) -> None:
        deadline = time.monotonic() + 60
        while not (found := list(self.queue.glob("*.req"))):
            if time.monotonic() > deadline:
                return
            time.sleep(0.05)
        running = found[0].with_suffix(".running")
        found[0].rename(running)
        self.remote_notebook().write_text('{"executed": true}')
        running.with_suffix(".log").write_text("built 8.4 on the worker\n")
        if outcome is None:
            self.worker(None)  # the worker job disappears mid-build
        else:
            running.rename(running.with_suffix(outcome))

    def sbatch_calls(self) -> list[str]:
        return [line for line in self.calls.read_text().splitlines() if line.startswith("sbatch")]

    def uv_syncs(self) -> list[str]:
        return [line for line in self.calls.read_text().splitlines() if line.startswith("uv sync")]


@pytest.fixture
def julia(tmp_path):
    return Setup(tmp_path)


def test_a_remote_build_syncs_queues_and_pulls_back_its_notebooks(julia):
    julia.worker("RUNNING")
    threading.Thread(target=julia.play_worker, args=(".done",)).start()
    result = julia.run("remote_build.sh", SECTION)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (julia.remote_repo / CLUSTER / "worker.py").exists(), "sources were not synced"
    assert julia.uv_syncs() == ["uv sync --locked --python 3.14"]
    assert (julia.base / "env-synced-lock.sha256").exists()
    assert "built 8.4 on the worker" in result.stdout
    assert julia.local_notebook().read_text() == '{"executed": true}'


def test_a_failed_environment_sync_queues_nothing(julia):
    julia.worker("RUNNING")
    result = julia.run("remote_build.sh", SECTION, FAKE_UV_SYNC_FAIL="1")
    assert result.returncode != 0
    assert "env sync failed" in result.stderr
    assert list(julia.queue.iterdir()) == []
    assert not julia.local_notebook().exists()


def test_a_lost_worker_fails_the_build_and_pulls_nothing_back(julia):
    julia.worker("RUNNING")
    threading.Thread(target=julia.play_worker, args=(None,)).start()
    result = julia.run("remote_build.sh", SECTION)
    assert result.returncode == 2, result.stdout + result.stderr
    assert f"worker job {JOB} is gone" in result.stdout
    assert "nothing was pulled back" in result.stdout
    assert not julia.local_notebook().exists()
    assert list(julia.queue.glob("*.abandoned"))


def test_a_batch_submission_syncs_first_at_normal_qos(julia):
    result = julia.run("submit_batch.sh", "--gpus", "2", "--time", "04:00:00", "8.4", "9.1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert julia.uv_syncs() == ["uv sync --locked --python 3.14"]
    (call,) = julia.sbatch_calls()
    assert "--qos=normal" in call
    assert call.endswith("batch.sbatch 8.4 9.1")


def test_a_failed_environment_sync_submits_no_batch(julia):
    result = julia.run("submit_batch.sh", "--time", "04:00:00", "8.4", FAKE_UV_SYNC_FAIL="1")
    assert result.returncode != 0
    assert julia.sbatch_calls() == []
