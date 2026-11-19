"""Julia's locked-environment sync (`cluster/env_sync.py`) on CPU fixtures.

A fake `uv` records its calls; fake `srun`, `salloc` and `sbatch` record any attempt to use a
GPU allocation.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "infrastructure/extension/cluster/env_sync.py"
pytestmark = pytest.mark.cpu
UV_VERSION = "0.13.9"

FAKE_UV = """\
#!/bin/sh
echo "uv $*" >> "$FAKE_CALLS"
case "$1" in
  --version) echo "uv ${FAKE_UV_VERSION} (x86_64-unknown-linux-gnu)" ;;
  sync) if [ -n "$FAKE_UV_SYNC_FAIL" ]; then echo "error: lock mismatch" >&2; exit 2; fi
        mkdir -p .venv/bin && touch .venv/bin/python ;;
esac
"""
FAKE_ALLOCATOR = '#!/bin/sh\necho "$(basename "$0") $*" >> "$FAKE_CALLS"\n'


class Julia:
    def __init__(self, tmp_path: Path) -> None:
        self.base = tmp_path / "arena-extended"
        self.repo = self.base / "repo"
        self.queue = self.base / "queue"
        self.state = self.base / "env-synced-lock.sha256"
        self.repo.mkdir(parents=True)
        self.queue.mkdir()
        (self.repo / ".python-version").write_text("3.14\n")
        self.calls = tmp_path / "calls.log"
        self.calls.touch()
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        self.uv = bin_dir / "uv"
        self.uv.write_text(FAKE_UV)
        for name in ("srun", "salloc", "sbatch"):
            (bin_dir / name).write_text(FAKE_ALLOCATOR)
        for tool in bin_dir.iterdir():
            tool.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_CALLS": str(self.calls),
            "FAKE_UV_VERSION": UV_VERSION,
        }

    def copy_lock(self, text: str) -> str:
        (self.repo / "uv.lock").write_text(text)
        return hashlib.sha256(text.encode()).hexdigest()

    def sync(self, expected: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo",
                str(self.repo),
                "--state",
                str(self.state),
                "--queue",
                str(self.queue),
                "--expected",
                expected,
                "--uv",
                str(self.uv),
                "--uv-version",
                UV_VERSION,
            ],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def uv_calls(self) -> list[str]:
        return [line for line in self.calls.read_text().splitlines() if line.startswith("uv ")]

    def syncs(self) -> list[str]:
        return [line for line in self.uv_calls() if line.startswith("uv sync")]

    def allocations(self) -> list[str]:
        return [line for line in self.calls.read_text().splitlines() if not line.startswith("uv ")]


@pytest.fixture
def julia(tmp_path):
    return Julia(tmp_path)


def test_a_lock_change_syncs_the_environment_without_cuda_kernels(julia):
    digest = julia.copy_lock("lock v1")
    julia.state.write_text("0" * 64 + "\n")
    result = julia.sync(digest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert julia.syncs() == ["uv sync --locked --python 3.14"]
    assert "cuda-kernels" not in julia.calls.read_text()
    assert "--all-extras" not in julia.calls.read_text()
    assert julia.state.read_text().strip() == digest
    assert julia.allocations() == []


def test_a_matching_environment_is_left_alone(julia):
    digest = julia.copy_lock("lock v1")
    assert julia.sync(digest).returncode == 0
    julia.calls.write_text("")
    result = julia.sync(digest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert julia.syncs() == []
    assert "matches" in result.stdout


def test_copying_the_lock_alone_does_not_count_as_a_sync(julia):
    digest = julia.copy_lock("lock v2")
    result = julia.sync(digest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(julia.syncs()) == 1


def test_a_failed_sync_stops_and_records_nothing(julia):
    old = julia.copy_lock("lock v1")
    julia.state.write_text(old + "\n")
    new = julia.copy_lock("lock v2")
    result = julia.sync(new, FAKE_UV_SYNC_FAIL="1")
    assert result.returncode != 0
    assert "sync failed" in result.stdout + result.stderr
    assert julia.state.read_text().strip() == old
    assert julia.sync(new).returncode == 0, "the next attempt must sync again"
    assert len(julia.syncs()) == 2


def test_a_lock_that_did_not_arrive_is_not_synced(julia):
    julia.copy_lock("lock v1")
    laptop = hashlib.sha256(b"lock v2").hexdigest()
    result = julia.sync(laptop)
    assert result.returncode != 0
    assert "differs" in result.stdout + result.stderr
    assert julia.syncs() == []


def test_a_running_build_is_never_synced_under(julia):
    digest = julia.copy_lock("lock v1")
    (julia.queue / "8.4-1-1.running").write_text("8.4 777\n")
    result = julia.sync(digest)
    assert result.returncode != 0
    assert "running" in result.stdout + result.stderr
    assert julia.syncs() == []
    assert not julia.state.exists()


def test_julia_uv_is_brought_to_the_laptop_version_first(julia):
    digest = julia.copy_lock("lock v1")
    result = julia.sync(digest, FAKE_UV_VERSION="0.12.23")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = julia.uv_calls()
    assert f"uv self update {UV_VERSION}" in calls
    assert calls.index(f"uv self update {UV_VERSION}") < calls.index(
        "uv sync --locked --python 3.14"
    )


def test_the_python_minor_from_the_policy_is_installed_at_its_latest_patch(julia):
    digest = julia.copy_lock("lock v1")
    assert julia.sync(digest).returncode == 0
    assert "uv python install --upgrade 3.14" in julia.uv_calls()


def test_a_missing_uv_stops_cleanly(julia):
    digest = julia.copy_lock("lock v1")
    julia.uv.unlink()
    result = julia.sync(digest)
    assert result.returncode == 1
    assert "cannot run Julia's uv" in result.stderr
    assert "Traceback" not in result.stderr
    assert not julia.state.exists()
