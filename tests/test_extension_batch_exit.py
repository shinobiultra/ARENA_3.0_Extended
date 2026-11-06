"""Exercise batch CLI exit codes with tiny builds, without model or GPU work."""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "infrastructure/extension/cluster/run_batch.py"
pytestmark = pytest.mark.cpu


def run_batch(tmp_path, sections, *, fault=None):
    """Keep the actual CLI/threads; replace only the expensive build boundary."""
    cluster = tmp_path / "infrastructure/extension/cluster"
    cluster.mkdir(parents=True)
    runner = cluster / "run_batch.py"
    shutil.copyfile(RUNNER, runner)
    (cluster.parent / "build.py").write_text(
        "import os, sys\n"
        "print('fixture build', sys.argv[1:], os.environ['CUDA_VISIBLE_DEVICES'], flush=True)\n"
        "raise SystemExit(7 if sys.argv[1] == 'bad' else 0)\n"
    )
    command = [sys.executable, str(runner)]
    if fault:
        # Inject launch failure or worker termination before running the real entrypoint.
        harness = (
            "import runpy, subprocess, sys\n"
            "original_run = subprocess.run\n"
            "def run(command, **kwargs):\n"
            "    if command[2] == 'lost':\n"
            f"        raise {fault}('fixture launch failure')\n"
            "    return original_run(command, **kwargs)\n"
            "subprocess.run = run\n"
            "script = sys.argv.pop(1)\n"
            "sys.argv[0] = script\n"
            "runpy.run_path(script, run_name='__main__')\n"
        )
        command = [sys.executable, "-c", harness, str(runner)]
    out = tmp_path / "output"
    result = subprocess.run(
        [*command, *sections, "--out", str(out), "--timeout", "13"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "fixture-gpu"},
        capture_output=True,
        text=True,
        timeout=15,
    )
    with (out / "summary.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return result, rows, out


def test_failed_build_exits_nonzero_and_retains_other_results(tmp_path):
    result, rows, out = run_batch(tmp_path, ["bad", "good"])
    assert result.returncode != 0, result.stdout + result.stderr
    assert {row["section"]: row["status"] for row in rows} == {
        "bad": "FAILED(7)", "good": "ok",
    }
    assert "1 ok, 1 failed" in result.stdout
    assert "fixture build" in (out / "bad.log").read_text()
    assert (out / "good.log").exists()


def test_successful_builds_exit_zero_and_keep_execution_arguments(tmp_path):
    result, rows, out = run_batch(tmp_path, ["first", "second"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert "2 ok, 0 failed" in result.stdout
    log = (out / "first.log").read_text()
    assert "'--execute', '--timeout', '13'" in log
    assert "fixture-gpu" in log


def test_launch_exception_is_recorded_and_worker_continues(tmp_path):
    result, rows, out = run_batch(tmp_path, ["lost", "good"], fault="OSError")
    assert result.returncode != 0, result.stdout + result.stderr
    assert {row["section"]: row["status"] for row in rows} == {
        "lost": "FAILED(OSError)", "good": "ok",
    }
    assert "fixture launch failure" in (out / "lost.log").read_text()
    assert "1 ok, 1 failed" in result.stdout


@pytest.mark.parametrize("sections", [["lost", "good"], ["lost", "lost", "good"]])
def test_worker_termination_marks_every_missing_result_failed(tmp_path, sections):
    result, rows, _ = run_batch(tmp_path, sections, fault="SystemExit")
    assert result.returncode != 0, result.stdout + result.stderr
    assert len(rows) == len(sections)
    assert sorted(row["section"] for row in rows) == sorted(sections)
    assert all(row["status"].startswith("FAILED(") for row in rows)
    assert "missing" in (result.stdout + result.stderr).lower()


def test_worker_termination_preserves_completed_successes(tmp_path):
    result, rows, out = run_batch(tmp_path, ["good", "lost", "never"], fault="SystemExit")
    assert result.returncode != 0, result.stdout + result.stderr
    assert {row["section"]: row["status"] for row in rows} == {
        "good": "ok", "lost": "FAILED(missing result)", "never": "FAILED(missing result)",
    }
    assert "1 ok, 2 failed" in result.stdout
    assert "fixture build" in (out / "good.log").read_text()
