"""Exercise the batch runner's exit codes and lesson time limit with tiny builds, without model or GPU work."""

import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

CLUSTER = Path(__file__).resolve().parents[1] / "infrastructure/extension/cluster"
pytestmark = pytest.mark.cpu

# Stands in for build.py. The section name picks the behaviour.
FIXTURE_BUILD = """\
import os, subprocess, sys, time
section = sys.argv[1]
print('fixture build', sys.argv[1:], os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
if section == 'bad':
    raise SystemExit(7)
if section == 'overrun':  # build.py stopping itself at its time limit
    print('TIMEOUT: overrun ran past its time limit', flush=True)
    raise SystemExit(124)
if section == 'hang':  # a build that ignores its limit, with a child process of its own
    child = subprocess.Popen(['sleep', '300'])
    open(os.environ['FIXTURE_PIDS'], 'w').write(f'{os.getpid()} {child.pid}')
    time.sleep(300)
"""


def run_batch(tmp_path, sections, *args, fault=None):
    """Keep the actual CLI/threads; replace only the expensive build boundary."""
    cluster = tmp_path / "infrastructure/extension/cluster"
    cluster.mkdir(parents=True)
    runner = cluster / "run_batch.py"
    shutil.copyfile(CLUSTER / "run_batch.py", runner)
    (cluster.parent / "build.py").write_text(FIXTURE_BUILD)
    command = [sys.executable, str(runner)]
    if fault:
        # Inject launch failure or worker termination before running the real entrypoint.
        harness = (
            "import runpy, subprocess, sys\n"
            "original_popen = subprocess.Popen\n"
            "def popen(command, *args, **kwargs):\n"
            "    if command[2] == 'lost':\n"
            f"        raise {fault}('fixture launch failure')\n"
            "    return original_popen(command, *args, **kwargs)\n"
            "subprocess.Popen = popen\n"
            "script = sys.argv.pop(1)\n"
            "sys.argv[0] = script\n"
            "runpy.run_path(script, run_name='__main__')\n"
        )
        command = [sys.executable, "-c", harness, str(runner)]
    out = tmp_path / "output"
    result = subprocess.run(
        [*command, *sections, "--out", str(out), *args],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "fixture-gpu",
            "FIXTURE_PIDS": str(tmp_path / "pids.txt"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    with (out / "summary.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return result, rows, out


def statuses(rows):
    return {row["section"]: row["status"] for row in rows}


def test_failed_build_exits_nonzero_and_retains_other_results(tmp_path):
    result, rows, out = run_batch(tmp_path, ["bad", "good"])
    assert result.returncode != 0, result.stdout + result.stderr
    assert statuses(rows) == {"bad": "FAILED(7)", "good": "ok"}
    assert "1 ok, 1 failed" in result.stdout
    assert "fixture build" in (out / "bad.log").read_text()
    assert (out / "good.log").exists()


def test_successful_builds_exit_zero_with_the_thirty_minute_lesson_limit(tmp_path):
    result, rows, out = run_batch(tmp_path, ["first", "second"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert "2 ok, 0 failed" in result.stdout
    log = (out / "first.log").read_text()
    assert "'--execute', '--time-limit', '1800'" in log
    assert "fixture-gpu" in log


def test_a_build_that_stops_at_its_time_limit_is_a_timeout_and_the_batch_continues(tmp_path):
    result, rows, out = run_batch(tmp_path, ["overrun", "later"], "--time-limit", "5")
    assert result.returncode != 0, result.stdout + result.stderr
    assert statuses(rows) == {"overrun": "FAILED(timeout)", "later": "ok"}
    assert "1 ok, 1 failed" in result.stdout
    assert "overrun: FAILED(timeout)" in result.stdout
    assert "'--time-limit', '5'" in (out / "later.log").read_text()


def test_a_hung_build_is_killed_at_its_limit_with_its_processes(tmp_path):
    start = time.monotonic()
    result, rows, out = run_batch(
        tmp_path, ["hang", "later"], "--time-limit", "2", "--kill-grace", "1"
    )
    assert time.monotonic() - start < 30
    assert result.returncode != 0, result.stdout + result.stderr
    assert statuses(rows) == {"hang": "FAILED(timeout)", "later": "ok"}
    assert "TIMEOUT" in (out / "hang.log").read_text()
    for pid in map(int, (tmp_path / "pids.txt").read_text().split()):
        assert not Path(f"/proc/{pid}").exists() or "\nState:\tZ" in Path(
            f"/proc/{pid}/status"
        ).read_text(), f"process {pid} outlived the timed-out build"


def test_launch_exception_is_recorded_and_worker_continues(tmp_path):
    result, rows, out = run_batch(tmp_path, ["lost", "good"], fault="OSError")
    assert result.returncode != 0, result.stdout + result.stderr
    assert statuses(rows) == {"lost": "FAILED(OSError)", "good": "ok"}
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
    assert statuses(rows) == {
        "good": "ok",
        "lost": "FAILED(missing result)",
        "never": "FAILED(missing result)",
    }
    assert "1 ok, 2 failed" in result.stdout
    assert "fixture build" in (out / "good.log").read_text()
