"""The 30-minute lesson limit in `build.py --execute`, shown on tiny CPU notebooks.

Each test starts a real Jupyter kernel (no model, no GPU) and shrinks the limit to seconds.
"""

import subprocess
import time
from pathlib import Path

import nbformat
import pytest

from infrastructure.extension import build

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.cpu


def notebook(path: Path, *sources: str) -> Path:
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source) for source in sources])
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nbformat.write(nb, path)
    return path


def is_gone(pid: int) -> bool:
    """A process is gone once it no longer exists or is only a zombie awaiting its reaper."""
    status = Path(f"/proc/{pid}/status")
    try:
        return "\nState:\tZ" in status.read_text()
    except FileNotFoundError:
        return True


def wait_until_gone(pid: int, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if is_gone(pid):
            return True
        time.sleep(0.1)
    return is_gone(pid)


def test_the_limit_is_thirty_minutes_per_lesson():
    assert build.LESSON_TIME_LIMIT_S == 1800


def test_a_lesson_within_the_limit_runs_every_cell(tmp_path):
    path = notebook(tmp_path / "lesson_solutions.ipynb", "x = 20 + 22", "print(x)")
    executed, _ = build.execute(path, deadline=time.monotonic() + 60)
    assert [cell.execution_count for cell in executed.cells] == [1, 2]
    assert executed.cells[1].outputs[0]["text"] == "42\n"


def test_an_overrunning_lesson_stops_and_its_kernel_processes_are_reaped(tmp_path):
    pids = tmp_path / "pids.txt"
    path = notebook(
        tmp_path / "lesson_solutions.ipynb",
        "import os, subprocess\n"
        f"open({str(pids)!r}, 'w').write(f\"{{os.getpid()}} {{subprocess.Popen(['sleep', '300']).pid}}\")",
        "import time\ntime.sleep(300)",
        "print('never reached')",
    )
    limit = 8
    start = time.monotonic()
    with pytest.raises(build.LessonTimeout) as caught:
        build.execute(path, deadline=start + limit)
    elapsed = time.monotonic() - start

    assert elapsed < limit + 20, f"the build ran {elapsed:.0f}s past an {limit}s limit"
    assert "time limit" in str(caught.value)
    kernel_pid, child_pid = map(int, pids.read_text().split())
    assert wait_until_gone(kernel_pid), "the kernel outlived the timed-out build"
    assert wait_until_gone(child_pid), "a process the lesson started outlived the timed-out build"

    failed = path.with_name("lesson_solutions.FAILED.ipynb")
    assert failed.exists(), "the partial notebook should be kept for diagnosis"
    assert nbformat.read(failed, as_version=4).cells[0].execution_count == 1
    assert not any(cell.get("outputs") for cell in nbformat.read(path, as_version=4).cells)


def test_a_lesson_already_past_its_deadline_does_not_start_a_kernel(tmp_path):
    path = notebook(tmp_path / "lesson_solutions.ipynb", "x = 1")
    with pytest.raises(build.LessonTimeout):
        build.execute(path, deadline=time.monotonic() - 1)


def test_failed_notebooks_stay_out_of_commits():
    sample = "chapter7_activation_to_language/exercises/part1_x/7.1_Lesson_solutions.FAILED.ipynb"
    result = subprocess.run(["git", "check-ignore", "-q", sample], cwd=ROOT, check=False)
    assert result.returncode == 0, f"{sample} is not ignored by git"
