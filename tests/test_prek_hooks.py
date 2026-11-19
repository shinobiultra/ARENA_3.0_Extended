"""The maintainer hooks (.pre-commit-config.yaml) select maintainer code and leave learner material alone.

Asks prek itself which files each hook would run on (`prek run --dry-run`), so the check
follows prek's own file matching.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
HOOKS = {"ruff-check", "ruff-format", "ty"}
# prek from PATH, or installed next to this interpreter (an unactivated venv, as in CI).
PREK = shutil.which("prek") or shutil.which("prek", path=str(Path(sys.executable).parent))
pytestmark = pytest.mark.cpu


def tracked(pattern: str) -> str:
    """The first tracked file matching `pattern`, so the examples stay real as the repo changes."""
    files = subprocess.run(
        ["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert files, f"no tracked file matches {pattern}"
    return files[0]


MAINTAINER = [
    "infrastructure/extension/build.py",
    "infrastructure/extension/cluster/run_batch.py",
    "scripts/ci_quality_gate.py",
    "tests/test_extension_batch_exit.py",
    tracked("arena_ext/*.py"),
    tracked("mastery_hub/*.py"),
]
LEARNER = {
    "original ARENA lesson": lambda: tracked("chapter0_fundamentals/exercises/*/solutions.py"),
    "original ARENA tooling": lambda: tracked("infrastructure/core/*.py"),
    "lesson master": lambda: tracked("infrastructure/chapters/chapter7_*/master_*.py"),
    "generated lesson solutions": lambda: tracked("chapter7_*/exercises/*/solutions.py"),
    "generated instructions page": lambda: tracked("chapter7_*/instructions/pages/*.py"),
}


def files_per_hook(paths: list[str]) -> dict[str, set[str]]:
    out = subprocess.run(
        [str(PREK), "run", "--dry-run", "--verbose", "--color", "never", "--files", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    selected: dict[str, set[str]] = {}
    hook = None
    for line in out.splitlines():
        if match := re.match(r"- hook id: (\S+)", line):
            hook = match.group(1)
            selected[hook] = set()
        elif hook and (match := re.match(r"\s+- (\S+\.pyi?)$", line)):
            selected[hook].add(match.group(1))
    return selected


def test_the_hooks_are_ruff_check_fix_ruff_format_and_ty_check():
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}
    assert set(hooks) == HOOKS
    assert hooks["ruff-check"].get("args") == ["--fix"]
    assert hooks["ty"]["entry"].startswith("ty check")


@pytest.mark.skipif(PREK is None, reason="prek is not installed")
def test_the_hooks_run_on_maintainer_code_and_skip_learner_material():
    learner = {kind: pick() for kind, pick in LEARNER.items()}
    selected = files_per_hook([*MAINTAINER, *learner.values()])
    assert set(selected) == HOOKS
    for hook, files in selected.items():
        assert files == set(MAINTAINER), f"{hook} would run on {sorted(files ^ set(MAINTAINER))}"
