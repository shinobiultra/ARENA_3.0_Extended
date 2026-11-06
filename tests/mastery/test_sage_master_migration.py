"""SAGE migration fidelity and tiny CPU oracles; never execute its native study."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import nbformat
import pytest

from scripts.evidence_bundles import resolve_historical_archive

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance"
SECTION = LEGACY.parent / "part4_sage"
MASTER = ROOT / "infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_4.py"
EVIDENCE = ROOT / "docs/mastery/evidence/sage_master_migration"
STEM = "16.4_SAGE_Global_Importance"
OLD_STEM = "16.12_SAGE_Global_Importance"
MODULE = "chapter16_shapley_attribution_baselines.exercises.part4_sage"
OLD_MODULE = "chapter16_shapley_attribution_baselines.exercises.part12_sage_global_importance"
OLD_IMPORT = f"from {OLD_MODULE} import (\n    sage_methods_tests as tests,\n)"
NEW_IMPORT = f"from {MODULE} import tests"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(cell):
    return "".join(cell["source"])


def codes(notebook):
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def notebook(kind):
    return nbformat.read(SECTION / f"{STEM}_{kind}.ipynb", as_version=4)


def old_notebook(kind):
    return json.loads((LEGACY / f"{OLD_STEM}_{kind}.ipynb").read_text())


def historical_route(text, ordinal):
    """One exact test-import substitution; scientific code receives no exclusions."""
    if ordinal == 0:
        assert text.count(NEW_IMPORT) == 1
        return text.replace(NEW_IMPORT, OLD_IMPORT)
    assert NEW_IMPORT not in text
    return text


@pytest.fixture(scope="module")
def methods():
    return importlib.import_module(MODULE + ".solutions")


def test_archive_is_recoverable_and_every_legacy_file_is_unchanged():
    before = json.loads((EVIDENCE / "before.json").read_text())
    assert sha(resolve_historical_archive(ROOT / before["archive"], before["archive_sha256"]).read_bytes()) == before["archive_sha256"]
    with tarfile.open(resolve_historical_archive(ROOT / before["archive"], before["archive_sha256"])) as saved:
        for name, expected in before["files"].items():
            assert sha(saved.extractfile(name).read()) == expected, name
            assert sha((ROOT / name).read_bytes()) == expected, name
    assert len(before["files"]) == 93
    receipt = json.loads((LEGACY / "expected_outputs/learner_execution.json").read_text())
    assert len(receipt["source_inputs"]) == 15
    for name, expected in receipt["source_inputs"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    for kind in ("exercises", "solutions"):
        assert sha((LEGACY / f"{OLD_STEM}_{kind}.ipynb").read_bytes()) == receipt[f"published_{kind}_sha256"]
    page = ROOT / "chapter16_shapley_attribution_baselines/instructions/pages/12_[16.12]_SAGE_Global_Importance.md"
    assert sha(page.read_bytes()) == receipt["published_page_sha256"]


def test_full_historical_code_and_outputs_have_bounded_exact_parity():
    parity = json.loads((EVIDENCE / "output_parity.json").read_text())
    assert parity["execution_kind"] == "retained_historical_reference"
    assert parity["fresh_model_execution"] is False
    assert parity["route_adaptation"]["old"] == OLD_IMPORT
    assert parity["route_adaptation"]["new"] == NEW_IMPORT
    old, new = old_notebook("solutions"), notebook("solutions")
    normalized = nbformat.reads(json.dumps(old), as_version=4)
    assert len(codes(old)) == len(codes(new)) == len(parity["cells"]) == 33
    for ordinal, (a, b, norm, pair) in enumerate(zip(codes(old), codes(new), codes(normalized), parity["cells"], strict=True)):
        x, y = source(a), source(b)
        comparable = historical_route(y, ordinal)
        assert [line for line in x.splitlines() if line.strip()] == [line for line in comparable.splitlines() if line.strip()]
        assert ast.dump(ast.parse(x)) == ast.dump(ast.parse(comparable))
        assert sha(x.encode()) == pair["published_source_sha256"]
        assert sha(y.encode()) == pair["generated_source_sha256"]
        assert sha(ast.dump(ast.parse(x)).encode()) == pair["published_ast_sha256"] == pair["normalized_ast_sha256"]
        assert sha(ast.dump(ast.parse(y)).encode()) == pair["generated_ast_sha256"]
        assert b.outputs == norm.outputs
        assert b.execution_count == norm.execution_count == pair["historical_execution_count"]
        assert sha(json.dumps(a.get("outputs", []), sort_keys=True).encode()) == pair["outputs_sha256"]
        assert pair["route_adaptation"] == ("setup_test_import_only" if ordinal == 0 else None)
    assert sum(bool(c.outputs) for c in codes(new)) == 31
    assert sum(c.execution_count is not None for c in codes(new)) == 33
    assert sum("image/png" in out.get("data", {}) for c in codes(new) for out in c.outputs) == 18


def test_retained_lesson_and_actual_recorded_execution_reconstruct_the_publication():
    builder = importlib.import_module(OLD_MODULE + ".build_notebooks")
    completed = builder.complete_learner({"cells": builder.lesson_cells()})
    old = old_notebook("solutions")
    assert [(c["cell_type"], source(c)) for c in completed["cells"]] == [(c["cell_type"], source(c)) for c in old["cells"]]
    with tarfile.open(LEGACY / "expected_outputs/full_own_answer_v4/execution.tar.gz") as stored:
        recorded = json.load(stored.extractfile("completed_learner.ipynb"))
    recorded = nbformat.reads(json.dumps(recorded), as_version=4)
    for ordinal, (a, b, current) in enumerate(zip(codes(old), codes(recorded), codes(notebook("solutions")), strict=True)):
        assert ast.dump(ast.parse(source(a))) == ast.dump(ast.parse(source(b)))
        assert ast.dump(ast.parse(source(b))) == ast.dump(ast.parse(historical_route(source(current), ordinal)))
        assert b.execution_count == current.execution_count
        assert b.outputs == current.outputs


def test_all_eighteen_canonical_figures_are_exact_historical_bytes():
    before = json.loads((EVIDENCE / "before.json").read_text())
    assert len(before["figures"]) == 18
    assert {p.name for p in (SECTION / "media").glob("*.png")} == set(before["figures"])
    for name, expected in before["figures"].items():
        assert sha((SECTION / "media" / name).read_bytes()) == expected
        assert sha((ROOT / "chapter16_shapley_attribution_baselines/instructions/assets" / name).read_bytes()) == expected
    ex = notebook("exercises")
    assert sum("data:image/png;base64," in str(o.get("data", {}).get("text/html", "")) for c in codes(ex) for o in c.outputs) == 18


def test_semantic_test_route_adaptation_is_byte_identical_and_explicit():
    assert (SECTION / "tests.py").read_bytes() == (LEGACY / "sage_methods_tests.py").read_bytes()
    setup = source(codes(notebook("solutions"))[0])
    assert NEW_IMPORT in setup and OLD_IMPORT not in setup
    for dependency in ("sage_data", "sage_reference", "run_native"):
        assert f"from {OLD_MODULE}.{dependency} import" in setup
    assert "Path(__file__).resolve().parents[3]" in (SECTION / "solutions.py").read_text()
    assert "not a fresh execution" in MASTER.read_text()
    assert "test-import route" in MASTER.read_text()


def test_student_stubs_complete_only_from_their_own_visible_answers():
    builder = importlib.import_module(OLD_MODULE + ".build_notebooks")
    ex, sol = notebook("exercises"), notebook("solutions")
    nbformat.validate(ex)
    nbformat.validate(sol)
    completed = builder.complete_learner(ex)
    stub_names = set()
    count = 0
    for index, cell in enumerate(ex.cells):
        if cell.cell_type == "code" and "raise NotImplementedError" in cell.source:
            count += 1
            stub_names.update(n.name for n in ast.parse(cell.source).body if isinstance(n, ast.FunctionDef))
            assert "tests.test_" in cell.source
    assert count == 9
    assert stub_names == {"coalition_masks", "marginal_predict", "loss_game", "exact_shapley", "gaussian_game", "path_contributions", "permutation_sage", "summarize_paths", "prefix_values"}
    for a, b in zip(codes(completed), codes(sol), strict=True):
        assert ast.dump(ast.parse(source(a))) == ast.dump(ast.parse(source(b)))
    assert all(c.execution_count is None for c in codes(ex))


def test_every_original_explanatory_line_survives():
    current = "\n".join(c.source for c in notebook("exercises").cells if c.cell_type == "markdown")
    current_lines = {line.strip() for line in current.splitlines() if line.strip()}
    for cell in old_notebook("exercises")["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = re.sub(r"<details><summary>Solution</summary>.*?</details>", "", source(cell), flags=re.S)
        for line in text.splitlines():
            line = line.strip()
            if not line or line in {"## Introduction", "### Learning objectives", "## Setup"} or line.startswith("# [16.12]"):
                continue
            if re.match(r"^## [1-7]️⃣ ", line):
                line = line.replace("## ", "# ", 1)
            assert line in current_lines, line


def test_full_lesson_controls_plays_negative_findings_and_forecasts_survive():
    nb = notebook("exercises")
    text = "\n".join(c.source for c in nb.cells)
    assert len(nb.cells) >= 130
    assert text.count("### Try It Yourself") == 2
    assert text.count("## Forecast before the result") == 18
    assert text.count("## Reflect after the result") == 18
    assert "Exercise - why can we average local loss explanations?" in text
    assert "Absolute value is nonlinear" in text
    for term in (
        "shuffled",
        "ridge",
        "training-mean",
        "singleton",
        "leave-one-out",
        "100 random feature orderings",
        "not a fresh execution",
        "119 audit rows",
        "guided practice",
        "mean imputation",
        "bias",
        "negative",
        "Limitations",
        "Reading",
        "audit results should be treated as new exploratory experiments",
        "doesn't reproduce",
        "marginal",
        "conditional",
        "758",
        "153",
        "119",
        "64"
    ):
        assert term.lower() in text.lower(), term
    forecasts = [c for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## Forecast before the result")]
    for cell in forecasts:
        assert "confidence (0–100%)" in cell.source
        assert "Record whether you have already seen it" in cell.source
        assert "Changed-condition forecast" in cell.source
    assert text.index("Forecast before the result — native-global") < text.index("prepared = prepare_study(")
    assert text.index("Forecast before the result — sampling-and-choices") < text.index("diagnostic = diagnostics(")
    assert text.index("Forecast before the result — audit-model-control") < text.index('audit = evaluate_study(')
    assert "from arena_ext" not in MASTER.read_text()


def test_import_is_safe_from_an_unrelated_directory(tmp_path):
    script = r'''
import contextlib, importlib, io, os, sys, tempfile
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import urllib.request
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
os.environ['ARENA_SAGE_OUTPUT_DIRECTORY'] = str(Path.cwd() / 'must_not_exist')
def forbidden(*args, **kwargs):
    raise AssertionError('Import ran a lesson side effect')
torch.set_num_threads = forbidden
torch.set_grad_enabled = forbidden
plt.subplots = forbidden
plt.rcParams.update = forbidden
urllib.request.urlopen = forbidden
tempfile.mkdtemp = forbidden
before = torch.cuda.is_initialized()
threads = torch.get_num_threads()
grad = torch.is_grad_enabled()
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    module = importlib.import_module('chapter16_shapley_attribution_baselines.exercises.part4_sage.solutions')
assert not captured.getvalue(), captured.getvalue()
assert not (Path.cwd() / 'must_not_exist').exists()
assert torch.cuda.is_initialized() == before
assert torch.get_num_threads() == threads
assert torch.is_grad_enabled() == grad
assert not hasattr(module, 'train') and not hasattr(module, 'prepared')
for name in ('gaussian_game', 'exact_shapley', 'marginal_predict', 'fit_models', 'prepare_study', 'evaluate_study', 'diagnostics', 'native_play'):
    assert callable(getattr(module, name)), name
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "Import safety passed"


ORACLES = ("coalition_masks", "marginal_predict", "loss_game", "exact_shapley", "gaussian_game", "path_contributions", "permutation_sage", "summarize_paths", "prefix_values")


@pytest.mark.parametrize("name", ORACLES)
def test_visible_generated_methods_against_independent_cpu_oracles(methods, name):
    checks = importlib.import_module(MODULE + ".tests")
    torch = importlib.import_module("torch")
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        getattr(checks, "test_" + name)(getattr(methods, name))
    finally:
        torch.set_default_dtype(old_dtype)
