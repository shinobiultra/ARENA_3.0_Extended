"""11.1 migration invariants and tiny CPU oracles; never execute the model study."""
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

from scripts.verified_presentation import verified_pre_wrap_notebook

ROOT = Path(__file__).resolve().parents[2]
SECTION = ROOT / "chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls"
MASTER = ROOT / "infrastructure/chapters/chapter11_representation_geometry/master_11_1.py"
EVIDENCE = ROOT / "docs/mastery/evidence/geometry_master_migration"
STEM = "11.1_PCA_SVD_and_Geometry_Controls"
MODULE = "chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(cell):
    return "".join(cell["source"])


def code_cells(notebook):
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def read_notebook(kind):
    return nbformat.read(SECTION / f"{STEM}_{kind}.ipynb", as_version=4)


@pytest.fixture(scope="module")
def methods():
    return importlib.import_module(MODULE + ".solutions")


def test_before_snapshot_and_unchanged_historical_sources():
    before = json.loads((EVIDENCE / "before.json").read_text())
    archive = ROOT / before["archive"]
    assert sha(archive.read_bytes()) == before["archive_sha256"]
    with tarfile.open(archive) as saved:
        for name, expected in before["files"].items():
            assert sha(saved.extractfile(name).read()) == expected, name
    # The old lesson, evidence, methods, tests and protocol remain physically present.
    for name, expected in before["source_inputs"].items():
        assert sha((SECTION / name).read_bytes()) == expected, name
    receipt = json.loads((SECTION / "expected_outputs/learner_execution.json").read_text())
    assert before["files"][str((SECTION / f"{STEM}_solutions.ipynb").relative_to(ROOT))] == receipt["published_solutions_sha256"]
    assert before["files"][str((SECTION / f"{STEM}_exercises.ipynb").relative_to(ROOT))] == receipt["published_exercises_sha256"]
    assert before["files"]["chapter11_representation_geometry/instructions/pages/01_[11.1]_PCA_SVD_and_Geometry_Controls.md"] == receipt["published_page_sha256"]
    for name, expected in before["figures"].items():
        assert sha((SECTION / "media" / name).read_bytes()) == expected
        assert sha((ROOT / "chapter11_representation_geometry/instructions/assets" / name).read_bytes()) == expected


def test_all_published_code_and_outputs_have_exact_parity_receipt():
    parity = json.loads((EVIDENCE / "output_parity.json").read_text())
    before = json.loads((EVIDENCE / "before.json").read_text())
    with tarfile.open(ROOT / before["archive"]) as saved:
        old = json.load(saved.extractfile(str((SECTION / f"{STEM}_solutions.ipynb").relative_to(ROOT))))
    new = read_notebook("solutions")
    old_normalized = nbformat.reads(json.dumps(old), as_version=4)
    assert len(code_cells(old)) == len(code_cells(new)) == len(parity["cells"]) == 35
    assert parity["fresh_model_execution"] is False
    assert parity["execution_kind"] == "retained_historical_reference"
    for old_cell, new_cell, pair in zip(code_cells(old), code_cells(new), parity["cells"], strict=True):
        a, b = source(old_cell), source(new_cell)
        assert [line for line in a.splitlines() if line.strip()] == [line for line in b.splitlines() if line.strip()]
        assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
        assert sha(a.encode()) == pair["published_source_sha256"]
        assert sha(b.encode()) == pair["generated_source_sha256"]
        assert sha(ast.dump(ast.parse(b)).encode()) == pair["ast_sha256"]
        assert new_cell["execution_count"] == old_cell["execution_count"]
        # nbformat normalizes multiline output strings/lists, so compare normalized nodes.
        normalized = code_cells(old_normalized)[pair["ordinal"]]
        assert new_cell["outputs"] == normalized["outputs"]
        assert sha(json.dumps(old_cell.get("outputs", []), sort_keys=True).encode()) == pair["outputs_sha256"]
    assert sum(c["execution_count"] is not None for c in code_cells(new)) == 35
    assert sum(bool(c["outputs"]) for c in code_cells(new)) == 25


def test_current_legacy_lesson_reconstructs_the_exact_archived_publication():
    before = json.loads((EVIDENCE / "before.json").read_text())
    with tarfile.open(ROOT / before["archive"]) as saved:
        old = json.load(saved.extractfile(str((SECTION / f"{STEM}_solutions.ipynb").relative_to(ROOT))))
    legacy = importlib.import_module(MODULE + ".build_notebooks")
    completed = legacy.complete_learner({"cells": legacy.lesson_cells()})
    assert [(c["cell_type"], source(c)) for c in completed["cells"]] == [
        (c["cell_type"], source(c)) for c in old["cells"]]
    receipt = json.loads((SECTION / "expected_outputs/learner_execution.json").read_text())
    for name, expected in receipt["source_inputs"].items():
        if name not in {"lesson.md", "geometry_plotting.py"}:
            assert sha((SECTION / name).read_bytes()) == expected, name


def test_recorded_execution_source_drift_is_bounded_and_named():
    """Compare real stored execution, including its explicit output-path override."""
    with tarfile.open(SECTION / "expected_outputs/own_full_release_v3/execution.tar.gz") as stored:
        recorded = json.load(stored.extractfile("completed_learner.ipynb"))
        old_plotting = stored.extractfile("source_snapshot/geometry_plotting.py").read()
    assert ast.dump(ast.parse(old_plotting)) == ast.dump(ast.parse((SECTION / "geometry_plotting.py").read_text()))
    for ordinal, (historical, canonical) in enumerate(zip(code_cells(recorded), code_cells(read_notebook("solutions")), strict=True)):
        a, b = ast.parse(source(historical)), ast.parse(source(canonical))
        if ordinal == 0:
            assert ast.unparse(a.body[-2]) == "assets_dir = Path('/home/bitwise/.cache/arena_extended/geometry/full-own-answer-v3/figures')"
            assert ast.unparse(a.body[-1]) == "assets_dir.mkdir(exist_ok=True)"
            a.body = a.body[:-2]
        assert ast.dump(a) == ast.dump(b), ordinal


def test_student_stubs_complete_from_their_own_visible_answers():
    exercises, solutions = read_notebook("exercises"), read_notebook("solutions")
    nbformat.validate(exercises)
    nbformat.validate(solutions)
    legacy = importlib.import_module(MODULE + ".build_notebooks")
    completed = []
    exercise_count = 0
    stub_names = set()
    for index, cell in enumerate(exercises.cells):
        if cell.cell_type != "code":
            continue
        text = cell.source
        if "raise NotImplementedError" in text:
            exercise_count += 1
            stub_names.update(node.name for node in ast.parse(text).body
                              if isinstance(node, ast.FunctionDef)
                              and any(isinstance(child, ast.Raise)
                                      and "NotImplementedError" in ast.unparse(child)
                                      for child in ast.walk(node)))
            following = []
            for next_cell in exercises.cells[index + 1:]:
                if next_cell.cell_type == "code":
                    break
                following.append(next_cell.source)
            blocks = re.findall(r"<summary>Solution</summary>(.*?)</details>", "\n".join(following), re.S)
            assert len(blocks) == 1
            answer = "\n\n".join(re.findall(r"```python\n(.*?)```", blocks[0], re.S))
            text = legacy.complete_from_answer(text, answer)
            assert "tests.test_" in text, "Every exercise immediately calls semantic tests."
        completed.append(text)
    assert exercise_count == 11
    taught_names = {node.name for node in ast.parse((SECTION / "geometry_methods.py").read_text()).body
                    if isinstance(node, ast.FunctionDef)}
    assert stub_names == taught_names and len(stub_names) == 20
    for student, reference in zip(completed, code_cells(solutions), strict=True):
        assert ast.dump(ast.parse(student)) == ast.dump(ast.parse(source(reference)))
    assert all(c.execution_count is None for c in exercises.cells if c.cell_type == "code")


def test_full_lesson_controls_play_and_prediction_prompts_survive():
    notebook = read_notebook("exercises")
    text = "\n".join(c.source for c in notebook.cells)
    master = MASTER.read_text()
    assert len(notebook.cells) >= 110
    assert len(re.findall(r"^### Exercise", text, re.M)) == 11
    assert len(re.findall(r"^## Forecast before the result", text, re.M)) == 7
    assert len(re.findall(r"^## Reflect after the result", text, re.M)) == 7
    for term in ("Historical reference", "historical reference results", "not a fresh", "guided practice", "Try It Yourself", "Limitations", "Further investigations", "Reading", "training", "development", "final", "transductive", "full-vocabulary", "independent-noise", "shuffled", "mean_only", "no_op", "same_rank_random_donor_plane", "equal_norm_random_edit", "class_order_permuted_circular_probe", "run_calendar_study", "run_geometry_projection_sweep", "run_calendar_angle_play"):
        assert term in text, term
    assert "from arena_ext" not in master
    assert len(re.findall(r'^# ! TAGS: \[html\]', master, re.M)) == 10
    # New prompt cells contain no fabricated confidence, answer or observation fields.
    forecasts = [c for c in notebook.cells if c.cell_type == "markdown" and c.source.startswith("## Forecast before the result")]
    assert all("confidence (0–100%)" in c.source and "Record whether you have already seen it" in c.source for c in forecasts)
    assert text.index("Forecast before the result — causal-controls") < text.index('primary = final_frame[')
    assert text.index("Forecast before the result — nonlinear-generalization") < text.index("projection_records = run_geometry_projection_sweep(")


def test_every_original_explanatory_line_is_preserved():
    before = json.loads((EVIDENCE / "before.json").read_text())
    with tarfile.open(ROOT / before["archive"]) as saved:
        old = json.load(saved.extractfile(str((SECTION / f"{STEM}_solutions.ipynb").relative_to(ROOT))))
    current = "\n".join(c.source for c in verified_pre_wrap_notebook(SECTION / f'{STEM}_exercises.ipynb').cells if c.cell_type == "markdown")
    current_lines = {line.strip() for line in current.splitlines() if line.strip()}
    for cell in old["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = re.sub(r"<details><summary>Solution</summary>.*?</details>\s*", "", source(cell), flags=re.S)
        for line in text.splitlines():
            line = line.strip()
            if not line or line in {"## Setup", "# What have we learned?"} or line.startswith("# [11.1]"):
                continue
            assert line in current_lines, line


def test_import_is_safe_from_an_unrelated_directory(tmp_path):
    script = r'''
import contextlib, importlib, io, os, sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
os.environ['ARENA_GEOMETRY_OUTPUT_DIRECTORY'] = str(Path.cwd() / 'must_not_exist')
def forbidden(*args, **kwargs):
    raise AssertionError('Import ran a lesson side effect')
torch.set_num_threads = forbidden
plt.style.use = forbidden
plt.subplots = forbidden
before = torch.cuda.is_initialized()
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    module = importlib.import_module('chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.solutions')
assert not captured.getvalue(), captured.getvalue()
assert not (Path.cwd() / 'must_not_exist').exists()
assert torch.cuda.is_initialized() == before
assert 'transformers' not in sys.modules
for name in ('fit_pca', 'replace_coordinates', 'capture_residual_positions', 'run_calendar_study', 'run_geometry_projection_sweep', 'run_calendar_angle_play', 'run_smoke_test', 'run_gpu_test'):
    assert callable(getattr(module, name)), name
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "Import safety passed"


ORACLES = [
    "test_pca_fits_only_training_and_preserves_total_variance",
    "test_ridge_has_unpenalized_intercept_and_known_shrinkage",
    "test_template_centering_is_explicitly_transductive",
    "test_angles_wrap_around_and_zero_abstains",
    "test_calendar_order_null_is_distinct_from_class_identification",
    "test_neighborhood_ranks_match_independent_library_and_duality",
    "test_procrustes_recovers_rotation_reflection_and_translation",
    "test_minimum_norm_coordinate_edit_matches_constraint_solution",
    "test_donor_subspace_does_not_copy_complement",
    "test_exact_ring_plane_changes_answer_and_preserves_nuisance",
    "test_semantic_positions_respect_both_padding_sides_and_cleanup_hooks",
    "test_mean_ablation_and_literal_author_coordinates_answer_different_questions",
    "test_native_capture_preserves_baseline_and_cleans_up_on_failure",
    "test_token_offsets_do_not_assume_identical_prompt_lengths",
    "test_restricted_calendar_success_does_not_hide_full_vocabulary_failure",
    "test_least_squares_repeated_fits_and_rank_deficiency",
]


@pytest.mark.parametrize("oracle", ORACLES)
def test_generated_visible_methods_against_independent_cpu_oracles(methods, oracle):
    tests = importlib.import_module(MODULE + ".geometry_methods_tests")
    torch = importlib.import_module("torch")
    previous_dtype = torch.get_default_dtype()
    try:
        # The preserved native-hook oracle explicitly creates float32 model outputs.
        # Reproduce its historical float32 default even after another test changes it.
        torch.set_default_dtype(torch.float32)
        getattr(tests, oracle)(methods)
    finally:
        torch.set_default_dtype(previous_dtype)
