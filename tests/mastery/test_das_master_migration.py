"""11.3 migration invariants and tiny CPU oracles; never execute the model study."""
from __future__ import annotations
from scripts.verified_publication_pages import historical_bytes

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

from scripts.verified_presentation import verified_pre_wrap_notebook

ROOT = Path(__file__).resolve().parents[2]
SECTION = ROOT / "chapter11_representation_geometry/exercises/part3_causal_abstraction_das"
MASTER = ROOT / "infrastructure/chapters/chapter11_representation_geometry/master_11_3.py"
EVIDENCE = ROOT / "docs/mastery/evidence/das_master_migration"
STEM = "11.3_Causal_Abstraction_and_Distributed_Alignment_Search"
MODULE = "chapter11_representation_geometry.exercises.part3_causal_abstraction_das"


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
    archive = resolve_historical_archive(ROOT / before["archive"], before["archive_sha256"])
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
    assert before["files"]["chapter11_representation_geometry/instructions/pages/03_[11.3]_Causal_Abstraction_and_Distributed_Alignment_Search.md"] == receipt["published_page_sha256"]
    for name, expected in before["figures"].items():
        assert sha((SECTION / "media" / name).read_bytes()) == expected
        assert sha((ROOT / "chapter11_representation_geometry/instructions/assets" / name).read_bytes()) == expected


def test_all_published_code_and_outputs_have_exact_parity_receipt():
    parity = json.loads((EVIDENCE / "output_parity.json").read_text())
    before = json.loads((EVIDENCE / "before.json").read_text())
    with tarfile.open(resolve_historical_archive(ROOT / before["archive"], before["archive_sha256"])) as saved:
        old = json.load(saved.extractfile(str((SECTION / f"{STEM}_solutions.ipynb").relative_to(ROOT))))
    new = read_notebook("solutions")
    old_normalized = nbformat.reads(json.dumps(old), as_version=4)
    assert len(code_cells(old)) == len(code_cells(new)) == len(parity["cells"]) == 32
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
    assert sum(c["execution_count"] is not None for c in code_cells(new)) == 32
    assert sum(bool(c["outputs"]) for c in code_cells(new)) == 29


def test_current_legacy_lesson_reconstructs_the_exact_archived_publication():
    before = json.loads((EVIDENCE / "before.json").read_text())
    with tarfile.open(resolve_historical_archive(ROOT / before["archive"], before["archive_sha256"])) as saved:
        old = json.load(saved.extractfile(str((SECTION / f"{STEM}_solutions.ipynb").relative_to(ROOT))))
    legacy = importlib.import_module(MODULE + ".build_notebooks")
    completed = legacy.complete_learner({"cells": legacy.lesson_cells()})
    assert [(c["cell_type"], source(c)) for c in completed["cells"]] == [
        (c["cell_type"], source(c)) for c in old["cells"]]
    receipt = json.loads((SECTION / "expected_outputs/learner_execution.json").read_text())
    for name, expected in receipt["source_inputs"].items():
        assert sha((SECTION / name).read_bytes()) == expected, name


def test_historical_receipts_match_sources_without_claiming_external_execution():
    before = json.loads((EVIDENCE / "before.json").read_text())
    parity = json.loads((EVIDENCE / "output_parity.json").read_text())
    for name, expected in parity["retained_receipts"].items():
        assert sha((ROOT / name).read_bytes()) == expected
    for name in ("learner_execution.json", "fresh_reproduction.json"):
        receipt = json.loads((SECTION / "expected_outputs" / name).read_text())
        for filename, expected in receipt["source_inputs"].items():
            assert sha((SECTION / filename).read_bytes()) == expected
    assert before["external_cache_directories_present"] == {"learner_directory": False, "fresh_directory": False}
    assert "not reopened" in parity["evidence_boundary"]
    assert parity["fresh_model_execution"] is False


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
            assert "test_" in text or "assert" in text, "Every exercise immediately calls semantic tests."
        completed.append(text)
    assert exercise_count == 9
    taught_names = {
        "high_level_interchange", "high_level_output", "orthonormal_columns",
        "subspace_interchange", "intervention_loss", "interchange_accuracy",
        "fit_alignment", "role_interchange_targets", "continue_native",
        "fit_role_probe", "fit_minibatch_alignment",
    }
    assert stub_names == taught_names and len(stub_names) == 11
    for student, reference in zip(completed, code_cells(solutions), strict=True):
        assert ast.dump(ast.parse(student)) == ast.dump(ast.parse(source(reference)))
    assert all(c.execution_count is None for c in exercises.cells if c.cell_type == "code")


def test_full_lesson_controls_play_and_prediction_prompts_survive():
    notebook = read_notebook("exercises")
    text = "\n".join(c.source for c in notebook.cells)
    master = MASTER.read_text()
    assert len(notebook.cells) >= 110
    assert len(re.findall(r"^### Exercise", text, re.M)) == 9
    assert len(re.findall(r"^## Forecast before the result", text, re.M)) == 12
    assert len(re.findall(r"^## Reflect after the result", text, re.M)) == 12
    for term in ("Historical reference", "retained historical reference", "not a fresh", "guided practice", "Try It Yourself", "Limitations", "Reading", "training", "development", "final", "71/96", "73/96", "72/96", "56", "32 random", "Unused copy", "Whole donor", "Shuffled", "Explicit counterfactual", "PLAY_ANGLE_DEGREES", "PLAY_LAYER", "PLAY_BASE_INDEX", "72 training", "48 development", "96"):
        assert term in text, term
    assert "from arena_ext" not in master
    assert len(re.findall(r'^# ! TAGS: \[html\]', master, re.M)) == 12
    forecasts = [c for c in notebook.cells if c.cell_type == "markdown" and c.source.startswith("## Forecast before the result")]
    assert all("confidence (0–100%)" in c.source and "Record whether you have already seen it" in c.source for c in forecasts)
    for key, marker in (("learned-subspace", "fits = {}"), ("site-selection", "site_results = []"), ("rank-selection", "rank_fits, rank_results, rank_logits"), ("frozen-final-role-transfer", "final_fits, final_bases = {}, {}"), ("actual-token-distributions", "fixed_examples = [")):
        assert text.index("Forecast before the result — " + key) < text.index(marker)
    assert "not show that" in text and "outperforms copying the whole residual vector" in text


def test_every_original_explanatory_line_is_preserved():
    before = json.loads((EVIDENCE / "before.json").read_text())
    with tarfile.open(resolve_historical_archive(ROOT / before["archive"], before["archive_sha256"])) as saved:
        old = json.load(saved.extractfile(str((SECTION / f"{STEM}_solutions.ipynb").relative_to(ROOT))))
    current = "\n".join(c.source for c in verified_pre_wrap_notebook(SECTION / f'{STEM}_exercises.ipynb').cells if c.cell_type == "markdown")
    current_lines = {line.strip() for line in current.splitlines() if line.strip()}
    for cell in old["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = re.sub(r"<details><summary>Solution</summary>\s*```python\n.*?```\s*</details>\s*", "", source(cell), flags=re.S)
        for line in text.splitlines():
            line = line.strip()
            if not line or line == "## Setup" or line.startswith("# [11.3]"):
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
os.environ['ARENA_DAS_OUTPUT_DIRECTORY'] = str(Path.cwd() / 'must_not_exist')
def forbidden(*args, **kwargs):
    raise AssertionError('Import ran a lesson side effect')
torch.set_num_threads = forbidden
plt.style.use = forbidden
plt.subplots = forbidden
before = torch.cuda.is_initialized()
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    module = importlib.import_module('chapter11_representation_geometry.exercises.part3_causal_abstraction_das.solutions')
assert not captured.getvalue(), captured.getvalue()
assert not (Path.cwd() / 'must_not_exist').exists()
assert not (Path.cwd() / 'das_outputs').exists()
assert torch.cuda.is_initialized() == before
assert 'transformers' not in sys.modules
for name in ('high_level_interchange', 'subspace_interchange', 'fit_alignment', 'continue_native', 'fit_role_probe', 'fit_minibatch_alignment', 'evaluate_native_pairs', 'train_native_alignment', 'run_smoke_test', 'run_gpu_test'):
    assert callable(getattr(module, name)), name
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "Import safety passed"


ORACLES = [
    "test_high_level_mechanisms",
    "test_projector_matches_explicit_rotate_swap_unrotate",
    "test_orthonormal_basis_limits_and_invalid_columns",
    "test_disjoint_interventions_commute_and_same_variable_overwrites",
    "test_qr_and_independent_finite_difference_gradient",
    "test_full_vocabulary_loss_and_empty_iia",
    "test_predictive_unused_copy_fails_causal_interchange",
    "test_alignment_optimization_changes_subspace_and_freezes_model",
]


@pytest.mark.parametrize("oracle", ORACLES)
def test_generated_visible_methods_against_independent_cpu_oracles(methods, oracle):
    tests = importlib.import_module(MODULE + ".das_methods_tests")
    torch = importlib.import_module("torch")
    previous_dtype, previous_threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        getattr(tests, oracle)(methods)
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("oracle,method", [("test_role_probe", "fit_role_probe"), ("test_minibatch_alignment", "fit_minibatch_alignment")])
def test_generated_training_methods_against_independent_oracles(methods, oracle, method):
    tests = importlib.import_module(MODULE + ".das_training_tests")
    torch = importlib.import_module("torch")
    previous_dtype, previous_threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        getattr(tests, oracle)(getattr(methods, method))
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


def test_frozen_ioi_data_contract_and_visible_role_targets(methods):
    torch = importlib.import_module("torch")
    checks = importlib.import_module(MODULE + ".das_training_tests")
    checks.test_ioi_intervention_contract()
    names = torch.tensor([[11, 29], [11, 29], [37, 53], [37, 53]])
    roles = torch.tensor([0, 1, 0, 1])
    assert methods.role_interchange_targets(names, roles).tolist() == [29, 11, 53, 37]
    assert torch.equal(names, torch.tensor([[11, 29], [11, 29], [37, 53], [37, 53]]))


def test_all_legacy_files_remain_byte_preserved_except_replaced_publication():
    before = json.loads((EVIDENCE / "before.json").read_text())
    replaced = {str((SECTION / filename).relative_to(ROOT)) for filename in (f"{STEM}_exercises.ipynb", f"{STEM}_solutions.ipynb", "solutions.py")}
    replaced.add("chapter11_representation_geometry/instructions/pages/03_[11.3]_Causal_Abstraction_and_Distributed_Alignment_Search.md")
    for name, expected in before["files"].items():
        if name not in replaced:
            assert sha((ROOT / name).read_bytes()) == expected, name


def test_guided_case_proposal_is_frozen_source_backed_and_has_no_filled_answers():
    proposal = json.loads((EVIDENCE / "guided_case_proposal.json").read_text())
    fingerprint = proposal.pop("configuration_sha256")
    assert sha(json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode()) == fingerprint
    for filename, expected in proposal["artifact_sha256"].items():
        assert sha(historical_bytes(ROOT / filename, expected)) == expected, filename
    summary = json.loads((ROOT / proposal["reference_artifact"]).read_text())
    for label in ("primary", "transfer"):
        value = summary
        for key in proposal[label]["reference_json_path"]:
            value = value[key]
        assert 0 <= value <= 1
        subgroup = summary["final_metrics"]["DAS/seed0"][proposal[label]["reference_json_path"][-2]]
        assert subgroup["n"] == proposal[label]["denominator"] == 96
    assert proposal["learner_exposure"] == "seen_or_unknown"
    assert proposal["execution_kind"] == "retained_historical_reference"
    assert "overlapping prompts" in proposal["transfer"]["boundary"]
    for response in proposal["learner_response_fields"]:
        assert response not in proposal


def test_checked_in_completed_execution_archive_matches_canonical_scientific_cells():
    """External caches are absent; the checked-in completed notebook is inspectable."""
    with tarfile.open(SECTION / "expected_outputs/full_own_answer_v3/execution.tar.gz") as stored:
        recorded = nbformat.reads(stored.extractfile("completed_learner.ipynb").read().decode(), as_version=4)
    current = read_notebook("solutions")
    assert len(code_cells(recorded)) == len(code_cells(current)) == 32
    for historical, canonical in zip(code_cells(recorded), code_cells(current), strict=True):
        assert ast.dump(ast.parse(historical.source)) == ast.dump(ast.parse(canonical.source))
        assert historical.outputs == canonical.outputs
        assert historical.execution_count == canonical.execution_count
