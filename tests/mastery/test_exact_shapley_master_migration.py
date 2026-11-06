"""Exact-game preservation and tiny CPU oracles; no neural training or CUDA."""
from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "chapter16_shapley_attribution_baselines"
OLD = CHAPTER / "exercises/part1_exact_shapley_ground_truth_games"
SECTION = CHAPTER / "exercises/part1_exact_shapley"
EVIDENCE = ROOT / "docs/mastery/evidence/exact_shapley_master_migration"
MASTER = ROOT / "infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_1.py"
MODULE = "chapter16_shapley_attribution_baselines.exercises.part1_exact_shapley"
OLD_STEM = "16.1_Exact_Shapley_on_Ground_Truth_Games"
STEM = "16.1_Exact_Shapley_on_Ground-Truth_Games"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(cell):
    return "".join(cell["source"])


def codes(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def read(kind):
    return nbformat.read(SECTION / f"{STEM}_{kind}.ipynb", as_version=4)


def old(kind):
    return json.loads((OLD / f"{OLD_STEM}_{kind}.ipynb").read_text())


def normalize_route(text, ordinal):
    if ordinal != 0:
        return text
    return text.replace(
        'section = "part1_exact_shapley"',
        'section = "part1_exact_shapley_ground_truth_games"',
    ).replace(
        'assets_dir = exercises_dir / section / "media"',
        'assets_dir = root_dir / chapter / "instructions" / "assets"',
    ).replace(
        "import part1_exact_shapley.tests as tests",
        "import part1_exact_shapley_ground_truth_games.tests as tests",
    )


def test_pre_edit_archive_and_all_old_files_are_physically_unchanged():
    from scripts.evidence_bundles import validate_review_bundle

    before = json.loads((EVIDENCE / "before.json").read_text())
    archive = ROOT / before["archive"]
    assert sha(archive.read_bytes()) == before["archive_sha256"]
    contract = validate_review_bundle(archive)
    assert len(contract["members"]) == 20
    assert before["nested_original_receipts"] == []
    with tarfile.open(archive) as saved:
        for name, expected in before["files"].items():
            assert sha(saved.extractfile(name).read()) == expected, name
    for name, expected in before["legacy_inventory"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    assert len(before["legacy_inventory"]) == 16
    for name in ("artifacts.lock.yml", "verification_report.json", "verification_report.schema.json"):
        assert (SECTION / name).read_bytes() == (OLD / name).read_bytes()
    assert json.loads((SECTION / "verification_report.json").read_text())["report_inputs"] == before["report_inputs"]


def test_all_11_historical_code_asts_counts_and_outputs_survive():
    original = old("solutions")
    normalized = nbformat.reads(json.dumps(original), as_version=4)
    current = read("solutions")
    receipt = json.loads((EVIDENCE / "output_parity.json").read_text())
    assert receipt["execution_kind"] == "retained_historical_reference"
    assert receipt["fresh_model_execution"] is False
    assert receipt["canonical_notebook_freshly_executed"] is False
    assert len(original["cells"]) == 24
    assert len(codes(original)) == len(receipt["cells"]) == 11
    assert len(codes(current)) == 12
    for ordinal, (a, b, record) in enumerate(zip(codes(original), codes(current)[:11], receipt["cells"], strict=True)):
        previous, routed = source(a), source(b)
        adapted = normalize_route(routed, ordinal)
        assert ast.dump(ast.parse(previous)) == ast.dump(ast.parse(adapted)), ordinal
        assert [line for line in previous.splitlines() if line.strip()] == [
            line for line in adapted.splitlines() if line.strip()
        ], ordinal
        assert record["published_source_sha256"] == sha(previous.encode())
        assert record["generated_source_sha256"] == sha(routed.encode())
        assert record["normalized_ast_sha256"] == sha(ast.dump(ast.parse(adapted)).encode())
        assert record["outputs_sha256"] == sha(json.dumps(a.get("outputs", []), sort_keys=True).encode())
        assert b.outputs == codes(normalized)[ordinal].outputs
        assert b.execution_count == a["execution_count"] == record["execution_count"]
        assert bool(record["named_adaptation"]) == (ordinal == 0)
    assert sum(bool(c.outputs) for c in codes(current)[:11]) == 10
    assert sum(len(c.outputs) for c in codes(current)[:11]) == 11
    assert codes(current)[11].execution_count is None and not codes(current)[11].outputs
    assert ast.dump(ast.parse(codes(current)[11].source)) == ast.dump(ast.parse(source(codes(old("exercises"))[11])))


def test_all_student_functions_complete_from_their_own_generated_answers():
    exercises, solutions = read("exercises"), read("solutions")
    nbformat.validate(exercises)
    nbformat.validate(solutions)
    completed, stubs = [], set()
    for index, cell in enumerate(exercises.cells):
        if cell.cell_type != "code":
            continue
        text = cell.source
        if "raise NotImplementedError" in text:
            following = []
            for other in exercises.cells[index + 1:]:
                if other.cell_type == "code":
                    break
                following.append(other.source)
            answers = re.findall(r"<summary>Solution</summary>(.*?)</details>", "\n".join(following), re.S)
            assert len(answers) == 1
            answer = "\n\n".join(re.findall(r"```python\n(.*?)```", answers[0], re.S))
            definitions = {
                n.name: ast.get_source_segment(answer, n)
                for n in ast.parse(answer).body if isinstance(n, ast.FunctionDef)
            }
            lines = text.splitlines(True)
            for node in reversed(ast.parse(text).body):
                if isinstance(node, ast.FunctionDef) and "raise NotImplementedError" in ast.get_source_segment(text, node):
                    stubs.add(node.name)
                    lines[node.lineno - 1:node.end_lineno] = [definitions[node.name] + "\n"]
            text = "".join(lines)
            assert "raise NotImplementedError" not in text
        completed.append(ast.dump(ast.parse(text)))
    assert completed == [ast.dump(ast.parse(c.source)) for c in codes(solutions)]
    assert stubs == {
        "all_coalitions", "normalize_coalition_values", "coalition_values_from_function",
        "game_from_dividends", "shapley_from_dividends", "marginal_contribution_rows",
        "exact_shapley_values", "additive_game", "conjunction_game", "shapley_efficiency_report",
        "permutation_shapley_values", "permutation_parity_report", "leave_one_out_values",
        "interaction_gap_report", "interaction_scale_sweep",
    }


def test_teaching_explanations_historical_formulations_and_specific_forecasts_survive():
    text = "\n".join(c.source for c in read("exercises").cells if c.cell_type == "markdown")
    compact = " ".join(text.split())
    for cell in old("exercises")["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        previous = source(cell).replace("<summary>Solution</summary>", "<summary>Historical worked formulation</summary>")
        for line in previous.splitlines():
            stripped = line.strip()
            if (not stripped or stripped.startswith(("# [16.1]", "<img", "> Difficulty:", "> Importance:"))
                    or stripped == "## Learning objectives"):
                continue
            assert " ".join(stripped.split()) in compact, stripped
    assert text.count("### Forecast —") == text.count("### Reflect —") == 9
    assert text.count("### Exercise -") == 7
    for phrase in ("historical reference evidence", "not a fresh execution", "guided practice", "confidence",
                   "causal reasoning", "exposure status", "dummy", "24 orderings", "additive", "-0.5",
                   "0.25", "Try It Yourself", "not a negative bar"):
        assert phrase.lower() in compact.lower(), phrase
    assert "arena_ext" not in MASTER.read_text()
    assert "arena_ext" not in (SECTION / "tests.py").read_text()


def test_saved_asset_inline_png_and_supporting_diagrams_remain_byte_identical():
    before = json.loads((EVIDENCE / "before.json").read_text())
    for name in ("exact_shapley_ground_truth_signature.png", "exact_shapley_signature_result.svg",
                 "exact_shapley_validation_loop.svg"):
        original = CHAPTER / "instructions/assets" / name
        assert sha((SECTION / "media" / name).read_bytes()) == before["files"][original.relative_to(ROOT).as_posix()]
    images = [out["data"]["image/png"] for cell in codes(old("solutions")) for out in cell.get("outputs", [])
              if "image/png" in out.get("data", {})]
    assert len(images) == 1
    expected = base64.b64decode("".join(images[0]))
    assert (SECTION / "media/exact_shapley_reference_output.png").read_bytes() == expected
    html = "\n".join("".join(out.get("data", {}).get("text/html", ""))
                     for cell in codes(read("exercises")) for out in cell.outputs)
    encoded = re.findall(r'data:image/png;base64,([^\"]+)', html)
    assert len(encoded) == 1 and base64.b64decode(encoded[0]) == expected


def test_generated_module_import_is_inert_outside_repository(tmp_path):
    script = '''
import contextlib, importlib, io, os, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
def forbidden(*args, **kwargs):
    raise AssertionError("Import executed a study, changed gradient state or queried CUDA")
torch.cuda.is_available = forbidden
torch.cuda.empty_cache = forbidden
torch.set_grad_enabled = forbidden
plt.subplots = forbidden
state = torch.random.get_rng_state().clone()
before = np.random.get_state()
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    m = importlib.import_module('chapter16_shapley_attribution_baselines.exercises.part1_exact_shapley.solutions')
assert not captured.getvalue() and not list(Path.cwd().iterdir())
assert torch.equal(state, torch.random.get_rng_state())
after = np.random.get_state()
assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
assert 'arena_ext' not in sys.modules and 'transformers' not in sys.modules
assert not m.MAIN
for name in ('exact_shapley_values', 'interaction_scale_sweep', 'run_gpu_test', 'run_full_experiment'):
    assert callable(getattr(m, name))
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "Import safety passed"


ORACLES = [
    ("test_all_coalitions_enumerates_the_power_set", ["all_coalitions"]),
    ("test_complete_table_normalization", ["all_coalitions", "normalize_coalition_values", "coalition_values_from_function"]),
    ("test_dividend_game_and_analytic_oracle", ["game_from_dividends", "shapley_from_dividends"]),
    ("test_weighted_marginal_rows", ["marginal_contribution_rows"]),
    ("test_additive_game_and_exact_shapley_recover_weights", ["additive_game", "exact_shapley_values"]),
    ("test_exact_shapley_requires_a_complete_coalition_table", ["exact_shapley_values"]),
    ("test_conjunction_game_splits_symmetric_credit_and_checks_efficiency", ["conjunction_game", "exact_shapley_values", "shapley_efficiency_report"]),
    ("test_permutation_parity_report_matches_exact_formula", ["conjunction_game", "permutation_parity_report"]),
    ("test_interaction_gap_report_catches_leave_one_out_overcount", ["conjunction_game", "interaction_gap_report"]),
    ("test_interaction_scale_sweep", ["interaction_scale_sweep"]),
]


@pytest.mark.parametrize("oracle,functions", ORACLES)
def test_visible_student_methods_with_local_independent_oracles(oracle, functions):
    methods = importlib.import_module(MODULE + ".solutions")
    tests = importlib.import_module(MODULE + ".tests")
    getattr(tests, oracle)(*[getattr(methods, name) for name in functions])


def test_original_cpu_play_across_signed_scales_and_player_choices():
    import torch

    methods = importlib.import_module(MODULE + ".solutions")
    play = codes(read("solutions"))[10].source
    for scale in (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0):
        for player in range(4):
            namespace = dict(vars(methods))
            changed = play.replace("PLAY_INTERACTION_SCALE = 0.5", f"PLAY_INTERACTION_SCALE = {scale}")
            changed = changed.replace("PLAY_PLAYER = 1", f"PLAY_PLAYER = {player}")
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(changed, "<finite-game-play>", "exec"), namespace)
            expected = torch.tensor([0.8 + 1.1 * scale, -0.2 + 0.8 * scale,
                                     0.4 + 0.65 * scale, 0.1 + 0.45 * scale], dtype=torch.float64)
            torch.testing.assert_close(namespace["play_shapley"], expected, atol=1e-12, rtol=0)
            assert abs(float((namespace["play_loo"] - namespace["play_shapley"]).sum()) - 4.5 * scale) < 1e-12
            assert len(namespace["play_rows"]) == 8
            assert abs(sum(weight for _, _, weight in namespace["play_rows"]) - 1.0) < 1e-12


def test_oracles_reject_broken_student_answers():
    import torch

    methods = importlib.import_module(MODULE + ".solutions")
    tests = importlib.import_module(MODULE + ".tests")
    with pytest.raises(AssertionError):
        tests.test_dividend_game_and_analytic_oracle(methods.game_from_dividends,
                                                   lambda *args: torch.zeros(4, dtype=torch.float64))
    with pytest.raises(AssertionError):
        tests.test_interaction_scale_sweep(lambda scales: {
            "shapley_efficiency_error": torch.zeros_like(scales),
            "oracle_max_error": torch.zeros_like(scales),
            "leave_one_out_overcount": torch.zeros_like(scales),
        })
