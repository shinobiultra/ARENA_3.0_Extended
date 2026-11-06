"""Gemma preservation, exact primitive checks and subset receipts; no model runs."""
from __future__ import annotations

import ast
import base64
import gzip
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

ROOT = Path(__file__).resolve().parents[2]
SECTION = ROOT / "chapter5_modern_architectures/exercises/part1_gemma_from_scratch"
EVIDENCE = ROOT / "docs/mastery/evidence/gemma_master_migration"
MASTER = ROOT / "infrastructure/chapters/chapter5_modern_architectures/master_5_1.py"
PAGE = ROOT / "chapter5_modern_architectures/instructions/pages/01_[5.1]_Gemma_from_Scratch.md"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(cell):
    return "".join(cell["source"])


def codes(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def read(kind):
    return nbformat.read(SECTION / f"5.1_Gemma_from_Scratch_{kind}.ipynb", as_version=4)


def archived(path):
    with tarfile.open(EVIDENCE / "published_before_master.tar.gz") as saved:
        return saved.extractfile(path.relative_to(ROOT).as_posix()).read()


def original(kind):
    return json.loads(archived(SECTION / f"5.1_Gemma_from_Scratch_{kind}.ipynb"))


def historical_setup_source(text):
    """Invert exactly one hash-pinned setup import compatibility route, never methods."""
    proof = json.loads((ROOT / "docs/mastery/evidence/gemma_reference_route_20261019/import_bridge.json").read_text())
    if proof["after_import"] not in text:
        return text
    assert sha(text.encode()) == proof["after_setup_source_sha256"]
    assert text.count(proof["after_import"]) == 1
    restored = text.replace(proof["after_import"], proof["before_import"])
    assert sha(restored.encode()) == proof["before_setup_source_sha256"]
    return restored


def normalized(text):
    return historical_setup_source(text).replace('assets_dir = section_dir / "media"', 'assets_dir = root_dir / chapter / "instructions" / "assets"').replace(
        'part1_gemma_from_scratch.legacy_solutions"', 'part1_gemma_from_scratch.solutions"')


def test_full_archive_and_historical_report_preserved():
    from scripts.evidence_bundles import validate_review_bundle

    before = json.loads((EVIDENCE / "before.json").read_text())
    archive = ROOT / before["archive"]
    assert sha(archive.read_bytes()) == before["archive_sha256"]
    assert len(validate_review_bundle(archive)["members"]) == 18
    with tarfile.open(archive) as saved:
        assert set(saved.getnames()) == set(before["files"])
        for name, record in before["files"].items():
            data = saved.extractfile(name).read()
            assert len(data) == record["size_bytes"] and sha(data) == record["sha256"]
    assert len(before["report_input_physical_checks"]) == 16
    assert all(row["matches"] for row in before["report_input_physical_checks"])
    assert (SECTION / "legacy_solutions.py").read_bytes() == archived(SECTION / "solutions.py")
    for name in ("verification_report.json", "artifacts.lock.yml", "verification_report.schema.json", "tests.py", "utils.py"):
        assert (SECTION / name).read_bytes() == archived(SECTION / name)


def test_all_historical_code_outputs_counts_and_prose_preserved():
    old = original("solutions")
    oldnorm = nbformat.reads(json.dumps(old), as_version=4)
    current = read("solutions")
    receipt = json.loads((EVIDENCE / "output_parity.json").read_text())
    assert receipt["historical_code_cells"] == 16 and receipt["new_code_cells"] == 5
    assert len(codes(current)) == 21 and not receipt["fresh_model_execution"]
    for before, normalized_cell, row in zip(codes(old), codes(oldnorm), receipt["cells"], strict=True):
        after = codes(current)[row["canonical_code_ordinal"]]
        assert ast.dump(ast.parse(source(before))) == ast.dump(ast.parse(normalized(after.source)))
        assert [line for line in source(before).splitlines() if line.strip()] == [line for line in normalized(after.source).splitlines() if line.strip()]
        assert after.outputs == normalized_cell.outputs
        assert after.execution_count == before["execution_count"] == row["execution_count"]
        assert sha(historical_setup_source(after.source).encode()) == row["generated_source_sha256"]
        assert after.metadata["arena_evidence"]["kind"] == "retained_historical_reference"
    assert sum(len(cell["outputs"]) for cell in codes(old)) == 18
    text = "\n".join(cell.source for cell in current.cells if cell.cell_type == "markdown")
    compact = " ".join(text.split())
    for index, cell in enumerate(old["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        historical = source(cell)
        if index == 0:
            historical = historical.split("\n", 1)[1].replace("## Learning Objectives", "## What you will implement")
        historical = historical.replace("<summary>Solution</summary>", "<summary>Historical worked formulation</summary>")
        historical = historical.replace("../../instructions/assets/", "media/")
        historical = re.sub(r'> \*\*Difficulty:\*\* (\w+)[^\n]*\n> \*\*Suggested time:\*\* ([^\n]+)',
            lambda m: '> ```yaml\n> Difficulty: ' + {"easy": "🔴⚪⚪⚪⚪", "medium": "🔴🔴🔴⚪⚪", "hard": "🔴🔴🔴🔴⚪"}[m[1]]
            + '\n> Importance: 🔵🔵🔵🔵⚪\n>\n> You should spend ' + m[2] + ' on this exercise.\n> ```', historical)
        if index == 24:
            historical = historical.replace('<details><summary>Interpreting the result</summary>', '</details>\n\n<details><summary>Interpreting the result</summary>', 1)
        assert " ".join(historical.split()) in compact, index


def test_student_methods_complete_from_their_own_answers():
    exercises, solutions = read("exercises"), read("solutions")
    nbformat.validate(exercises)
    nbformat.validate(solutions)
    completed = []
    stub_count = 0
    for index, cell in enumerate(exercises.cells):
        if cell.cell_type != "code":
            continue
        text = cell.source
        if "raise NotImplementedError" in text:
            stub_count += text.count("raise NotImplementedError")
            answer = exercises.cells[index + 1]
            assert answer.cell_type == "markdown" and "<summary>Solution</summary>" in answer.source
            blocks = re.findall(r"```python\n(.*?)```", answer.source, re.S)
            assert len(blocks) == 1
            text = blocks[0]
            assert "raise NotImplementedError" not in text
            assert any(marker in text for marker in (
                "tests.test_", "assert observed_first_divergence", "Projection checks passed", "GELU formula check passed",
            ))
        completed.append(ast.dump(ast.parse(text)))
    assert stub_count == 17
    assert completed == [ast.dump(ast.parse(cell.source)) for cell in codes(solutions)]


def test_forecasts_truthful_scope_dropdown_and_media_routes():
    text = "\n".join(cell.source for cell in read("exercises").cells if cell.cell_type == "markdown")
    assert text.count("### Forecast") == text.count("### Reflect") == 18
    assert text.count("<details>") == text.count("</details>") == 58
    for term in ("substantially duplicates", "guided practice", "unseen", "confidence", "exposure", "Gemma3-270m", "SmolLM2", "outside", "not a fresh", "1e-8", "gelu_pytorch_tanh", "F.silu", "coordinate permutation", "not established"):
        assert term in text
    page = PAGE.read_text()
    for name in ("gemma_decoder_block.svg", "gemma_historical_cell_23_output_0.png", "gemma_historical_cell_26_output_0.png", "gemma_cpu_cache_mask.png", "gemma_cpu_convention_comparison.png"):
        assert f'../../exercises/part1_gemma_from_scratch/media/{name}' in page
        assert (SECTION / "media" / name).is_file()
    assert PAGE.with_suffix(".py").exists()
    for path in (MASTER, PAGE):
        assert max(map(len, path.read_text().splitlines())) <= 240


def test_all_historical_images_preserved_byte_for_byte():
    for name in ("gemma_parity_signature.png", "gemma_decoder_block.svg"):
        assert (SECTION / "media" / name).read_bytes() == archived(ROOT / "chapter5_modern_architectures/instructions/assets" / name)
    count = 0
    for index, cell in enumerate(original("solutions")["cells"]):
        for ordinal, output in enumerate(cell.get("outputs", [])):
            image = output.get("data", {}).get("image/png")
            if image:
                count += 1
                assert (SECTION / "media" / f"gemma_historical_cell_{index}_output_{ordinal}.png").read_bytes() == base64.b64decode("".join(image))
    assert count == 2


def test_cpu_subset_is_real_and_bound_to_only_selected_cells():
    receipt = json.loads((EVIDENCE / "cpu_subset.json").read_text())
    executed = nbformat.reads(gzip.decompress((EVIDENCE / "cpu_subset_executed.ipynb.gz").read_bytes()).decode(), as_version=4)
    assert sha((EVIDENCE / "cpu_subset_executed.ipynb.gz").read_bytes()) == receipt["executed_subset_sha256"]
    assert receipt["execution_kind"] == "fresh_cpu_primitive_subset"
    assert receipt["canonical_code_ordinals"] == [0, 2, 3, 4, 14]
    assert receipt["mask_cases_checked"] == 24 and receipt["default_wrong_bits"] == 9
    assert receipt["default_allowed_counts"] == [3, 3] and receipt["global_allowed_counts"] == [4, 5]
    assert receipt["threads"] == 1 and not receipt["cuda_initialized"]
    assert not any(receipt[key] for key in ("model_forward", "training", "downloads", "gpu", "whole_notebook_execution"))
    assert len(executed.cells) == 7
    current = codes(read("solutions"))
    for row, cell in zip(receipt["cells"], executed.cells[1:-1], strict=True):
        assert cell.source == historical_setup_source(current[row["canonical_code_ordinal"]].source)
        assert sha(cell.source.encode()) == row["source_sha256"]
        assert cell.execution_count == row["subset_execution_count"]
        assert all(output.output_type != "error" for output in cell.outputs)
    play = current[14]
    assert play.outputs == executed.cells[5].outputs and play.execution_count == executed.cells[5].execution_count
    assert play.metadata["arena_evidence"]["kind"] == "fresh_cpu_primitive_subset"
    assert sha((ROOT / receipt["figure"]).read_bytes()) == receipt["figure_sha256"]
    provenance = read("solutions").metadata["arena_execution_provenance"]
    assert provenance["fresh_model_execution"] is True
    assert provenance["historical_cells_freshly_executed"] is False
    assert read("solutions").metadata["arena_execution_provenance"]["canonical_notebook_freshly_executed"] is False


def test_import_does_not_execute_models_or_change_rng(tmp_path):
    script = '''import contextlib, hashlib, importlib, io, pathlib, sys
import torch as t
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root))
import chapter5_modern_architectures.exercises.part1_gemma_from_scratch.gemma_reference as reference
def forbidden(*args,**kwargs):raise AssertionError("Model execution or CUDA during import")
reference.GemmaForCausalLM.__init__=forbidden
t.cuda.init=forbidden
rng=t.get_rng_state().clone(); dtype=t.get_default_dtype(); threads=t.get_num_threads()
section=root/'chapter5_modern_architectures/exercises/part1_gemma_from_scratch'
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in section.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
before=hashes(); output=io.StringIO()
with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
    module=importlib.import_module('chapter5_modern_architectures.exercises.part1_gemma_from_scratch.solutions')
assert not module.MAIN and t.equal(rng,t.get_rng_state())
assert dtype==t.get_default_dtype() and threads==t.get_num_threads()
assert not output.getvalue(),output.getvalue()
assert before==hashes()
assert 'transformers' not in sys.modules
assert 'chapter5_modern_architectures.exercises.part1_gemma_from_scratch.legacy_solutions' not in sys.modules
print('inert import passed')
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_exact_mask_and_negative_control_without_model():
    import torch as t

    lesson = importlib.import_module("chapter5_modern_architectures.exercises.part1_gemma_from_scratch.solutions")
    kwargs = dict(query_length=2, key_length=5, sliding_window=3, device=t.device("cpu"))
    correct = lesson.build_causal_attention_mask(past_length=3, **kwargs)
    expected = t.tensor([[[[False, True, True, True, False], [False, False, True, True, True]]]])
    assert t.equal(correct, expected)
    wrong = lesson.build_causal_attention_mask(past_length=0, **kwargs)
    with pytest.raises(AssertionError):
        t.testing.assert_close(wrong, expected)
    assert int((correct != wrong).sum()) == 9


def test_release_and_compatibility_hooks_remain_lazy():
    tree = ast.parse((SECTION / "solutions.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    authoring = json.loads((EVIDENCE / "authoring.json").read_text())
    for name in ["run_gpu_test", "run_full_experiment", *authoring["compatibility_helpers"]]:
        assert name in functions
    assert 'part1_gemma_from_scratch.legacy_solutions' in ast.get_source_segment((SECTION / "solutions.py").read_text(), functions["_release_solution_module"])
    assert (SECTION / "verification_report.json").read_bytes() == archived(SECTION / "verification_report.json")


def test_convention_probe_receipt_binds_actual_new_cells_and_fixed_gates():
    receipt = json.loads((EVIDENCE / "convention_execution.json").read_text())
    protocol_bytes = (EVIDENCE / "convention_protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    review = json.loads((EVIDENCE / "convention_pre_run_review.json").read_text())
    assert receipt["protocol_sha256"] == review["protocol_sha256"] == sha(protocol_bytes)
    assert review["cleared_for_single_run"] is True
    assert protocol["gates"]["corrected_max_abs_lte"] == 5e-6
    assert protocol["gates"]["corrected_mse_lte"] == 1e-12
    assert protocol["config"]["initializer_range"] == 0.02 and protocol["seed"] == 5101
    assert protocol["runtime"]["hard_timeout_seconds"] == 90
    assert sha((EVIDENCE / "convention_frozen_master.py").read_bytes()) == protocol["files"][MASTER.relative_to(ROOT).as_posix()]
    frozen_notebook = gzip.decompress((EVIDENCE / "convention_frozen_canonical_input.ipynb.gz").read_bytes())
    assert sha(frozen_notebook) == protocol["files"][(SECTION / "5.1_Gemma_from_Scratch_solutions.ipynb").relative_to(ROOT).as_posix()]
    assert receipt["runtime"]["model_forward_calls"] == {"reference": 1, "learner": 4}
    assert receipt["runtime"]["threads"] == receipt["runtime"]["interop_threads"] == 1
    assert not receipt["runtime"]["cuda_initialized"] and not receipt["whole_notebook_execution"]
    result = receipt["result"]
    assert result["input_ids"] == protocol["input_ids"]
    assert result["state_dict_keys"] == len(result["state_dict_sha256"]) == 21
    assert len(result["rows"]) == 4 and result["corrected_passed"]
    corrected = result["rows"][-1]
    assert corrected["max_abs"] <= 5e-6 and corrected["mse"] <= 1e-12
    executed_bytes = (EVIDENCE / "convention_execution.ipynb.gz").read_bytes()
    assert sha(executed_bytes) == receipt["execution_notebook_sha256"]
    executed = nbformat.reads(gzip.decompress(executed_bytes).decode(), as_version=4)
    current = codes(read("solutions"))
    for row, cell in zip(receipt["cells"], executed.cells[1:-1], strict=True):
        actual = current[row["canonical_code_ordinal"]]
        assert actual.source == cell.source and sha(cell.source.encode()) == row["source_sha256"]
        assert actual.outputs == cell.outputs
        assert actual.execution_count == cell.execution_count == row["subset_execution_count"]
        assert actual.metadata["arena_evidence"]["kind"] == "fresh_cpu_convention_comparison"
    assert sha((ROOT / receipt["figure"]).read_bytes()) == receipt["figure_sha256"]


def test_guided_proposal_schema_and_visible_method_identity():
    from mastery_hub.units.extension_predictions.lesson import validate_case

    proposal = json.loads((EVIDENCE / "proposed_case.json").read_text())
    validate_case(proposal)
    assert proposal["section"] == "5.1" and proposal["status"] == "guided_practice"
    assert proposal["data_role"] == "monitored"
    assert proposal["prediction_type"] == "numeric" and proposal["transfer_prediction_type"] == "binary"
    assert "GELU-only" in proposal["transfer"] and "already recorded" in proposal["transfer"]
    assert "868 logits" in proposal["question"]
    # evidence_detail is rendered before commitment, so it must not disclose
    # the target or measured control outcomes even for guided practice.
    assert "1.862645" not in proposal["evidence_detail"]
    assert "3.997058" not in proposal["evidence_detail"]
    assert "all four have" not in proposal["evidence_detail"]
    for key in ("master_path", "exercise_path", "solutions_path"):
        assert (ROOT / proposal[key]).is_file()
    # A hub case must name methods learners can actually find and implement.
    methods = set()
    for cell in codes(read("solutions")):
        for node in ast.parse(cell.source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                methods.add(node.name)
    assert set(proposal["method_functions"]) <= methods


def test_convention_basis_inverse_is_not_assumed_and_input_is_unmodified():
    import torch as t

    lesson = importlib.import_module("chapter5_modern_architectures.exercises.part1_gemma_from_scratch.solutions")
    weights = t.arange(48).reshape(16, 3)
    before = weights.clone()
    converted = lesson.hf_split_to_interleaved_projection(weights, 2)
    t.testing.assert_close(weights, before)
    assert not t.equal(lesson.hf_split_to_interleaved_projection(converted, 2), weights)
    for weight, heads in [(t.zeros(3, 2), 1), (t.zeros(4), 1), (t.zeros(4, 2), 0)]:
        with pytest.raises(ValueError):
            lesson.hf_split_to_interleaved_projection(weight, heads)


def test_local_reference_copy_and_exact_setup_route_bridge():
    proof = json.loads((ROOT / "docs/mastery/evidence/gemma_reference_route_20261019/import_bridge.json").read_text())
    freeze = json.loads((ROOT / "docs/mastery/evidence/gemma_reference_route_20261019/before.json").read_text())
    assert proof["allowed_canonical_code_ordinal"] == 0
    assert (ROOT / proof["reference_copy"]).read_bytes() == (ROOT / proof["reference_source"]).read_bytes()
    assert sha((ROOT / proof["reference_copy"]).read_bytes()) == proof["reference_sha256"] == freeze[proof["reference_source"]]["sha256"]
    assert sha((SECTION / "legacy_solutions.py").read_bytes()) == freeze[(SECTION / "legacy_solutions.py").relative_to(ROOT).as_posix()]["sha256"]
    assert "arena_ext" not in MASTER.read_text()
    old = gzip.decompress((ROOT / freeze[MASTER.relative_to(ROOT).as_posix()]["snapshot"]).read_bytes()).decode()
    assert MASTER.read_text().replace(proof["after_import"], proof["before_import"]) == old
    # Mutation of a student method cannot pass the setup-route bridge.
    setup = codes(read("solutions"))[0].source
    after = setup.replace(proof["before_import"], proof["after_import"])
    assert sha(after.encode()) == proof["after_setup_source_sha256"]
    assert sha(historical_setup_source(after).encode()) == proof["before_setup_source_sha256"]
    with pytest.raises(AssertionError):
        historical_setup_source(after + "\n# unrelated mutation")
