"""Full 15.3 preservation and independent tiny CPU fixtures; no trained-model runs."""
from __future__ import annotations
from scripts.verified_publication_pages import historical_bytes

import ast
import base64
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
SECTION = ROOT / "chapter15_peft_misalignment/exercises/part3_qlora_from_scratch"
EVIDENCE = ROOT / "docs/mastery/evidence/qlora_master_migration"
MASTER = ROOT / "infrastructure/chapters/chapter15_peft_misalignment/master_15_3.py"
MODULE = "chapter15_peft_misalignment.exercises.part3_qlora_from_scratch"
STEM = "15.3_QLoRA_from_Scratch"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(cell):
    return "".join(cell["source"])


def code_cells(notebook):
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def notebook(kind):
    return nbformat.read(SECTION / f"{STEM}_{kind}.ipynb", as_version=4)


def archived(name):
    before = json.loads((EVIDENCE / "before.json").read_text())
    with tarfile.open(ROOT / before["archive"]) as saved:
        return saved.extractfile(name).read()


def old_notebook(kind):
    return json.loads(archived(str((SECTION / f"{STEM}_{kind}.ipynb").relative_to(ROOT))))


@pytest.fixture(scope="module")
def methods():
    # Reproduce the historical float32 process default, then restore ambient state.
    import torch
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        return importlib.import_module(MODULE + ".solutions")
    finally:
        torch.set_default_dtype(previous)


@pytest.fixture
def cpu_state():
    import torch
    previous_dtype, previous_threads = torch.get_default_dtype(), torch.get_num_threads()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        yield torch
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(previous_threads)


def test_archive_and_original_report_inputs_are_exact():
    before = json.loads((EVIDENCE / "before.json").read_text())
    archive = ROOT / before["archive"]
    assert sha(archive.read_bytes()) == before["archive_sha256"]
    assert all(before["source_input_matches"].values())
    with tarfile.open(archive) as saved:
        for name, expected in before["files"].items():
            assert sha(saved.extractfile(name).read()) == expected, name
        for name, expected in before["source_inputs"].items():
            assert sha(saved.extractfile(str((SECTION / name).relative_to(ROOT))).read()) == expected, name



def test_every_original_legacy_file_and_saved_instruction_asset_remains():
    before = json.loads((EVIDENCE / "before.json").read_text())
    replaced = {str((SECTION / p).relative_to(ROOT)) for p in (f"{STEM}_exercises.ipynb", f"{STEM}_solutions.ipynb", "solutions.py")}
    replaced.add("chapter15_peft_misalignment/instructions/pages/03_[15.3]_QLoRA_from_Scratch.md")
    for name, expected in before["files"].items():
        if name not in replaced and (name.startswith(str(SECTION.relative_to(ROOT)) + "/") or "/instructions/assets/" in name):
            assert sha((ROOT / name).read_bytes()) == expected, name
    old_module = str((SECTION / "solutions.py").relative_to(ROOT))
    assert (SECTION / "legacy_reference.py").read_bytes() == archived(old_module)


def test_all_twenty_one_scientific_cells_preserve_ast_lines_outputs_and_counts():
    old = old_notebook("solutions")
    normalized = nbformat.reads(json.dumps(old), as_version=4)
    current = notebook("solutions")
    parity = json.loads((EVIDENCE / "output_parity.json").read_text())
    assert parity["execution_kind"] == "retained_historical_reference"
    assert parity["fresh_model_execution"] is False
    assert len(code_cells(old)) == len(code_cells(current)) == len(parity["cells"]) == 21
    for ordinal, (historical, canonical, normalized_old, receipt) in enumerate(zip(code_cells(old), code_cells(current), code_cells(normalized), parity["cells"], strict=True)):
        a, b = source(historical), source(canonical)
        adapted = a.replace("import solutions as reference", "import legacy_reference as reference").replace("part3_qlora_from_scratch.solutions import (", "part3_qlora_from_scratch.legacy_reference import (")
        assert (a != adapted) == (ordinal == 0)
        assert [line for line in adapted.splitlines() if line.strip()] == [line for line in b.splitlines() if line.strip()]
        assert ast.dump(ast.parse(adapted)) == ast.dump(ast.parse(b))
        assert sha(a.encode()) == receipt["published_source_sha256"]
        assert sha(adapted.encode()) == receipt["published_route_adapted_source_sha256"]
        assert sha(b.encode()) == receipt["generated_source_sha256"]
        assert canonical.outputs == normalized_old.outputs
        assert canonical.execution_count == historical["execution_count"]
        assert sha(json.dumps(historical["outputs"], sort_keys=True).encode()) == receipt["outputs_sha256"]
    assert sum(c.execution_count is not None for c in code_cells(current)) == 21
    assert sum(bool(c.outputs) for c in code_cells(current)) == 14
    assert sum(len(c.outputs) for c in code_cells(current)) == 70


def test_four_canonical_pngs_are_exact_historical_notebook_output_bytes():
    old = old_notebook("solutions")
    figures = json.loads((EVIDENCE / "figures.json").read_text())
    indices = [36, 38, 41, 44]
    names = ["qlora_nf4_toy_signature.png", "qlora_real_signature_panel.png", "qlora_outlier_play.png", "qlora_precision_play.png"]
    for i, name in zip(indices, names, strict=True):
        images = [o["data"]["image/png"] for o in old["cells"][i]["outputs"] if "image/png" in o.get("data", {})]
        assert len(images) == 1
        data = base64.b64decode("".join(images[0]))
        assert (SECTION / "media" / name).read_bytes() == data
        assert sha(data) == figures[name]


def complete_answer(stub, answer):
    lines, answers = stub.splitlines(True), answer.splitlines(True)
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    mapping = {n.name: n for n in ast.parse(answer).body if isinstance(n, definitions)}
    for node in reversed(ast.parse(stub).body):
        if isinstance(node, definitions) and any(isinstance(c, ast.Raise) and "NotImplementedError" in ast.unparse(c) for c in ast.walk(node)):
            candidate = mapping[node.name]
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
            answer_start = min([candidate.lineno, *[d.lineno for d in candidate.decorator_list]]) - 1
            lines[start:node.end_lineno] = ["".join(answers[answer_start:candidate.end_lineno]) + "\n"]
    return "".join(lines)


def test_all_seven_exercises_complete_from_their_own_answers():
    ex, sol = notebook("exercises"), notebook("solutions")
    nbformat.validate(ex)
    nbformat.validate(sol)
    completed, stub_names = [], set()
    exercise_count = 0
    for i, c in enumerate(ex.cells):
        if c.cell_type != "code":
            continue
        text = c.source
        if "NotImplementedError" in text:
            exercise_count += 1
            stub_names.update(n.name for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and any(isinstance(x, ast.Raise) and "NotImplementedError" in ast.unparse(x) for x in ast.walk(n)))
            following = []
            for next_cell in ex.cells[i + 1:]:
                if next_cell.cell_type == "code":
                    break
                following.append(next_cell.source)
            blocks = re.findall(r"<summary>Solution</summary>(.*?)</details>", "\n".join(following), re.S)
            assert len(blocks) == 1, i
            answer = "\n".join(re.findall(r"```python\n(.*?)```", blocks[0], re.S))
            text = complete_answer(text, answer)
            assert "NotImplementedError" not in text
        completed.append(text)
    assert exercise_count == 7 and len(stub_names) == 31
    assert "QLoRALinear" in stub_names
    for a, b in zip(completed, code_cells(sol), strict=True):
        assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(source(b)))
    assert all(c.execution_count is None for c in code_cells(ex))


def test_all_original_teaching_lines_survive_except_named_heading_adaptation():
    current = "\n".join(c.source for c in notebook("exercises").cells if c.cell_type == "markdown")
    lines = {line.strip() for line in current.splitlines() if line.strip()}
    for c in old_notebook("solutions")["cells"]:
        if c["cell_type"] != "markdown":
            continue
        text = re.sub(r"<details>\s*<summary>Solution</summary>\s*```python\n.*?```\s*</details>\s*", "", source(c), flags=re.S)
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("# [15.3]") or stripped in ("## Learning Objectives", "## Setup"):
                continue
            assert stripped in lines, stripped
    assert "## Original learning goals" in current
    for word in ("Historical reference", "not a fresh", "guided practice", "unreleased", "0.03125", "0.01", "name-only", "same-norm", "Limitations", "Try It Yourself", "BF16", "Student-t", "two-label conditional NLL"):
        assert word in current, word
    assert current.count("## Forecast before the result") == current.count("## Reflect after the result") == 8
    assert len(re.findall(r"^### Exercise", current, re.M)) == 7
    assert len(re.findall(r"^# ! TAGS: \[html\]", MASTER.read_text(), re.M)) == 4
    assert current.index("Forecast before the result — pretrained-rule-and-memory") < current.index("September 5 reference run")



def test_all_original_notebook_definitions_remain_authoritative():
    old_source = "\n".join(source(c) for c in code_cells(old_notebook("solutions")))
    new_source = "\n".join(source(c) for c in code_cells(notebook("solutions")))
    old = {n.name: n for n in ast.parse(old_source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new = {n.name: n for n in ast.parse(new_source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert old.keys() == new.keys() and len(old) == 43
    for name in old:
        assert ast.dump(old[name]) == ast.dump(new[name]), name
    assert "reference.run_cuda_release" in ast.unparse(new["run_cuda_release"])



def test_import_is_safe_from_unrelated_directory(tmp_path):
    script = r'''
import contextlib, importlib, io, os, sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
def forbidden(*args, **kwargs):
    raise AssertionError('Import ran a lesson side effect')
plt.style.use = forbidden
plt.subplots = forbidden
torch.set_num_threads = forbidden
torch.cuda.is_available = forbidden
torch.manual_seed = forbidden
before = torch.cuda.is_initialized()
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    module = importlib.import_module('chapter15_peft_misalignment.exercises.part3_qlora_from_scratch.solutions')
assert not captured.getvalue(), captured.getvalue()
assert torch.cuda.is_initialized() == before
assert 'bitsandbytes' not in sys.modules and 'transformers' not in sys.modules
assert not list(Path.cwd().iterdir())
for name in ('nf4_codebook', 'quantize_nf4', 'audit_linear4bit', 'QLoRALinear', 'counterbalance_audit', 'train_on_codebook', 'load_nf4_model', 'run_cuda_release'):
    assert callable(getattr(module, name)), name
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "Import safety passed"


CPU_ORACLES = [
    ("check_nf4_codebook", ("nf4_codebook",)),
    ("check_blockwise_absmax", ("blockwise_absmax",)),
    ("check_manual_nf4", ("quantize_nf4", "dequantize_nf4")),
    ("check_double_quantization", ("double_quantize_scales", "dequantize_scales", "nf4_storage_accounting")),
    ("check_linear4bit_storage", ("linear4bit_storage_accounting",)),
    ("check_quant_state_audit", ("audit_linear4bit",)),
    ("check_ste", ("ste_nf4",)),
    ("check_toy_qlora_forward", ("toy_qlora_forward",)),
    ("check_real_qlora_wrapper", ("QLoRALinear",)),
    ("check_dequantized_base_parity", ("dequantized_base_max_diff",)),
    ("check_inject_qlora", ("inject_qlora", "QLoRALinear")),
    ("check_nf4_model_audit", ("audit_nf4_model",)),
    ("check_counterbalanced_data", ("make_codebook_examples", "make_unseen_subject_examples", "make_marker_control_examples", "counterbalance_audit", "name_only_baseline")),
    ("check_random_labels", ("make_codebook_examples", "randomize_training_labels")),
    ("check_spectrum", ("summarize_spectrum",)),
    ("check_outlier_stress", ("quantization_stress_test",)),
]


@pytest.mark.parametrize("oracle,names", CPU_ORACLES)
def test_generated_methods_against_bounded_independent_cpu_oracles(methods, cpu_state, oracle, names):
    checks = importlib.import_module(MODULE + ".tests")
    getattr(checks, oracle)(*[getattr(methods, name) for name in names])
    assert not cpu_state.cuda.is_initialized()


def test_historical_failed_gate_and_every_receipt_hash_remain_explicit():
    receipt = json.loads((SECTION / "expected_outputs/reference_metrics.json").read_text())
    assert receipt["course_release_accepted"] is False
    assert receipt["precision_diagnostic"]["bf16_max_abs_error"] == 0.03125
    assert receipt["precision_diagnostic"]["fp32_max_abs_error"] < 1e-5
    assert receipt["cuda"]["accepted"] is False
    for name, expected in receipt["source_sha256"].items():
        assert sha(archived(str((SECTION / name).relative_to(ROOT)))) == expected, name
    for name, expected in receipt["figure_sha256"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name


def test_numeric_nf4_quantizer_matches_brute_force_nearest_code_and_preserves_input(methods, cpu_state):
    t = cpu_state
    values = t.tensor([-1., -0.31, 0.18, 0.74, 0., 0., 0., 0., -0.05])
    original = values.clone()
    quantized = methods.quantize_nf4(values, block_size=4)
    codebook = methods.nf4_codebook()
    expected = []
    for start in range(0, len(values), 4):
        block = values[start:start+4]
        scale = float(block.abs().max())
        for value in block:
            normalized = 0. if scale == 0 else float(value) / scale
            index = min(range(16), key=lambda i: abs(normalized - float(codebook[i])))
            expected.append(float(codebook[index]) * scale)
    t.testing.assert_close(methods.dequantize_nf4(quantized), t.tensor(expected))
    assert t.equal(values, original)


def test_guided_case_freezes_the_metric_inputs_and_unmeasured_changed_condition():
    proposal = json.loads((EVIDENCE / "guided_case_proposal.json").read_text())
    fingerprint = proposal.pop("configuration_sha256")
    assert sha(json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode()) == fingerprint
    for path, expected in proposal["artifact_sha256"].items():
        assert sha(historical_bytes(ROOT / path, expected)) == expected, path
    reference = json.loads((ROOT / proposal["reference_artifact"]).read_text())
    value = reference
    for key in proposal["primary"]["reference_json_path"]:
        value = value[key]
    assert isinstance(value, float) and value > 0
    assert proposal["input_recipe"]["block_size"] == 128
    assert proposal["input_recipe"]["raw_outlier_value"] == 12.0
    assert proposal["transfer"]["raw_outlier_value"] == 6.0
    assert proposal["transfer"]["reference_json_path"] is None
    assert "hardcodes12.0" in proposal["api_boundary"]
    for response in proposal["learner_response_fields"]:
        assert response not in proposal


def test_imported_cuda_release_requires_an_explicit_learner_bundle_before_delegating(methods, monkeypatch):
    monkeypatch.delitem(vars(methods), "learner_functions", raising=False)
    called = []
    monkeypatch.setattr(methods.reference, "run_cuda_release", lambda **kwargs: called.append(kwargs))
    with pytest.raises(ValueError, match="implementation"):
        methods.run_cuda_release(max_vram_gb=7.0)
    assert called == []


def explicit_completed_notebook_callbacks(methods):
    """Compile only completed definitions, with no setup, tests, training or calls."""
    namespace = dict(vars(methods), __name__="completed_notebook_fixture")
    definitions = []
    for cell in code_cells(old_notebook("solutions")):
        definitions.extend(node for node in ast.parse(source(cell)).body
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "completed_notebook_fixture.py", "exec"), namespace)
    return {field: namespace["QLoRALinear" if field == "qlora_linear" else field]
            for field in methods.LearnerFunctions.__dataclass_fields__}


def test_imported_cuda_api_preserves_explicit_callbacks_and_frozen_recipe(methods, monkeypatch):
    callbacks = explicit_completed_notebook_callbacks(methods)
    implementation = methods.make_learner_functions(**callbacks)
    called = []
    sentinel = object()
    monkeypatch.setattr(methods.reference, "run_cuda_release", lambda **kwargs: called.append(kwargs) or sentinel)
    assert methods.run_cuda_release(7.0, implementation=implementation) is sentinel
    assert called == [{"implementation": implementation, "max_vram_gb": 7.0, "steps": 32, "batch_size": 12}]
    for name, callback in callbacks.items():
        assert getattr(called[0]["implementation"], name) is callback
    monkeypatch.setitem(vars(methods), "learner_functions", implementation)
    assert methods.run_cuda_release(8.0) is sentinel
    assert called[-1]["implementation"] is implementation
    assert called[-1]["max_vram_gb"] == 8.0


def test_imported_factory_and_api_do_not_substitute_or_accept_reference_callbacks(methods, monkeypatch):
    reference_bundle = methods.reference.reference_learner_functions()
    called = []
    monkeypatch.setattr(methods.reference, "run_cuda_release", lambda **kwargs: called.append(kwargs))
    with pytest.raises(ValueError, match="non-reference learner callbacks"):
        methods.make_learner_functions(**vars(reference_bundle))
    with pytest.raises(ValueError, match="non-reference learner callbacks"):
        methods.run_cuda_release(implementation=reference_bundle)
    module_callbacks = {
        field: vars(methods)["QLoRALinear" if field == "qlora_linear" else field]
        for field in methods.LearnerFunctions.__dataclass_fields__
    }
    with pytest.raises(ValueError, match="module-owned solutions callbacks"):
        methods.make_learner_functions(**module_callbacks)
    callbacks = explicit_completed_notebook_callbacks(methods)
    def direct_module_alias(*args, **kwargs):
        return methods.nf4_codebook(*args, **kwargs)
    callbacks["nf4_codebook"] = direct_module_alias
    with pytest.raises(ValueError, match="direct delegation"):
        methods.make_learner_functions(**callbacks)
    assert called == []


def test_original_missing_widget_state_is_disclosed_without_fabricating_progress_data():
    original = old_notebook("solutions")
    assert not original.get("metadata", {}).get("widgets")
    views = [output["data"]["application/vnd.jupyter.widget-view+json"]
             for cell in code_cells(original) for output in cell["outputs"]
             if "application/vnd.jupyter.widget-view+json" in output.get("data", {})]
    assert len(views) == 25
    assert not notebook("solutions").metadata.get("widgets")
