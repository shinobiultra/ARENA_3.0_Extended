"""Full 13.2 preservation and independent tiny CPU fixtures; no trained-model runs."""
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

from scripts.verified_presentation import verified_pre_wrap_notebook

ROOT = Path(__file__).resolve().parents[2]
SECTION = ROOT / "chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching"
EVIDENCE = ROOT / "docs/mastery/evidence/denoising_master_migration"
MASTER = ROOT / "infrastructure/chapters/chapter13_image_generation_interpretability/master_13_2.py"
MODULE = "chapter13_image_generation_interpretability.exercises.part2_denoising_time_causal_patching"
STEM = "13.2_Denoising-Time_Causal_Patching"


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
    # COLOR_VALUES is created at import by the byte-preserved legacy module.
    # Reproduce its historical float32 process default, then restore ambient state.
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
    assert all(before["report_input_matches"].values())
    with tarfile.open(archive) as saved:
        for name, expected in before["files"].items():
            assert sha(saved.extractfile(name).read()) == expected, name
        for item in before["report_inputs"]["files"]:
            assert sha(saved.extractfile(item["path"]).read()) == item["sha256"], item["path"]


def test_every_original_legacy_file_and_saved_instruction_asset_remains():
    before = json.loads((EVIDENCE / "before.json").read_text())
    replaced = {str((SECTION / p).relative_to(ROOT)) for p in (f"{STEM}_exercises.ipynb", f"{STEM}_solutions.ipynb", "solutions.py")}
    replaced.add("chapter13_image_generation_interpretability/instructions/pages/02_[13.2]_Denoising-Time_Causal_Patching.md")
    for name, expected in before["files"].items():
        if name not in replaced and (name.startswith(str(SECTION.relative_to(ROOT)) + "/") or "/instructions/assets/" in name):
            assert sha((ROOT / name).read_bytes()) == expected, name
    old_module = str((SECTION / "solutions.py").relative_to(ROOT))
    assert (SECTION / "legacy_reference.py").read_bytes() == archived(old_module)


def test_all_sixteen_scientific_cells_preserve_ast_lines_outputs_and_counts():
    old = old_notebook("solutions")
    normalized = nbformat.reads(json.dumps(old), as_version=4)
    current = notebook("solutions")
    parity = json.loads((EVIDENCE / "output_parity.json").read_text())
    assert parity["execution_kind"] == "retained_historical_reference"
    assert parity["fresh_model_execution"] is False
    assert len(code_cells(old)) == len(code_cells(current)) == len(parity["cells"]) == 16
    for ordinal, (historical, canonical, normalized_old, receipt) in enumerate(zip(code_cells(old), code_cells(current), code_cells(normalized), parity["cells"], strict=True)):
        a, b = source(historical), source(canonical)
        adapted = a.replace("import solutions as reference", "import legacy_reference as reference")
        assert (a != adapted) == (ordinal == 0)
        assert [line for line in adapted.splitlines() if line.strip()] == [line for line in b.splitlines() if line.strip()]
        assert ast.dump(ast.parse(adapted)) == ast.dump(ast.parse(b))
        assert sha(a.encode()) == receipt["published_source_sha256"]
        assert sha(adapted.encode()) == receipt["published_route_adapted_source_sha256"]
        assert sha(b.encode()) == receipt["generated_source_sha256"]
        assert canonical.outputs == normalized_old.outputs
        assert canonical.execution_count == historical["execution_count"]
        assert sha(json.dumps(historical["outputs"], sort_keys=True).encode()) == receipt["outputs_sha256"]
    assert sum(c.execution_count is not None for c in code_cells(current)) == 16
    assert sum(bool(c.outputs) for c in code_cells(current)) == 13
    assert sum(len(c.outputs) for c in code_cells(current)) == 26


def test_four_canonical_pngs_are_exact_historical_notebook_output_bytes():
    old = old_notebook("solutions")
    figures = json.loads((EVIDENCE / "figures.json").read_text())
    indices = [24, 26, 37, 39]
    names = ["denoising_time_toy_signature.png", "denoising_time_toy_play.png", "denoising_time_sd15_signature.png", "denoising_time_sd15_play.png"]
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


def test_all_ten_exercises_complete_from_their_own_answers():
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
            assert "tests.test_" in text
        completed.append(text)
    assert exercise_count == 10 and len(stub_names) == 25
    assert {"ResidualConvBlock", "HookableObjectDenoiser"} <= stub_names
    for a, b in zip(completed, code_cells(sol), strict=True):
        assert ast.dump(ast.parse(a)) == ast.dump(ast.parse(source(b)))
    assert all(c.execution_count is None for c in code_cells(ex))


def test_all_original_teaching_lines_survive_except_named_heading_adaptation():
    current = "\n".join(c.source for c in verified_pre_wrap_notebook(SECTION / f'{STEM}_exercises.ipynb').cells if c.cell_type == "markdown")
    lines = {line.strip() for line in current.splitlines() if line.strip()}
    for c in old_notebook("solutions")["cells"]:
        if c["cell_type"] != "markdown":
            continue
        text = re.sub(r"<details>\s*<summary>Solution</summary>\s*```python\n.*?```\s*</details>\s*", "", source(c), flags=re.S)
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("# [13.2]") or stripped == "## Learning Objectives":
                continue
            assert stripped in lines, stripped
    assert "## Original learning goals" in current
    for word in ("Historical reference", "not a fresh", "guided practice", "UNet circuit", "exact trained toy", "CLIP", "Common bug", "Limitations and Claim Boundary", "Anomaly hunting", "Try It Yourself: Real Model"):
        assert word in current, word
    assert current.count("## Forecast before the result") == current.count("## Reflect after the result") == 4
    assert len(re.findall(r"^### Exercise", current, re.M)) == 10
    assert len(re.findall(r"^# ! TAGS: \[html\]", MASTER.read_text(), re.M)) == 4
    assert current.index("Forecast before the result — sd15-time-sweep") < current.index("The pinned CUDA 13.2 release run")


def test_original_notebook_report_wrappers_remain_authoritative():
    old_source = "\n".join(source(c) for c in code_cells(old_notebook("solutions")))
    new_source = "\n".join(source(c) for c in code_cells(notebook("solutions")))
    old = {n.name: n for n in ast.parse(old_source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new = {n.name: n for n in ast.parse(new_source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert old.keys() == new.keys() and len(old) == 30
    for name in old:
        assert ast.dump(old[name]) == ast.dump(new[name]), name
    for name in ("run_smoke_test", "run_gpu_test", "run_full_experiment"):
        assert "reference." + name in ast.unparse(new[name])


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
before = torch.cuda.is_initialized()
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    module = importlib.import_module('chapter13_image_generation_interpretability.exercises.part2_denoising_time_causal_patching.solutions')
assert not captured.getvalue(), captured.getvalue()
assert torch.cuda.is_initialized() == before
assert 'diffusers' not in sys.modules and 'transformers' not in sys.modules
assert not list(Path.cwd().iterdir())
for name in ('q_sample', 'causal_patch_sweep', 'apply_latent_patch', 'run_sd15_latent_trajectory', 'run_sd15_patch_case', 'run_smoke_test', 'run_gpu_test'):
    assert callable(getattr(module, name)), name
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "Import safety passed"


TENSOR_ORACLES = [
    ("test_diffusion_schedule_has_exact_endpoint_and_valid_noise", ("make_diffusion_schedule", "q_sample")),
    ("test_q_sample_matches_closed_form_endpoints", ("q_sample",)),
    ("test_rendered_world_has_exact_labels_and_masks", ("render_object_world",)),
    ("test_sinusoidal_timestep_embedding_has_paired_coordinates", ("sinusoidal_timestep_embedding",)),
    ("test_activation_replacement_changes_only_selected_coordinates", ("make_activation_replacement_hook",)),
    ("test_regional_causal_metric_has_exact_oracles", ("regional_causal_metric",)),
    ("test_regional_causal_metric_has_exact_anchors", ("regional_causal_metric",)),
    ("test_center_mask_is_preregistered_and_centered", ("make_center_mask",)),
    ("test_latent_patch_preserves_the_complement_exactly", ("apply_latent_patch",)),
    ("test_random_latent_patch_is_seeded_and_same_size", ("apply_random_latent_patch",)),
    ("test_calibrated_recovery_anchors_recipient_and_donor", ("calibrated_recovery",)),
    ("test_latent_transfer_metric_rewards_regional_transfer", ("latent_transfer_metric",)),
    ("test_sd15_trajectory_rejects_invalid_step_count", ("run_sd15_latent_trajectory",)),
    ("test_real_sd15_case_contract_is_pinned_and_counterfactual", ()),
]


@pytest.mark.parametrize("oracle,names", TENSOR_ORACLES)
def test_generated_methods_against_existing_independent_tensor_oracles(methods, cpu_state, oracle, names):
    checks = importlib.import_module(MODULE + ".tests")
    getattr(checks, oracle)(*[getattr(methods, name) for name in names])


def test_native_ddim_loop_with_independent_two_step_cfg_and_patch_oracle(methods, cpu_state, monkeypatch):
    t = cpu_state
    class Scheduler:
        def set_timesteps(self, count, device):
            self.timesteps = t.arange(count - 1, -1, -1, device=device)
        def scale_model_input(self, value, timestep):
            return value
        def step(self, noise, timestep, current, eta, return_dict):
            assert eta == 0.0 and return_dict is False
            return (current - noise,)
    def encode_prompt(**kwargs):
        return t.tensor([[2.0]]), t.tensor([[1.0]])
    def unet(value, timestep, encoder_hidden_states, return_dict):
        assert encoder_hidden_states.tolist() == [[1.0], [2.0]]
        return (t.cat([t.ones_like(value[:1]), 2 * t.ones_like(value[:1])]),)
    pipe = SimpleNamespace(scheduler=Scheduler(), encode_prompt=encode_prompt, unet=unet)
    monkeypatch.setattr(methods, "decode_sd15_latents", lambda _, x: (None, x[:, :3].clone()))
    initial = t.zeros(1, 4, 2, 2)
    baseline = methods.run_sd15_latent_trajectory(pipe, "positive", "negative", initial, num_inference_steps=2, guidance_scale=3)
    # 1 + 3*(2-1) = 4; two deterministic scheduler updates yield -8.
    assert t.equal(baseline.latent_states[-1], t.full_like(initial, -8))
    assert len(baseline.latent_states) == 3 and baseline.scheduler_timesteps == (1, 0)
    mask = t.tensor([[True, False], [False, False]])
    donor = (initial.clone(), t.full_like(initial, 10), t.full_like(initial, 20))
    patched = methods.run_sd15_latent_trajectory(pipe, "positive", "negative", initial, num_inference_steps=2, guidance_scale=3, patch_step_index=1, donor_states=donor, spatial_mask=mask)
    assert t.equal(patched.latent_states[-1][:, :, 0, 0], t.full((1, 4), 6.0))
    assert t.equal(patched.latent_states[-1][:, :, 1, 1], t.full((1, 4), -8.0))
    assert t.equal(initial, t.zeros_like(initial)) and t.equal(donor[1], t.full_like(initial, 10))


def test_cache_hooks_preserve_time_layer_identity_and_detach_copies(methods, cpu_state):
    t = cpu_state
    layers = {name: t.nn.Identity() for name in methods.PATCH_LAYERS}
    fixture = SimpleNamespace(patch_layer=layers.__getitem__)
    cache = methods.ActivationCache(values={}, timestep=3)
    handles = methods.register_activation_cache(fixture, cache)
    try:
        x = t.ones(1, 2, 2, 2, requires_grad=True)
        for name, layer in layers.items():
            layer(x)
        assert set(cache.values) == {(3, name) for name in layers}
        x.detach().fill_(5)
        assert all(t.equal(value, t.ones_like(value)) and not value.requires_grad for value in cache.values.values())
    finally:
        for handle in handles:
            handle.remove()
    assert all(not layer._forward_hooks for layer in layers.values())


def test_guided_proposal_fingerprints_exact_case_and_changed_time_without_answers():
    proposal = json.loads((EVIDENCE / "guided_case_proposal.json").read_text())
    fingerprint = proposal.pop("configuration_sha256")
    assert sha(json.dumps(proposal, sort_keys=True, separators=(",", ":")).encode()) == fingerprint
    for path, expected in proposal["artifact_sha256"].items():
        assert sha(historical_bytes(ROOT / path, expected)) == expected, path
    report = json.loads((ROOT / proposal["reference_artifact"]).read_text())
    for label in ("primary", "transfer"):
        value = report
        for key in proposal[label]["reference_json_path"]:
            value = value[key]
        assert isinstance(value, float)
    assert proposal["primary"]["patch_step_index"] == 2
    assert proposal["transfer"]["patch_step_index"] == 19
    assert proposal["case"]["seed"] == 4 and proposal["patch_fraction"] == 0.5
    assert proposal["execution_kind"] == "retained_historical_reference"
    assert "historically selected" in proposal["primary"]["selection_boundary"]
    for name in proposal["learner_response_fields"]:
        assert name not in proposal
