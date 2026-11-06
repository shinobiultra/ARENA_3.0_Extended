"""Influence/TracIn migration fidelity and tiny CPU oracles; never execute its native study."""
from __future__ import annotations

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

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin"
SECTION = LEGACY.parent / "part10_influence_tracin"
MASTER = ROOT / "infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_10.py"
EVIDENCE = ROOT / "docs/mastery/evidence/influence_tracin_master_migration"
STEM = "16.10_Influence_Functions_and_TracIn"
OLD_STEM = "16.9_Influence_Functions_and_TracIn"
MODULE = "chapter16_shapley_attribution_baselines.exercises.part10_influence_tracin"
OLD_MODULE = "chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin"
OLD_IMPORT = f"from {OLD_MODULE} import (\n    attribution_methods_tests as tests,\n)"
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
    assert sha((ROOT / before["archive"]).read_bytes()) == before["archive_sha256"]
    with tarfile.open(ROOT / before["archive"]) as saved:
        for name, expected in before["files"].items():
            assert sha(saved.extractfile(name).read()) == expected, name
            assert sha((ROOT / name).read_bytes()) == expected, name
    assert len(before["files"]) == 48
    receipt = json.loads((LEGACY / "expected_outputs/learner_execution.json").read_text())
    assert len(receipt["source_inputs"]) == 13
    assert len(before['preserved_inventory']) == 91
    for name, expected in before['preserved_inventory'].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    for name, expected in receipt["source_inputs"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    fresh = json.loads((LEGACY / 'expected_outputs/fresh_reproduction.json').read_text())
    assert len(fresh['source_inputs']) == 14
    for name, expected in fresh['source_inputs'].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    for kind in ("exercises", "solutions"):
        assert sha((LEGACY / f"{OLD_STEM}_{kind}.ipynb").read_bytes()) == receipt[f"published_{kind}_sha256"]
    page = ROOT / "chapter16_shapley_attribution_baselines/instructions/pages/09_[16.9]_Influence_Functions_and_TracIn.md"
    assert sha(page.read_bytes()) == receipt["published_page_sha256"]


def test_full_historical_code_and_outputs_have_bounded_exact_parity():
    parity = json.loads((EVIDENCE / "output_parity.json").read_text())
    assert parity["source"] == "legacy 16.9 Influence Functions and TracIn published completed-own-answer solution"
    assert parity["execution_kind"] == "retained_historical_reference"
    assert parity["fresh_model_execution"] is False
    assert parity["route_adaptation"]["old"] == OLD_IMPORT
    assert parity["route_adaptation"]["new"] == NEW_IMPORT
    old, new = old_notebook("solutions"), notebook("solutions")
    normalized = nbformat.reads(json.dumps(old), as_version=4)
    assert len(codes(old)) == len(codes(new)) == len(parity["cells"]) == 40
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
    assert sum(c.execution_count is not None for c in codes(new)) == 40
    assert sum("image/png" in out.get("data", {}) for c in codes(new) for out in c.outputs) == 18


def test_retained_lesson_and_actual_recorded_execution_reconstruct_the_publication():
    builder = importlib.import_module(OLD_MODULE + ".build_notebooks")
    completed = builder.complete_learner({"cells": builder.lesson_cells()})
    old = old_notebook("solutions")
    assert [(c["cell_type"], source(c)) for c in completed["cells"]] == [(c["cell_type"], source(c)) for c in old["cells"]]
    with tarfile.open(LEGACY / "expected_outputs/full_own_answer_v3/execution.tar.gz") as stored:
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
    hashes=[]
    for cell in codes(ex):
        for output in cell.outputs:
            html=str(output.get('data',{}).get('text/html',''))
            hashes.extend(sha(base64.b64decode(data)) for data in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',html))
    assert sorted(hashes)==sorted(before['figures'].values()) and len(hashes)==18


def test_semantic_test_route_adaptation_is_byte_identical_and_explicit():
    assert (SECTION / "tests.py").read_bytes() == (LEGACY / "attribution_methods_tests.py").read_bytes()
    setup = source(codes(notebook("solutions"))[0])
    assert NEW_IMPORT in setup and OLD_IMPORT not in setup
    for dependency in ("attribution_data",):
        assert f"from {OLD_MODULE}.{dependency} import" in setup
    assert "Path(__file__).resolve().parents[3]" in (SECTION / "solutions.py").read_text()
    assert "not a fresh execution" in MASTER.read_text()
    assert "setup semantic-test import changes" in MASTER.read_text()


def test_student_stubs_complete_only_from_their_own_visible_answers():
    builder = importlib.import_module(OLD_MODULE + ".build_notebooks")
    ex, sol = notebook("exercises"), notebook("solutions")
    nbformat.validate(ex);nbformat.validate(sol)
    completed=[];stub_names=set();count=0
    for index, cell in enumerate(ex.cells):
        if cell.cell_type != 'code':continue
        text=cell.source
        if 'raise NotImplementedError' in text:
            count+=1
            stub_names.update(n.name for n in ast.parse(text).body if isinstance(n,ast.FunctionDef))
            following=[]
            for other in ex.cells[index+1:]:
                if other.cell_type=='code':break
                following.append(other.source)
            blocks=re.findall(r'<summary>Solution</summary>(.*?)</details>', '\n'.join(following),re.S)
            assert len(blocks)==1
            answer='\n\n'.join(re.findall(r'```python\n(.*?)```',blocks[0],re.S))
            text=builder.complete_from_answer(text,answer)
            assert 'tests.test_' in text
        completed.append(text)
    assert count==11
    assert stub_names==set(ORACLES)
    for a,b in zip(completed,codes(sol),strict=True):
        assert ast.dump(ast.parse(a))==ast.dump(ast.parse(source(b)))
    assert all(c.execution_count is None for c in codes(ex))


def test_every_original_explanatory_line_survives():
    current="\n".join(c.source for c in notebook('exercises').cells if c.cell_type=='markdown')
    lines={line.strip().removeprefix('> ').strip() for line in current.splitlines() if line.strip()}
    adaptations=json.loads((EVIDENCE/'route_adaptations.json').read_text())['prose_crosslink']
    for cell in old_notebook('exercises')['cells']:
        if cell['cell_type']!='markdown':continue
        text=re.sub(r'<details>\s*<summary>Solution</summary>.*?</details>','',source(cell),flags=re.S)
        text=text.replace(adaptations['old'],adaptations['new'])
        for line in text.splitlines():
            line=line.strip()
            if not line or line in {'## Introduction','## Learning Objectives'} or line.startswith('# [16.9]'):continue
            # Converter standardizes exercise metadata fences to quoted YAML.
            if line=='```text':line='```yaml'
            assert line in lines,line


def test_full_lesson_controls_plays_negative_findings_and_forecasts_survive():
    nb=notebook('exercises');text='\n'.join(c.source for c in nb.cells)
    assert len(nb.cells)>=140
    assert text.count('Try It Yourself')==2
    assert text.count('## Forecast before the result')==17
    assert text.count('## Reflect after the result')==17
    for term in ('wrong sign for 54','34 of 8192','0.997','930','six','128','48','64','32 matched random',
                 'clean','corrupted','wrong normalization','unfinished','damping','ghost','not a benchmark',
                 'not a Shapley value','historical and monitored','not a fresh execution','guided practice',
                 'Limitations','Checkpoint choice matters','62/64','regularizer','exact path'):
        assert term.lower() in text.lower(),term
    forecasts=[c for c in nb.cells if c.cell_type=='markdown' and c.source.startswith('## Forecast before the result')]
    assert all('confidence (0–100%)' in c.source and 'Record whether you have already seen it' in c.source
               and 'Changed-condition forecast' in c.source for c in forecasts)
    assert text.index('Forecast before the result — native-deletion') < text.index('fitted, ranking_results = prepare_study(')
    assert text.index('Forecast before the result — native-deletion') < text.index('final_results = evaluate_final(')
    assert 'from arena_ext' not in MASTER.read_text()
    assert 'Current canonical' in text and 'Data Shapley' in text


def test_import_is_safe_from_an_unrelated_directory(tmp_path):
    script = r'''
import contextlib, importlib, io, os, sys, tempfile
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import urllib.request
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
os.environ['ARENA_ATTRIBUTION_OUTPUT_DIRECTORY'] = str(Path.cwd() / 'must_not_exist')
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
    module = importlib.import_module('chapter16_shapley_attribution_baselines.exercises.part10_influence_tracin.solutions')
assert not captured.getvalue(), captured.getvalue()
assert not (Path.cwd() / 'must_not_exist').exists()
assert torch.cuda.is_initialized() == before
assert torch.get_num_threads() == threads
assert torch.is_grad_enabled() == grad
assert not hasattr(module, 'data') and not hasattr(module, 'fitted')
for name in ('logistic_losses', 'conjugate_gradient', 'upweight_influence', 'tracin_checkpoint_scores', 'prepare_study', 'evaluate_final', 'query_deletion_play'):
    assert callable(getattr(module, name)), name
print('Import safety passed')
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "Import safety passed"


ORACLES = ('logistic_losses','logistic_gradients','weighted_objective','objective_derivatives',
           'logistic_hvp','conjugate_gradient','fit_weighted_logistic','upweight_influence',
           'sgd_trace','trace_loss_reductions','tracin_checkpoint_scores','ghost_linear_products','class_matched_removals')


@pytest.mark.parametrize('name',ORACLES)
def test_visible_generated_methods_against_independent_cpu_oracles(methods,name,monkeypatch):
    checks=importlib.import_module(MODULE+'.tests')
    torch=importlib.import_module('torch');previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        monkeypatch.setattr(checks,'answers',methods)
        getattr(checks,'test_'+name)(getattr(methods,name))
    finally:
        torch.set_default_dtype(previous)


def test_regularized_removal_normalization_against_independent_refits(methods,monkeypatch):
    checks=importlib.import_module(MODULE+'.tests')
    monkeypatch.setattr(checks,'answers',methods)
    checks.test_regularized_removal_normalization()
