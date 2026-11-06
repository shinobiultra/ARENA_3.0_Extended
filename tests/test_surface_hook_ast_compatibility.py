"""Regression controls for compact legacy hooks. Never execute notebook/model code."""
import ast
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / 'infrastructure/core/config.yaml').is_file())
HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('ARENA_SURFACE_SOURCE', ROOT / 'scripts/audit_course_surface.py'))
spec = importlib.util.spec_from_file_location('surface_compat_candidate', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.ROOT = ROOT

GPU = '''def run_gpu_test(max_vram_gb:float=24.0):
    from part3_shapley_interactions_shapiq.solutions import run_gpu_test as run
    return run(max_vram_gb=max_vram_gb)
'''
FULL = '''def run_full_experiment(max_vram_gb:float=24.0):
    return run_gpu_test(max_vram_gb=max_vram_gb)
'''
DELEGATE = (ROOT / 'chapter16_shapley_attribution_baselines/exercises/part3_shapley_interactions_shapiq/solutions.py').read_text()


def notebook(tmp_path, source, delegate=DELEGATE):
    here = tmp_path / 'exercises' / 'part5'
    here.mkdir(parents=True)
    target = here.parent / 'part3_shapley_interactions_shapiq' / 'solutions.py'
    target.parent.mkdir()
    target.write_text(delegate)
    path = here / 'example_exercises.ipynb'
    path.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': source}]}))
    return path


def recognized(tmp_path, source, delegate=DELEGATE):
    return audit._exercise_notebook_declares_verification_contract(
        notebook(tmp_path, source, delegate)
    )


def test_current_16_5_compact_hooks_are_recognized():
    path = ROOT / ('chapter16_shapley_attribution_baselines/exercises/'
                   'part5_shapley_interactions/16.5_Shapley_Interactions_exercises.ipynb')
    assert audit._exercise_notebook_declares_verification_contract(path)


def test_compact_hooks_forward_budget_without_running_delegate(tmp_path):
    # The complete real delegate is read as source only; no CUDA/native code runs.
    assert recognized(tmp_path, GPU + FULL + '\nreport = "verification_report.json"\n')


@pytest.mark.parametrize('source', [
    GPU.replace(':float=', ': float = ') + FULL,
    GPU + FULL.replace(':float=', ': float = '),
    GPU.replace('max_vram_gb:float=24.0', 'max_vram_gb: float = (24.0)') + FULL,
])
def test_equivalent_argument_formatting_is_accepted(tmp_path, source):
    assert recognized(tmp_path, source + '\nreport = "verification_report.json"\n')


@pytest.mark.parametrize('mutation', [
    'argument', 'default', 'annotation', 'extra_argument', 'keyword_only',
    'nested', 'false_branch', 'async', 'missing_gpu', 'missing_full',
    'stub_gpu', 'stub_full', 'empty_gpu', 'empty_full', 'ellipsis_gpu',
    'literal_gpu', 'wrong_forward', 'wrong_full_call', 'no_gpu_call',
    'rebound_gpu', 'duplicate_gpu', 'decorated_gpu', 'invalid_syntax',
])
def test_compact_fallback_rejects_noncallable_or_changed_contracts(tmp_path, mutation):
    gpu, full = GPU, FULL
    if mutation == 'argument': gpu = gpu.replace('max_vram_gb', 'other_budget')
    elif mutation == 'default': gpu = gpu.replace('24.0', '12.0')
    elif mutation == 'annotation': gpu = gpu.replace(':float=', ':int=')
    elif mutation == 'extra_argument': gpu = gpu.replace('=24.0)', '=24.0, other=None)')
    elif mutation == 'keyword_only': gpu = gpu.replace('(max_vram', '(*, max_vram')
    elif mutation == 'nested': gpu = 'def hidden():\n' + '\n'.join('    '+line for line in gpu.splitlines())+'\n'
    elif mutation == 'false_branch': gpu = 'if False:\n' + '\n'.join('    '+line for line in gpu.splitlines())+'\n'
    elif mutation == 'async': gpu = 'async '+gpu
    elif mutation == 'missing_gpu': gpu = ''
    elif mutation == 'missing_full': full = ''
    elif mutation == 'stub_gpu': gpu = 'def run_gpu_test(max_vram_gb:float=24.0):\n    raise NotImplementedError\n'
    elif mutation == 'stub_full': full = 'def run_full_experiment(max_vram_gb:float=24.0):\n    raise NotImplementedError\n'
    elif mutation == 'empty_gpu': gpu = 'def run_gpu_test(max_vram_gb:float=24.0):\n    pass\n'
    elif mutation == 'empty_full': full = 'def run_full_experiment(max_vram_gb:float=24.0):\n    "Only documentation"\n'
    elif mutation == 'ellipsis_gpu': gpu = 'def run_gpu_test(max_vram_gb:float=24.0):\n    ...\n'
    elif mutation == 'literal_gpu': gpu = 'def run_gpu_test(max_vram_gb:float=24.0):\n    return {"accepted": True}\n'
    elif mutation == 'wrong_forward': full = full.replace('=max_vram_gb)', '=12.0)')
    elif mutation == 'wrong_full_call': full = full.replace('return run_gpu_test(', 'return unrelated(')
    elif mutation == 'no_gpu_call': gpu = gpu.replace('return run(', 'return dict(')
    elif mutation == 'rebound_gpu': full += 'run_gpu_test = None\n'
    elif mutation == 'duplicate_gpu': full += gpu
    elif mutation == 'decorated_gpu': gpu = '@unrelated\n'+gpu
    elif mutation == 'invalid_syntax': gpu = 'not valid Python!!!\n'
    assert not recognized(tmp_path, gpu + full + '\nreport = "verification_report.json"\n')


@pytest.mark.parametrize('delegate', [
    '', 'run_gpu_test = None\n',
    DELEGATE + 'run_gpu_test = None\n',
    DELEGATE + 'from unrelated import run_gpu_test\n',
    DELEGATE.replace(': float = 24.0', ': int = 24'),
    '@unrelated\n' + DELEGATE,
    DELEGATE + DELEGATE,
    'def run_gpu_test(max_vram_gb:float=24.0):\n    pass\n',
    'def run_gpu_test(max_vram_gb:float=24.0):\n    raise NotImplementedError\n',
    'def run_gpu_test(max_vram_gb:float=24.0):\n    return {}\n',
])
def test_compact_fallback_requires_substantive_local_delegate(tmp_path, delegate):
    assert not recognized(tmp_path, GPU + FULL + '\nreport = "verification_report.json"\n', delegate)


def test_missing_evidence_surface_still_rejected(tmp_path):
    assert not recognized(tmp_path, GPU + FULL)


def test_existing_literal_branch_keeps_prior_recognition(tmp_path):
    # Deliberately preserve historical detector behavior; new fallback is narrower.
    source = ('def run_gpu_test(max_vram_gb: float = 24.0):\n    return {}\n'
              'def run_full_experiment(max_vram_gb: float = 24.0):\n    return {}\n'
              'report = "verification_report.json"\n')
    assert recognized(tmp_path, source)


# Exact six independent NO-GO inputs; only helper/module routing is retargeted.
INDEPENDENT_GPU = GPU.replace("part3_shapley_interactions_shapiq.solutions", "local_reference")
INDEPENDENT_GOOD='def run_gpu_test(max_vram_gb: float = 24.0):\n    return measure(max_vram_gb=max_vram_gb)\n'

def independent_recognized(tmp_path,delegate):
    section=tmp_path/'exercises'/'part';section.mkdir(parents=True)
    (section.parent/'local_reference.py').write_text(delegate)
    path=section/'exercises.ipynb';path.write_text(json.dumps({'cells':[{'cell_type':'code','source':INDEPENDENT_GPU+FULL+'\nevidence="verification_report.json"\n'}]}))
    return audit._exercise_notebook_declares_verification_contract(path)

@pytest.mark.parametrize('delegate',[
    INDEPENDENT_GOOD+'class run_gpu_test:\n    pass\n',
    INDEPENDENT_GOOD+'async def run_gpu_test(max_vram_gb: float = 24.0):\n    return {}\n',
    'def run_gpu_test(max_vram_gb: float = 24.0):\n    return dict(accepted=True)\n',
    'def run_gpu_test(max_vram_gb: float = 24.0):\n    return measure(max_vram_gb=999.0)\n',
    'def run_gpu_test(max_vram_gb: float = 24.0):\n    if False:\n        return measure(max_vram_gb=max_vram_gb)\n    return {}\n',
    'def run_gpu_test(max_vram_gb: float = 24.0):\n    measure=None\n    return measure(max_vram_gb=max_vram_gb)\n',
])
def test_noncallable_placeholder_or_lost_budget_rejected(tmp_path,delegate):
    assert not independent_recognized(tmp_path,delegate)



BAD_BODIES = [
    'return dict(accepted=True)',
    'return run_neural_interaction_preflight(max_vram_gb=999.0)',
    'if False:\n    return run_neural_interaction_preflight(max_vram_gb=max_vram_gb)\nreturn {}',
    'run_neural_interaction_preflight=None\nreturn run_neural_interaction_preflight(max_vram_gb=max_vram_gb)',
]


def replaced_body(body):
    tree = ast.parse(DELEGATE)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_gpu_test')
    lines = DELEGATE.splitlines(keepends=True)
    replacement = 'def run_gpu_test(max_vram_gb: float = 24.0) -> dict:\n' + ''.join('    '+line+'\n' for line in body.splitlines())
    return ''.join(lines[:node.lineno-1]) + replacement + ''.join(lines[node.end_lineno:])


@pytest.mark.parametrize('delegate', [
    DELEGATE + '\nclass run_gpu_test:\n    pass\n',
    DELEGATE + '\nasync def run_gpu_test(max_vram_gb: float = 24.0):\n    return {}\n',
    *[replaced_body(body) for body in BAD_BODIES],
])
def test_supported_route_rejects_each_reviewed_bad_form(tmp_path, delegate):
    assert not recognized(tmp_path, GPU + FULL + '\nreport="verification_report.json"\n', delegate)


def test_supported_route_ignores_only_formatting_and_comments(tmp_path):
    delegate = DELEGATE.replace('def run_gpu_test(max_vram_gb: float = 24.0)', 'def run_gpu_test( max_vram_gb:float=24.0 )')
    assert recognized(tmp_path, GPU + FULL + '\nreport="verification_report.json"\n', delegate + '\n# Pure formatting remains compatible.\n')


@pytest.mark.parametrize('tail', [
    '\nrun_neural_interaction_preflight = None\n',
    '\ndel run_gpu_test\n',
    '\nclass run_neural_interaction_preflight:\n    pass\n',
])
def test_complete_module_pin_rejects_other_binding_changes(tmp_path, tail):
    assert not recognized(tmp_path, GPU + FULL + '\nreport="verification_report.json"\n', DELEGATE + tail)


def test_alias_cannot_rebind_forwarded_budget(tmp_path):
    # Exact independent V2 review mutant: import shadows the numeric budget.
    gpu = GPU.replace('as run\n', 'as max_vram_gb\n').replace('return run(', 'return max_vram_gb(')
    path = notebook(tmp_path, gpu + FULL + '\nevidence="verification_report.json"\n')
    spec = importlib.util.spec_from_file_location('unchanged_original_surface', ROOT / 'scripts/audit_course_surface.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    assert not original._exercise_notebook_declares_verification_contract(path)
    assert not audit._exercise_notebook_declares_verification_contract(path)
