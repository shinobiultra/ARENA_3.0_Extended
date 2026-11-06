"""An archived receipt validator cannot stand in for an experiment implementation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from scripts import verification_fixture_contract as contract

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = Path(contract.CHAPTER)
SPARSE = ('part18_spex', 'part19_proxyspex')
SECTIONS = (*SPARSE, 'part1_ordinary_fixture')


def asset_checks(tmp_path, monkeypatch, section):
    spec = importlib.util.spec_from_file_location(
        'entrypoint_role_asset_checks', ROOT / 'tests/test_extension_verification_assets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / CHAPTER / section
    directory.mkdir(parents=True)
    validator = tmp_path / 'arena_ext/legacy_sparse_evidence/native_report.py'
    validator.parent.mkdir(parents=True)
    validator.write_text(
        'def build_report(): return {}\n'
        'def has_native_evidence(): return True\n'
        'def receipt_blockers(): return []\n')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'extension_exercise_dirs', lambda: [directory])
    return module, directory, validator


@pytest.mark.parametrize('section', SECTIONS)
def test_experiment_selector_never_falls_back_to_validator(tmp_path, monkeypatch, section):
    _, directory, validator = asset_checks(tmp_path, monkeypatch, section)
    assert contract.experiment_entrypoint(tmp_path, directory) == directory / 'solutions.py'
    expected_validator = validator if section in SPARSE else directory / 'solutions.py'
    assert contract.verification_entrypoint(tmp_path, directory) == expected_validator


@pytest.mark.parametrize('section', SPARSE)
def test_missing_experiment_remains_failure_despite_valid_validator(tmp_path, monkeypatch, section):
    module, _, _ = asset_checks(tmp_path, monkeypatch, section)
    with pytest.raises(FileNotFoundError):
        module.test_extension_full_experiment_hooks_are_callable_not_placeholders()


@pytest.mark.parametrize('section', SECTIONS)
def test_actual_experiment_placeholder_still_rejected(tmp_path, monkeypatch, section):
    module, directory, _ = asset_checks(tmp_path, monkeypatch, section)
    (directory / 'solutions.py').write_text(
        'def run_full_experiment():\n    raise NotImplementedError("fixture only")\n')
    with pytest.raises(AssertionError):
        module.test_extension_full_experiment_hooks_are_callable_not_placeholders()


@pytest.mark.parametrize('section', SECTIONS)
def test_implemented_hook_positive_control_without_execution(tmp_path, monkeypatch, section):
    module, directory, _ = asset_checks(tmp_path, monkeypatch, section)
    (directory / 'solutions.py').write_text(
        'def run_full_experiment():\n    return {"fixture": True}\n')
    module.test_extension_full_experiment_hooks_are_callable_not_placeholders()


def test_historical_validator_predicates_stay_exact_and_read_only():
    package = ROOT / 'arena_ext/legacy_sparse_evidence'
    provenance = json.loads((package / 'PROVENANCE.json').read_text())
    old_name = 'chapter16_shapley_attribution_baselines/exercises/part18_spex/native_report.py'
    identity = provenance['source_files'][old_name]
    old = subprocess.check_output(['git', 'cat-file', 'blob', identity['git_blob']], cwd=ROOT)
    assert hashlib.sha256(old).hexdigest() == identity['sha256']
    before = {node.name: node for node in ast.parse(old).body if isinstance(node, ast.FunctionDef)}
    after = {node.name: node for node in ast.parse((package / 'native_report.py').read_text()).body
             if isinstance(node, ast.FunctionDef)}
    retained = provenance['retained_functions']['native_report.py']
    assert set(after) == set(retained)
    assert {'has_native_evidence', 'receipt_blockers'} <= set(after)
    assert not {'build_report', 'run_full_experiment'} & set(after)
    for name in retained:
        assert ast.dump(before[name]) == ast.dump(after[name]), name
