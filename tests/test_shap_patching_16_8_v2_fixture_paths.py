"""DEV fixture-path regressions; never read a historical report or run its model."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch

import pytest

C = Path(__file__).resolve().parents[1]
W = C.parent
V = C / 'docs/mastery/evidence/shap_patching_16_8_migration_20261021/v2/v1_source_tree'
R = C
CH = 'chapter16_shapley_attribution_baselines'
PART = 'part8_shap_vs_patching'
LEGACY = R / CH / 'exercises/part8_shapley_mechinterp_agreement'
E = C / 'docs/mastery/evidence/shap_patching_16_8_migration_20261021/v2'


def v1_bytes(relative):
    original = V / 'candidate' / relative
    if original.exists():
        return original.read_bytes()
    with tarfile.open(E / 'v1_changed_source_preimages.tar.gz') as handle:
        return handle.extractfile(str(relative)).read()


@contextmanager
def imported_candidate():
    # Third-party dependency imports are outside the observed lesson boundary.
    import torch
    import matplotlib
    import matplotlib.pyplot
    import IPython.display
    root = Path(os.environ.get('ARENA_SHAP_FIXTURE_CANDIDATE', str(C)))
    events = []
    def forbidden(*args, **kwargs):
        events.append('forbidden runtime event')
        raise AssertionError('Source import attempted target/runtime work')
    old_path = list(sys.path)
    sys.path[:0] = [str(root), str(R)]
    try:
        spec = importlib.util.spec_from_file_location('shap_fixture_path_DEV', root / CH / 'exercises' / PART / 'solutions.py')
        module = importlib.util.module_from_spec(spec)
        with patch.object(torch, 'tensor', forbidden), patch.object(torch, 'zeros', forbidden), patch.object(torch, 'ones', forbidden), patch.object(torch, 'set_default_dtype', forbidden), patch.object(matplotlib, 'use', forbidden), patch.object(Path, 'cwd', forbidden):
            spec.loader.exec_module(module)
        assert events == []
        yield module
    finally:
        sys.path[:] = old_path


@pytest.mark.parametrize('check', ['report', 'both_notebooks'])
def test_historical_fixture_read_paths_are_original(check, capsys):
    with imported_candidate() as module:
        events = []
        if check == 'report':
            class RecordingStop(Exception):
                pass
            def sentinel_read(path, *args, **kwargs):
                events.append(str(path))
                raise RecordingStop('DEV: stop before reading or parsing any report')
            with patch.object(Path, 'read_text', sentinel_read):
                with pytest.raises(RecordingStop):
                    module.agreement_tests.test_committed_gpu_report_records_agreement_and_controls()
            expected = [str(LEGACY / 'verification_report.json')]
        else:
            # A source-only DEV string drives the loop far enough to record both
            # notebook inputs. It is not either notebook or a teaching result.
            dev_text = '\n'.join([
                'By the end of this notebook', 'Core Question', 'Learning Objectives',
                'Signature Result', 'Try It Yourself', 'Bonus Anomaly Hunt', 'Limitations',
                '<summary>Expected output</summary>', '<summary>Help',
                '<summary>Interpreting the result</summary>', '<summary>Solution</summary>',
                'Exercise - ' * 6,
            ])
            def sentinel_read(path, *args, **kwargs):
                events.append(str(path))
                return dev_text
            with patch.object(Path, 'read_text', sentinel_read):
                module.agreement_tests.test_notebooks_have_arena_learner_surface()
            capsys.readouterr()  # Suppress a historical check's print on DEV text.
            expected = [str(LEGACY / f'16.8_Do_SHAPley_and_Mechanistic_Interpretability_Agree_{kind}.ipynb') for kind in ['exercises', 'solutions']]
        if receipt_dir := os.environ.get('ARENA_SHAP_FIXTURE_RECEIPT_DIR'):
            receipt = Path(receipt_dir) / (check + '.json')
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({
                'check': check,
                'candidate_module': module.__file__,
                'runtime_test_module': module.agreement_tests.__file__,
                'observed_paths': events,
                'expected_legacy_paths': expected,
                'paths_match': events == expected,
                'fixture_contents_read': False,
                'fixture_substitute': 'raising stop before report parse' if check == 'report' else 'explicit DEV surface text',
                'scientific_or_model_call': False,
            }, indent=2) + '\n')
        assert events == expected, 'Fixture identity must come from the original module file, never a copied module directory'


def test_actual_transitive_test_module_origins_are_legacy_and_import_is_inert():
    with imported_candidate() as module:
        assert Path(module.agreement_tests.__file__).resolve() == LEGACY / 'tests.py'
        assert Path(module.relu_tests.__file__).resolve() == R / CH / 'exercises/part6_shap_vs_activation_patching/tests.py'
        assert module.agreement_tests.__name__ == CH + '.exercises.part8_shapley_mechinterp_agreement.tests'
        assert module.relu_tests.__name__ == CH + '.exercises.part6_shap_vs_activation_patching.tests'
        assert not hasattr(module, 'TOY_LINEAR_WEIGHTS')
        assert not hasattr(module, 'finite_circuit_value')


def test_v2_module_inverse_changes_only_two_test_imports_and_raw_bytes_match():
    old = ast.parse(v1_bytes(Path(CH) / 'exercises' / PART / 'solutions.py'))
    new = ast.parse((C / CH / 'exercises' / PART / 'solutions.py').read_text())
    names = ['relu_tests', 'agreement_tests']
    old_imports = {n.names[0].name: n for n in old.body if isinstance(n, ast.ImportFrom) and n.module == CH + '.exercises.' + PART and len(n.names) == 1 and n.names[0].name in names}
    assert set(old_imports) == set(names)
    changed = []
    for index, node in enumerate(new.body):
        if isinstance(node, ast.ImportFrom) and len(node.names) == 1 and node.names[0].asname in names:
            alias = node.names[0].asname
            changed.append(alias)
            new.body[index] = old_imports[alias]
    assert sorted(changed) == sorted(names)
    assert ast.dump(old) == ast.dump(new)
    actual = W / 'cli_probe' / CH / 'exercises' / PART / 'solutions.py'
    if not actual.exists():
        actual = E / 'raw_generated/solutions.py'
    assert actual.read_bytes() == (C / CH / 'exercises' / PART / 'solutions.py').read_bytes()


def test_all_original_v1_assertion_asts_are_preserved():
    name = 'tests/test_shap_patching_16_8_migration.py'
    original = ast.parse(v1_bytes(Path(name)))
    current = ast.parse((C / name).read_text())
    old = [ast.dump(n) for n in ast.walk(original) if isinstance(n, ast.Assert)]
    new = [ast.dump(n) for n in ast.walk(current) if isinstance(n, ast.Assert)]
    assert old == new
    assert len(old) == 71
