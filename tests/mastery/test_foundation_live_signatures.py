"""Tiny callback and plot-data checks; no notebook kernel, model or CUDA calls."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'infrastructure/core/config.yaml').is_file())
MODULE = Path(os.environ.get('FOUNDATION_LIVE_PROVIDER', ROOT / 'mastery_hub/practice_addons/foundation_live_signatures/lesson.py'))


def provider():
    spec = importlib.util.spec_from_file_location('foundation_live_candidate', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Axis:
    def __init__(self): self.calls = []
    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record


def plotting(monkeypatch):
    pyplot = ModuleType('matplotlib.pyplot')
    pyplot.axes = []
    pyplot.shown = 0
    def subplots(*args, **kwargs):
        count = args[1] if len(args) > 1 else 1
        axes = [Axis() for _ in range(count)]
        pyplot.axes.extend(axes)
        return SimpleNamespace(tight_layout=lambda: None), axes if count > 1 else axes[0]
    def show(): pyplot.shown += 1
    pyplot.subplots = subplots
    pyplot.show = show
    package = ModuleType('matplotlib')
    package.pyplot = pyplot
    monkeypatch.setitem(sys.modules, 'matplotlib', package)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', pyplot)
    return pyplot


def code(section):
    return [c['source'] for c in provider().additions(section) if c['cell_type'] == 'code']


def run(source, namespace):
    exec(compile(source, '<live-signature-cell>', 'exec'), namespace)


def test_provider_is_deterministic_fresh_and_blank():
    for section in ('0.6', '1.6'):
        first = provider().additions(section)
        second = provider().additions(section)
        assert len(first) == 6
        assert first == second
        assert len({c['id'] for c in first}) == len(first)
        assert [c['cell_type'] for c in first] == ['markdown', 'code', 'markdown', 'markdown', 'code', 'markdown']
        for c in first:
            if c['cell_type'] == 'code':
                assert c['execution_count'] is None and c['outputs'] == []
                ast.parse(c['source'])
        first[1]['outputs'].append({'not': 'shared'})
        assert provider().additions(section) == second


def test_provider_rejects_unrelated_sections():
    with pytest.raises(ValueError): provider().additions('14.2')


def test_provider_imports_only_stdlib_and_no_eager_work():
    tree = ast.parse(MODULE.read_text())
    imports = {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports |= {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imports <= {'__future__', 'textwrap'}
    assert not any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) for n in tree.body)
    for section in ('0.6', '1.6'):
        for source in code(section):
            tree = ast.parse(source)
            assert not any(isinstance(n, (ast.ImportFrom, ast.Import)) and any(
                word in ast.unparse(n) for word in ('solutions', 'reference', 'arena_ext')) for n in ast.walk(tree))
            assert not any(isinstance(n, ast.Attribute) and n.attr in {'savefig', 'cuda', 'from_pretrained'} for n in ast.walk(tree))


def leakage_callbacks(state, calls):
    def leak(features, labels):
        calls.append(('leak', features, labels))
        return SimpleNamespace(leaked_feature_accuracy=state[0], shifted_no_leak_accuracy=state[1])
    def cherry(effects, selected_indices):
        calls.append(('cherry', list(selected_indices)))
        return SimpleNamespace(selected_mean_effect=sum(effects[i] for i in selected_indices) / len(selected_indices),
                               population_mean_effect=sum(effects) / len(effects))
    return {'label_leakage_report': leak, 'cherry_pick_report': cherry,
            't': SimpleNamespace(tensor=lambda data, **kwargs: data, float64='float64')}


def test_06_plotted_data_are_live_callback_results(monkeypatch):
    plt = plotting(monkeypatch)
    calls = []
    namespace = leakage_callbacks([0.8, 0.2], calls)
    run(code('0.6')[0], namespace)
    bars = [call for ax in plt.axes for call in ax.calls if call[0] == 'bar']
    assert list(bars[0][1][1]) == [0.8, 0.2]
    assert list(bars[1][1][1]) == [9.0, 2.6, 2.6]
    assert [c[0] for c in calls] == ['leak', 'cherry', 'cherry']
    assert plt.shown == 1
    plt.axes.clear()
    run(code('0.6')[0], leakage_callbacks([0.6, 0.4], []))
    changed = next(call for ax in plt.axes for call in ax.calls if call[0] == 'bar')
    assert list(changed[1][1]) == [0.6, 0.4]


def test_06_play_changes_selection_and_recomputes_own_callback(monkeypatch):
    plt = plotting(monkeypatch)
    calls = []
    namespace = leakage_callbacks([1.0, 0.0], calls)
    plot, play = code('0.6')
    run(plot, namespace)
    before = len(calls)
    run(play, namespace)
    assert calls[before:] == [('cherry', [0, 1, 8, 9])]
    last_bar = [call for ax in plt.axes for call in ax.calls if call[0] == 'bar'][-1]
    assert list(last_bar[1][1]) == [9.0, 5.0, 2.6]


def memory_callback(calls, parameter_scale=1.0):
    def estimate(**kwargs):
        calls.append(dict(kwargs))
        parameter = kwargs['num_parameters'] * 2 / 1024**3 * parameter_scale
        activation = kwargs['batch_size'] * kwargs['context_length'] * kwargs['hidden_size'] * 2 * 2 / 1024**3
        kv = (2 * kwargs['batch_size'] * kwargs['context_length'] * kwargs.get('num_layers', 0)
              * kwargs.get('num_key_value_heads', 0) * kwargs.get('head_dim', 0) * 2 / 1024**3)
        return SimpleNamespace(parameter_gb=parameter, kv_cache_gb=kv, activation_gb=activation,
                               total_gb=parameter + kv + activation + kwargs['overhead_gb'])
    return estimate


def test_16_callback_inputs_and_all_six_curves_are_live(monkeypatch):
    plt = plotting(monkeypatch)
    calls = []
    run(code('1.6')[0], {'estimate_inference_memory': memory_callback(calls)})
    assert len(calls) == 8
    assert [c['context_length'] for c in calls[::2]] == [128, 512, 2048, 8192]
    assert all(c['batch_size'] == 1 for c in calls)
    assert all(c['dtype'] == 'bfloat16' and c['overhead_gb'] == 0.25 for c in calls)
    assert all('num_layers' not in c and 'num_key_value_heads' not in c and 'head_dim' not in c for c in calls[1::2])
    plots = [c for ax in plt.axes for c in ax.calls if c[0] == 'plot']
    assert len(plots) == 5
    by_label = {c[2]['label']: list(c[1][1]) for c in plots}
    assert by_label['KV cache'][-1] == 0.5
    assert by_label['Activations'][-1] == 0.03125
    assert by_label['Total'][-1] - by_label['Omitted KV (underestimate)'][-1] == 0.5
    assert any(c[0] == 'axhline' and c[1][0] == 3.0 for ax in plt.axes for c in ax.calls)
    plt.axes.clear()
    run(code('1.6')[0], {'estimate_inference_memory': memory_callback([], parameter_scale=2.0)})
    mutated = next(c for ax in plt.axes for c in ax.calls if c[0] == 'plot' and c[2]['label'] == 'Parameters')
    assert list(mutated[1][1]) == [2 * by_label['Parameters'][0]] * 4


def test_16_play_changes_only_batch_and_preserves_parameter_curve(monkeypatch):
    plt = plotting(monkeypatch)
    calls = []
    namespace = {'estimate_inference_memory': memory_callback(calls)}
    plot, play = code('1.6')
    run(plot, namespace)
    initial = [dict(c) for c in calls]
    calls.clear()
    run(play, namespace)
    assert len(calls) == 8
    assert all(c['batch_size'] == 2 for c in calls)
    assert [{k:v for k,v in c.items() if k != 'batch_size'} for c in calls] == [
        {k:v for k,v in c.items() if k != 'batch_size'} for c in initial]
    last_total = [c for ax in plt.axes for c in ax.calls if c[0] == 'plot' and c[2]['label'] == 'Total'][-1]
    assert last_total[1][1][-1] > 3.0


@pytest.mark.parametrize('section, callback', [('0.6', 'label_leakage_report'), ('0.6', 'cherry_pick_report'), ('1.6', 'estimate_inference_memory')])
def test_unimplemented_student_callback_is_not_replaced(monkeypatch, section, callback):
    plotting(monkeypatch)
    namespace = leakage_callbacks([1.0, 0.0], []) if section == '0.6' else {}
    def unfinished(*args, **kwargs): raise NotImplementedError('student must implement the existing method')
    namespace[callback] = unfinished
    with pytest.raises(NotImplementedError, match='student must implement'):
        run(code(section)[0], namespace)


@pytest.mark.parametrize('section', ['0.6', '1.6'])
def test_missing_student_callback_raises(monkeypatch, section):
    plotting(monkeypatch)
    namespace = {'t': SimpleNamespace(tensor=lambda data, **kwargs: data, float64='float64')}
    with pytest.raises(NameError): run(code(section)[0], namespace)


def test_interpretations_disclose_controls_units_and_limits():
    m = provider()
    text06 = '\n'.join(c['source'] for c in m.additions('0.6') if c['cell_type']=='markdown')
    text16 = '\n'.join(c['source'] for c in m.additions('1.6') if c['cell_type']=='markdown')
    assert 'anti-correlated' in text06 and 'independent null' in text06
    assert 'GiB' in text16 and 'no GPU allocation' in text16
    assert all(word in text06 + text16 for word in ['Forecast', '<summary>Help', '<summary>Expected', 'Limitations'])


def exact_core(section):
    """Load only pinned analytic definitions, never either source module's setup."""
    from dataclasses import dataclass
    root = next(p for p in HERE.parents if (p / 'infrastructure/core/config.yaml').is_file())
    if section == '0.6':
        relative = 'chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice/solutions.py'
        selected = {'LabelLeakageReport', 'CherryPickReport', 'binary_accuracy',
                    'label_leakage_report', 'cherry_pick_report'}
        import torch
        namespace = {'dataclass': dataclass, 't': torch, '__name__': __name__}
    else:
        relative = 'chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice/solutions.py'
        selected = {'MemoryBudget', 'bytes_per_value', 'estimate_inference_memory'}
        namespace = {'dataclass': dataclass, '__name__': __name__}
    tree = ast.parse((root / relative).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected]
    nodes += [n for n in tree.body if isinstance(n, ast.AnnAssign)
              and isinstance(n.target, ast.Name) and n.target.id == '_DTYPE_BYTES'] if section == '1.6' else []
    future = ast.parse('from __future__ import annotations').body[0]
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, '<exact-tiny-analytic-core>', 'exec'), namespace)
    return namespace


def test_exact_06_core_tiny_fixture_regenerates_plot_and_play(monkeypatch):
    plt = plotting(monkeypatch)
    namespace = exact_core('0.6')
    plot, play = code('0.6')
    run(plot, namespace)
    bars = [c for ax in plt.axes for c in ax.calls if c[0] == 'bar']
    assert list(bars[0][1][1]) == [1.0, 0.0]
    assert list(bars[1][1][1]) == [9.0, 2.6, 2.6]
    run(play, namespace)
    assert namespace['live06_play'].selected_mean_effect == 5.0
    assert namespace['live06_play'].population_mean_effect == 2.6


def test_exact_16_core_accounting_is_gib_and_play_crosses_budget(monkeypatch):
    plotting(monkeypatch)
    namespace = exact_core('1.6')
    plot, play = code('1.6')
    run(plot, namespace)
    before = namespace['live16_full_budgets'][-1]
    omitted = namespace['live16_omitted_kv_budgets'][-1]
    assert before.kv_cache_gb == 0.5
    assert before.activation_gb == 0.03125
    assert before.total_gb - omitted.total_gb == 0.5
    assert before.total_gb < 3.0
    run(play, namespace)
    after = namespace['live16_full_budgets'][-1]
    assert after.parameter_gb == before.parameter_gb
    assert after.kv_cache_gb == 2 * before.kv_cache_gb
    assert after.activation_gb == 2 * before.activation_gb
    assert after.total_gb > 3.0


def test_only_literal_cpu_tensor_fixtures_are_allocated():
    for section in ('0.6', '1.6'):
        for source in code(section):
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'tensor':
                    device = [keyword.value for keyword in node.keywords if keyword.arg == 'device']
                    assert len(device) == 1 and isinstance(device[0], ast.Constant) and device[0].value == 'cpu'
