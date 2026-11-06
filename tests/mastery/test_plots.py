import json

import numpy as np
import pytest


def modules():
    from arena_ext.mastery import plots, quadratic
    return plots, quadratic


def test_plot_values_equal_trace():
    plots, q = modules()
    config = q.QuadraticConfig(steps=8)
    trace = q.simulate(config)
    figures = plots.build_figures(trace, config)
    assert set(figures) == {'trajectory', 'loss', 'updates'}
    trajectory = next(t for t in figures['trajectory'].data if t.name == 'Accepted trajectory')
    np.testing.assert_array_equal(trajectory.x, trace.states[:, 0])
    np.testing.assert_array_equal(trajectory.y, trace.states[:, 1])
    np.testing.assert_array_equal(figures['loss'].data[0].y, trace.losses)
    np.testing.assert_array_equal(figures['updates'].data[0].y, trace.update_norms)
    assert 'analytic toy' in figures['trajectory'].layout.title.text.lower()


def test_step_selection_synchronizes_markers_without_changing_trace():
    plots, q = modules()
    config = q.QuadraticConfig(steps=8)
    trace = q.simulate(config)
    figures = plots.build_figures(trace, config, step=3)
    marker = next(t for t in figures['trajectory'].data if t.name == 'Selected step')
    assert marker.x[0] == trace.states[3, 0]
    assert figures['loss'].data[1].y[0] == trace.losses[3]
    assert figures['updates'].data[1].y[0] == trace.update_norms[3]


def test_log_zero_loss_is_omitted_and_annotated():
    plots, q = modules()
    config = q.QuadraticConfig(x0=(0, 0), steps=3)
    figures = plots.build_figures(q.simulate(config), config, log_loss=True)
    loss = figures['loss']
    assert loss.layout.yaxis.type == 'log'
    assert all(y is None for y in loss.data[0].y)
    assert any('zero' in a.text.lower() and 'omitted' in a.text.lower() for a in loss.layout.annotations)
    json.loads(loss.to_json())


def test_nonzero_log_loss_values_are_not_changed():
    plots, q = modules()
    config = q.QuadraticConfig(steps=5)
    trace = q.simulate(config)
    loss = plots.build_figures(trace, config, log_loss=True)['loss']
    np.testing.assert_array_equal(loss.data[0].y, trace.losses)


def test_empty_trace_and_extreme_trace_export_without_nonfinite_plot_values():
    plots, q = modules()
    for config in (q.QuadraticConfig(x0=(1e308, 0)), q.QuadraticConfig(lambda_min=1e-308, lambda_max=1e-308, x0=(1e154, 0))):
        trace = q.simulate(config)
        figures = plots.build_figures(trace, config)
        for fig in figures.values():
            encoded = fig.to_json()
            assert 'NaN' not in encoded and 'Infinity' not in encoded
        assert any(trace.stop_reason in a.text for a in figures['trajectory'].layout.annotations)


@pytest.mark.parametrize('step', [-1, 4, True, 1.5])
def test_invalid_step_selection(step):
    plots, q = modules()
    config = q.QuadraticConfig(steps=3)
    with pytest.raises(ValueError):
        plots.build_figures(q.simulate(config), config, step=step)
