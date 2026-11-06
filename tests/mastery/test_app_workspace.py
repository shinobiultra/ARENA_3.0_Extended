"""Compact information architecture and result boundaries on real Streamlit reruns.

AppTest proves nesting and behavior, not browser pixel layout or human usability.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def app(tmp_path, monkeypatch):
    monkeypatch.setenv('ARENA_MASTERY_DATA_DIR', str(tmp_path / 'private'))
    return AppTest.from_file(str(ROOT / 'mastery_hub/app.py'), default_timeout=20).run()


def button(at, label):
    return next(item for item in at.button if item.label == label)


def expander(at, prefix):
    return next(item for item in at.expander if item.label.startswith(prefix))


def path_to_key(node, key, ancestors=()):
    if getattr(node, 'key', None) == key:
        return (*ancestors, node)
    for child in getattr(node, 'children', {}).values():
        path = path_to_key(child, key, (*ancestors, node))
        if path:
            return path
    return ()


def test_essential_settings_and_prediction_share_one_workspace(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    rate_path = path_to_key(at.main, 'eta')
    prediction_path = path_to_key(at.main, 'prediction')
    workspace = rate_path[1]
    assert workspace is prediction_path[1]
    assert workspace.children[0] is rate_path[2]
    assert workspace.children[1] is prediction_path[2]
    assert not any(node.type == 'expander' for node in rate_path)
    assert not any(node.type == 'expander' for node in path_to_key(at.main, 'beta'))
    advanced = expander(at, 'Advanced settings')
    assert not advanced.proto.expanded
    assert {item.key for item in advanced.get('number_input')} == {
        'lambda_min', 'lambda_max', 'angle_deg', 'noise_std', 'x0_0', 'x0_1', 'seed', 'steps',
    }
    # Commit and reveal are adjacent actions in one horizontal group.
    commit = button(at, 'Commit prediction')
    action_group = next(node for node in workspace.children[1].children.values()
                        if node.type == 'flex_container' and commit in list(node.get('button')))
    assert [item.label for item in action_group.get('button')] == ['Commit prediction', 'Reveal result']
    assert not at.exception


def test_default_has_one_primary_result_and_closed_secondary_material(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    assert at.radio(key='mode').value == 'Explore'
    assert len(at.get('plotly_chart')) == 1
    assert at.radio(key='result_view').value == 'Loss'
    assert not at.sidebar.get('number_input')
    # Every source link/download is behind a closed shelf; none fills the sidebar.
    assert not [node for node in at.sidebar.children.values() if node.type in ('markdown', 'download_button')]
    for label in ('Original materials', 'More prediction notebooks', 'Course map',
                  'Optional study duration', 'Start here', 'Step data', 'Chart options',
                  'Controls and limits', 'Optional reflection', 'Import an exported'):
        assert not expander(at, label).proto.expanded
    for node in at.main.get('dataframe'):
        assert any(node in list(panel.get('dataframe')) for panel in at.expander)
    assert not any(item.key and item.key.startswith('reflection_') for item in at.text_area)
    assert any(item.value.startswith('Next:') for item in at.caption)
    assert not at.exception


def test_challenge_next_action_tracks_commit_and_has_no_result_leak(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    at.selectbox(key='case_template').select('Momentum transfer').run()
    assert at.session_state['progress']['current_attempt']['assessment'] == 'unseen'
    assert any('Local exposure: unseen' in item.value and
               'guided practice, not an independent assessment' in item.value
               for item in at.caption)
    assert any('choose an answer' in item.value for item in at.caption)
    assert not at.get('plotly_chart')
    assert not at.main.get('dataframe')
    assert not any('spectral radius:' in item.value for item in at.caption)
    assert button(at, 'Reveal result').disabled
    at.text_area(key='reasoning').set_value('Both eigenmodes should decay with this momentum.').run()
    button(at, 'Commit prediction').click().run()
    assert any('reveal your locked prediction' in item.value for item in at.caption)
    assert not at.get('plotly_chart')
    assert not at.success
    button(at, 'Reveal result').click().run()
    assert any('test a control' in item.value for item in at.caption)
    assert len(at.get('plotly_chart')) == 1
    assert any('Brier score' in item.value for item in at.success)
    assert not at.exception


def test_view_choices_keep_step_values_and_scientific_progress_unchanged(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    before = deepcopy(at.session_state['progress'])
    at.slider(key='selected_step').set_value(1).run()
    for view, expected_y, units in [
        ('Loss', .405, 'loss units'),
        ('Trajectory', 0., 'coordinate units'),
        ('Update size', 1.004987562112089, 'coordinate units'),
    ]:
        at.radio(key='result_view').set_value(view).run()
        assert len(at.get('plotly_chart')) == 1
        spec = json.loads(at.get('plotly_chart')[0].proto.spec)
        marker = next(trace for trace in spec['data'] if trace.get('name') == 'Selected step')
        assert marker['y'] == pytest.approx([expected_y])
        assert units in spec['layout']['yaxis']['title']['text']
        assert spec['layout']['height'] == 420
        assert not spec['layout']['showlegend']
        assert spec['layout']['meta']['stop_reason']
        assert at.slider(key='selected_step').value == 1
        assert at.session_state['progress'] == before
    at.checkbox(key='show_legend').check().run()
    assert json.loads(at.get('plotly_chart')[0].proto.spec)['layout']['showlegend']
    at.checkbox(key='baseline_overlay').uncheck().run()
    assert all(trace.get('name') != 'Baseline' for trace in json.loads(at.get('plotly_chart')[0].proto.spec)['data'])
    assert at.session_state['progress'] == before
    assert not at.exception


def test_compact_plot_retains_zero_loss_log_warning(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.number_input(key='x0_0').set_value(0.).run()
    at.number_input(key='x0_1').set_value(0.).run()
    at.checkbox(key='log_loss').check().run()
    spec = json.loads(at.get('plotly_chart')[0].proto.spec)
    assert any('Exact zero losses omitted' in note['text'] for note in spec['layout']['annotations'])
    assert not any(note['text'].startswith(('Stop:', 'Float64 2-D')) for note in spec['layout']['annotations'])
    loss = next(trace for trace in spec['data'] if trace.get('name') == 'Recorded loss')
    assert all(value is None for value in loss['y'])
    assert not at.exception


def test_replay_is_opt_in_with_an_independent_single_result_view(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    at.text_area(key='reasoning').set_value('The baseline contracts both modes.').run()
    button(at, 'Commit prediction').click().run()
    button(at, 'Reveal result').click().run()
    assert len(at.get('plotly_chart')) == 1
    before = deepcopy(at.session_state['progress'])
    button(at, 'Replay saved attempt').click().run()
    at.radio(key='replay_result_view').set_value('Trajectory').run()
    at.slider(key='replay_selected_step').set_value(2).run()
    assert len(at.get('plotly_chart')) == 2
    assert at.radio(key='result_view').value == 'Loss'
    assert at.slider(key='selected_step').value == 0
    assert at.slider(key='replay_selected_step').value == 2
    assert at.session_state['progress'] == before
    assert not at.exception


def test_explore_uses_wrapping_control_strip_and_view_neutral_next_action(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    rate_path = path_to_key(at.main, 'eta')
    assert not any(node.type == 'column' for node in rate_path)
    strip = rate_path[-2]
    assert strip.proto.flex_container.direction == 2  # native horizontal flex; buttons wrap, not five stacked columns
    assert strip.proto.flex_container.wrap
    assert {node.key for node in strip.get('number_input')} == {'eta', 'beta'}
    assert [node.key for node in strip.get('selectbox')] == ['case_template']
    assert [node.label for node in strip.get('button')] == ['Reset to baseline']
    playback = path_to_key(at.main, 'play')[-2]
    assert playback.proto.flex_container.direction == 2
    assert playback.proto.flex_container.wrap
    assert len(playback.get('button')) == 5
    assert not any(node.type == 'column' for node in path_to_key(at.main, 'play'))
    assert not at.main.get('title')
    assert not at.info  # next guidance is compact text, not a large alert box
    for view in ('Trajectory', 'Update size'):
        at.radio(key='result_view').set_value(view).run()
        guidance = next(item.value for item in at.caption if item.value.startswith('Next:'))
        assert 'inspect the result' in guidance
        assert 'Loss' not in guidance
    assert not at.exception


def test_save_conflict_requests_reload_and_preserves_newer_exposures(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    older = deepcopy(at.session_state['progress'])
    at.number_input(key='eta').set_value(.17).run()
    durable_path = tmp_path / 'private/session.json'
    newer_bytes = durable_path.read_bytes()
    newer = json.loads(newer_bytes)
    assert len(newer['exposed_cases']) > len(older['exposed_cases'])
    # Simulate a stale browser's in-memory ledger, as can happen after hot reload
    # or a concurrent local session. Guard must reject rather than erase history.
    at.session_state['progress'] = older
    at.number_input(key='eta').set_value(.1).run()
    assert any('Progress was not saved' in item.value for item in at.error)
    assert any('Reload this page' in item.value for item in at.info)
    assert durable_path.read_bytes() == newer_bytes
    assert not at.exception


@pytest.mark.parametrize('status', ['committed', 'revealed'])
def test_reloaded_locked_form_restores_immutable_forecast_and_disables_edits(tmp_path, monkeypatch, status):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    at.number_input(key='eta').set_value(.21).run()
    at.radio(key='prediction').set_value('Not strictly stable').run()
    at.slider(key='confidence').set_value(.9).run()
    reason = 'The high-curvature mode has multiplier -1.1, so it expands.'
    at.text_area(key='reasoning').set_value(reason).run()
    button(at, 'Commit prediction').click().run()
    if status == 'revealed':
        button(at, 'Reveal result').click().run()
    immutable = deepcopy(at.session_state['progress'])
    durable_path = tmp_path / 'private/session.json'
    recorded_bytes = durable_path.read_bytes()
    reloaded = app(tmp_path, monkeypatch)
    assert reloaded.session_state['progress']['current_attempt']['status'] == status
    assert reloaded.radio(key='prediction').value == 'Not strictly stable'
    assert reloaded.slider(key='confidence').value == .9
    assert reloaded.text_area(key='reasoning').value == reason
    for element in (reloaded.radio(key='prediction'), reloaded.slider(key='confidence'),
                    reloaded.text_area(key='reasoning')):
        assert element.disabled
    assert button(reloaded, 'Commit prediction').disabled
    assert button(reloaded, 'Reveal result').disabled == (status == 'revealed')
    reloaded.run()
    assert reloaded.session_state['progress'] == immutable
    assert durable_path.read_bytes() == recorded_bytes
    assert not reloaded.exception
    # A changed scientific configuration starts a genuinely editable new attempt.
    reloaded.number_input(key='eta').set_value(.17).run()
    assert reloaded.session_state['progress']['current_attempt']['status'] == 'uncommitted'
    for element in (reloaded.radio(key='prediction'), reloaded.slider(key='confidence'),
                    reloaded.text_area(key='reasoning')):
        assert not element.disabled
    assert reloaded.session_state['progress']['history'] == immutable['history']
