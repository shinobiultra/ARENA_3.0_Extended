"""Real Streamlit reruns must preserve predictions and use changed scientific inputs."""
from pathlib import Path
from copy import deepcopy
import json
from time import monotonic

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def app(tmp_path, monkeypatch):
    monkeypatch.setenv('ARENA_MASTERY_DATA_DIR', str(tmp_path / 'private'))
    return AppTest.from_file(str(ROOT / 'mastery_hub/app.py'), default_timeout=20).run()


def button(at, label):
    return next(item for item in at.button if item.label == label)


def test_offline_cpu_session_and_real_controls(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    assert not at.exception
    assert any('analytic toy' in c.value.lower() for c in at.caption)
    before = at.session_state['progress']['config']['eta']
    at.number_input(key='eta').set_value(0.21).run()
    assert not at.exception
    assert at.session_state['progress']['config']['eta'] == 0.21
    assert before != 0.21
    assert at.session_state['progress']['exposed_cases']
    button(at, 'Reset to baseline').click().run()
    assert at.session_state['progress']['config']['eta'] == 0.1


def test_commit_before_reveal_and_config_change_invalidates(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    assert button(at, 'Reveal result').disabled
    at.text_area(key='reasoning').set_value('I predict contraction along both eigendirections.').run()
    button(at, 'Commit prediction').click().run()
    assert at.session_state['progress']['current_attempt']['status'] == 'committed'
    at.number_input(key='eta').set_value(0.21).run()
    assert at.session_state['progress']['current_attempt']['status'] == 'uncommitted'
    assert button(at, 'Reveal result').disabled
    assert any('invalidat' in item.value.lower() for item in at.warning)


def test_completed_attempt_survives_reruns_and_reload(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    at.text_area(key='reasoning').set_value('Both eigenmodes should contract.').run()
    button(at, 'Commit prediction').click().run()
    button(at, 'Reveal result').click().run()
    assert not at.exception
    history = at.session_state['progress']['history']
    assert len(history) == 1
    assert history[0]['attempt']['assessment'] == 'practice'
    at.run()
    assert at.session_state['progress']['history'] == history
    reloaded = app(tmp_path, monkeypatch)
    assert reloaded.session_state['progress']['history'] == history
    assert reloaded.session_state['progress']['current_attempt']['status'] == 'revealed'


def test_step_selection_and_all_controls_recompute_config(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    values = dict(lambda_min=2.0, lambda_max=12.0, angle_deg=35.0, x0_0=0.5, x0_1=-0.5, eta=0.04, beta=0.4, noise_std=0.2, seed=7, steps=35)
    for key, value in values.items():
        at.number_input(key=key).set_value(value).run()
        assert not at.exception, key
    cfg = at.session_state['progress']['config']
    assert cfg['x0'] == [0.5, -0.5]
    assert cfg['steps'] == 35
    assert cfg['noise_std'] == 0.2
    at.slider(key='selected_step').set_value(5).run()
    assert at.session_state['selected_step'] == 5
    assert not at.exception


def test_replay_and_private_reflection_do_not_rewrite_history(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    at.text_area(key='reasoning').set_value('Both modes contract under the baseline.').run()
    button(at, 'Commit prediction').click().run()
    button(at, 'Reveal result').click().run()
    history = at.session_state['progress']['history'].copy()
    attempt_id = history[0]['attempt_id']
    at.text_area(key='reflection_' + attempt_id).set_value('Noise changes a limiting trajectory, not the deterministic oracle.').run()
    button(at, 'Replay saved attempt').click().run()
    assert not at.exception
    assert at.session_state['progress']['history'] == history
    assert at.session_state['progress']['reflections'][attempt_id].startswith('Noise')


def test_pending_challenge_mode_and_prediction_survive_reload(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    at.selectbox(key='case_template').select('Curvature interpolation').run()
    at.text_area(key='reasoning').set_value('The proposed eigendirection contraction should persist.').run()
    button(at, 'Commit prediction').click().run()
    pending = at.session_state['progress']['current_attempt'].copy()
    reloaded = app(tmp_path, monkeypatch)
    assert reloaded.radio(key='mode').value == 'Challenge'
    assert reloaded.session_state['progress']['current_attempt'] == pending
    assert not button(reloaded, 'Reveal result').disabled


def test_irrelevant_seed_change_cannot_manufacture_unseen_stability(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    at.number_input(key='seed').set_value(12).run()
    assert at.session_state['progress']['current_attempt']['assessment'] == 'practice'


def test_newly_exposed_pending_case_becomes_practice_before_scoring(tmp_path, monkeypatch):
    from copy import deepcopy
    from arena_ext.mastery.challenges import commit_prediction, expose_case, new_attempt, reveal
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    at.number_input(key='eta').set_value(0.17).run()
    at.text_area(key='reasoning').set_value('Both eigenmodes lie within the stable region.').run()
    button(at, 'Commit prediction').click().run()
    progress = deepcopy(at.session_state['progress'])
    pending = progress['current_attempt']
    assert pending['assessment'] == 'unseen'
    # An imported completed record exposes the answer while this prediction is pending.
    progress['exposed_cases'] = expose_case(progress['exposed_cases'], pending['challenge_id'], pending['config_fingerprint'])
    at.session_state['progress'] = progress
    at.run()
    assert at.session_state['progress']['current_attempt']['assessment'] == 'practice'
    button(at, 'Reveal result').click().run()
    assert at.session_state['progress']['history'][-1]['attempt']['assessment'] == 'practice'
    assert not at.exception


def test_route_shows_prerequisite_original_and_notebook(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    assert any('Prerequisite' in item.value and 'Basics' in item.value for item in at.sidebar.markdown)
    assert any('Original foundation' in item.value and 'chapter0_fundamentals' in item.value for item in at.sidebar.markdown)
    assert any('companion notebook' in item.value.lower() for item in at.markdown)


def test_playback_advances_actual_selection_and_manual_controls_pause(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    before = deepcopy(at.session_state['progress'])
    button(at, 'Play').click().run()
    assert at.session_state['playing']
    at.session_state['playback_tick'] = monotonic() - 1
    at.run()
    assert at.slider(key='selected_step').value == 1
    # One primary chart at a time. Each selector must retain the same step and
    # the independent hand calculation for x1=(.9,0), L1=.405.
    button(at, 'Pause').click().run()
    expected = {
        'Trajectory': ([0.9], [0.0]),
        'Loss': ([1], [0.405]),
        'Update size': ([1], [1.004987562112089]),
    }
    for view, (x, y) in expected.items():
        at.radio(key='result_view').set_value(view).run()
        assert len(at.get('plotly_chart')) == 1
        selected = next(trace for trace in json.loads(at.get('plotly_chart')[0].proto.spec)['data']
                        if trace.get('name') == 'Selected step')
        assert selected['x'] == pytest.approx(x)
        assert selected['y'] == pytest.approx(y)
        assert at.slider(key='selected_step').value == 1
    paused = at.session_state['selected_step']
    at.session_state['playback_tick'] = monotonic() - 10
    at.run()
    assert at.slider(key='selected_step').value == paused
    button(at, 'Next step').click().run()
    assert at.slider(key='selected_step').value == paused + 1
    button(at, 'Previous step').click().run()
    assert at.slider(key='selected_step').value == paused
    button(at, 'Reset playback').click().run()
    assert at.slider(key='selected_step').value == 0
    assert not at.session_state['playing']
    assert at.session_state['progress'] == before
    assert not at.exception


def test_playback_stops_at_end_and_resets_on_scientific_changes(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.number_input(key='steps').set_value(2).run()
    at.slider(key='selected_step').set_value(1).run()
    button(at, 'Play').click().run()
    at.session_state['playback_tick'] = monotonic() - 1
    at.run()
    assert at.slider(key='selected_step').value == 2
    assert not at.session_state['playing']
    button(at, 'Play').click().run()
    assert at.slider(key='selected_step').value == 0
    at.slider(key='selected_step').set_value(1).run()
    assert not at.session_state['playing']
    button(at, 'Play').click().run()
    at.number_input(key='eta').set_value(0.12).run()
    assert at.slider(key='selected_step').value == 0
    assert not at.session_state['playing']
    button(at, 'Play').click().run()
    button(at, 'Reset to baseline').click().run()
    assert at.slider(key='selected_step').value == 0
    assert not at.session_state['playing']
    assert at.session_state['progress']['config']['eta'] == 0.1
    assert not at.exception


def test_optional_timer_starts_explicitly_and_freezes_without_progress_changes(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    before = deepcopy(at.session_state['progress'])
    assert not at.checkbox(key='study_timer_enabled').value
    assert all(item.label != 'Start / resume study timer' for item in at.button)
    assert sorted(path.name for path in (tmp_path / 'private').iterdir()) == ['session.json']
    at.checkbox(key='study_timer_enabled').check().run()
    assert at.session_state['study_started_at'] is None
    assert at.session_state['study_elapsed'] == 0
    button(at, 'Start / resume study timer').click().run()
    assert at.session_state['study_started_at'] is not None
    at.session_state['study_started_at'] = monotonic() - 3
    at.run()
    elapsed = float(next(item.value for item in at.metric if item.label == 'Elapsed (seconds)'))
    remaining = float(next(item.value for item in at.metric if item.label == 'Remaining (seconds)'))
    assert elapsed >= 3
    assert elapsed + remaining == pytest.approx(1500, abs=1)
    button(at, 'Stop study timer').click().run()
    stopped = at.session_state['study_elapsed']
    at.run()
    assert at.session_state['study_started_at'] is None
    assert at.session_state['study_elapsed'] == stopped
    assert at.session_state['progress'] == before
    button(at, 'Start / resume study timer').click().run()
    at.checkbox(key='study_timer_enabled').uncheck().run()
    assert at.session_state['study_started_at'] is None
    assert at.session_state['study_elapsed'] >= stopped
    at.checkbox(key='study_timer_enabled').check().run()
    button(at, 'Reset study timer').click().run()
    assert at.session_state['study_elapsed'] == 0
    assert at.session_state['progress'] == before
    assert not at.exception


def test_timer_completes_locally_and_does_not_invalidate_locked_prediction(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    at.text_area(key='reasoning').set_value('Both eigendirections contract.').run()
    button(at, 'Commit prediction').click().run()
    pending = deepcopy(at.session_state['progress'])
    at.checkbox(key='study_timer_enabled').check().run()
    at.selectbox(key='study_duration').select('Short · 10 minutes').run()
    button(at, 'Start / resume study timer').click().run()
    at.session_state['study_started_at'] = monotonic() - 1000
    at.run()
    assert at.session_state['study_elapsed'] == 600
    assert at.session_state['study_started_at'] is None
    assert next(item.value for item in at.metric if item.label == 'Remaining (seconds)') == '0'
    assert at.session_state['progress'] == pending
    assert not button(at, 'Reveal result').disabled
    reloaded = app(tmp_path, monkeypatch)
    assert not reloaded.checkbox(key='study_timer_enabled').value
    assert reloaded.session_state['progress'] == pending
    reloaded.number_input(key='eta').set_value(0.21).run()
    assert reloaded.session_state['progress']['current_attempt']['status'] == 'uncommitted'
    assert button(reloaded, 'Reveal result').disabled
    assert not at.exception


def test_replay_playback_is_separate_from_current_view_and_immutable_export(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.radio(key='mode').set_value('Challenge').run()
    at.text_area(key='reasoning').set_value('Both eigenmodes contract.').run()
    button(at, 'Commit prediction').click().run()
    button(at, 'Reveal result').click().run()
    before = deepcopy(at.session_state['progress'])
    at.slider(key='selected_step').set_value(4).run()
    button(at, 'Replay saved attempt').click().run()
    button(at, 'Play replay').click().run()
    at.session_state['replay_playback_tick'] = monotonic() - 1
    at.run()
    assert at.slider(key='replay_selected_step').value == 1
    assert at.slider(key='selected_step').value == 4
    assert at.session_state['progress'] == before
    # Durable JSON must contain only the original scientific progress fields.
    persisted = json.loads((tmp_path / 'private' / 'session.json').read_text())
    assert persisted == before
    assert not at.exception


def test_no_accepted_updates_has_no_fabricated_playback_steps(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    for key, value in [('lambda_min', 1000.0), ('lambda_max', 1000.0),
                       ('x0_0', 100.0), ('x0_1', 100.0), ('eta', 10.0)]:
        at.number_input(key=key).set_value(value).run()
    assert button(at, 'Play').disabled
    button(at, 'Next step').click().run()
    assert at.session_state['selected_step'] == 0
    assert not at.session_state['playing']
    assert not at.exception


def test_playback_and_timer_actions_keep_each_others_session_controls(tmp_path, monkeypatch):
    at = app(tmp_path, monkeypatch)
    at.checkbox(key='study_timer_enabled').check().run()
    at.selectbox(key='study_duration').select('Short · 10 minutes').run()
    button(at, 'Start / resume study timer').click().run()
    button(at, 'Play').click().run()
    assert at.checkbox(key='study_timer_enabled').value
    assert at.selectbox(key='study_duration').value == 'Short · 10 minutes'
    assert at.session_state['study_started_at'] is not None
    at.session_state['playback_tick'] = monotonic() - 1
    at.run()
    step = at.slider(key='selected_step').value
    button(at, 'Stop study timer').click().run()
    assert at.session_state['playing']
    assert at.slider(key='selected_step').value >= step
    button(at, 'Pause').click().run()
    assert at.checkbox(key='study_timer_enabled').value
    assert at.session_state['study_started_at'] is None
    at.number_input(key='steps').set_value(1).run()
    button(at, 'Play').click().run()
    at.session_state['playback_tick'] = monotonic() - 1
    at.run()
    assert not at.session_state['playing']
    assert at.checkbox(key='study_timer_enabled').value
    assert at.selectbox(key='study_duration').value == 'Short · 10 minutes'
    assert not at.exception
