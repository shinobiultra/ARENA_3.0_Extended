"""Local prediction-first optimizer companion. Launch from the repository root."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from arena_ext.mastery.catalog import load_catalog
from arena_ext.mastery.challenges import commit_prediction, expose_case, new_attempt, reveal
from arena_ext.mastery.plots import build_figures
from arena_ext.mastery.quadratic import QuadraticConfig, SIMULATOR_VERSION, simulate, spectral_radius
from arena_ext.mastery.records import (
    default_data_dir, export_attempt, fingerprint, load_session, loads_attempt,
    replay, save_session, validate_record,
)

CHALLENGE_ID = 'deterministic-stability-v1'
EQUIVALENCE_ID = 'deterministic-stability-eigenmodes-v1'
DEFAULT = asdict(QuadraticConfig())
TEMPLATES = {
    'Current settings': {},
    'Curvature interpolation': {'lambda_max': 13.0, 'eta': 0.12, 'beta': 0.0},
    'Curvature extrapolation': {'lambda_max': 35.0, 'eta': 0.065, 'beta': 0.0},
    'Momentum transfer': {'lambda_max': 18.0, 'eta': 0.15, 'beta': 0.6},
}


def set_controls(config):
    for key, value in config.items():
        if key not in ('simulator_version', 'x0'):
            st.session_state[key] = value
    st.session_state['x0_0'], st.session_state['x0_1'] = config['x0']
    st.session_state['selected_step'] = 0
    pause_playback()


def reset_controls():
    set_controls(DEFAULT)
    st.session_state['case_template'] = 'Current settings'


def apply_template():
    set_controls({**DEFAULT, **TEMPLATES[st.session_state['case_template']]})


def source_digest():
    paths = [Path(__file__), *sorted((ROOT / 'arena_ext/mastery').glob('*.py'))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return 'source-sha256:' + digest.hexdigest() + f'; Python {sys.version.split()[0]}; NumPy {np.__version__}'


def scientific_config(config):
    value = asdict(config)
    value['x0'] = list(config.x0)
    return {**value, 'simulator_version': SIMULATOR_VERSION}


def controls(config):
    """Run the named controls; no model, network, or training is involved."""
    baseline = QuadraticConfig()
    zero = simulate(replace(config, eta=0.0))
    optimum = simulate(replace(config, x0=(0.0, 0.0), noise_std=0.0))
    unstable = simulate(replace(baseline, eta=0.23))
    theta = np.deg2rad(37)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = simulate(replace(baseline, angle_deg=37, x0=tuple(rotation @ baseline.x0)))
    return {
        'zero_learning_rate_no_movement': bool(np.all(zero.states == zero.states[0])) if len(zero.states) else False,
        'noiseless_optimum_is_fixed': bool(np.all(optimum.losses == 0)),
        'unstable_step_loss_increases': bool(unstable.losses[-1] > unstable.losses[0]),
        'rotation_preserves_loss_trajectory': bool(np.allclose(rotated.losses, simulate(baseline).losses, rtol=1e-12, atol=1e-12)),
    }


def envelope(attempt, config, outcomes):
    return validate_record(dict(
        schema_version=1, unit_id='u02_optimization', challenge_id=CHALLENGE_ID,
        attempt_id=attempt['attempt_id'], config=config,
        config_fingerprint=fingerprint(config),
        source_revision='CS312 homepage checked 2026-10-15; design PR2 c8dec641421ba2fd833a693890e66cc82371a475',
        code_revision=source_digest(), seeds=[config['seed']], evidence_class='analytic_toy',
        attempt=attempt, control_outcomes=outcomes,
        next_action='Implement the heavy-ball step in the companion notebook, then try an unseen regime.',
        reflection='',
    ))


def pause_playback(prefix=''):
    st.session_state[prefix + 'playing'] = False
    st.session_state[prefix + 'playback_tick'] = None


def change_playback(prefix, last, action):
    step_key = prefix + 'selected_step'
    step = st.session_state.get(step_key, 0)
    pause_playback(prefix)
    if action == 'play':
        st.session_state[step_key] = 0 if step >= last else step
        st.session_state[prefix + 'playing'] = True
        st.session_state[prefix + 'playback_tick'] = monotonic()
    elif action == 'previous':
        st.session_state[step_key] = max(0, step - 1)
    elif action == 'next':
        st.session_state[step_key] = min(last, step + 1)
    elif action == 'reset':
        st.session_state[step_key] = 0


def show_trace(trace, config, *, prefix=''):
    """Place a fixed-trace view and return its deferred fragment renderer."""
    if not len(trace.states):
        st.error('The initial loss cannot be represented in float64. No finite trace was recorded.')
        pause_playback(prefix)
        return
    last = len(trace.states) - 1
    signature = fingerprint(scientific_config(config))
    if st.session_state.get(prefix + 'view_config') != signature:
        st.session_state[prefix + 'view_config'] = signature
        st.session_state[prefix + 'selected_step'] = 0
        pause_playback(prefix)
    # Hidden Challenge widgets are cleaned up by Streamlit; restore the view
    # default and detach its value before an action requests a full rerun.
    step_key = prefix + 'selected_step'
    st.session_state[step_key] = st.session_state.get(step_key, 0)
    for name in ('log_loss', 'baseline_overlay', 'result_view', 'show_legend'):
        if prefix + name in st.session_state:
            st.session_state[prefix + name] = st.session_state[prefix + name]
    playing = st.session_state.get(prefix + 'playing', False)
    suffix = ' replay' if prefix else ''
    st.radio('Result view' + suffix, ['Loss', 'Trajectory', 'Update size'], horizontal=True,
             label_visibility='collapsed', key=prefix + 'result_view')
    with st.container(horizontal=True, wrap=True, vertical_alignment='center', gap='small'):
        for label, action, disabled in (
            ('Play', 'play', not last or playing),
            ('Pause', 'pause', not playing),
            ('Previous step', 'previous', False),
            ('Next step', 'next', False),
            ('Reset playback', 'reset', False),
        ):
            # Outside the timed fragment: actions register the next interval
            # before its renderer runs, preserving the ordinary rerun lifecycle.
            st.button(label + suffix, key=prefix + action, disabled=disabled,
                      on_click=change_playback, args=(prefix, last, action))

    view = st.container()
    baseline = simulate(QuadraticConfig())

    @st.fragment(run_every=0.5 if playing else None)
    def trace_view():
        _trace_view(trace, config, baseline, prefix=prefix, last=last, scheduled=playing)

    def render():
        with view:
            trace_view()

    return render


def _trace_view(trace, config, baseline, *, prefix, last, scheduled):
    playing_key = prefix + 'playing'
    tick_key = prefix + 'playback_tick'
    step_key = prefix + 'selected_step'
    # A manual slider callback pauses first; refresh to remove its old interval.
    if scheduled != st.session_state[playing_key]:
        st.rerun()
    if st.session_state.get(step_key, 0) > last:
        st.session_state[step_key] = last
    if st.session_state[playing_key]:
        now = monotonic()
        previous = st.session_state.get(tick_key)
        if previous is not None and now - previous >= 0.5:
            st.session_state[step_key] = min(last, st.session_state[step_key] + 1)
            st.session_state[tick_key] = now
        if st.session_state[step_key] >= last:
            pause_playback(prefix)
            st.rerun()
    step = st.slider('Inspect step', 0, max(1, last), key=step_key, label_visibility='collapsed', on_change=pause_playback, args=(prefix,)) if last else 0
    step = min(step, last)
    st.caption(f"{'Playing' if st.session_state[playing_key] else 'Paused'} · step {step} of {last} · loss {trace.losses[step]:.6g} · x=({trace.states[step, 0]:.4g}, {trace.states[step, 1]:.4g}). Play advances every 0.5 s; inspecting pauses.")
    chart = st.empty()
    with st.expander('Chart options · baseline and scale'):
        log_loss = st.checkbox('Log loss axis', value=False, key=prefix + 'log_loss')
        st.session_state.setdefault(prefix + 'baseline_overlay', True)
        compare = st.checkbox('Overlay the baseline', key=prefix + 'baseline_overlay')
        show_legend = st.checkbox('Show chart legend', value=False, key=prefix + 'show_legend')
    figs = build_figures(trace, config, step=step, log_loss=log_loss)
    if compare:
        figs['trajectory'].add_trace(go.Scatter(x=baseline.states[:, 0], y=baseline.states[:, 1], mode='lines', line=dict(dash='dot', color='#f59e0b'), name='Baseline'))
        for name, values in [('loss', baseline.losses), ('updates', baseline.update_norms)]:
            display = [None if log_loss and name == 'loss' and value == 0 else float(value) for value in values]
            figs[name].add_trace(go.Scatter(x=list(range(len(display))), y=display, mode='lines', line=dict(dash='dot', color='#f59e0b'), name='Baseline'))
    selected = {'Loss': 'loss', 'Trajectory': 'trajectory', 'Update size': 'updates'}[st.session_state[prefix + 'result_view']]
    figure = figs[selected]
    # The app has its own nearby explanation; retain scientific caveats such as
    # exact-zero log omissions while avoiding repeated full-page plot footnotes.
    annotations = [annotation for annotation in figure.layout.annotations
                   if not annotation.text.startswith(('Stop:', 'Float64 2-D'))]
    titles = {'loss': 'Loss over accepted steps', 'trajectory': 'Path across the loss landscape',
              'updates': 'Update size over accepted steps'}
    figure.layout.annotations = annotations
    if selected == 'updates':
        figure.update_yaxes(title='Update norm (coordinate units)')
    figure.update_layout(title_text=titles[selected], height=420,
                         margin=dict(t=85, b=65, l=65, r=20), showlegend=show_legend)
    chart.plotly_chart(figure, width='stretch', key=prefix + selected + '_plot')
    explanations = {
        'loss': 'Lower loss means closer to the quadratic optimum. The marker follows the inspection slider; the full recorded curve stays visible.',
        'trajectory': 'Coordinates show where each update lands; contour lines show equal analytic loss. The open marker follows the inspection slider.',
        'updates': 'Update size is the distance moved at each step, in coordinate units. Step 0 has no update, so its size is zero.',
    }
    st.caption(explanations[selected] + ' The dotted baseline uses H=diag(1,10), η=.1, β=0.' if compare else explanations[selected])
    st.caption(f'Recorded {last} accepted updates · stop: {trace.stop_reason}. Float64 2-D quadratic, not a transformer landscape. Loss cap 10¹²; rejected updates are omitted, never clipped.')
    with st.expander('Step data · coordinates, loss and update size'):
        st.dataframe({'step': list(range(len(trace.states))), 'x₁': trace.states[:, 0], 'x₂': trace.states[:, 1], 'loss': trace.losses, 'update norm': trace.update_norms}, hide_index=True, width='stretch')



STUDY_DURATIONS = {'Short · 10 minutes': 600, 'Standard · 25 minutes': 1500, 'Deep · 50 minutes': 3000}


def stop_study_timer():
    started = st.session_state.get('study_started_at')
    if started is not None:
        duration = STUDY_DURATIONS[st.session_state['study_duration']]
        st.session_state['study_elapsed'] = min(duration, st.session_state.get('study_elapsed', 0.0) + max(0.0, monotonic() - started))
    st.session_state['study_started_at'] = None


def reset_study_timer():
    st.session_state['study_started_at'] = None
    st.session_state['study_elapsed'] = 0.0


def start_study_timer():
    st.session_state['study_started_at'] = monotonic()


def show_study_timer():
    if 'study_duration' in st.session_state:
        st.session_state['study_duration'] = st.session_state['study_duration']
    with st.expander('Optional study duration · local session only'):
        enabled = st.checkbox('Enable study timer', value=False, key='study_timer_enabled', on_change=stop_study_timer)
        st.caption('Off by default. No reminders or calendar events. Timer state lasts only in this browser session; saved learning progress is separate.')
        if not enabled:
            return
        st.session_state.setdefault('study_elapsed', 0.0)
        st.session_state.setdefault('study_started_at', None)
        st.session_state.setdefault('study_duration', 'Standard · 25 minutes')
        running = st.session_state['study_started_at'] is not None
        st.selectbox('Study duration', list(STUDY_DURATIONS), index=None, key='study_duration', disabled=running, on_change=reset_study_timer)
        duration = STUDY_DURATIONS[st.session_state['study_duration']]
        start, stop, reset = st.columns(3)
        start.button('Start / resume study timer', disabled=running or st.session_state['study_elapsed'] >= duration, on_click=start_study_timer)
        stop.button('Stop study timer', disabled=not running, on_click=stop_study_timer)
        reset.button('Reset study timer', on_click=reset_study_timer)
        view = st.container()

        @st.fragment(run_every=1 if running else None)
        def timer_view():
            started = st.session_state['study_started_at']
            elapsed = min(duration, st.session_state['study_elapsed'] + (max(0.0, monotonic() - started) if started is not None else 0.0))
            if started is not None and elapsed >= duration:
                stop_study_timer()
                st.rerun()
            left, right = st.columns(2)
            left.metric('Elapsed (seconds)', str(int(elapsed)))
            right.metric('Remaining (seconds)', str(max(0, duration - int(elapsed))))
            if elapsed >= duration:
                st.success('Study duration reached. Reflect or take a break when ready.')
            st.caption(f"{'Running' if running else 'Stopped'} · elapsed counts wall time after Start until Stop, disabling, or completion. Stop preserves elapsed; Start resumes. Reset or changing duration clears this timer only.")

        def render():
            with view:
                timer_view()

        return render


def main():
    st.set_page_config(page_title='ARENA · Mastery Hub', page_icon='🧭', layout='wide')
    if 'progress' not in st.session_state:
        try:
            st.session_state['progress'] = load_session()
        except (ValueError, OSError) as exc:
            st.error(f'Could not read saved progress: {exc}. The existing file has been preserved. Move or repair it before restarting.')
            st.stop()
        progress = st.session_state['progress']
        if progress['config'] is not None and progress['config']['simulator_version'] != SIMULATOR_VERSION:
            st.error('Saved simulator version differs from this implementation. The file is preserved; export or migrate it explicitly before resuming.')
            st.stop()
        set_controls(progress['config'] or DEFAULT)
        st.session_state['mode'] = progress['mode']
    progress = st.session_state['progress']
    with st.container(horizontal=True, wrap=True, vertical_alignment='center'):
        st.subheader('Optimizer lab', width='content')
        mode = st.radio('Mode', ['Explore', 'Challenge'], horizontal=True,
                        label_visibility='collapsed', key='mode', width='content')
    st.caption('Analytic toy · CPU only · Will these updates converge?')
    next_step = st.empty()
    progress['mode'] = mode
    trace_renderers = []
    with st.sidebar:
        st.header('Study tools')
        st.caption('Optimizer CPU pilot · learning materials and local session tools.')
        with st.expander('Start here · the model and a worked example'):
            st.latex(r'L(x)=\tfrac12x^T Hx,\quad H=R\,\mathrm{diag}(\lambda_{min},\lambda_{max})R^T')
            st.latex(r'v_{t+1}=\beta v_t+Hx_t+\epsilon_t,\qquad x_{t+1}=x_t-\eta v_{t+1},\quad v_0=0')
            st.write('Momentum is unnormalized heavy-ball: there is no (1 − β) multiplier. Noise is independent Gaussian gradient noise with coordinate standard deviation σ.')
            st.write('Worked example: H=diag(1,10), x₀=(1,1), η=.1, β=0, σ=0 gives x₁=(.9,0), L₁=.405. A 2-D quadratic does not reproduce a transformer loss landscape.')
            st.markdown('**Companion notebook:** implement the Hessian, update, loop and stability reasoning in `mastery_hub/units/u02_optimization/exercises.ipynb`. Use its hints and tests when needed.')
            st.download_button('Download companion exercises notebook', (ROOT / 'mastery_hub/units/u02_optimization/exercises.ipynb').read_bytes(), file_name='optimizer_exercises.ipynb', mime='application/x-ipynb+json')
            st.download_button('Download executed solutions notebook', (ROOT / 'mastery_hub/units/u02_optimization/solutions.ipynb').read_bytes(), file_name='optimizer_solutions.ipynb', mime='application/x-ipynb+json')

        with st.expander('Original materials · complete source shelf'):
            source_index = json.loads((ROOT / 'docs/mastery/source_links.json').read_text())
            st.caption('Canonical originals plus provenance · checked ' + source_index['checked_on'])
            show_quiz_answers = st.checkbox(
                'Show original quiz answers after attempting the quiz',
                value=False, key='show_original_quiz_answers',
            )
            for source_number, source in enumerate(source_index['links'], 1):
                if source.get('answer_material', False) and not show_quiz_answers:
                    continue
                st.markdown(f"{source_number}. [{source['label'] or 'Original source'}]({source['url']})")
            st.caption('Quiz answer links become visible above after your explicit choice. Reading them exposes those original quiz answers.')
            st.caption('This is the checked homepage link index; source publication does not imply companion completion.')
        with st.expander('More prediction notebooks'):
            workbook_path = ROOT / 'mastery_hub/units/extension_predictions/exercises.ipynb'
            if workbook_path.exists():
                st.download_button('Download ARENA prediction workbook', workbook_path.read_bytes(), file_name='arena_extension_prediction_workbook.ipynb', mime='application/x-ipynb+json')
                checkpoint_count = len(json.loads((ROOT / 'mastery_hub/extension_checkpoints.json').read_text())['checkpoints'])
                st.caption(f'Method-specific prediction practice across {checkpoint_count} master-backed lessons. Public reference results are guided practice.')
            circuit_path = ROOT / 'mastery_hub/units/circuit_prediction/exercises.ipynb'
            if circuit_path.exists():
                st.download_button('Download blind circuit challenge', circuit_path.read_bytes(), file_name='circuit_prediction_exercises.ipynb', mime='application/x-ipynb+json')
                st.caption('8.8 companion: implement the circuit methods, commit a forecast, then run the paired readout-gain experiment on CPU. Run from this checkout; no model download is needed.')
            shapley_path = ROOT / 'mastery_hub/units/shapley_prediction/exercises.ipynb'
            if shapley_path.exists():
                st.download_button('Download blind Data Shapley challenge', shapley_path.read_bytes(), file_name='shapley_prediction_exercises.ipynb', mime='application/x-ipynb+json')
                st.caption('16.9 companion: predict how coalition normalization changes example credit, then run the exact eight-example training comparison on CPU. Run from this checkout; no model download is needed.')
            sage_path = ROOT / 'mastery_hub/units/sage_prediction/exercises.ipynb'
            if sage_path.exists():
                st.download_button('Download blind SAGE challenge', sage_path.read_bytes(), file_name='sage_prediction_exercises.ipynb', mime='application/x-ipynb+json')
                st.caption('16.4 companion: predict how two missing-input rules allocate predictive performance. Implement the Gaussian game, commit, then inspect paired SAGE bars and coalition controls. CPU only; no model training or download.')
            ssd_path = ROOT / 'mastery_hub/units/ssd_prediction/exercises.ipynb'
            if ssd_path.exists():
                st.download_button('Download blind SSD challenge', ssd_path.read_bytes(), file_name='ssd_prediction_exercises.ipynb', mime='application/x-ipynb+json')
                st.caption('5.4 companion: implement recurrent, quadratic and chunked state updates. Commit a carry-effect forecast, then inspect state heatmaps and equivalence controls. CPU only; no model training or download.')
            graph_path = ROOT / 'mastery_hub/units/graph_prediction/exercises.ipynb'
            if graph_path.exists():
                st.download_button(
                    'Download 8.7 graph/path prediction challenge', graph_path.read_bytes(),
                    file_name='graph_prediction_exercises.ipynb', mime='application/x-ipynb+json',
                )
                st.caption(
                    '8.7 companion: implement graph attribution and path recovery, pass distinct DEV fixtures, '
                    'then commit before the explicit CPU experiment. Fixed public synthetic setup; '
                    'no automatic run, training or download, and no claim of unseen mastery.'
                )
            sae_path = ROOT / 'mastery_hub/units/sae_prediction/exercises.ipynb'
            if sae_path.exists():
                st.download_button(
                    'Download 6.1 SAE forward-rule prediction companion', sae_path.read_bytes(),
                    file_name='sae_prediction_exercises.ipynb', mime='application/x-ipynb+json',
                )
                st.caption(
                    '6.1 companion: implement seven visible methods for four fixed forward rules and pass distinct DEV fixtures. '
                    'The public target remains unexecuted; forecast before any separately authorized CPU attempt. '
                    'No automatic run, training or model/data download; this is not a learned-SAE ranking.'
                )
        with st.expander('Course map and original ARENA'):
            units = load_catalog(ROOT / 'mastery_hub/catalog.json')
            by_id = {unit['id']: unit for unit in units}
            for unit in units:
                st.markdown(f"[{unit['title']}]({unit['source_url']})")
                scope = ' · CPU pilot only; full unit remains planned' if unit['id'] == 'u02_optimization' else ''
                st.caption(f"Source: {unit['source_status']} · Companion: {unit['companion_status']}{scope}")
                if unit['prerequisites']:
                    links = ', '.join(f"[{by_id[prereq]['title']}]({by_id[prereq]['source_url']})" for prereq in unit['prerequisites'])
                    st.markdown('Prerequisite reading: ' + links)
                if unit['entrypoint'] is not None:
                    st.markdown('[Open optimizer companion](#optimizer-lab)')
                    for original in unit['original_paths']:
                        kind = 'tree' if (ROOT / original).is_dir() else 'blob'
                        original_url = f'https://github.com/ARENA-education/ARENA_materials/{kind}/6a46100c4f219b2764d3a30fe501e810ee500d86/{quote(original)}'
                        st.markdown(f'[Original foundation: {original}]({original_url})')
            st.caption('The app runs the optimizer pilot. Extension prediction notebooks are available above; the other Alchemy companions remain planned.')
            st.markdown('[Original ARENA](https://github.com/ARENA-education/ARENA_materials) · [Extension](https://github.com/shinobiultra/ARENA_3.0_Extended)')
        timer_renderer = show_study_timer()
        with st.expander('Local progress and privacy'):
            st.caption(f'Private progress: {default_data_dir()}')
            st.caption('No analytics or automatic upload. Run locally; this is self-study, not tamper-proof exam software.')

    settings, prediction_panel = st.columns([1, 1], gap='large') if mode == 'Challenge' else (st.container(), None)
    with settings:
        with st.container(horizontal=True, wrap=True, vertical_alignment='bottom', gap='small'):
            st.selectbox('Transfer regime', list(TEMPLATES), key='case_template', on_change=apply_template,
                         width=190, help='Choose a regime before prediction. Previously exposed settings count as practice.')
            st.number_input('Learning rate η', min_value=0.0, max_value=10.0, step=0.01,
                            format='%.4f', key='eta', width=130, help='η sets the size of the velocity step.')
            st.number_input('Momentum β', min_value=0.0, max_value=0.999, step=0.05,
                            key='beta', width=130, help='β carries the previous velocity into the next update.')
            st.button('Reset to baseline', on_click=reset_controls)
        with st.expander(f"Advanced settings · λ=({st.session_state['lambda_min']:g}, {st.session_state['lambda_max']:g}), σ={st.session_state['noise_std']:g}, {st.session_state['steps']} steps"):
            a, b = st.columns(2)
            a.number_input('Small curvature λmin', min_value=0.001, max_value=1000.0, key='lambda_min', help='Positive eigenvalue along the shallow eigendirection.')
            a.number_input('Large curvature λmax', min_value=0.001, max_value=1000.0, key='lambda_max', help='Largest curvature sets the strict GD upper boundary 2/λmax.')
            a.number_input('Rotation (degrees)', min_value=0.0, max_value=179.99, key='angle_deg')
            a.number_input('Gradient noise σ', min_value=0.0, max_value=10.0, step=0.1, key='noise_std')
            b.number_input('Starting x₁', min_value=-100.0, max_value=100.0, key='x0_0')
            b.number_input('Starting x₂', min_value=-100.0, max_value=100.0, key='x0_1')
            b.number_input('Seed', min_value=0, max_value=2**32 - 1, step=1, key='seed')
            b.number_input('Steps', min_value=1, max_value=1000, step=1, key='steps')
    try:
        config = QuadraticConfig(**{key: st.session_state[key] for key in DEFAULT if key != 'x0'}, x0=(st.session_state['x0_0'], st.session_state['x0_1']))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    fullconfig = scientific_config(config)
    key = fingerprint(fullconfig)
    equivalence_key = fingerprint({name: fullconfig[name] for name in ('lambda_min', 'lambda_max', 'eta', 'beta')})
    equivalent_seen = any(pair['challenge_id'] == EQUIVALENCE_ID and pair['config_fingerprint'] == equivalence_key for pair in progress['exposed_cases'])
    current = progress['current_attempt']
    if current is not None and current['config_fingerprint'] != key:
        if current['status'] == 'committed':
            st.warning('Changed settings invalidated your pending prediction. Commit a new prediction for this configuration.')
        current = None
    exact_seen = any(pair['challenge_id'] == CHALLENGE_ID and pair['config_fingerprint'] == key for pair in progress['exposed_cases'])
    if current is not None and current['status'] != 'revealed' and (equivalent_seen or exact_seen):
        current = {**current, 'assessment': 'practice'}
        progress['exposed_cases'] = expose_case(progress['exposed_cases'], CHALLENGE_ID, key)
    if mode == 'Explore':
        progress['exposed_cases'] = expose_case(progress['exposed_cases'], CHALLENGE_ID, key)
        progress['exposed_cases'] = expose_case(progress['exposed_cases'], EQUIVALENCE_ID, equivalence_key)
        equivalent_seen = True
        # Moving from a committed challenge into Explore exposes its result;
        # preserve the answer but make its assessment practice before revealing.
        if current is not None and current['status'] == 'committed':
            current = {**current, 'assessment': 'practice'}
    if current is None:
        current = new_attempt(CHALLENGE_ID, key, exposed_cases=progress['exposed_cases'])
        if equivalent_seen:
            current = {**current, 'assessment': 'practice'}
    progress['config'], progress['current_attempt'] = fullconfig, current

    if mode == 'Challenge':
        with prediction_panel:
            st.subheader('Predict before reveal')
            st.write('With **noise set to zero**, will every starting point converge to the origin? Predict strict asymptotic stability for these H, η and β; σ affects only the later simulation.')
            st.caption(f"Local exposure: {current['assessment']} · guided practice, not an independent assessment. Predictions lock before reveal; changing settings starts a new attempt.")
            locked = current['status'] in ('committed', 'revealed')
            if locked:
                # The immutable record is authoritative after reload (widget
                # defaults are not a second forecast). Restore before creation;
                # normal reruns preserve playback/timer widget lifecycles.
                st.session_state['prediction'] = 'Stable' if current['prediction'] else 'Not strictly stable'
                st.session_state['confidence'] = current['confidence']
                st.session_state['reasoning'] = current['reasoning']
            prediction = st.radio('Prediction', ['Stable', 'Not strictly stable'], horizontal=True,
                                  key='prediction', disabled=locked)
            confidence = st.slider('Confidence in my selected answer', 0.5, 1.0, 0.7, 0.05,
                                   key='confidence', disabled=locked)
            reasoning = st.text_area('Why do you expect that?', key='reasoning', height=90, disabled=locked)
            commit_action, reveal_action = st.columns(2)
            if commit_action.button('Commit prediction', type='primary', disabled=current['status'] != 'uncommitted' or not reasoning.strip()):
                current = commit_prediction(current, prediction == 'Stable', confidence, reasoning)
                progress['current_attempt'] = current
            if current['status'] in ('committed', 'revealed'):
                st.info(f"Locked answer: {'Stable' if current['prediction'] else 'Not strictly stable'} · confidence {current['confidence']:.0%}. Reason: {current['reasoning']}")
            if reveal_action.button('Reveal result', disabled=current['status'] != 'committed'):
                current = reveal(current, key, bool(spectral_radius(config) < 1.0))
                progress['current_attempt'] = current
                progress['exposed_cases'] = expose_case(progress['exposed_cases'], CHALLENGE_ID, key)
                progress['exposed_cases'] = expose_case(progress['exposed_cases'], EQUIVALENCE_ID, equivalence_key)
                record = envelope(current, fullconfig, controls(config))
                if all(row['attempt_id'] != current['attempt_id'] for row in progress['history']):
                    progress['history'].append(record)
            if current['status'] == 'revealed':
                st.success(f"Observed: {'strictly stable' if current['observed'] else 'not strictly stable'}. Brier score {current['score']:.3f} (lower is better).")
                with st.expander('What does this score teach?'):
                    st.markdown(
                        'Confidence is the probability you assigned to **your selected answer**. '
                        'For example, 90% confidence gives a squared error of **0.01** when that answer is right '
                        'and **0.81** when it is wrong; 50% gives **0.25** either way. '
                        'A confident mistake costs more because it rules out an outcome that actually occurred. '
                        'One case measures this forecast; learning calibration requires many comparable forecasts.'
                    )

    if mode == 'Challenge' and current['status'] == 'uncommitted':
        next_step.caption('Next: choose an answer and reason, then Commit prediction → Reveal result.')
    elif mode == 'Challenge' and current['status'] == 'committed':
        next_step.caption('Next: reveal your locked prediction, then inspect the result.')
    elif mode == 'Challenge':
        next_step.caption('Next: inspect the result and test a control, then predict a new regime.')
    else:
        next_step.caption('Next: adjust η or β and inspect the result. Challenge locks a prediction before reveal.')

    if mode == 'Explore' or current['status'] == 'revealed':
        trace = simulate(config)
        rho = spectral_radius(config)
        trace_renderers.append(show_trace(trace, config))
        st.caption(f'Deterministic spectral radius: **{rho:.6g}**. Strict stability requires ρ < 1; ρ = 1 includes nondecaying boundaries. Noise does not inherit convergence to zero.')
        with st.expander('Controls and limits · test your explanation'):
            st.write('Try it yourself: η=0 gives no movement. With baseline curvature, η=.23 grows the loss. Explain why before testing each change. Previously viewed stability settings count as practice, even after changing only the seed or starting point.')
            st.dataframe({'control': list(controls(config)), 'passed': list(controls(config).values())}, hide_index=True)
            st.write('Falsify “a flat-looking finite curve proves stability”: choose x₀ entirely in a stable eigendirection while a different eigendirection is unstable. Then add a small component in the unstable direction. Rotate H and x₀ together for the coordinate control; rotating only H changes the experiment.')
            st.markdown('**Explain before changing the next setting:** Which control contradicts your original explanation? Separate the all-start, noiseless stability claim from this finite trajectory and its noise floor. Predict what a small unstable-direction component will change, then test it in Explore.')
        if mode == 'Explore' and current['status'] == 'uncommitted' and current['assessment'] != 'practice':
            progress['current_attempt'] = new_attempt(CHALLENGE_ID, key, exposed_cases=progress['exposed_cases'])
    else:
        pause_playback()

    history = progress['history']
    with st.expander(f'Optional reflection, saved attempts and replay · {len(history)} completed'):
        progress['next_action'] = st.text_input('One next action', value=progress['next_action'] or 'Implement the heavy-ball update in the notebook; predict an unseen regime next.', key='next_action')
        unseen = [r for r in history if r['attempt']['assessment'] == 'unseen']
        st.caption(f'{len(history)} completed attempts · {len(unseen)} unseen · {len(history)-len(unseen)} practice. A small sample is evidence about these cases, not a mastery badge.')
        if history:
            ids = [r['attempt_id'] for r in history]
            selected = st.selectbox('Saved attempt', ids, format_func=lambda value: f"{ids.index(value)+1} · {next(r for r in history if r['attempt_id']==value)['attempt']['assessment']}")
            record = next(r for r in history if r['attempt_id'] == selected)
            note = st.text_area('What surprised you? Which assumption failed?', value=progress['reflections'].get(selected, ''), key='reflection_' + selected)
            progress['reflections'][selected] = note
            downloadable = validate_record({**record, 'reflection': note, 'next_action': progress['next_action']})
            data = json.dumps(downloadable, indent=2, allow_nan=False)
            st.download_button('Download attempt JSON', data, file_name=f'{selected}.json', mime='application/json')
            md = f"# Optimizer reflection\n\nEvidence: analytic_toy\n\nPrediction: {record['attempt']['prediction']}\n\nReasoning: {record['attempt']['reasoning']}\n\nObserved: {record['attempt']['observed']}\n\nReflection: {note}\n\nNext action: {progress['next_action']}\n\nConfiguration fingerprint: {record['config_fingerprint']}\n"
            st.download_button('Download reflection Markdown', md, file_name=f'{selected}.md', mime='text/markdown')
            if st.button('Save an explicit local export'):
                try:
                    path = export_attempt(downloadable, default_data_dir() / 'exports')
                    st.success(f'Saved {path}')
                except (ValueError, OSError) as exc:
                    st.warning(str(exc))
            if st.button('Replay saved attempt'):
                st.session_state['replay_id'] = selected
            if st.session_state.get('replay_id') == selected:
                try:
                    trace_renderers.append(show_trace(replay(record), QuadraticConfig(**{k: v for k, v in record['config'].items() if k != 'simulator_version'}), prefix='replay_'))
                    st.caption('Recomputed analytic toy from the recorded seed/config; no real-model training claim.')
                except ValueError as exc:
                    st.error(str(exc))
    with st.expander('Import an exported JSON attempt'):
        uploaded = st.file_uploader('Strict JSON only · at most 1 MiB', type=['json'])
        if st.button('Import attempt', disabled=uploaded is None):
            try:
                imported = loads_attempt(uploaded.getvalue())
                if any(r['attempt_id'] == imported['attempt_id'] for r in history):
                    raise ValueError('That attempt is already present')
                if imported['attempt']['status'] != 'revealed':
                    raise ValueError('Import a completed revealed attempt')
                progress['history'].append(imported)
                progress['exposed_cases'] = expose_case(progress['exposed_cases'], imported['challenge_id'], imported['config_fingerprint'])
                if imported['challenge_id'] == CHALLENGE_ID:
                    imported_equivalence = fingerprint({name: imported['config'][name] for name in ('lambda_min', 'lambda_max', 'eta', 'beta')})
                    progress['exposed_cases'] = expose_case(progress['exposed_cases'], EQUIVALENCE_ID, imported_equivalence)
                progress['reflections'][imported['attempt_id']] = imported['reflection']
                save_session(progress)
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    st.caption('CPU pilot: independently reviewed; learner walkthrough still required. Real-model, scaling, architecture and transfer companions remain planned.')
    try:
        save_session(progress)
    except (ValueError, OSError) as exc:
        st.error(f'Progress was not saved: {exc}. Existing records were preserved.')
        if isinstance(exc, ValueError):
            st.info('Reload this page to load the latest saved progress before continuing. Your current browser session may be older than the saved record.')
    # Register all ordinary widgets before a timed fragment may request a full
    # rerun on completion. This prevents Streamlit cleaning up later controls.
    for render in [*trace_renderers, timer_renderer]:
        if render is not None:
            render()


if __name__ == '__main__':
    main()
