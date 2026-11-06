"""Plot actual analytic-toy traces without clipping or invented log losses."""
from numbers import Integral

import numpy as np
import plotly.graph_objects as go

from .quadratic import QuadraticConfig, QuadraticTrace

_ASSUMPTIONS = ('Analytic 2-D positive-definite quadratic; float64; unnormalized heavy-ball. '
                'Controls: eta=0 (no movement), noiseless x0=0 (fixed optimum), unstable eta, '
                'and joint rotation of H and x0. Noise does not imply convergence to zero.')


def build_figures(trace: QuadraticTrace, config: QuadraticConfig, step: int | None = None,
                  log_loss: bool = False) -> dict[str, go.Figure]:
    """Return trajectory/loss/updates figures, with a synchronized selected step.

    Contours are evaluations of the supplied analytic quadratic, not measurements.
    They are omitted when a finite meaningful contour grid is not representable.
    In log display exact zero losses are omitted with an annotation, never shifted.
    An empty trace explicitly means no initial loss was representable.
    """
    count = len(trace.losses)
    if step is None:
        selected = count - 1 if count else None
    elif isinstance(step, (bool, np.bool_)) or not isinstance(step, Integral) or not 0 <= step < count:
        raise ValueError('step must index an accepted state')
    else:
        selected = int(step)
    trajectory = go.Figure()
    if count:
        extent = float(np.max(np.abs(trace.states)))
        # Keep the contour arithmetic finite while preserving all trajectory data.
        if extent < 1e100 and config.lambda_max < 1e100:
            radius = max(1.0, extent * 1.15)
            axis = np.linspace(-radius, radius, 65)
            xx, yy = np.meshgrid(axis, axis)
            angle = np.deg2rad(config.angle_deg)
            a = np.cos(angle) * xx + np.sin(angle) * yy
            b = -np.sin(angle) * xx + np.cos(angle) * yy
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                zz = .5 * (config.lambda_min * a*a + config.lambda_max * b*b)
            if np.isfinite(zz).all():
                trajectory.add_trace(go.Contour(x=axis.tolist(), y=axis.tolist(), z=zz.tolist(),
                    name='Analytic loss contours', showscale=False, contours={'coloring': 'lines'},
                    line={'width': 1}, opacity=.45, hovertemplate='x=%{x:.3g}, y=%{y:.3g}<br>L=%{z:.3g}<extra>Analytic quadratic</extra>'))
        else:
            trajectory.add_annotation(text='Contour grid omitted at extreme numeric scale.', xref='paper', yref='paper', x=.5, y=.95, showarrow=False)
    trajectory.add_trace(go.Scatter(x=trace.states[:, 0].tolist(), y=trace.states[:, 1].tolist(),
        mode='lines+markers', name='Accepted trajectory', customdata=list(range(count)),
        hovertemplate='step=%{customdata}<br>x1=%{x:.6g}<br>x2=%{y:.6g}<extra></extra>'))
    trajectory.add_trace(go.Scatter(x=[0], y=[0], mode='markers', name='Analytic optimum', marker={'symbol': 'x', 'size': 12}))
    if selected is not None:
        trajectory.add_trace(go.Scatter(x=[float(trace.states[selected, 0])], y=[float(trace.states[selected, 1])],
            mode='markers', name='Selected step', marker={'size': 14, 'symbol': 'circle-open'}))
    indices = list(range(count))
    displayed_losses = [None if log_loss and value == 0 else float(value) for value in trace.losses]
    loss = go.Figure(go.Scatter(x=indices, y=displayed_losses, mode='lines+markers', name='Recorded loss', connectgaps=False))
    updates = go.Figure(go.Scatter(x=indices, y=trace.update_norms.tolist(), mode='lines+markers', name='Recorded update norm'))
    if selected is not None:
        loss.add_trace(go.Scatter(x=[selected], y=[displayed_losses[selected]], mode='markers', name='Selected step', marker={'size': 12}))
        updates.add_trace(go.Scatter(x=[selected], y=[float(trace.update_norms[selected])], mode='markers', name='Selected step', marker={'size': 12}))
    if log_loss and np.any(trace.losses == 0):
        loss.add_annotation(text='Exact zero losses omitted on log axis.<br>See the linear view or data table.',
                            xref='paper', yref='paper', x=.5, y=.95, showarrow=False)
    loss.update_yaxes(type='log' if log_loss else 'linear')
    figures = {'trajectory': trajectory, 'loss': loss, 'updates': updates}
    for name, figure in figures.items():
        # Reserve actual pixel space for captions: paper coordinates alone
        # compress their separation when the chart's plotting area gets short.
        figure.update_layout(
            template='plotly_white', height=700,
            title={'text': f'Analytic toy: {name}', 'x': 0, 'xanchor': 'left'},
            margin={'t': 135, 'b': 230, 'l': 70, 'r': 25},
            legend={'orientation': 'h', 'y': 1.02, 'yanchor': 'bottom',
                    'x': 0, 'xanchor': 'left', 'font': {'size': 11}},
            meta={'assumptions': _ASSUMPTIONS, 'stop_reason': trace.stop_reason})
        figure.update_xaxes(automargin=True, title_standoff=16)
        figure.update_yaxes(automargin=True, title_standoff=12)
        figure.add_annotation(
            text=f'Stop: {trace.stop_reason}; accepted updates: {max(0, count - 1)}.' +
                 ('<br>No representable initial loss.' if not count else ''),
            xref='paper', yref='paper', x=0, y=0, yshift=-75,
            xanchor='left', yanchor='top', align='left', font={'size': 11}, showarrow=False)
        figure.add_annotation(
            text='Float64 2-D quadratic toy.<br>'
                 'Unnormalized heavy-ball; not a<br>'
                 'transformer loss landscape.<br>'
                 'Controls: eta=0, noiseless optimum,<br>'
                 'unstable eta, joint rotation.<br>'
                 'Oracle: zero noise; noisy runs need<br>'
                 'not converge to zero.',
            xref='paper', yref='paper', x=0, y=0, yshift=-119,
            xanchor='left', yanchor='top', align='left', font={'size': 11}, showarrow=False)
    trajectory.update_xaxes(title='x1 (coordinate units)')
    trajectory.update_yaxes(title='x2 (coordinate units)', scaleanchor='x', scaleratio=1)
    loss.update_xaxes(title='Accepted step (initial state = 0)')
    loss.update_yaxes(title='L(x) (loss units)')
    updates.update_xaxes(title='Accepted step (initial state = 0)')
    updates.update_yaxes(title='||x[t] - x[t-1]||₂ (coordinate units); initial = 0')
    return figures
