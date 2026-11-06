"""Plotting only: all adapter, metric and intervention computations are in the master."""
import matplotlib.pyplot as plt


def signature(times, spectra, curves, errors, alignments):
    """Plot adapter spectrum, thresholded behavior and intervention cost together."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for j in range(2):
        axes[0].plot(times, spectra[:, j], marker='o', label=f'singular value {j+1}')
    axes[0].plot(times, alignments, '--', label='top output / proxy alignment')
    axes[0].set(xlabel='Planted checkpoint time', ylabel='Singular value / alignment', title='Which mode is dominant?')
    for label, values in curves.items():
        axes[1].plot(times, values, marker='.', label=label)
    axes[1].axhline(.5, color='grey', linestyle=':', label='frozen 50% threshold')
    axes[1].set(xlabel='Planted checkpoint time', ylabel='Proxy label flip fraction', ylim=(-.05, 1.05), title='Proxy drift and controls')
    for label, values in errors.items():
        axes[2].plot(times, values, marker='.', label=label)
    axes[2].set(xlabel='Planted checkpoint time', ylabel='Narrow task RMSE', title='Did mitigation preserve the task?')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    return fig
