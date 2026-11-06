"""Plotting only; every probe, intervention and metric lives in the notebook."""
import matplotlib.pyplot as plt
import numpy as np


def labelled_heatmap(ax, values, rows, columns, title, *, vmin=0., vmax=1.):
    image = ax.imshow(values, vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
    ax.set(xticks=np.arange(len(columns)), xticklabels=columns,
           yticks=np.arange(len(rows)), yticklabels=rows, title=title,
           xlabel='Evaluated dataset', ylabel='Readout / fitted probe')
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if value < .6 else 'black')
    return image
