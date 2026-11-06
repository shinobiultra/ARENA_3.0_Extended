"""Labeled plot plumbing only; unlearning and evaluation remain student code."""
import matplotlib.pyplot as plt
import numpy as np


def metric_heatmap(ax, matrix, methods, columns):
    image = ax.imshow(matrix, cmap='viridis', vmin=0., vmax=1., aspect='auto')
    ax.set(xticks=np.arange(len(columns)), xticklabels=columns,
           yticks=np.arange(len(methods)), yticklabels=methods,
           title='The same low forget score can hide different mechanisms',
           xlabel='Distinct observable / attack', ylabel='Restricted CPU method')
    ax.tick_params(axis='x', rotation=25)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if value < .6 else 'black')
    return image
