"""Plotting labels and text only; geometry and interventions are notebook implementations."""
import numpy as np


def label_points(ax, coordinates, labels, title):
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=np.arange(len(labels)), cmap='hsv', s=65)
    for point, label in zip(coordinates, labels):
        ax.annotate(label, point, xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set(xlabel='Coordinate 1', ylabel='Coordinate 2', title=title, aspect='equal')
    ax.axhline(0, color='gray', lw=.5)
    ax.axvline(0, color='gray', lw=.5)


def token_heatmap(ax, predicted, tokens):
    ax.imshow(predicted, cmap='hsv', vmin=0, vmax=len(tokens), aspect='auto')
    for (i, j), token_id in np.ndenumerate(predicted):
        ax.text(j, i, tokens[token_id][:3], ha='center', va='center', fontsize=8)
    ax.set(xticks=np.arange(len(tokens)), xticklabels=tokens,
           yticks=np.arange(len(predicted)), yticklabels=np.arange(len(predicted)),
           xlabel='Input weekday', ylabel='Days added', title='Rotate the plane, then decode the token')
    ax.tick_params(axis='x', rotation=25)
