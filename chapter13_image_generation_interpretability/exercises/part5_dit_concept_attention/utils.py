"""Plotting only; all scientific methods and metrics live in the notebook."""
import numpy as np
import matplotlib.pyplot as plt

def image_grid(images, titles, heading):
    fig, axes = plt.subplots(1, len(images), figsize=(3*len(images), 3.2), squeeze=False)
    for ax, image, title in zip(axes[0], images, titles):
        ax.imshow(np.clip(image, 0, 1), interpolation="nearest")
        ax.set_title(title); ax.set_xlabel("x pixel"); ax.set_ylabel("y pixel")
    fig.suptitle(heading); fig.tight_layout()
    return fig
