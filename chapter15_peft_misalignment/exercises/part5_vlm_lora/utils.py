"""Image and figure plumbing; the VLM, adapters, controls and metrics live in the master."""
import matplotlib.pyplot as plt
import numpy as np


def show_scenes(images, queries, answers):
    """Show literal RGB examples with question and target answer labels."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), constrained_layout=True)
    for ax, i in zip(axes, [0, 2, 4, 6]):
        ax.imshow(images[i], interpolation='nearest')
        ax.set(title=f"What color is {'left' if queries[i] == 0 else 'right'}?\nTarget: {answers[i]}", xlabel='image column', ylabel='image row')
    return fig


def signature(names, vqa, language, recoveries, singular_values):
    """Place capability, causal grounding and weight spectrum on shared condition labels."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    x = np.arange(len(names))
    axes[0].bar(x-.18, vqa, width=.36, label='shifted-camera VQA')
    axes[0].bar(x+.18, language, width=.36, label='language-only color identity')
    axes[0].axhline(.5, linestyle=':', color='grey', label='balanced binary chance')
    axes[0].set(xticks=x, xticklabels=names, ylim=(0, 1.12), ylabel='Exact answer accuracy', title='Which adaptation preserves language?')
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].legend(fontsize=8)
    image = axes[1].imshow(recoveries, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[1].set(xticks=[0, 1, 2], xticklabels=['target object', 'other object', 'background'], yticks=x, yticklabels=names, xlabel='Patched visual-token source', ylabel='Adapter condition', title='Counterfactual target-margin recovery')
    for i in range(recoveries.shape[0]):
        for j in range(3):
            axes[1].text(j, i, f'{recoveries[i,j]:.2f}', ha='center', va='center', fontsize=9)
    fig.colorbar(image, ax=axes[1], label='Fraction of clean/corrupt margin gap')
    for label, values in singular_values.items():
        axes[2].plot(np.arange(1, len(values)+1), values, 'o-', label=label)
    axes[2].set(xlabel='Singular value index', ylabel='Singular value', title='Same-rank and same-norm update controls')
    axes[2].legend(fontsize=8)
    return fig
