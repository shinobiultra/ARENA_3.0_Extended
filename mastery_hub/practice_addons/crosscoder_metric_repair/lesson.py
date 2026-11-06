"""Notebook source for the bounded 6.5 CPU metric companion.

Core implementations are emitted into notebook cells, never imported by learners.
Only plotting is shared plumbing. No models, training or downloads.
"""
from textwrap import dedent


PEARSON = '''
def pearson_columns(reference, candidates):
    """Absolute Pearson r: reference[n], candidates[n, k]; constants score 0."""
    x = np.asarray(reference, dtype=float)
    y = np.asarray(candidates, dtype=float)
    if x.ndim != 1 or y.ndim != 2 or y.shape[0] != x.size or x.size < 2:
        raise ValueError("Expected reference[n], candidates[n,k], n >= 2")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite values required")
    x = x - x.mean()
    y = y - y.mean(axis=0)
    numerator = x @ y
    denominator = np.linalg.norm(x) * np.linalg.norm(y, axis=0)
    return np.clip(np.abs(np.divide(numerator, denominator,
                    out=np.zeros_like(numerator), where=denominator > 0)), 0, 1)
'''

AUC = '''
def rank_auc(scores, labels):
    """Probability positive beats negative, counting ties as one half."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Expected equal one-dimensional scores and labels")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Finite scores and binary labels required")
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not positive.size or not negative.size:
        raise ValueError("Both classes required")
    differences = positive[:, None] - negative[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))
'''

FIT = '''
def fit_orientation(train_scores, train_labels):
    """Select a sign using TRAIN only; tie convention is +1."""
    return 1 if rank_auc(train_scores, train_labels) >= 0.5 else -1
'''

EVALUATE = '''
def evaluate_frozen(eval_scores, eval_labels, orientation):
    """Measure EVAL with the supplied frozen TRAIN sign. Never refit here."""
    if orientation not in (-1, 1):
        raise ValueError("Orientation must be -1 or +1")
    return rank_auc(orientation * np.asarray(eval_scores, dtype=float), eval_labels)
'''

CHECKS = [
    '''x = np.array([0., 1., 2., 3.])
columns = np.column_stack([x, -x, 2*x + 7, np.ones(4), [1, -1, -1, 1]])
np.testing.assert_allclose(pearson_columns(x, columns), [1, 1, 1, 0, 0], atol=1e-12)
np.testing.assert_allclose(pearson_columns(np.ones(4), x[:, None]), [0])
print("Pearson: [1, 1, 1, 0, 0]; constant reference: [0]")''',
    '''labels = np.array([0, 0, 1, 1])
assert rank_auc([0, 1, 2, 3], labels) == 1
assert rank_auc([3, 2, 1, 0], labels) == 0
assert rank_auc([1, 1, 1, 1], labels) == 0.5
# Positive scores 1,2 versus negative scores 0,1: 1 + .5 + 1 + 1.
assert rank_auc([0, 1, 1, 2], labels) == 0.875
assert rank_auc([0, 3, 1, 2], labels) == 0.5
print("AUC increasing/reversed/tied/partial-tie/shuffled: 1, 0, .5, .875, .5")''',
    '''train_scores = np.array([3., 2., 1., 0.])
train_labels = np.array([0, 0, 1, 1])
orientation = fit_orientation(train_scores, train_labels)
assert orientation == -1
assert fit_orientation(-train_scores, train_labels) == 1
assert fit_orientation(np.ones(4), train_labels) == 1
print("TRAIN sign: -1; inverted TRAIN: +1; TRAIN tie: +1")''',
    '''# TRAIN has fitted -1. EVAL reverses its relationship to the label.
eval_scores = np.array([0., 1., 2., 3.])
eval_labels = np.array([0, 0, 1, 1])
assert evaluate_frozen(eval_scores, eval_labels, orientation) == 0
assert evaluate_frozen(eval_scores, 1-eval_labels, orientation) == 1
assert evaluate_frozen([0, 3, 1, 2], eval_labels, orientation) == 0.5
assert orientation == -1
print("Frozen EVAL AUC: 0; changed labels: 1; fixed-shuffled ranks: .5; sign stays -1")''',
]


def plot_metrics(sample_sizes, legacy, pearson, frozen, orientation_free, shuffled):
    """Plot already-computed measurements; the notebook owns every metric."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(sample_sizes, legacy, "o-", label="Legacy mixed normalization", color="#a34b33")
    axes[0].plot(sample_sizes, pearson, "s-", label="Centered Pearson |r|", color="#167568")
    axes[0].axhline(0, color="#666666", linestyle=":", label="Orthogonal/constant control = 0")
    axes[0].set(xlabel="Number of identical example pairs", ylabel="Feature match score",
                title="An exact copy should score 1", xticks=sample_sizes, ylim=(-.05, 1.12))
    bars = axes[1].bar(["Frozen TRAIN\nsign", "Orientation-free\nmax(AUC, 1−AUC)",
                         "Fixed-shuffled\nrank control"], [frozen, orientation_free, shuffled],
                        color=["#a34b33", "#3465a4", "#888888"])
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].axhline(.5, color="#666666", linestyle=":")
    axes[1].set(ylabel="EVAL AUC", title="EVAL reverses the TRAIN relationship", ylim=(0, 1.15))
    axes[0].legend(fontsize=8, loc="lower right")
    return fig


def notebook_cells(worked=False):
    """Return plain cell descriptions, keeping all taught algorithms visible."""
    cells = []

    def md(text):
        cells.append(("markdown", dedent(text).strip()))

    def code(text):
        cells.append(("code", dedent(text).strip()))

    md('''
    # 6.5 companion: when a perfect match scores .75

    **Question:** Can a metric say “perfect feature match” or “perfect behavioral
    prediction” while measuring the wrong thing?

    Four identical example pairs should have absolute Pearson correlation 1.
    A feature whose ranking reverses after TRAIN should fail a direction-sensitive
    EVAL check. In this short companion you will repair these two contracts,
    draw the failure, and predict a changed case. This is guided practice with
    public answers, not an unseen assessment.

    Prerequisites: arrays, means, dot products, and the feature matching/behavior
    distinction in [6.5 Crosscoders and Model Diffing](../../../chapter6_sparse_feature_methods/instructions/pages/05_[6.5]_Crosscoders_and_Model_Diffing.md).
    A feature column has one activation per example; a behavior label is 0 or 1.
    Here exact four-row fixtures replace trained activations so each answer can
    be counted by hand. No crosscoder is trained or evaluated by this companion.

    ## Learning objectives

    1. Compute absolute Pearson correlation without mixing sample and population normalization.
    2. Count positive/negative ranking pairs, including ties.
    3. Fit score orientation on TRAIN, then freeze it for EVAL.
    4. Use controls to explain why a high orientation-free score can hide reversal.

    Allow about 20–30 minutes for four small functions. Execution uses CPU NumPy
    and Matplotlib only. Install these in your own environment if absent; the
    notebook itself never installs packages or downloads data.

    **Predict before running:** with four pairs, what does averaging products of
    sample-standardized identical values return: .75 or 1? What happens at eight
    pairs? Write the arithmetic, not a confidence score with no meaning.
    ''')
    code('''import numpy as np
import sys
from pathlib import Path
from IPython.display import display
get_ipython().run_line_magic("matplotlib", "inline")
# Works from repository root or this companion directory.
root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "mastery_hub").is_dir())
sys.path.insert(0, str(root))
from mastery_hub.practice_addons.crosscoder_metric_repair.lesson import plot_metrics
''')
    md(r'''
    ## 1. Repair feature matching

    Let centered vectors be $x_c=x-\bar x$ and each column
    $y_{c,j}=y_j-\bar y_j$. Compute
    $|r_j|=|x_c^T y_{c,j}|/(\|x_c\|\|y_{c,j}\|)$.
    `reference` has shape `[n]`; `candidates` has shape `[n,k]`.
    This exercise intentionally ignores sign for feature matching: negation has
    the same absolute match. Behavioral ranking below will keep sign.

    **Exercise (5–7 min):** implement `pearson_columns`. Return `[k]` floats.
    A constant column or reference scores **0 by convention**: Pearson is
    mathematically undefined there, so 0 must not be interpreted as evidence of
    independence. Reject nonfinite values, wrong shapes and fewer than two rows.
    ''')
    specs = [
        (PEARSON, 'def pearson_columns(reference, candidates):',
         'Center first; use one dot product and column norms. Divide only where the denominator is nonzero.',
         'The literal oracle is [1, 1, 1, 0, 0]. The last column is centered and orthogonal to x. A sample-std/population-mean implementation yields .75 for the first column and must fail.'),
        (AUC, 'def rank_auc(scores, labels):',
         'Build positive[:, None] minus negative[None, :]. Count wins and half of ties.',
         'The partial-tie example has four pairs: three wins and one tie, so (3+.5)/4=.875. Reversed ranking is 0, not 1.'),
        (FIT, 'def fit_orientation(train_scores, train_labels):',
         'Compare the TRAIN AUC with .5; do not request any EVAL argument.',
         'TRAIN has lower scores for positives, so choose -1. A tie chooses +1 deterministically; it does not establish useful signal.'),
        (EVALUATE, 'def evaluate_frozen(eval_scores, eval_labels, orientation):',
         'Multiply scores by the supplied sign before measuring. There is no maximization or fitting here.',
         'Frozen -1 reverses increasing EVAL scores: zero positive-negative wins. Flipping EVAL labels changes AUC to 1 while the fitted sign stays -1.'),
    ]
    for i, (source, signature, hint, interpretation) in enumerate(specs):
        if i == 1:
            md('''
            ## 2. Repair behavioral ranking

            AUC is the fraction of all positive-negative example pairs where the
            positive score is larger; ties contribute one half. This definition
            works without choosing a threshold. Both classes must be present.

            **Exercise (5 min):** implement `rank_auc(scores[n], labels[n])`.
            Return a scalar in [0,1]. Reject invalid shapes, nonfinite scores,
            nonbinary labels and single-class inputs. Use the four-pair example
            in the immediate test as your independent oracle.
            ''')
        elif i == 2:
            md('''
            ## 3. Select a sign on TRAIN

            A sparse feature's sign can be arbitrary. Choose whether higher or
            lower scores indicate label 1 using TRAIN, then commit to that rule.
            `max(AUC,1-AUC)` answers “is either orientation separable?” It cannot
            answer “does my already-chosen orientation still predict behavior?”

            **Exercise (2–3 min):** `fit_orientation` takes TRAIN scores and
            labels only. Return +1 when AUC >= .5, otherwise -1. No threshold,
            feature selection, or EVAL label is part of this contract.
            ''')
        elif i == 3:
            md('''
            ## 4. Freeze before EVAL

            TRAIN scores `[3,2,1,0]` and labels `[0,0,1,1]` fit sign -1.
            EVAL scores `[0,1,2,3]` now reverse that relationship. Predict the
            frozen AUC before revealing the test. Should a failed ranking be
            rescued by looking at EVAL labels again?

            **Exercise (2–3 min):** `evaluate_frozen` takes EVAL scores, labels,
            and the existing sign. Validate the sign is -1 or +1, then measure
            the oriented scores with your `rank_auc`. It must never call
            `fit_orientation` or select another sign from EVAL.
            ''')
        code(source if worked else signature + '\n    raise NotImplementedError("Implement this contract above")')
        code(CHECKS[i])
        md(f'<details><summary>Hint</summary>\n\n{hint}\n\n</details>\n\n'
           f'<details><summary>Expected output and interpretation</summary>\n\n{interpretation}\n\n</details>\n\n'
           f'<details><summary>Solution</summary>\n\n```python\n{source.strip()}\n```\n\n</details>')
    md('''
    ## See both errors together

    The left panel uses exact copies at two sample sizes, plus a verified zero
    control. The legacy expression divides by sample standard deviation but
    averages the products with denominator n. Since squared sample z-scores
    sum to n−1, its answer is (n−1)/n. Increasing n hides the bug rather than
    fixing it. The right panel uses the same frozen TRAIN sign for every EVAL
    score. Its fixed-shuffled ranks are deliberately chosen to have exactly
    two of four positive-negative wins, not selected from random seeds.
    ''')
    code('''import matplotlib.pyplot as plt
sample_sizes = [4, 8]
legacy_values, pearson_values = [], []
for n in sample_sizes:
    x = np.arange(n, dtype=float)
    legacy_values.append(float(np.mean(((x-x.mean()) / x.std(ddof=1)) ** 2)))
    pearson_values.append(float(pearson_columns(x, x[:, None])[0]))
np.testing.assert_allclose(legacy_values, [.75, .875])
np.testing.assert_allclose(pearson_values, [1, 1])
# Orthogonal and constant controls were checked in Part 1, not fabricated by plotting.
np.testing.assert_allclose(pearson_columns(np.arange(4.),
    np.column_stack([[1, -1, -1, 1], np.ones(4)])), [0, 0], atol=1e-12)
frozen = evaluate_frozen(eval_scores, eval_labels, orientation)
raw_auc = rank_auc(eval_scores, eval_labels)
orientation_free = max(raw_auc, 1-raw_auc)
shuffled = evaluate_frozen([0, 3, 1, 2], eval_labels, orientation)
assert (frozen, orientation_free, shuffled) == (0, 1, .5)
fig = plot_metrics(sample_sizes, legacy_values, pearson_values,
                   frozen, orientation_free, shuffled)
display(fig)
plt.close(fig)  # explicit display above; avoid a duplicate inline auto-display
''')
    md('''
    <details><summary>Help — what does the figure establish?</summary>

    The perfect-copy baseline stays at 1 under Pearson, while the legacy score
    moves .75 → .875 just because the dataset grew. The orthogonal/constant
    control is 0. On reversed EVAL, frozen AUC is 0, orientation-free AUC is 1,
    and the fixed-shuffled control is .5. The last result is an exact constructed
    control, not an empirical estimate of chance performance. Both AUC values
    are valid for their stated questions; calling orientation-free AUC “signed
    prediction” is the error. EVAL labels measure a frozen rule, never select it.

    </details>

    ## Try it yourself: change the condition, keep the commitment

    Before running, predict what `EVAL_SIGN=-1` and `N=8` will do. Explain which
    result changes because of sample size and which because of reversal. Edit
    only these two knobs. TRAIN and its fitted sign remain fixed. The answers
    are exposed guided transfer, not independent scientific evaluation.
    ''')
    code('''EVAL_SIGN = 1  # try -1: restores TRAIN's direction
N = 4           # try 8; choose an even integer >= 4
assert EVAL_SIGN in (-1, 1) and isinstance(N, int) and N >= 4 and N % 2 == 0
play_scores = EVAL_SIGN * np.arange(N, dtype=float)
play_labels = np.array([0]*(N//2) + [1]*(N//2))
play_auc = evaluate_frozen(play_scores, play_labels, orientation)
play_legacy = np.mean(((play_scores-play_scores.mean()) / play_scores.std(ddof=1))**2)
play_pearson = pearson_columns(play_scores, play_scores[:, None])[0]
print(f"Frozen TRAIN sign {orientation:+d}; EVAL AUC {play_auc:.3f}; "
      f"legacy identity {play_legacy:.3f}; Pearson identity {play_pearson:.3f}")
assert orientation == -1  # never refit on play/EVAL labels
''')
    md('''
    Expected default: `Frozen TRAIN sign -1; EVAL AUC 0.000; legacy identity
    0.750; Pearson identity 1.000`. With sign -1 and n=8: AUC 1.000, legacy
    .875, Pearson 1.000. Revise your original explanation: can a larger sample
    conceal a normalization mistake? Can an orientation-free metric reveal a
    reversal? What would happen if all scores tied?

    ## Return to 6.5: scope and limitations

    Feature matching and behavior prediction support different claims in the
    paired sparse-feature study. Absolute correlation can match a negated
    activation column; a TRAIN-frozen behavior ranking must preserve its chosen
    direction. Neither metric establishes causal intervention success or model
    ownership by itself. This CPU companion isolates measurement contracts.
    It does not recompute, replace or relabel any historical 6.5 result, fit a
    crosscoder, test a native base/chat pair, or establish learner acceptance.
    The EVAL examples are tiny, deterministic and public. There are no confidence
    intervals or population-generalization claims. Pairwise AUC takes O(P×Q)
    memory for P positives and Q negatives; use this transparent implementation
    for the small teaching fixture, not a large corpus.

    Further investigation: hold the sign fixed and introduce one tied score.
    Enumerate all pairs before running. Why can AUC .5 mean tied scores or a
    mixture of correct and incorrect rankings? For the full planted-model,
    ownership, SVD and intervention context, return to the canonical
    [6.5 exercise notebook](../../../chapter6_sparse_feature_methods/exercises/part5_crosscoders/6.5_Crosscoders_and_Model_Diffing_exercises.ipynb)
    and [reading page](../../../chapter6_sparse_feature_methods/instructions/pages/05_[6.5]_Crosscoders_and_Model_Diffing.md).
    Original ARENA remains the append-only base; this is a small repair companion,
    not a replacement chapter or new scientific result.
    ''')
    return cells
