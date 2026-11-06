"""PROTOTYPE for issue 245 (throwaway, never merged): the 12.1 held-out quiz key and a local scorer.

In the real design the scoring half (bins, rankings, the score table) would live once in a shared
helper, and each lesson would only declare its QUESTIONS/KEY/RUNS. Here everything sits in one file.

Every number in RUNS comes from 12.1 as it is today:
  * "batch 64, lr 5e-4" rows: the Part 2 "Training with the sigmoid loss" expected output (one seed).
  * every other row: the Bonus "Batch size, temperature and the learning rate" text (the lesson
    author's exploratory runs, 2,000 steps, same tiny CLIP and data).
No new run was made for this prototype.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

# ---------------------------------------------------------------- per-lesson part

# Image-to-text R@1 on the 120 fresh test images, one entry per seed.
RUNS: dict[tuple[str, str], list[float]] = {
    ("CLIP", "batch 64, lr 5e-4\n(Part 2)"): [0.942],
    ("SigLIP", "batch 64, lr 5e-4\n(Part 2)"): [0.958],
    ("CLIP", "batch 32\n(Bonus)"): [0.93],
    ("SigLIP", "batch 32\n(Bonus)"): [0.94],
    ("CLIP", "batch 16\n(Q3)"): [0.93, 0.86, 0.92],
    ("SigLIP", "batch 16\n(Q3)"): [0.57, 0.63, 0.73],
    ("CLIP", "lr 2e-3\n(Q2: A, C)"): [0.96, 0.97],
    ("SigLIP", "lr 2e-3\n(Q2: A, C)"): [0.80, 0.58],
    ("CLIP", "lr 5e-3\n(Q2: B, D; Q4)"): [0.96, 0.98],
    ("SigLIP", "lr 5e-3\n(Q2: B, D; Q4)"): [0.008, 0.008],
}
SIGLIP_LR_5E3_FINAL_LOSS = 5.151  # both seeds, mean of the last 50 steps

# Q1 uses fixed logits (4 images x 5 candidate captions), from the old blind CLIP companion.
Q1_LOGITS = np.array([[(((7 * i + 5 * j + 3) % 17) - 8) / 4 for j in range(5)] for i in range(4)])
Q1_TRUE_CAPTION = [1, 2, 0, 1]  # the correct caption for each image, among the first three

KEY = {
    "Q1a_softmax": "falls",
    "Q1a_sigmoid": "unchanged",
    "Q1b": "c",
    "Q2": "D < C < A = B",
    "Q3": "a",
    "Q4": "d",
}
POINTS = {"Q1a_softmax": 1, "Q1a_sigmoid": 1, "Q1b": 2, "Q2": 6, "Q3": 5, "Q4": 5}
KIND = {"Q1a_softmax": "choice", "Q1a_sigmoid": "choice", "Q1b": "bins", "Q2": "ranking", "Q3": "bins", "Q4": "bins"}

# ---------------------------------------------------------------- shared part

BINS = "abcde"


def _ranks(ranking: str) -> dict[str, int]:
    """'D < C < A = B' -> {'D': 0, 'C': 1, 'A': 2, 'B': 2}."""
    ranks = {}
    for position, group in enumerate(ranking.replace(" ", "").upper().split("<")):
        for label in group.split("="):
            ranks[label] = position
    return ranks


def _score_one(kind: str, answer: str, key: str, points: int) -> float:
    answer = answer.strip()
    if not answer:
        return 0.0
    if kind == "choice":
        return float(points) if answer.lower() == key else 0.0
    if kind == "bins":  # full points for the right interval, half for an adjacent one
        distance = abs(BINS.index(answer.lower()) - BINS.index(key))
        return float(points) if distance == 0 else points / 2 if distance == 1 else 0.0
    if kind == "ranking":  # one point per pair of runs whose order (<, > or =) you got right
        mine, truth = _ranks(answer), _ranks(key)
        if set(mine) != set(truth):
            return 0.0
        pairs = list(combinations(sorted(truth), 2))
        right = sum(np.sign(mine[x] - mine[y]) == np.sign(truth[x] - truth[y]) for x, y in pairs)
        return points * right / len(pairs)
    raise ValueError(kind)


def score(my_answers: dict[str, str]) -> None:
    """Print a score table for your answers. Nothing is sent anywhere or saved."""
    total = 0.0
    print(f"{'question':<13}{'your answer':<17}{'answer':<17}points")
    for q, key in KEY.items():
        got = _score_one(KIND[q], my_answers.get(q, ""), key, POINTS[q])
        total += got
        print(f"{q:<13}{my_answers.get(q, '') or '-':<17}{key:<17}{got:g} / {POINTS[q]}")
    print(f"{'total':<47}{total:g} / {sum(POINTS.values())}")


def reveal_q1(zero_shot_probabilities) -> None:
    """Q1's reveal is live: it runs *your* Part 3 function on the fixed logits. It takes no time."""
    rows = np.arange(4)
    for kind in ("softmax", "sigmoid"):
        before = np.asarray(zero_shot_probabilities(Q1_LOGITS[:, :3], kind))[rows, Q1_TRUE_CAPTION]
        after = np.asarray(zero_shot_probabilities(Q1_LOGITS, kind))[rows, Q1_TRUE_CAPTION]
        print(f"{kind}: probability of the correct caption, 3 candidates -> 5 candidates")
        for i in rows:
            print(f"  image {i}: {before[i]:.3f} -> {after[i]:.3f}   ({100 * (after[i] - before[i]):+5.1f} points)")
        print(f"  mean change: {100 * np.mean(after - before):+.1f} percentage points\n")


def plot_runs(path: str | None = None):
    """The reveal figure: every run behind Q2-Q4, one dot per seed, bars at the seed mean."""
    import matplotlib.pyplot as plt

    settings = list(dict.fromkeys(s for _, s in RUNS))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for offset, (loss, colour) in zip((-0.18, 0.18), (("CLIP", "#4c72b0"), ("SigLIP", "#dd8452"))):
        for x, setting in enumerate(settings):
            seeds = RUNS[(loss, setting)]
            ax.bar(x + offset, np.mean(seeds), width=0.34, color=colour, alpha=0.35,
                   label=f"{loss} loss" if x == 0 else None)
            ax.scatter([x + offset] * len(seeds), seeds, color=colour, s=18, zorder=3)
    ax.axhline(1 / 120, color="grey", ls=":", lw=1)
    ax.text(len(settings) - 0.5, 1 / 120 + 0.02, "chance (1/120)", ha="right", fontsize=8, color="grey")
    ax.set_xticks(range(len(settings)), settings, fontsize=8)
    ax.set_ylabel("image-to-text R@1")
    ax.set_ylim(0, 1.02)
    ax.set_title("Tiny CLIP, 2,000 steps: one dot per seed (one seed where only one dot)", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
