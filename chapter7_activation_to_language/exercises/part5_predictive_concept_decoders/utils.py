"""Dataset templates and plotting helpers for [7.5] Predictive Concept Decoders.

Nothing in here is part of the method you're learning: it's the word lists and template
code that build the GPT-2 dataset, and matplotlib code for the figures. All of the bottleneck,
training, metric and intervention logic lives in the notebook.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t

# ---------------------------------------------------------------------------------------------
# The templated GPT-2 dataset
# ---------------------------------------------------------------------------------------------

CONCEPT_NAMES = ["female", "plural", "positive", "outdoors"]

FEMALE_NOUNS = (
    "woman girl queen mother actress lady bride nun princess sister wife aunt daughter grandmother waitress niece "
    "mom grandma girlfriend duchess empress heroine goddess widow stepmother policewoman businesswoman landlady "
    "schoolgirl maid ballerina hostess priestess witch heiress housewife granddaughter stewardess"
).split()
MALE_NOUNS = (
    "man boy king father actor gentleman groom monk prince brother husband uncle son grandfather waiter nephew "
    "dad grandpa boyfriend duke emperor hero god widower stepfather policeman businessman landlord "
    "schoolboy butler wrestler host priest wizard heir househusband grandson steward"
).split()
FEMALE_PLURALS = (
    "women girls queens mothers actresses ladies brides nuns princesses sisters wives aunts daughters grandmothers "
    "waitresses nieces moms grandmas girlfriends duchesses empresses heroines goddesses widows stepmothers "
    "policewomen businesswomen landladies schoolgirls maids ballerinas hostesses priestesses witches heiresses "
    "housewives granddaughters stewardesses"
).split()
MALE_PLURALS = (
    "men boys kings fathers actors gentlemen grooms monks princes brothers husbands uncles sons grandfathers "
    "waiters nephews dads grandpas boyfriends dukes emperors heroes gods widowers stepfathers policemen "
    "businessmen landlords schoolboys butlers wrestlers hosts priests wizards heirs househusbands grandsons stewards"
).split()
FEMALE_NAMES = (
    "Mary Anna Sarah Emma Lisa Laura Julia Maria Emily Jessica Rachel Rebecca Linda Susan Karen Helen Alice Grace "
    "Lucy Sophie Olivia Hannah Kate Jane Amy Megan Nicole Elizabeth Victoria Natalie"
).split()
MALE_NAMES = (
    "John David Michael James Robert Tom Peter Paul Mark Daniel Richard Steven Kevin Brian George Edward Henry Jack "
    "Sam Ben Luke Adam Chris Matthew Andrew Joseph Thomas William Charles Frank"
).split()
POSITIVE_EVENTS = [
    "won the big prize", "got a wonderful gift", "passed the exam easily", "had a fantastic day",
    "heard the great news", "found a lovely puppy", "received a big promotion", "enjoyed a delicious meal",
    "won the championship", "got a big raise", "made a new friend", "saw a beautiful sunset",
    "celebrated a happy birthday", "received a warm welcome", "found a hidden treasure", "won the lottery",
    "had a wonderful time", "got the dream job", "finished the race first", "received a kind letter",
    "had a great vacation", "laughed at a funny joke", "earned a gold medal", "got an amazing surprise",
    "heard some wonderful music",
]
NEGATIVE_EVENTS = [
    "lost the big game", "got a terrible injury", "failed the exam badly", "had an awful day",
    "heard the sad news", "lost a beloved pet", "received a harsh punishment", "suffered a painful accident",
    "lost the job", "caught a bad cold", "broke a precious vase", "missed the last train",
    "had a horrible nightmare", "got into a nasty fight", "lost all the money", "attended a sad funeral",
    "got a parking ticket", "burned the dinner", "fell down the stairs", "got terrible news",
    "cried all night", "lost the house", "got stung by a bee", "crashed the car", "heard a frightening scream",
]
INDOOR_PLACES = [
    "at home", "in the kitchen", "at the office", "in the library", "at the hospital", "in the museum",
    "at school", "in the classroom", "at the bank", "in the hotel", "at the restaurant", "in the church",
    "at the theater", "in the basement", "at the mall", "in the bedroom",
]
OUTDOOR_PLACES = [
    "in the park", "at the beach", "in the forest", "on the mountain", "in the garden", "by the river",
    "on the farm", "in the desert", "at the lake", "in the field", "on the island", "in the jungle",
    "by the sea", "in the meadow", "on the hill", "in the woods",
]
FRAMES = [
    "{S} {e} {p}.",
    "Yesterday, {s} {e} {p}.",
    "{P}, {s} {e}.",
    "Last week {s} {e} {p}.",
    "According to the report, {s} {e} {p}.",
    "One morning {s} {e} {p}.",
]


def _split_words(words: list[str], split: str, train_frac: float = 0.8) -> list[str]:
    n_train = round(len(words) * train_frac)
    return words[:n_train] if split == "train" else words[n_train:]


def make_template_prompts(split: str, n: int, seed: int = 0) -> tuple[list[str], t.Tensor]:
    """
    Returns `n` templated one-sentence stories and a (n, 4) float tensor of binary concept labels, with
    columns in the order of `CONCEPT_NAMES` (female, plural, positive, outdoors).

    The four concepts are sampled independently and uniformly. Every word list (nouns, names, events,
    places) is split 80/20 into train/test words, so a `split="test"` prompt never contains a subject,
    event or place that appears in any `split="train"` prompt. The sentence frames are shared.
    """
    assert split in ("train", "test")
    rng = random.Random(seed)
    f_nouns, m_nouns, f_plurals, m_plurals, f_names, m_names = (
        _split_words(w, split) for w in [FEMALE_NOUNS, MALE_NOUNS, FEMALE_PLURALS, MALE_PLURALS, FEMALE_NAMES, MALE_NAMES]
    )
    positive, negative, indoor, outdoor = (
        _split_words(w, split) for w in [POSITIVE_EVENTS, NEGATIVE_EVENTS, INDOOR_PLACES, OUTDOOR_PLACES]
    )
    prompts, labels = [], []
    for _ in range(n):
        female, plural, pos, out = [rng.random() < 0.5 for _ in range(4)]
        use_name = rng.random() < 0.4
        names = f_names if female else m_names
        if not plural:
            subject = rng.choice(names) if use_name else "the " + rng.choice(f_nouns if female else m_nouns)
        else:
            subject = " and ".join(rng.sample(names, 2)) if use_name else "the " + rng.choice(f_plurals if female else m_plurals)
        event = rng.choice(positive if pos else negative)
        place = rng.choice(outdoor if out else indoor)
        frame = rng.choice(FRAMES)
        prompts.append(
            frame.format(S=subject[0].upper() + subject[1:], s=subject, e=event, p=place, P=place[0].upper() + place[1:])
        )
        labels.append([female, plural, pos, out])
    return prompts, t.tensor(labels, dtype=t.float)


# ---------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------


def plot_training_curves(curves: dict[str, list[float]], title: str, ylabel: str, baselines: dict[str, float] | None = None, logy: bool = True):
    """One line per training curve, plus dashed horizontal lines for baselines."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, values in curves.items():
        ax.plot(values, label=name, alpha=0.9)
    for (name, value), color in zip((baselines or {}).items(), ["gray", "black", "tab:red"]):
        ax.axhline(value, ls="--", color=color, label=name)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_alignment(cos_mats: dict[str, t.Tensor], true_labels: list[str], n_quiet: int, title: str):
    """
    Side-by-side heatmaps of |cosine similarity| between each model's latents (rows) and the planted
    ground-truth directions (columns). A vertical line separates the quiet (behavioural) concepts from
    the loud (nuisance) features.
    """
    n = len(cos_mats)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    for ax, (name, mat) in zip(axes[0], cos_mats.items()):
        mat = mat.detach().abs().cpu().numpy().T  # (latent, true)
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.axvline(n_quiet - 0.5, color="tab:red", lw=2)
        ax.set_xticks(range(len(true_labels)))
        ax.set_xticklabels(true_labels, rotation=90, fontsize=8)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_ylabel("latent")
        ax.set_xlabel("ground-truth direction  (quiet | loud)")
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), label="|cosine similarity|", shrink=0.8)
    fig.suptitle(title)
    plt.show()


def plot_bars(values: dict[str, float], title: str, ylabel: str, highlight: int = 0, fmt: str = "{:.3f}", logy: bool = False):
    """Bar chart for comparing a method (highlighted) against baselines and controls."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    names, vals = list(values), list(values.values())
    colors = ["tab:gray"] * len(vals)
    colors[highlight] = "tab:blue"
    bars = ax.bar(names, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()


def plot_grouped_bars(
    df: pd.DataFrame, title: str, ylabel: str, ylim: tuple[float, float] | None = None, chance: float | None = None, ax=None
):
    """Grouped bar chart: one group per row of `df`, one bar per column."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.8))
    n_rows, n_cols = df.shape
    width = 0.8 / n_cols
    colors = plt.cm.tab10(np.arange(n_cols))
    for j, col in enumerate(df.columns):
        ax.bar(np.arange(n_rows) + (j - (n_cols - 1) / 2) * width, df[col].values, width, label=col, color=colors[j])
    if chance is not None:
        ax.axhline(chance, color="black", ls="--", lw=1, label="chance")
    ax.set_xticks(np.arange(n_rows))
    ax.set_xticklabels(df.index)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.grid(alpha=0.3, axis="y")
    if show:
        plt.tight_layout()
        plt.show()


def plot_heatmaps(mats: dict[str, t.Tensor], row_label: str, col_labels: list[str], title: str, vmin: float = 0.5, vmax: float = 1.0):
    """Side-by-side annotated heatmaps (e.g. latent x concept AUROC), one per entry of `mats`."""
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.6), squeeze=False)
    for ax, (name, mat) in zip(axes[0], mats.items()):
        mat = mat.detach().cpu().numpy() if isinstance(mat, t.Tensor) else np.asarray(mat)
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] < (vmin + vmax) / 2 else "black")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(mat.shape[0]))
        ax.set_ylabel(row_label)
        ax.set_title(name, fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), label="held-out AUROC", shrink=0.8)
    fig.suptitle(title)
    plt.show()


def plot_faithfulness_scatter(points: dict[str, tuple[t.Tensor, t.Tensor]], title: str, max_points: int = 4000):
    """
    Scatter of predicted change (x) vs actual change (y), one colour per entry. Each entry's legend label
    includes its Pearson correlation. The dashed diagonal is perfect faithfulness.
    """
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    lo, hi = 0.0, 0.0
    for name, (x, y) in points.items():
        x, y = x.detach().float().cpu(), y.detach().float().cpu()
        r = t.corrcoef(t.stack([x, y]))[0, 1].item()
        if len(x) > max_points:
            idx = t.randperm(len(x), generator=t.Generator().manual_seed(0))[:max_points]
            x, y = x[idx], y[idx]
        ax.scatter(x, y, s=4, alpha=0.35, label=f"{name} (r = {r:.2f})")
        lo, hi = min(lo, x.min().item(), y.min().item()), max(hi, x.max().item(), y.max().item())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfectly faithful")
    ax.set_xlabel("predicted change in log-prob (decoder)")
    ax.set_ylabel("actual change in log-prob (model)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    leg = ax.legend(fontsize=7, markerscale=3)
    for handle in leg.legend_handles:
        handle.set_alpha(1)
    plt.tight_layout()
    plt.show()


def plot_faithfulness_bars(df: pd.DataFrame, title: str):
    """
    `df` has a MultiIndex (latent, condition) and columns "actual KL" and "predicted KL". Draws one panel
    per latent, with paired bars (actual vs predicted) for each intervention condition.
    """
    latents = list(dict.fromkeys(df.index.get_level_values(0)))
    fig, axes = plt.subplots(1, len(latents), figsize=(4.2 * len(latents), 4.2), sharey=True, squeeze=False)
    for ax, latent in zip(axes[0], latents):
        sub = df.loc[latent]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["actual KL"], 0.4, label="actual (GPT-2)", color="tab:orange")
        ax.bar(x + 0.2, sub["predicted KL"], 0.4, label="predicted (PCD)", color="tab:blue")
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"remove {latent}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("KL(clean || edited), mean over probes")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
