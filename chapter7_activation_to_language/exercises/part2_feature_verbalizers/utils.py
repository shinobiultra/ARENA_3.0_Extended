"""Data loading, LLM plumbing and plotting helpers for [7.2] Feature Verbalizers.

Nothing in here is part of the method you're learning. The functions below load text, load the
explainer LLM, run batched forward passes / generation through it, and draw figures. Everything that
*is* the method (choosing examples, writing prompts, simulating activations, computing scores, the
controls and the interventions) lives in the notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------


def _tokenize_texts(
    model: HookedTransformer, texts: list[str], n_sequences: int, seq_len: int, seed: int, max_chars: int | None
) -> t.Tensor:
    rng = np.random.default_rng(seed)
    rows = []
    for i in rng.permutation(len(texts)):
        text = texts[i] if max_chars is None else texts[i][:max_chars]
        toks = model.to_tokens(text, truncate=True)[0]
        if len(toks) >= seq_len:
            rows.append(toks[:seq_len])
        if len(rows) == n_sequences:
            break
    return t.stack(rows).cpu()


def load_pile_tokens(model: HookedTransformer, n_sequences: int = 4000, seq_len: int = 64, seed: int = 0) -> t.Tensor:
    """
    Returns a [n_sequences, seq_len] tensor of tokenized text from NeelNanda/pile-10k (a 10,000-document sample of
    the Pile: web text, code, papers, legal text, ...). Each sequence is the start of a different document, with BOS
    prepended. The parquet file is ~33MB.
    """
    path = hf_hub_download(
        "NeelNanda/pile-10k", "data/train-00000-of-00001-4746b8785c874cc7.parquet", repo_type="dataset"
    )
    texts = pd.read_parquet(path)["text"].tolist()
    return _tokenize_texts(model, texts, n_sequences, seq_len, seed, max_chars=2000)


def load_tinystories_tokens(
    model: HookedTransformer, n_sequences: int = 2000, seq_len: int = 64, seed: int = 0
) -> t.Tensor:
    """
    Returns a [n_sequences, seq_len] tensor of tokenized TinyStories text (simple children's stories), with BOS
    prepended. We use the validation split, which is a single ~20MB file.
    """
    path = hf_hub_download(
        "roneneldan/TinyStories", "data/validation-00000-of-00001-869c898b519ad725.parquet", repo_type="dataset"
    )
    texts = pd.read_parquet(path)["text"].tolist()
    return _tokenize_texts(model, texts, n_sequences, seq_len, seed, max_chars=None)


# ---------------------------------------------------------------------------------------------------
# The explainer / simulator LLM
# ---------------------------------------------------------------------------------------------------


def load_explainer(name: str, device: t.device | str):
    """
    Loads a HuggingFace chat model in bfloat16 and its tokenizer (with left padding, which is what we want for
    batched generation and for reading logits at the end of each prompt).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(name, dtype=t.bfloat16).to(device).eval()
    llm.requires_grad_(False)
    return tokenizer, llm


def chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """
    Applies the tokenizer's chat template and adds the start of the assistant's turn. For Qwen3 models we turn off
    "thinking mode", so the assistant answers directly (the template inserts an empty <think></think> block).
    """
    kwargs = {"enable_thinking": False} if "qwen3" in tokenizer.name_or_path.lower() else {}
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


@t.inference_mode()
def generate_chat(
    tokenizer, llm, conversations: list[list[dict[str, str]]], max_new_tokens: int = 40, batch_size: int = 16
) -> list[str]:
    """
    Greedy generation for a list of conversations (each a list of {"role", "content"} messages). Returns the
    assistant's reply to each conversation, as a string.
    """
    prompts = [chat_prompt(tokenizer, messages) for messages in conversations]
    replies = []
    for start in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[start : start + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(llm.device)
        out = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        replies += tokenizer.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
    return [reply.strip() for reply in replies]


@t.inference_mode()
def logits_at_positions(
    tokenizer,
    llm,
    input_ids: list[list[int]],
    positions: list[list[int]],
    token_ids: list[int],
    batch_size: int = 32,
) -> list[t.Tensor]:
    """
    Runs the LLM on each sequence in `input_ids`, and returns the logits of the tokens `token_ids` (only those, not the
    whole vocabulary) at the given positions.

    Args:
        input_ids:  one list of token ids per sequence (sequences can have different lengths)
        positions:  for each sequence, the positions whose *next-token* logits we want
        token_ids:  the vocabulary entries we care about (e.g. the ids of "0", ..., "9", or of "Yes" and "No")

    Returns:
        A list with one float tensor of shape (len(positions[k]), len(token_ids)) per sequence.

    Implementation notes: sequences are sorted by length and left-padded in batches, with position ids computed from
    the attention mask (so padding doesn't shift positions). We take the final hidden states and multiply them by only
    the rows of the unembedding we need, which is much cheaper than computing logits over the full ~150k-token vocab.
    """
    W_U = llm.get_output_embeddings().weight[token_ids]  # (n_tokens, d_model)
    order = sorted(range(len(input_ids)), key=lambda k: len(input_ids[k]))
    results: list[t.Tensor | None] = [None] * len(input_ids)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        max_len = max(len(input_ids[k]) for k in idx)
        ids = t.full((len(idx), max_len), tokenizer.pad_token_id, dtype=t.long)
        mask = t.zeros((len(idx), max_len), dtype=t.long)
        for row, k in enumerate(idx):
            ids[row, max_len - len(input_ids[k]) :] = t.tensor(input_ids[k])
            mask[row, max_len - len(input_ids[k]) :] = 1
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        hidden = llm.model(
            input_ids=ids.to(llm.device), attention_mask=mask.to(llm.device), position_ids=position_ids.to(llm.device)
        ).last_hidden_state
        for row, k in enumerate(idx):
            pos = t.tensor(positions[k], dtype=t.long) + (max_len - len(input_ids[k]))
            results[k] = (hidden[row, pos.to(llm.device)] @ W_U.T).float().cpu()
    return results


# ---------------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------------


def _clean_token(tok: str) -> str:
    return tok.replace("\n", "↵").replace("$", r"\$")


def plot_activation_distribution(
    acts: t.Tensor, top_values: list[float], quantile_values: list[float], random_values: list[float], title: str
):
    """
    Histogram of a unit's nonzero activations (log-scale counts), with the activations of the examples we sampled
    drawn as tick marks underneath: top-k examples, quantile-sampled examples and random examples.
    """
    acts = acts.flatten()
    nonzero = acts[acts > 0].numpy()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(nonzero, bins=60, color="tab:gray", alpha=0.6, label=f"all nonzero activations (n={len(nonzero):,})")
    ax.set_yscale("log")
    ymin = 0.3
    for values, color, name, y in [
        (top_values, "tab:red", "top-k examples (what the explainer sees)", 0.55),
        (quantile_values, "tab:blue", "quantile-sampled examples", 0.42),
        (random_values, "tab:green", "random examples (max activation in window)", 0.32),
    ]:
        ax.scatter(values, [y] * len(values), marker="|", s=250, color=color, label=name, zorder=3)
    ax.set_ylim(ymin * 0.9, None)
    ax.set_xlabel("activation")
    ax.set_ylabel("number of tokens (log scale)")
    frac = (acts > 0).float().mean().item()
    ax.set_title(f"{title}\n(fires on {frac:.2%} of tokens)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_simulation(
    windows: list[tuple[list[str], t.Tensor, t.Tensor]], titles: list[str], suptitle: str, max_tokens: int = 20
):
    """
    For each window (str_toks, true_acts, simulated_acts), draw two rows of coloured cells: the true activations
    (normalised by the max over all windows shown for that row's panel) and the simulator's predictions (on its 0-9
    scale). Each window is one panel.
    """
    n = len(windows)
    fig, axes = plt.subplots(n, 1, figsize=(15, 1.55 * n + 0.6))
    axes = np.atleast_1d(axes)
    for ax, (str_toks, true, pred), title in zip(axes, windows, titles):
        str_toks, true, pred = str_toks[:max_tokens], true[:max_tokens], pred[:max_tokens]
        true_norm = (true / true.max().clamp(min=1e-6)).numpy()
        pred_norm = (pred / 9).numpy()
        grid = np.stack([true_norm, pred_norm])
        ax.imshow(grid, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        for j, tok in enumerate(str_toks):
            for i in range(2):
                ax.text(j, i, _clean_token(tok), ha="center", va="center", fontsize=7.5,
                        color="white" if grid[i, j] > 0.6 else "black", parse_math=False)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["true", "simulated"])
        ax.set_xticks([])
        ax.set_title(title, fontsize=9, loc="left")
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_score_dots(
    labels: list[str], series: dict[str, list[float]], null95: list[float] | None, title: str, ylabel: str,
    xlabel: str = "unit",
):
    """
    One column per unit (sorted by the first series, in decreasing order), one marker per series, plus an optional grey
    bar for the 95th percentile of a permutation null.
    """
    names = list(series)
    order = np.argsort(series[names[0]])[::-1]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    if null95 is not None:
        ax.bar(x, np.array(null95)[order], width=0.7, color="lightgray", label="95th percentile of shuffled-activation null")
    markers = ["o", "s", "^", "D", "v"]
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange"]
    for k, name in enumerate(names):
        ax.scatter(x, np.array(series[name])[order], marker=markers[k], color=colors[k], label=name, zorder=3,
                   s=40 if k == 0 else 25)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(np.array(labels)[order], rotation=90, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_matrix(matrix: t.Tensor, row_labels: list[str], col_labels: list[str], title: str, xlabel: str, ylabel: str,
                cbar_label: str, vmin: float | None = None, vmax: float | None = None, cmap: str = "viridis",
                fmt: str = "{:.2f}", figsize: tuple[float, float] = (8, 6.5)):
    """Annotated heatmap."""
    m = matrix.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    lo, hi = (np.nanmin(m) if vmin is None else vmin), (np.nanmax(m) if vmax is None else vmax)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if m[i, j] < lo + 0.6 * (hi - lo) else "black")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    plt.show()


def plot_unit_types(scores: dict[str, list[float]], title: str, ylabel: str):
    """Box plot plus jittered points, one box per kind of unit."""
    names = list(scores)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([scores[n] for n in names], widths=0.5, showfliers=False)
    rng = np.random.default_rng(0)
    for k, name in enumerate(names):
        vals = np.array(scores[name])
        ax.scatter(k + 1 + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14, alpha=0.7, zorder=3)
        ax.text(k + 1, max(vals.max(), 0) + 0.04, f"mean {vals.mean():.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_grouped_bars(groups: list[str], series: dict[str, list[float]], title: str, ylabel: str,
                      errors: dict[str, list[float]] | None = None, ylim: tuple[float, float] | None = None,
                      hline: float | None = None, hline_label: str | None = None):
    """Grouped bar chart: one group per entry of `groups`, one bar per series."""
    names = list(series)
    x = np.arange(len(groups))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, name in enumerate(names):
        vals = np.array(series[name])
        err = None if errors is None else np.array(errors[name])
        bars = ax.bar(x + (k - (len(names) - 1) / 2) * width, vals, width, yerr=err, capsize=3, label=name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    if hline is not None:
        ax.axhline(hline, color="black", ls="--", lw=0.8, label=hline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_lines(x: list, curves: dict[str, list[float]], title: str, xlabel: str, ylabel: str,
               errors: dict[str, list[float]] | None = None, xticklabels: list[str] | None = None,
               hline: float | None = None, hline_label: str | None = None):
    """Line plot with one line (and optional error band) per curve."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    styles = ["-o", "--s", ":^", "-.v", "-d"]
    for (name, ys), style in zip(curves.items(), styles):
        ys = np.array(ys)
        ax.plot(range(len(x)) if xticklabels else x, ys, style, label=name, markersize=5)
        if errors is not None and name in errors:
            err = np.array(errors[name])
            ax.fill_between(range(len(x)) if xticklabels else x, ys - err, ys + err, alpha=0.15)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=0.8, label=hline_label)
    if xticklabels:
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
