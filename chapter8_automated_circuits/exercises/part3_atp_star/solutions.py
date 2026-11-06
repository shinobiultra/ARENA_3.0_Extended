# %%


import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter8_automated_circuits"
section = "part3_atp_star"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part3_atp_star.tests as tests
import part3_atp_star.utils as utils

# We reuse the IOI task, metric and helpers from [8.2]
from part2_attribution_patching_eap.solutions import Task, make_ioi_task, spearman_correlation

MAIN = __name__ == "__main__"

# %%

if MAIN:
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.requires_grad_(False)
    
    ioi = make_ioi_task(model, 64, seed=0)
    ioi = Task(ioi.clean_tokens.to(device), ioi.corrupt_tokens.to(device), ioi.answers.to(device), ioi.metric)
    
    with t.no_grad():
        clean_metric = ioi.metric(model(ioi.clean_tokens), ioi.answers)
        corrupt_metric = ioi.metric(model(ioi.corrupt_tokens), ioi.answers)
    print(f"Clean logit diff:   {clean_metric.mean():.3f}")
    print(f"Corrupt logit diff: {corrupt_metric.mean():.3f}")

# %%

KINDS = ["q", "k", "v", "z"]


def node_hook_names(n_layers: int) -> list[str]:
    return [f"blocks.{layer}.attn.hook_{kind}" for layer in range(n_layers) for kind in KINDS + ["pattern"]]


@t.no_grad()
def cache_nodes(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> dict[str, Tensor]:
    names = set(node_hook_names(model.cfg.n_layers))
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)
    return {name: cache[name] for name in names}

# %%

if MAIN:
    clean_cache = cache_nodes(model, ioi.clean_tokens)
    corrupt_cache = cache_nodes(model, ioi.corrupt_tokens)
    print(clean_cache["blocks.0.attn.hook_q"].shape, clean_cache["blocks.0.attn.hook_pattern"].shape)

# %%

@t.no_grad()
def exact_node_effects(
    model: HookedTransformer,
    task: Task,
    clean_cache: dict[str, Tensor],
    corrupt_cache: dict[str, Tensor],
    per_prompt: bool = False,
) -> Tensor:
    """Exact noising effect of every (kind, layer, head) node, either per prompt or averaged over prompts."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    clean = task.metric(model(task.clean_tokens), task.answers)
    effects = t.zeros(len(KINDS), n_layers, n_heads, len(task), device=clean.device)
    for k, kind in enumerate(KINDS):
        for layer in range(n_layers):
            name = f"blocks.{layer}.attn.hook_{kind}"
            for head in range(n_heads):

                def patch(act: Tensor, hook, head=head, name=name) -> Tensor:
                    act[:, :, head] = corrupt_cache[name][:, :, head]
                    return act

                patched = task.metric(model.run_with_hooks(task.clean_tokens, fwd_hooks=[(name, patch)]), task.answers)
                effects[k, layer, head] = patched - clean
    return effects if per_prompt else effects.mean(-1)


if MAIN:
    tests.test_exact_node_effects(exact_node_effects, cache_nodes, model, ioi)

# %%

if MAIN:
    t0 = time.time()
    exact_per_prompt = exact_node_effects(model, ioi, clean_cache, corrupt_cache, per_prompt=True)
    exact = exact_per_prompt.mean(-1).cpu()
    print(f"Exact patching of 576 nodes took {time.time() - t0:.1f}s")
    
    names = utils.flat_node_names(model.cfg.n_layers, model.cfg.n_heads)
    top = exact.flatten().abs().argsort(descending=True)[:10]
    print("\nLargest exact effects:")
    for i in top.tolist():
        print(f"  {names[i]:>6}: {exact.flatten()[i]:+.3f}")

# %%

def node_gradients(model: HookedTransformer, task: Task, drop_layer: int | None = None) -> dict[str, Tensor]:
    """
    Gradient of the summed per-prompt metric (on clean prompts) with respect to every node activation and attention
    pattern. If drop_layer is given, the backward pass doesn't flow through that layer's attention/MLP outputs.
    """
    names = set(node_hook_names(model.cfg.n_layers))
    store = {}

    def keep(act: Tensor, hook) -> Tensor:
        act.requires_grad_(True)
        act.retain_grad()
        store[hook.name] = act
        return act

    hooks = [(lambda name: name in names, keep)]
    if drop_layer is not None:
        hooks += [
            (f"blocks.{drop_layer}.hook_attn_out", lambda act, hook: act.detach()),
            (f"blocks.{drop_layer}.hook_mlp_out", lambda act, hook: act.detach()),
        ]
    with t.enable_grad():
        metric = task.metric(model.run_with_hooks(task.clean_tokens, fwd_hooks=hooks), task.answers)
        metric.sum().backward()
    return {name: (act.grad if act.grad is not None else t.zeros_like(act)).detach() for name, act in store.items()}


if MAIN:
    tests.test_node_gradients(node_gradients, model, ioi)

# %%

def atp_estimates(
    clean_cache: dict[str, Tensor], corrupt_cache: dict[str, Tensor], grads: dict[str, Tensor]
) -> Float[Tensor, "kind layer head batch"]:
    n_layers = len([name for name in clean_cache if name.endswith("hook_q")])
    rows = []
    for kind in KINDS:
        per_layer = []
        for layer in range(n_layers):
            name = f"blocks.{layer}.attn.hook_{kind}"
            # (batch, seq, head, d_head) -> (head, batch)
            per_layer.append(((corrupt_cache[name] - clean_cache[name]) * grads[name]).sum((1, 3)).T)
        rows.append(t.stack(per_layer))
    return t.stack(rows)


if MAIN:
    tests.test_atp_estimates(atp_estimates, cache_nodes, node_gradients, model, ioi)

# %%

if MAIN:
    grads = node_gradients(model, ioi)
    atp_per_prompt = atp_estimates(clean_cache, corrupt_cache, grads)
    atp = atp_per_prompt.mean(-1).cpu()
    
    utils.plot_estimate_vs_exact(
        {"Attribution patching": atp}, exact, "AtP vs exact patching, 576 attention nodes, IOI (64 prompts)",
        highlight=["v4.11", "q10.7", "k10.7", "q0.5", "z0.10"],
    )

# %%

def recall_curve(scores: Tensor, exact: Tensor, threshold: float) -> Float[Tensor, " n_nodes"]:
    scores, exact = scores.flatten().cpu(), exact.flatten().cpu()
    important = exact.abs() > threshold
    order = scores.abs().argsort(descending=True, stable=True)
    return important[order].float().cumsum(0) / important.sum()


if MAIN:
    tests.test_recall_curve(recall_curve)

# %%

def random_recall_band(exact: Tensor, threshold: float, n_orders: int = 200, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    generator = t.Generator().manual_seed(seed)
    curves = t.stack([recall_curve(t.rand(exact.numel(), generator=generator), exact, threshold) for _ in range(n_orders)])
    return curves.mean(0), curves.quantile(0.05, dim=0), curves.quantile(0.95, dim=0)


def nodes_for_full_recall(scores: Tensor, exact: Tensor, threshold: float) -> int:
    """How many nodes you must verify (in score order) before every important node has been found."""
    curve = recall_curve(scores, exact, threshold)
    return int((curve < 1.0).sum().item()) + 1

# %%

if MAIN:
    THRESHOLD = 0.1
    n_important = int((exact.abs() > THRESHOLD).sum())
    print(f"{n_important} of 576 nodes are important (|exact effect| > {THRESHOLD})")
    print(f"AtP needs {nodes_for_full_recall(atp, exact, THRESHOLD)} verifications to find all of them")
    
    utils.plot_recall_curves(
        {"AtP": recall_curve(atp, exact, THRESHOLD)},
        n_important,
        "Finding the important nodes: AtP vs random order",
        random_band=random_recall_band(exact, THRESHOLD),
    )

# %%

if MAIN:
    important = (exact.flatten().abs() > THRESHOLD).nonzero().flatten()
    atp_rank = atp.flatten().abs().argsort(descending=True).argsort()
    worst = important[atp_rank[important].argsort(descending=True)][:8]
    rows = [
        {"node": names[i], "exact effect": exact.flatten()[i].item(), "AtP estimate": atp.flatten()[i].item(), "AtP rank": atp_rank[i].item() + 1}
        for i in worst.tolist()
    ]
    display(pd.DataFrame(rows).set_index("node").style.format({"exact effect": "{:+.3f}", "AtP estimate": "{:+.3f}"}))

# %%

def pattern_from_qk(
    q: Float[Tensor, "batch seq n_heads d_head"], k: Float[Tensor, "batch seq n_heads d_head"]
) -> Float[Tensor, "batch n_heads seq_q seq_k"]:
    scores = t.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
    seq = scores.shape[-1]
    mask = t.triu(t.ones(seq, seq, dtype=t.bool, device=scores.device), diagonal=1)
    return scores.masked_fill(mask, float("-inf")).softmax(dim=-1)


if MAIN:
    tests.test_pattern_from_qk(pattern_from_qk, model, ioi)

# %%

if MAIN:
    layer, head, prompt = 10, 7, 0
    q_clean = clean_cache[f"blocks.{layer}.attn.hook_q"][prompt : prompt + 1]
    k_clean = clean_cache[f"blocks.{layer}.attn.hook_k"][prompt : prompt + 1]
    k_corrupt = corrupt_cache[f"blocks.{layer}.attn.hook_k"][prompt : prompt + 1]
    
    k_patched = k_clean.clone()
    k_patched[:, :, head] = k_corrupt[:, :, head]
    clean_row = pattern_from_qk(q_clean, k_clean)[0, head, -1]
    patched_row = pattern_from_qk(q_clean, k_patched)[0, head, -1]
    
    # First-order prediction: A + J @ (change in scores), with J = diag(A) - A A^T
    scale = math.sqrt(model.cfg.d_head)
    delta_scores = (q_clean[0, -1, head] @ (k_patched[0, :, head] - k_clean[0, :, head]).T) / scale
    jacobian = t.diag(clean_row) - t.outer(clean_row, clean_row)
    linear_row = clean_row + jacobian @ delta_scores
    
    tokens = [model.to_string(tok) for tok in ioi.clean_tokens[prompt]]
    utils.plot_saturation(clean_row, patched_row, linear_row, tokens, f"Head {layer}.{head}, final position, key patched (prompt {prompt})")

# %%

def qk_fix_estimates(
    clean_cache: dict[str, Tensor], corrupt_cache: dict[str, Tensor], grads: dict[str, Tensor]
) -> Float[Tensor, "2 layer head batch"]:
    n_layers = len([name for name in clean_cache if name.endswith("hook_q")])
    out = []
    for which in ["q", "k"]:
        per_layer = []
        for layer in range(n_layers):
            q = clean_cache[f"blocks.{layer}.attn.hook_q"]
            k = clean_cache[f"blocks.{layer}.attn.hook_k"]
            if which == "q":
                q = corrupt_cache[f"blocks.{layer}.attn.hook_q"]
            else:
                k = corrupt_cache[f"blocks.{layer}.attn.hook_k"]
            name = f"blocks.{layer}.attn.hook_pattern"
            delta = pattern_from_qk(q, k) - clean_cache[name]
            # (batch, head, q, k) -> (head, batch)
            per_layer.append((delta * grads[name]).sum((-1, -2)).T)
        out.append(t.stack(per_layer))
    return t.stack(out)


if MAIN:
    tests.test_qk_fix_estimates(qk_fix_estimates, cache_nodes, node_gradients, model, ioi)

# %%

def atp_qk_fixed(clean_cache: dict[str, Tensor], corrupt_cache: dict[str, Tensor], grads: dict[str, Tensor]) -> Tensor:
    """Per-prompt AtP estimates for all nodes, with queries and keys replaced by their QK-fixed estimates."""
    estimates = atp_estimates(clean_cache, corrupt_cache, grads).clone()
    estimates[:2] = qk_fix_estimates(clean_cache, corrupt_cache, grads)
    return estimates

# %%

if MAIN:
    qk_per_prompt = atp_qk_fixed(clean_cache, corrupt_cache, grads)
    atp_qk = qk_per_prompt.mean(-1).cpu()
    
    utils.plot_estimate_vs_exact(
        {"AtP": atp, "AtP + QK fix": atp_qk}, exact, "Does the QK fix help?", highlight=["q10.7", "k10.7", "k11.1", "q0.5"]
    )
    for k, kind in enumerate(KINDS):
        r_atp = t.corrcoef(t.stack([atp[k].flatten(), exact[k].flatten()]))[0, 1]
        r_fix = t.corrcoef(t.stack([atp_qk[k].flatten(), exact[k].flatten()]))[0, 1]
        print(f"{utils.KIND_LABELS[kind]:>12}: Pearson r with exact  AtP {r_atp:.3f}  ->  AtP + QK fix {r_fix:.3f}")

# %%

if MAIN:
    print(f"Verifications needed to find all {n_important} important nodes:")
    print(f"  AtP:          {nodes_for_full_recall(atp, exact, THRESHOLD)}")
    print(f"  AtP + QK fix: {nodes_for_full_recall(atp_qk, exact, THRESHOLD)}")
    for name in ["k11.1", "q0.5"]:
        i = names.index(name)
        print(f"  {name}: exact {exact.flatten()[i]:+.3f}, AtP {atp.flatten()[i]:+.3f}, AtP + QK fix {atp_qk.flatten()[i]:+.3f}")

# %%

def toy_forward(x: Tensor, node_override: Tensor | None = None, drop_layer: int | None = None) -> Tensor:
    """The two-layer cancellation toy. Returns y."""
    r0 = x
    n = 2 * x if node_override is None else node_override
    r1 = r0 + (n.detach() if drop_layer == 0 else n)
    m = -t.relu(r1 - 2)
    return r1 + (m.detach() if drop_layer == 1 else m)


def toy_estimates(x_clean: Tensor, x_corrupt: Tensor) -> dict[str, float]:
    n_clean, n_corrupt = 2 * x_clean, 2 * x_corrupt
    exact = (toy_forward(x_clean, node_override=n_corrupt) - toy_forward(x_clean)).item()

    def atp_with(drop_layer: int | None) -> float:
        n = n_clean.clone().requires_grad_(True)
        y = toy_forward(x_clean, node_override=n, drop_layer=drop_layer)
        if not y.requires_grad:  # dropping the node's own layer cuts every path from n to y
            return 0.0
        (grad,) = t.autograd.grad(y, n)
        return ((n_corrupt - n_clean) * grad).item()

    n_layers = 2
    graddrop = sum(abs(atp_with(layer)) for layer in range(n_layers)) / (n_layers - 1)
    return {"exact": exact, "atp": atp_with(None), "graddrop": graddrop}


if MAIN:
    tests.test_toy(toy_forward, toy_estimates)
    print({key: value + 0.0 for key, value in toy_estimates(t.tensor(1.0), t.tensor(0.0)).items()})  # + 0.0 turns -0.0 into 0.0

# %%

if MAIN:
    tests.test_node_gradients(node_gradients, model, ioi, check_drop=True)

# %%

def graddrop_scores(
    model: HookedTransformer, task: Task, clean_cache: dict[str, Tensor], corrupt_cache: dict[str, Tensor]
) -> Float[Tensor, "kind layer head"]:
    n_layers = model.cfg.n_layers
    total = None
    for drop_layer in range(n_layers):
        grads = node_gradients(model, task, drop_layer=drop_layer)
        estimates = atp_qk_fixed(clean_cache, corrupt_cache, grads).abs()
        total = estimates if total is None else total + estimates
    return (total / (n_layers - 1)).mean(-1)


if MAIN:
    tests.test_graddrop_scores(graddrop_scores, cache_nodes, model, ioi)

# %%

if MAIN:
    t0 = time.time()
    atp_star = graddrop_scores(model, ioi, clean_cache, corrupt_cache).cpu()
    print(f"AtP* scores for 576 nodes took {time.time() - t0:.2f}s")
    
    utils.plot_recall_curves(
        {
            "AtP": recall_curve(atp, exact, THRESHOLD),
            "AtP + QK fix": recall_curve(atp_qk, exact, THRESHOLD),
            "AtP* (QK fix + GradDrop)": recall_curve(atp_star, exact, THRESHOLD),
        },
        n_important,
        "Finding the important nodes: does AtP* reduce false negatives?",
        random_band=random_recall_band(exact, THRESHOLD),
    )

# %%

if MAIN:
    # GradDrop without the QK fix, for the ablation
    total = None
    for drop_layer in range(model.cfg.n_layers):
        estimates = atp_estimates(clean_cache, corrupt_cache, node_gradients(model, ioi, drop_layer=drop_layer)).abs()
        total = estimates if total is None else total + estimates
    graddrop_no_qk = (total / (model.cfg.n_layers - 1)).mean(-1).cpu()
    
    variants = {
        "AtP": atp.abs(),
        "AtP + QK fix": atp_qk.abs(),
        "AtP + QK fix, mean of |per-prompt|\n(no GradDrop)": qk_per_prompt.abs().mean(-1).cpu(),
        "GradDrop without QK fix": graddrop_no_qk,
        "AtP* (QK fix + GradDrop)": atp_star,
    }
    thresholds = [0.05, 0.1, 0.2]
    table = {name: {th: nodes_for_full_recall(scores, exact, th) for th in thresholds} for name, scores in variants.items()}
    display(pd.DataFrame(table).T.rename(columns=lambda th: f"|exact| > {th} ({int((exact.abs() > th).sum())} nodes)"))
    utils.plot_ingredients(table, "Verifications needed to find every important node (lower is better)")

# %%

if MAIN:
    for name in ["v4.11", "z0.10", "k11.1"]:
        k, rest = KINDS.index(name[0]), name[1:]
        layer, head = map(int, rest.split("."))
        per_exact = exact_per_prompt[k, layer, head]
        per_atp = atp_per_prompt[k, layer, head]
        r = t.corrcoef(t.stack([per_exact, per_atp]))[0, 1].item()
        print(
            f"{name}: corr = {r:.2f}, exact < 0 on {(per_exact < 0).float().mean():.0%} of prompts, "
            f"mean exact = {per_exact.mean():+.3f}, mean AtP = {per_atp.mean():+.3f}, mean |AtP| = {per_atp.abs().mean():.3f}"
        )
        if name == "v4.11":
            utils.plot_per_prompt(per_exact, per_atp, name)

# %%

if MAIN:
    CORRUPTION = "s2"
    
    play = make_ioi_task(model, 64, seed=0, corruption=CORRUPTION)
    play = Task(play.clean_tokens.to(device), play.corrupt_tokens.to(device), play.answers.to(device), play.metric)
    play_clean, play_corrupt = cache_nodes(model, play.clean_tokens), cache_nodes(model, play.corrupt_tokens)
    play_exact = exact_node_effects(model, play, play_clean, play_corrupt).cpu()
    play_grads = node_gradients(model, play)
    play_atp = atp_estimates(play_clean, play_corrupt, play_grads).mean(-1).cpu()
    play_qk_pp = atp_qk_fixed(play_clean, play_corrupt, play_grads)
    play_star = graddrop_scores(model, play, play_clean, play_corrupt).cpu()
    
    print(f"Corruption '{CORRUPTION}': {int((play_exact.abs() > THRESHOLD).sum())} important nodes")
    for name, scores in {
        "AtP": play_atp,
        "AtP + QK fix": play_qk_pp.mean(-1).cpu(),
        "mean |per-prompt| (no GradDrop)": play_qk_pp.abs().mean(-1).cpu(),
        "AtP*": play_star,
    }.items():
        print(f"  {name:<32} verifications for full recall: {nodes_for_full_recall(scores, play_exact, THRESHOLD)}")

# %%

def neuron_atp(model: HookedTransformer, task: Task, per_prompt: bool = False) -> Tensor:
    """
    AtP estimate for every MLP neuron: shape (n_layers, d_mlp) averaged over prompts, or (n_layers, d_mlp, batch)
    if per_prompt is True.
    """
    names = [f"blocks.{layer}.mlp.hook_post" for layer in range(model.cfg.n_layers)]
    with t.no_grad():
        _, corrupt = model.run_with_cache(task.corrupt_tokens, names_filter=lambda name: name in names)
    store = {}

    def keep(act: Tensor, hook) -> Tensor:
        act.requires_grad_(True)
        act.retain_grad()
        store[hook.name] = act
        return act

    with t.enable_grad():
        metric = task.metric(model.run_with_hooks(task.clean_tokens, fwd_hooks=[(lambda n: n in names, keep)]), task.answers)
        metric.sum().backward()
    estimates = t.stack([((corrupt[name] - store[name].detach()) * store[name].grad).sum(1).T for name in names])
    return estimates if per_prompt else estimates.mean(-1)


if MAIN:
    tests.test_neuron_atp(neuron_atp, model, ioi)

# %%

@t.no_grad()
def group_effects(model: HookedTransformer, task: Task, groups: list[list[tuple[int, int]]]) -> Float[Tensor, " n_groups"]:
    names = [f"blocks.{layer}.mlp.hook_post" for layer in range(model.cfg.n_layers)]
    _, corrupt = model.run_with_cache(task.corrupt_tokens, names_filter=lambda name: name in names)
    clean = task.metric(model(task.clean_tokens), task.answers)
    effects = []
    for group in groups:
        by_layer: dict[int, list[int]] = {}
        for layer, idx in group:
            by_layer.setdefault(layer, []).append(idx)
        hooks = []
        for layer, idxs in by_layer.items():
            name = f"blocks.{layer}.mlp.hook_post"

            def patch(act: Tensor, hook, idxs=idxs, name=name) -> Tensor:
                act[:, :, idxs] = corrupt[name][:, :, idxs]
                return act

            hooks.append((name, patch))
        patched = task.metric(model.run_with_hooks(task.clean_tokens, fwd_hooks=hooks), task.answers)
        effects.append((patched - clean).mean())
    return t.stack(effects)


def exact_neuron_effects(model: HookedTransformer, task: Task, neurons: list[tuple[int, int]]) -> Float[Tensor, " n_neurons"]:
    return group_effects(model, task, [[neuron] for neuron in neurons])


if MAIN:
    tests.test_group_effects(group_effects, exact_neuron_effects, model, ioi)
    tests.test_exact_neuron_effects(exact_neuron_effects, model, ioi)

# %%

if MAIN:
    t0 = time.time()
    n_atp = neuron_atp(model, ioi).cpu()
    print(f"AtP for all {n_atp.numel():,} neurons took {time.time() - t0:.2f}s")
    
    d_mlp = model.cfg.d_mlp
    BUDGET = 100
    top_flat = n_atp.flatten().abs().argsort(descending=True)[:BUDGET]
    generator = t.Generator().manual_seed(0)
    random_flat = t.randperm(n_atp.numel(), generator=generator)[:300]
    selected = [(i // d_mlp, i % d_mlp) for i in t.cat([top_flat, random_flat]).tolist()]
    
    t0 = time.time()
    neuron_exact = exact_neuron_effects(model, ioi, selected).cpu()
    print(f"Exact patching of {len(selected)} neurons took {time.time() - t0:.1f}s")
    
    is_top = t.zeros(len(selected), dtype=t.bool)
    is_top[:BUDGET] = True
    sel_atp = n_atp.flatten()[t.cat([top_flat, random_flat])]
    utils.plot_neurons(sel_atp, neuron_exact, is_top, "MLP neurons: AtP vs exact patching")
    print(f"Mean |exact effect|: AtP's top {BUDGET}: {neuron_exact[is_top].abs().mean():.4f}, random: {neuron_exact[~is_top].abs().mean():.4f}")
    print(f"Pearson r (AtP vs exact): top {BUDGET}: {t.corrcoef(t.stack([sel_atp[is_top], neuron_exact[is_top]]))[0, 1]:.3f}, random: {t.corrcoef(t.stack([sel_atp[~is_top], neuron_exact[~is_top]]))[0, 1]:.3f}")
    print(f"Largest |exact effect| among the random neurons: {neuron_exact[~is_top].abs().max():.4f}")

# %%

if MAIN:
    verified = set(top_flat.tolist())
    remainder = t.tensor([i for i in range(n_atp.numel()) if i not in verified])
    GROUP_SIZE, N_GROUPS = 1000, 30
    generator = t.Generator().manual_seed(1)
    groups = []
    for _ in range(N_GROUPS):
        members = remainder[t.randperm(len(remainder), generator=generator)[:GROUP_SIZE]]
        groups.append([(i // d_mlp, i % d_mlp) for i in members.tolist()])
    remainder_effects = group_effects(model, ioi, groups).cpu()
    
    # For comparison: groups of the same size drawn from all neurons, including the verified top 100
    all_groups = []
    for _ in range(N_GROUPS):
        members = t.randperm(n_atp.numel(), generator=generator)[:GROUP_SIZE]
        all_groups.append([(i // d_mlp, i % d_mlp) for i in members.tolist()])
    all_effects = group_effects(model, ioi, all_groups).cpu()
    
    top_group = group_effects(model, ioi, [[(i // d_mlp, i % d_mlp) for i in top_flat.tolist()]]).cpu()
    print(f"Patching all {BUDGET} verified top neurons together: {top_group.item():+.3f}")
    print(f"Groups of {GROUP_SIZE} unverified neurons: mean {remainder_effects.mean():+.3f}, most extreme {remainder_effects[remainder_effects.abs().argmax()]:+.3f}")
    print(f"Groups of {GROUP_SIZE} neurons from all neurons: mean {all_effects.mean():+.3f}, most extreme {all_effects[all_effects.abs().argmax()]:+.3f}")
    
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(remainder_effects.numpy(), bins=15, alpha=0.7, label=f"groups of {GROUP_SIZE} unverified neurons")
    ax.hist(all_effects.numpy(), bins=15, alpha=0.7, label=f"groups of {GROUP_SIZE} neurons from all neurons")
    ax.axvline(top_group.item(), color="tab:red", lw=2, label=f"the {BUDGET} verified top neurons, together")
    ax.set_xlabel("change in logit diff when the whole group is patched")
    ax.set_ylabel("number of groups")
    ax.set_title("Group tests: is anything important left unverified?")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# %%

def bisect_group(
    group: list[tuple[int, int]], effect_fn: Callable[[list[tuple[int, int]]], float], threshold: float
) -> tuple[list[tuple[tuple[int, int], float]], int]:
    """
    Recursively halves `group`, descending only into halves with |effect| >= threshold, down to single neurons.
    Returns (found, n_calls): the single neurons with |effect| >= threshold together with their effects, and the
    number of calls made to effect_fn.
    """
    n_calls = 0

    def recurse(subgroup: list[tuple[int, int]]) -> list[tuple[tuple[int, int], float]]:
        nonlocal n_calls
        n_calls += 1
        effect = effect_fn(subgroup)
        if abs(effect) < threshold:
            return []
        if len(subgroup) == 1:
            return [(subgroup[0], effect)]
        mid = len(subgroup) // 2
        return recurse(subgroup[:mid]) + recurse(subgroup[mid:])

    found = recurse(group)
    return found, n_calls


if MAIN:
    tests.test_bisect_group(bisect_group)

# %%

if MAIN:
    worst = int(remainder_effects.abs().argmax())
    found, n_calls = bisect_group(groups[worst], lambda group: group_effects(model, ioi, [group]).item(), threshold=0.1)
    
    print(f"Group {worst} (effect {remainder_effects[worst]:+.3f}): bisection used {n_calls} group patches")
    abs_atp = n_atp.abs()
    for (layer, idx), effect in found:
        rank = int((abs_atp > abs_atp[layer, idx]).sum()) + 1
        print(f"  L{layer}N{idx}: exact effect {effect:+.3f}, AtP estimate {n_atp[layer, idx]:+.4f} (AtP rank {rank:,} of {n_atp.numel():,})")

# %%

if MAIN:
    (layer, idx), _ = found[0]
    name = f"blocks.{layer}.mlp.hook_post"
    with t.no_grad():
        clean_act = model.run_with_cache(ioi.clean_tokens, names_filter=name)[1][name][:, :, idx]
        corrupt_act = model.run_with_cache(ioi.corrupt_tokens, names_filter=name)[1][name][:, :, idx]
    
    change = (corrupt_act - clean_act).abs().mean(0)
    str_tokens = [model.to_string(tok) for tok in ioi.clean_tokens[0]]
    print("Largest mean |activation change| by position (tokens shown for prompt 0):")
    for pos in change.argsort(descending=True)[:3].tolist():
        print(f"  position {pos:>2} {str_tokens[pos]!r:<9} change {change[pos]:.3f}  (clean mean {clean_act[:, pos].mean():+.3f}, corrupt mean {corrupt_act[:, pos].mean():+.3f})")
    
    
    def patch_neuron(act: Tensor, hook) -> Tensor:
        act[:, :, idx] = corrupt_act
        return act
    
    
    with t.no_grad():
        clean_metric = ioi.metric(model(ioi.clean_tokens), ioi.answers)
        patched_metric = ioi.metric(model.run_with_hooks(ioi.clean_tokens, fwd_hooks=[(name, patch_neuron)]), ioi.answers)
    per_exact = (patched_metric - clean_metric).cpu()
    neuron_pp = neuron_atp(model, ioi, per_prompt=True).cpu()
    per_atp = neuron_pp[layer, idx]
    utils.plot_per_prompt(per_exact, per_atp, f"L{layer}N{idx}")
    
    abs_scores = neuron_pp.abs().mean(-1)
    rank_abs = int((abs_scores > abs_scores[layer, idx]).sum()) + 1
    print(
        f"corr = {t.corrcoef(t.stack([per_exact, per_atp]))[0, 1]:.2f}, exact < 0 on {(per_exact < 0).float().mean():.0%} of prompts, "
        f"mean exact = {per_exact.mean():+.3f}, mean AtP = {per_atp.mean():+.3f}, mean |AtP| = {per_atp.abs().mean():.3f}"
    )
    print(f"Rank by |mean AtP|: {int((abs_atp > abs_atp[layer, idx]).sum()) + 1:,}   rank by mean |per-prompt AtP|: {rank_abs:,}")

# %%
