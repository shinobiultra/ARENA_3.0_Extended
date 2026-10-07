# %%


import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import einops
import numpy as np
import pandas as pd
import torch as t
from IPython.display import display
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter8_automated_circuits"
section = "part2_attribution_patching_eap"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_attribution_patching_eap.tests as tests
import part2_attribution_patching_eap.utils as utils

MAIN = __name__ == "__main__"

# %%

if MAIN:
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.requires_grad_(False)
    
    print(f"{model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer, d_model={model.cfg.d_model}")

# %%

@dataclass
class Task:
    clean_tokens: Int[Tensor, "batch seq"]
    corrupt_tokens: Int[Tensor, "batch seq"]
    answers: Tensor  # task-specific, e.g. (batch, 2) token ids of the correct and incorrect answer
    metric: Callable[[Tensor, Tensor], Tensor]  # (logits, answers) -> per-prompt metric of shape (batch,)

    def __len__(self) -> int:
        return len(self.clean_tokens)

    def __getitem__(self, sl: slice) -> "Task":
        """A sub-task containing the prompts selected by the slice `sl`, e.g. task[:32]."""
        return Task(self.clean_tokens[sl], self.corrupt_tokens[sl], self.answers[sl], self.metric)

    def chunks(self, batch_size: int) -> Iterator["Task"]:
        """Yields consecutive sub-tasks of at most `batch_size` prompts."""
        for i in range(0, len(self), batch_size):
            yield self[i : i + batch_size]

# %%

def ioi_metric(logits: Float[Tensor, "batch seq d_vocab"], answers: Int[Tensor, "batch 2"]) -> Float[Tensor, " batch"]:
    """Returns logit(IO) - logit(S) at the final position, for each prompt."""
    final_logits = logits[:, -1]
    return final_logits.gather(-1, answers[:, :1])[:, 0] - final_logits.gather(-1, answers[:, 1:])[:, 0]


if MAIN:
    tests.test_ioi_metric(ioi_metric)

# %%

def make_ioi_task(model: HookedTransformer, n: int, seed: int, corruption: str = "abc") -> Task:
    clean, corrupt, io_names, s_names = utils.make_ioi_prompts(n, seed=seed, corruption=corruption)
    answers = t.stack(
        [model.to_tokens(io_names, prepend_bos=False)[:, 0], model.to_tokens(s_names, prepend_bos=False)[:, 0]], dim=1
    )
    return Task(model.to_tokens(clean), model.to_tokens(corrupt), answers, ioi_metric)


@t.no_grad()
def mean_metric(model: HookedTransformer, task: Task, tokens: Tensor) -> float:
    """The task's metric on `tokens` (either task.clean_tokens or task.corrupt_tokens), averaged over prompts."""
    return task.metric(model(tokens), task.answers).mean().item()

# %%

if MAIN:
    ioi = make_ioi_task(model, 128, seed=0)
    ioi_heldout = make_ioi_task(model, 128, seed=1)
    
    for i in range(2):
        print("clean:  ", model.to_string(ioi.clean_tokens[i, 1:]))
        print("corrupt:", model.to_string(ioi.corrupt_tokens[i, 1:]))
        print("answers:", [model.to_string(tok) for tok in ioi.answers[i]], "\n")
    
    with t.no_grad():
        clean_ld = ioi.metric(model(ioi.clean_tokens), ioi.answers)
        corrupt_ld = ioi.metric(model(ioi.corrupt_tokens), ioi.answers)
    print(f"Clean logit diff:   {clean_ld.mean():.3f}  (IO preferred in {(clean_ld > 0).float().mean():.0%} of prompts)")
    print(f"Corrupt logit diff: {corrupt_ld.mean():.3f}")

# %%

def get_cache_fwd_and_bwd(
    model: HookedTransformer, tokens: Int[Tensor, "batch seq"], metric_fn: Callable[[Tensor], Tensor], names: list[str]
) -> tuple[float, dict[str, Tensor], dict[str, Tensor]]:
    """
    Runs the model on `tokens`, backpropagates from metric_fn(logits), and returns (metric value, activations,
    gradients), where the dicts map each hook name in `names` to the activation / gradient at that hook.
    """
    activations, gradients = {}, {}

    def make_differentiable(embed: Tensor, hook) -> Tensor:
        return embed.requires_grad_(True)

    def forward_hook(act: Tensor, hook) -> None:
        activations[hook.name] = act.detach().clone()

    def backward_hook(grad: Tensor, hook) -> None:
        gradients[hook.name] = grad.detach().clone()

    in_names = lambda name: name in names
    with model.hooks(
        fwd_hooks=[("hook_embed", make_differentiable), (in_names, forward_hook)], bwd_hooks=[(in_names, backward_hook)]
    ):
        value = metric_fn(model(tokens))
        value.backward()
    return value.item(), activations, gradients


if MAIN:
    tests.test_get_cache_fwd_and_bwd(get_cache_fwd_and_bwd, model)

# %%

@t.no_grad()
def get_activations(model: HookedTransformer, tokens: Int[Tensor, "batch seq"], names: list[str]) -> dict[str, Tensor]:
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)
    return {name: cache[name] for name in names}

# %%

def get_cache_fwd_and_bwd_chunked(
    model: HookedTransformer, task: Task, names: list[str], batch_size: int = 32
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Clean-run activations and gradients of the task's mean metric at `names`, computed chunk by chunk."""
    acts, grads = {name: [] for name in names}, {name: [] for name in names}
    for chunk in task.chunks(batch_size):
        metric_fn = lambda logits: chunk.metric(logits, chunk.answers).sum() / len(task)
        _, chunk_acts, chunk_grads = get_cache_fwd_and_bwd(model, chunk.clean_tokens, metric_fn, names)
        for name in names:
            acts[name].append(chunk_acts[name])
            grads[name].append(chunk_grads[name])
    return {name: t.cat(v) for name, v in acts.items()}, {name: t.cat(v) for name, v in grads.items()}

# %%

def attribution_patch_heads(
    clean_acts: dict[str, Tensor], clean_grads: dict[str, Tensor], corrupt_acts: dict[str, Tensor], n_layers: int
) -> Float[Tensor, "layer head"]:
    """AP estimate of the change in the metric from corrupting each head's output (hook_z), at every position."""
    scores = []
    for layer in range(n_layers):
        name = f"blocks.{layer}.attn.hook_z"
        delta = corrupt_acts[name] - clean_acts[name]
        scores.append(einops.einsum(delta, clean_grads[name], "batch pos head d_head, batch pos head d_head -> head"))
    return t.stack(scores)


if MAIN:
    tests.test_attribution_patch_heads(attribution_patch_heads, model)

# %%

if MAIN:
    z_names = [f"blocks.{layer}.attn.hook_z" for layer in range(model.cfg.n_layers)]
    
    start = time.time()
    clean_acts, clean_grads = get_cache_fwd_and_bwd_chunked(model, ioi, z_names)
    corrupt_acts = get_activations(model, ioi.corrupt_tokens, z_names)
    ap_heads = attribution_patch_heads(clean_acts, clean_grads, corrupt_acts, model.cfg.n_layers).cpu()
    ap_time = time.time() - start
    print(f"Attribution patching: 144 head scores in {ap_time:.2f}s")

# %%

@t.no_grad()
def exact_patch_heads(model: HookedTransformer, task: Task, corrupt_acts: dict[str, Tensor]) -> Float[Tensor, "layer head"]:
    """Exact change in the mean metric from patching each head's corrupted hook_z into the clean run."""
    clean_value = task.metric(model(task.clean_tokens), task.answers).mean()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        name = f"blocks.{layer}.attn.hook_z"
        for head in range(model.cfg.n_heads):

            def patch_head(z: Tensor, hook) -> Tensor:
                z[:, :, head] = corrupt_acts[name][:, :, head]
                return z

            logits = model.run_with_hooks(task.clean_tokens, fwd_hooks=[(name, patch_head)])
            results[layer, head] = task.metric(logits, task.answers).mean() - clean_value
    return results


if MAIN:
    tests.test_exact_patch_heads(exact_patch_heads, model, make_ioi_task(model, 8, seed=5))

# %%

if MAIN:
    start = time.time()
    exact_heads = exact_patch_heads(model, ioi, corrupt_acts)
    exact_time = time.time() - start
    print(f"Exact patching: 144 head scores in {exact_time:.2f}s ({exact_time / ap_time:.0f}x slower than AP)")
    
    pearson = t.corrcoef(t.stack([exact_heads.flatten(), ap_heads.flatten()]))[0, 1].item()
    print(f"Pearson correlation between AP and exact: {pearson:.3f}\n")
    
    head_labels = [f"{layer}.{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    print("Largest disagreements:")
    for i in (ap_heads - exact_heads).abs().flatten().topk(4).indices.tolist():
        print(f"  head {head_labels[i]:>5}: exact {exact_heads.flatten()[i]:+.3f}, AP {ap_heads.flatten()[i]:+.3f}")
    utils.plot_head_heatmaps(
        {"Exact patching (144 forward passes)": exact_heads, "Attribution patching (1 fwd + 1 bwd)": ap_heads},
        title="IOI: change in logit difference when each head's output is corrupted",
    )
    utils.plot_scatter(
        exact_heads,
        {"attention heads": ap_heads},
        title=f"Attribution patching vs exact patching, per head (r = {pearson:.2f})",
        labels=head_labels,
        n_labels=8,
    )

# %%

def spearman_correlation(x: Tensor, y: Tensor) -> float:
    """Spearman rank correlation between two tensors of the same number of elements (ties ignored)."""
    rank_x = x.flatten().argsort().argsort().float()
    rank_y = y.flatten().argsort().argsort().float()
    return t.corrcoef(t.stack([rank_x, rank_y]))[0, 1].item()


if MAIN:
    tests.test_spearman_correlation(spearman_correlation)

# %%

def baseline_head_scores(
    clean_acts: dict[str, Tensor], clean_grads: dict[str, Tensor], corrupt_acts: dict[str, Tensor], n_layers: int, seed: int = 0
) -> dict[str, Float[Tensor, "layer head"]]:
    """Returns the activation-only, gradient-only and shuffled-pairing baselines described above."""
    names = [f"blocks.{layer}.attn.hook_z" for layer in range(n_layers)]
    delta = t.stack([corrupt_acts[name] - clean_acts[name] for name in names])  # (layer, batch, pos, head, d_head)
    grads = t.stack([clean_grads[name] for name in names])
    n_heads = delta.shape[3]

    delta_flat = einops.rearrange(delta, "layer batch pos head d -> batch pos (layer head) d")
    grads_flat = einops.rearrange(grads, "layer batch pos head d -> batch pos (layer head) d")
    perm = t.randperm(n_layers * n_heads, generator=t.Generator().manual_seed(seed)).to(delta.device)
    shuffled = einops.einsum(delta_flat, grads_flat[:, :, perm], "batch pos lh d, batch pos lh d -> lh")

    return {
        "activation change only": delta.norm(dim=-1).sum(dim=(1, 2)),
        "gradient only": grads.norm(dim=-1).sum(dim=(1, 2)),
        "AP, shuffled pairing": shuffled.reshape(n_layers, n_heads),
    }


if MAIN:
    tests.test_baseline_head_scores(baseline_head_scores)

# %%

if MAIN:
    baselines = baseline_head_scores(clean_acts, clean_grads, corrupt_acts, model.cfg.n_layers)
    rank_agreement = {"attribution patching": spearman_correlation(ap_heads.abs(), exact_heads.abs())}
    for name, scores in baselines.items():
        rank_agreement[name] = spearman_correlation(scores.cpu().abs(), exact_heads.abs())
    for name, rho in rank_agreement.items():
        print(f"{name:>24}: Spearman with |exact| = {rho:.3f}")
    print("\nSigned agreement (Pearson correlation with the signed exact effect):")
    for name, scores in [("attribution patching", ap_heads), ("AP, shuffled pairing", baselines["AP, shuffled pairing"].cpu())]:
        print(f"{name:>24}: {t.corrcoef(t.stack([scores.flatten(), exact_heads.flatten()]))[0, 1]:+.3f}")
    
    utils.plot_bars(
        rank_agreement,
        title="Which scores rank heads by the size of their exact patching effect?",
        ylabel="Spearman correlation with |exact effect|",
    )

# %%

def attribution_patch_resid(
    clean_acts: dict[str, Tensor], clean_grads: dict[str, Tensor], corrupt_acts: dict[str, Tensor], n_layers: int
) -> Float[Tensor, "layer pos"]:
    """AP estimate of the change in the metric from corrupting resid_pre at each (layer, position)."""
    scores = []
    for layer in range(n_layers):
        name = f"blocks.{layer}.hook_resid_pre"
        delta = corrupt_acts[name] - clean_acts[name]
        scores.append(einops.einsum(delta, clean_grads[name], "batch pos d_model, batch pos d_model -> pos"))
    return t.stack(scores)


if MAIN:
    tests.test_attribution_patch_resid(attribution_patch_resid)

# %%

@t.no_grad()
def exact_patch_resid(model: HookedTransformer, task: Task, corrupt_acts: dict[str, Tensor]) -> Float[Tensor, "layer pos"]:
    """Exact change in the mean metric from patching corrupted resid_pre into the clean run, per (layer, position)."""
    clean_value = task.metric(model(task.clean_tokens), task.answers).mean()
    seq_len = task.clean_tokens.shape[1]
    results = t.zeros(model.cfg.n_layers, seq_len)
    for layer in range(model.cfg.n_layers):
        name = f"blocks.{layer}.hook_resid_pre"
        for pos in range(seq_len):

            def patch_resid(resid: Tensor, hook) -> Tensor:
                resid[:, pos] = corrupt_acts[name][:, pos]
                return resid

            logits = model.run_with_hooks(task.clean_tokens, fwd_hooks=[(name, patch_resid)])
            results[layer, pos] = task.metric(logits, task.answers).mean() - clean_value
    return results


if MAIN:
    tests.test_exact_patch_resid(exact_patch_resid, model, make_ioi_task(model, 8, seed=5))

# %%

if MAIN:
    resid_names = [f"blocks.{layer}.hook_resid_pre" for layer in range(model.cfg.n_layers)]
    clean_resid, clean_resid_grads = get_cache_fwd_and_bwd_chunked(model, ioi, resid_names)
    corrupt_resid = get_activations(model, ioi.corrupt_tokens, resid_names)
    
    ap_resid = attribution_patch_resid(clean_resid, clean_resid_grads, corrupt_resid, model.cfg.n_layers).cpu()
    exact_resid = exact_patch_resid(model, ioi, corrupt_resid)
    
    print(f"Pearson:  {t.corrcoef(t.stack([ap_resid.flatten(), exact_resid.flatten()]))[0, 1]:.3f}")
    print(f"Spearman: {spearman_correlation(ap_resid, exact_resid):.3f}\n")
    print("Layer 0 at the name positions:")
    for pos in [2, 4, 10]:
        label = utils.IOI_POSITION_LABELS[pos]
        print(f"  {label:>6}: exact {exact_resid[0, pos]:+.2f}, AP {ap_resid[0, pos]:+.2f}")
    
    utils.plot_layer_pos_heatmaps(
        {"Exact patching (180 forward passes)": exact_resid, "Attribution patching": ap_resid},
        utils.IOI_POSITION_LABELS,
        title="IOI: change in logit difference when resid_pre is corrupted at one (layer, position)",
    )

# %%

def metric_along_path(
    model: HookedTransformer,
    task: Task,
    hook_name: str,
    index: tuple,
    alphas: list[float],
    clean_acts: dict[str, Tensor],
    corrupt_acts: dict[str, Tensor],
) -> tuple[list[float], list[float]]:
    """
    Returns (values, slopes): f(alpha) and f'(alpha) for each alpha, where f is the change in the mean metric when
    act[index] at `hook_name` is set to clean + alpha * (corrupt - clean) in a run on the clean prompts.
    """
    clean_value = mean_metric(model, task, task.clean_tokens)
    clean, corrupt = clean_acts[hook_name], corrupt_acts[hook_name]
    delta = corrupt[index] - clean[index]
    values, slopes = [], []
    for alpha in alphas:
        stored = {}

        def interpolate(act: Tensor, hook) -> Tensor:
            act = act.clone()
            act[index] = clean[index] + alpha * delta
            act.requires_grad_(True)
            stored["act"] = act
            return act

        logits = model.run_with_hooks(task.clean_tokens, fwd_hooks=[(hook_name, interpolate)])
        value = task.metric(logits, task.answers).mean()
        (grad,) = t.autograd.grad(value, stored["act"])
        values.append(value.item() - clean_value)
        slopes.append((grad[index] * delta).sum().item())
    return values, slopes


if MAIN:
    tests.test_metric_along_path(metric_along_path, model, make_ioi_task(model, 8, seed=5))

# %%

if MAIN:
    COMPONENTS = {
        "head 10.7 output": ("blocks.10.attn.hook_z", np.s_[:, :, 7]),
        "head 6.9 query": ("blocks.6.attn.hook_q", np.s_[:, :, 9]),
        "MLP0 output at S2": ("blocks.0.hook_mlp_out", np.s_[:, 10]),
        "resid_pre 0 at name 1": ("blocks.0.hook_resid_pre", np.s_[:, 2]),
    }
    ioi_small = ioi[:32]  # metric_along_path does a backward pass on all prompts at once, so we use a subset
    component_hooks = sorted({hook_name for hook_name, _ in COMPONENTS.values()})
    clean_component_acts = get_activations(model, ioi_small.clean_tokens, component_hooks)
    corrupt_component_acts = get_activations(model, ioi_small.corrupt_tokens, component_hooks)
    
    alphas = [i / 20 for i in range(21)]
    paths = {}
    for name, (hook_name, index) in COMPONENTS.items():
        values, slopes = metric_along_path(
            model, ioi_small, hook_name, index, alphas, clean_component_acts, corrupt_component_acts
        )
        paths[name] = {"alphas": alphas, "values": values, "ap": slopes[0], "exact": values[-1]}
        print(f"{name:>22}: exact {values[-1]:+.3f}   AP {slopes[0]:+.3f}")
    
    utils.plot_paths(paths, title="The metric along the straight line from clean (α=0) to corrupted (α=1) activation")

# %%

if MAIN:
    hook_name, index = COMPONENTS["head 6.9 query"]
    s1_pos = (ioi_small.clean_tokens[:, :10] == ioi_small.clean_tokens[:, 10:11]).float().argmax(dim=1)  # position of S1
    batch_idx = t.arange(len(ioi_small))
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        stored = {}
    
        def interpolate(q: Tensor, hook) -> Tensor:
            q[index] = clean_component_acts[hook_name][index] + alpha * (
                corrupt_component_acts[hook_name][index] - clean_component_acts[hook_name][index]
            )
            return q
    
        def store_pattern(pattern: Tensor, hook) -> None:
            stored["p"] = pattern[batch_idx, 9, 10, s1_pos + 1]  # head 9, query position S2, key position S1+1
    
        with t.no_grad():
            model.run_with_hooks(
                ioi_small.clean_tokens, fwd_hooks=[(hook_name, interpolate), ("blocks.6.attn.hook_pattern", store_pattern)]
            )
        print(f"alpha = {alpha:.1f}: head 6.9 attention from S2 to S1+1 = {stored['p'].mean():.3f}")

# %%

def integrated_gradients_component(
    model: HookedTransformer,
    task: Task,
    hook_name: str,
    index: tuple,
    n_steps: int,
    clean_acts: dict[str, Tensor],
    corrupt_acts: dict[str, Tensor],
) -> float:
    """IG estimate of the patching effect of act[index] at hook_name, with n_steps left-Riemann steps."""
    alphas = [k / n_steps for k in range(n_steps)]
    _, slopes = metric_along_path(model, task, hook_name, index, alphas, clean_acts, corrupt_acts)
    return sum(slopes) / n_steps


if MAIN:
    tests.test_integrated_gradients_component(integrated_gradients_component, metric_along_path, model, make_ioi_task(model, 8, seed=5))

# %%

if MAIN:
    step_counts = [1, 2, 4, 8, 16, 32]
    ig_errors = {}
    for name, (hook_name, index) in COMPONENTS.items():
        estimates = [
            integrated_gradients_component(model, ioi_small, hook_name, index, m, clean_component_acts, corrupt_component_acts)
            for m in step_counts
        ]
        ig_errors[name] = [abs(est - paths[name]["exact"]) for est in estimates]
        paths[name]["ig"] = estimates[step_counts.index(8)]
        print(f"{name:>22}: exact {paths[name]['exact']:+.3f}  IG: " + "  ".join(f"m={m}: {e:+.3f}" for m, e in zip(step_counts, estimates)))
    
    utils.plot_ig_convergence(step_counts, ig_errors, title="Integrated gradients converge to the exact patching effect")

# %%

if MAIN:
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)

# %%

N_LAYERS, N_HEADS = 12, 12  # GPT-2 Small

SENDER_NAMES = ["input"] + [
    name for layer in range(N_LAYERS) for name in [f"a{layer}.{head}" for head in range(N_HEADS)] + [f"m{layer}"]
]
RECEIVER_NAMES = [
    name
    for layer in range(N_LAYERS)
    for name in [f"a{layer}.{head}<{qkv}>" for qkv in "qkv" for head in range(N_HEADS)] + [f"m{layer}"]
] + ["logits"]
N_SENDERS, N_RECEIVERS = len(SENDER_NAMES), len(RECEIVER_NAMES)


def head_sender(layer: int, head: int) -> int:
    return 1 + layer * (N_HEADS + 1) + head


def mlp_sender(layer: int) -> int:
    return 1 + layer * (N_HEADS + 1) + N_HEADS


def head_receiver(layer: int, qkv: str, head: int) -> int:
    return layer * (3 * N_HEADS + 1) + "qkv".index(qkv) * N_HEADS + head


def mlp_receiver(layer: int) -> int:
    return layer * (3 * N_HEADS + 1) + 3 * N_HEADS


LOGITS_RECEIVER = N_LAYERS * (3 * N_HEADS + 1)

# %%

if MAIN:
    assert SENDER_NAMES[head_sender(9, 9)] == "a9.9" and SENDER_NAMES[mlp_sender(3)] == "m3"
    assert RECEIVER_NAMES[head_receiver(5, "k", 5)] == "a5.5<k>" and RECEIVER_NAMES[mlp_receiver(0)] == "m0"
    assert RECEIVER_NAMES[LOGITS_RECEIVER] == "logits"
    print(f"{N_SENDERS} senders, {N_RECEIVERS} receivers")

# %%

def build_edge_mask() -> Bool[Tensor, "sender receiver"]:
    """True where sender u writes to the residual stream before receiver v reads from it."""
    write_time = [0]
    for layer in range(N_LAYERS):
        write_time += [2 * layer + 1] * N_HEADS + [2 * layer + 2]
    read_time = []
    for layer in range(N_LAYERS):
        read_time += [2 * layer + 1] * (3 * N_HEADS) + [2 * layer + 2]
    read_time += [2 * N_LAYERS + 1]
    return t.tensor(write_time)[:, None] < t.tensor(read_time)[None, :]


if MAIN:
    tests.test_build_edge_mask(build_edge_mask, SENDER_NAMES, RECEIVER_NAMES)

EDGE_MASK = build_edge_mask()

# %%

@t.no_grad()
def sender_activations(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch pos sender d_model"]:
    """Outputs of every sender (input embedding, each head, each MLP) in the residual stream basis."""
    names = ["blocks.0.hook_resid_pre"]
    for layer in range(model.cfg.n_layers):
        names += [f"blocks.{layer}.attn.hook_z", f"blocks.{layer}.hook_mlp_out"]
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)

    outputs = [cache["blocks.0.hook_resid_pre"][:, :, None]]
    for layer in range(model.cfg.n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"]
        outputs.append(einops.einsum(z, model.W_O[layer], "batch pos head d_head, head d_head d_model -> batch pos head d_model"))
        outputs.append(cache[f"blocks.{layer}.hook_mlp_out"][:, :, None])
    return t.cat(outputs, dim=2)


if MAIN:
    tests.test_sender_activations(sender_activations, model)

# %%

def receiver_gradients(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    metric_fn: Callable[[Tensor], Tensor],
    input_embed: Float[Tensor, "batch seq d_model"] | None = None,
) -> Float[Tensor, "batch pos receiver d_model"]:
    """Gradient of metric_fn(logits) with respect to the input of every receiver."""
    batch, seq = tokens.shape
    grads = t.zeros(batch, seq, N_RECEIVERS, model.cfg.d_model, device=tokens.device)

    def store_grad(grad: Tensor, hook) -> None:
        layer, hook_type = hook.layer(), hook.name.split(".")[-1]
        if hook_type == "hook_resid_post":
            grads[:, :, LOGITS_RECEIVER] = grad
        elif hook_type == "hook_mlp_in":
            grads[:, :, mlp_receiver(layer)] = grad
        else:
            qkv = hook_type[len("hook_")]  # "q", "k" or "v"
            start = head_receiver(layer, qkv, 0)
            grads[:, :, start : start + N_HEADS] = grad

    def embed_hook(embed: Tensor, hook) -> Tensor:
        if input_embed is not None:
            embed = input_embed.clone()
        return embed.requires_grad_(True)

    names = [f"blocks.{layer}.hook_{qkv}_input" for layer in range(model.cfg.n_layers) for qkv in "qkv"]
    names += [f"blocks.{layer}.hook_mlp_in" for layer in range(model.cfg.n_layers)]
    names += [f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    with model.hooks(fwd_hooks=[("hook_embed", embed_hook)], bwd_hooks=[(lambda name: name in names, store_grad)]):
        metric_fn(model(tokens)).backward()
    return grads


if MAIN:
    tests.test_receiver_gradients(receiver_gradients, model)

# %%

def eap_scores(model: HookedTransformer, task: Task, batch_size: int = 16) -> Float[Tensor, "sender receiver"]:
    """EAP score of every edge: (corrupt - clean sender output) . (gradient at receiver input on the clean run)."""
    scores = t.zeros(N_SENDERS, N_RECEIVERS, device=model.cfg.device)
    for chunk in task.chunks(batch_size):
        delta = sender_activations(model, chunk.corrupt_tokens) - sender_activations(model, chunk.clean_tokens)
        metric_fn = lambda logits: chunk.metric(logits, chunk.answers).sum() / len(task)
        grads = receiver_gradients(model, chunk.clean_tokens, metric_fn)
        scores += einops.einsum(delta, grads, "batch pos sender d, batch pos receiver d -> sender receiver")
    return scores * EDGE_MASK.to(scores.device)


if MAIN:
    tests.test_eap_scores(eap_scores, sender_activations, receiver_gradients, model, make_ioi_task(model, 6, seed=5))

# %%

if MAIN:
    start = time.time()
    ioi_eap = eap_scores(model, ioi)
    print(f"EAP: {int(EDGE_MASK.sum())} edge scores for {len(ioi)} prompts in {time.time() - start:.1f}s\n")
    
    
    def top_edges_table(scores: Tensor, k: int = 15) -> pd.DataFrame:
        flat = scores.abs().masked_fill(~EDGE_MASK.to(scores.device), -1).flatten()
        rows = []
        for idx in flat.topk(k).indices.tolist():
            u, v = divmod(idx, N_RECEIVERS)
            rows.append({"edge": f"{SENDER_NAMES[u]} -> {RECEIVER_NAMES[v]}", "score": round(scores[u, v].item(), 3)})
        return pd.DataFrame(rows)
    
    
    display(top_edges_table(ioi_eap))

# %%

def receiver_hook(v: int) -> tuple[str, int | None]:
    """Returns (hook name, head index or None) for receiver v."""
    if v == LOGITS_RECEIVER:
        return f"blocks.{N_LAYERS - 1}.hook_resid_post", None
    layer, offset = divmod(v, 3 * N_HEADS + 1)
    if offset == 3 * N_HEADS:
        return f"blocks.{layer}.hook_mlp_in", None
    return f"blocks.{layer}.hook_{'qkv'[offset // N_HEADS]}_input", offset % N_HEADS


assert receiver_hook(head_receiver(5, "k", 5)) == ("blocks.5.hook_k_input", 5)

# %%

@t.no_grad()
def patch_single_edges(
    model: HookedTransformer, task: Task, edges: list[tuple[int, int]], batch_size: int = 32
) -> Float[Tensor, " n_edges"]:
    """Exact change in the mean metric from corrupting each edge (u, v) on its own, in the clean run."""
    patched_sums = t.zeros(len(edges))
    clean_sum = 0.0
    for chunk in task.chunks(batch_size):
        delta = sender_activations(model, chunk.corrupt_tokens) - sender_activations(model, chunk.clean_tokens)
        clean_sum += chunk.metric(model(chunk.clean_tokens), chunk.answers).sum().item()
        for i, (u, v) in enumerate(edges):
            hook_name, head = receiver_hook(v)

            def add_delta(x: Tensor, hook) -> Tensor:
                if head is None:
                    return x + delta[:, :, u]
                x = x.clone()
                x[:, :, head] += delta[:, :, u]
                return x

            logits = model.run_with_hooks(chunk.clean_tokens, fwd_hooks=[(hook_name, add_delta)])
            patched_sums[i] += chunk.metric(logits, chunk.answers).sum().item()
    return (patched_sums - clean_sum) / len(task)


if MAIN:
    tests.test_patch_single_edges(patch_single_edges, sender_activations, model, make_ioi_task(model, 6, seed=5))

# %%

if MAIN:
    valid_edges = EDGE_MASK.nonzero().tolist()  # list of [u, v], in a fixed order
    flat_eap = ioi_eap[EDGE_MASK.to(ioi_eap.device)].cpu()  # EAP score of each valid edge, same order
    
    top_idx = flat_eap.abs().topk(200).indices
    random_idx = t.randperm(len(valid_edges), generator=t.Generator().manual_seed(0))[:200]
    check_idx = t.cat([top_idx, random_idx])
    
    start = time.time()
    check_exact = patch_single_edges(model, ioi, [tuple(valid_edges[i]) for i in check_idx.tolist()])
    print(f"Exact patching of {len(check_idx)} edges took {time.time() - start:.0f}s")
    
    exact_top, exact_random = check_exact[:200], check_exact[200:]
    eap_top, eap_random = flat_eap[top_idx], flat_eap[random_idx]
    for name, ex, ap in [("top-200 by |EAP|", exact_top, eap_top), ("200 random edges", exact_random, eap_random)]:
        pearson = t.corrcoef(t.stack([ex, ap]))[0, 1].item()
        print(f"{name:>17}: Pearson {pearson:.3f}, Spearman {spearman_correlation(ex, ap):.3f}, "
              f"mean |exact effect| {ex.abs().mean():.4f}")
    
    edge_labels = [f"{SENDER_NAMES[valid_edges[i][0]]}->{RECEIVER_NAMES[valid_edges[i][1]]}" for i in top_idx.tolist()]
    print("\nThe five top-200 edges where EAP is furthest from the exact effect:")
    for j in (eap_top - exact_top).abs().topk(5).indices.tolist():
        print(f"  {edge_labels[j]:>18}: exact {exact_top[j]:+.3f}, EAP {eap_top[j]:+.3f}")
    
    utils.plot_scatter(
        exact_top,
        {"top-200 edges by |EAP|": eap_top, "200 random edges (control)": (exact_random, eap_random)},
        title="EAP vs exact single-edge patching (IOI)",
        xlabel="exact effect of corrupting the edge",
        ylabel="EAP estimate",
        labels=edge_labels,
        n_labels=6,
    )

# %%

IOI_HEAD_ROLES = {
    "a9.9": "name mover", "a9.6": "name mover", "a10.0": "name mover",
    "a10.7": "neg. name mover", "a11.10": "neg. name mover",
    "a7.3": "S-inhibition", "a7.9": "S-inhibition", "a8.6": "S-inhibition", "a8.10": "S-inhibition",
    "a5.5": "induction", "a6.9": "induction", "a5.8": "induction", "a5.9": "induction",
    "a0.1": "duplicate token", "a0.10": "duplicate token", "a3.0": "duplicate token",
    "a2.2": "previous token", "a4.11": "previous token",
}

# %%

if MAIN:
    utils.plot_circuit_graph(
        ioi_eap, EDGE_MASK, SENDER_NAMES, RECEIVER_NAMES, top_n=50,
        title="IOI: the 50 edges with the largest |EAP score|", highlight=IOI_HEAD_ROLES,
    )

# %%

@t.no_grad()
def run_with_circuit(
    model: HookedTransformer, task: Task, in_circuit: Bool[Tensor, "sender receiver"], batch_size: int = 32
) -> float:
    """Mean metric over the task when every edge outside `in_circuit` carries its corrupted value."""
    not_in_circuit = (EDGE_MASK.to(in_circuit.device) & ~in_circuit).float().to(model.cfg.device)
    total = 0.0
    for chunk in task.chunks(batch_size):
        corrupt_out = sender_activations(model, chunk.corrupt_tokens)
        diff = t.zeros_like(corrupt_out)

        def store_input(x: Tensor, hook) -> None:
            diff[:, :, 0] = x - corrupt_out[:, :, 0]

        def store_heads(z: Tensor, hook) -> None:
            layer, start = hook.layer(), head_sender(hook.layer(), 0)
            out = einops.einsum(z, model.W_O[layer], "batch pos head d_head, head d_head d_model -> batch pos head d_model")
            diff[:, :, start : start + N_HEADS] = out - corrupt_out[:, :, start : start + N_HEADS]

        def store_mlp(x: Tensor, hook) -> None:
            u = mlp_sender(hook.layer())
            diff[:, :, u] = x - corrupt_out[:, :, u]

        def patch_head_inputs(x: Tensor, hook) -> Tensor:
            qkv = hook.name.split(".")[-1][len("hook_")]
            start = head_receiver(hook.layer(), qkv, 0)
            mask = not_in_circuit[:, start : start + N_HEADS]
            return x - einops.einsum(diff, mask, "batch pos sender d, sender head -> batch pos head d")

        def patch_single_input(x: Tensor, hook) -> Tensor:
            v = LOGITS_RECEIVER if hook.name.endswith("resid_post") else mlp_receiver(hook.layer())
            return x - einops.einsum(diff, not_in_circuit[:, v], "batch pos sender d, sender -> batch pos d")

        hooks = [("blocks.0.hook_resid_pre", store_input)]
        for layer in range(model.cfg.n_layers):
            hooks += [(f"blocks.{layer}.hook_{qkv}_input", patch_head_inputs) for qkv in "qkv"]
            hooks += [
                (f"blocks.{layer}.attn.hook_z", store_heads),
                (f"blocks.{layer}.hook_mlp_in", patch_single_input),
                (f"blocks.{layer}.hook_mlp_out", store_mlp),
            ]
        hooks += [(f"blocks.{model.cfg.n_layers - 1}.hook_resid_post", patch_single_input)]
        logits = model.run_with_hooks(chunk.clean_tokens, fwd_hooks=hooks)
        total += chunk.metric(logits, chunk.answers).sum().item()
    return total / len(task)


if MAIN:
    tests.test_run_with_circuit(run_with_circuit, patch_single_edges, model, make_ioi_task(model, 6, seed=5))

# %%

def top_k_circuit(scores: Float[Tensor, "sender receiver"], k: int) -> Bool[Tensor, "sender receiver"]:
    """The k valid edges with the largest |score|."""
    flat = scores.abs().masked_fill(~EDGE_MASK.to(scores.device), -1.0).flatten()
    circuit = t.zeros_like(flat, dtype=t.bool)
    circuit[flat.topk(k).indices] = True
    return circuit.reshape(scores.shape)

# %%

def faithfulness_curve(
    model: HookedTransformer, task: Task, scores: Float[Tensor, "sender receiver"], ks: list[int], batch_size: int = 32
) -> list[float]:
    """Normalised faithfulness of the top-k circuit (by |score|), for each k in ks."""
    clean_value = mean_metric(model, task, task.clean_tokens)
    corrupt_value = mean_metric(model, task, task.corrupt_tokens)
    return [
        (run_with_circuit(model, task, top_k_circuit(scores, k), batch_size) - corrupt_value) / (clean_value - corrupt_value)
        for k in ks
    ]


if MAIN:
    tests.test_faithfulness_curve(faithfulness_curve, model, make_ioi_task(model, 6, seed=5))

# %%

if MAIN:
    KS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    
    def random_faithfulness(model: HookedTransformer, task: Task, ks: list[int], n_seeds: int = 5) -> list[list[float]]:
        """Faithfulness curves of random edge sets (random scores), one curve per seed."""
        generator = t.Generator().manual_seed(0)
        return [faithfulness_curve(model, task, t.rand(N_SENDERS, N_RECEIVERS, generator=generator), ks) for _ in range(n_seeds)]
    
    
    ioi_random_curves = random_faithfulness(model, ioi_heldout, KS)
    ioi_eap_curve = faithfulness_curve(model, ioi_heldout, ioi_eap, KS)
    print(f"{'edges':>8} {'EAP':>7} {'random (mean)':>14}")
    for k, f_eap, f_rand in zip(KS, ioi_eap_curve, np.mean(ioi_random_curves, axis=0)):
        print(f"{k:>8} {f_eap:>7.2f} {f_rand:>14.2f}")

# %%

def eap_ig_scores(
    model: HookedTransformer, task: Task, n_steps: int = 5, batch_size: int = 16
) -> Float[Tensor, "sender receiver"]:
    """EAP-IG score of every edge, integrating receiver gradients along the clean -> corrupt input-embedding path."""
    scores = t.zeros(N_SENDERS, N_RECEIVERS, device=model.cfg.device)
    for chunk in task.chunks(batch_size):
        delta = sender_activations(model, chunk.corrupt_tokens) - sender_activations(model, chunk.clean_tokens)
        with t.no_grad():
            clean_embed, corrupt_embed = model.embed(chunk.clean_tokens), model.embed(chunk.corrupt_tokens)
        metric_fn = lambda logits: chunk.metric(logits, chunk.answers).sum() / len(task)
        for k in range(n_steps):
            alpha = k / n_steps
            embed = clean_embed + alpha * (corrupt_embed - clean_embed)
            grads = receiver_gradients(model, chunk.clean_tokens, metric_fn, input_embed=embed)
            scores += einops.einsum(delta, grads, "batch pos sender d, batch pos receiver d -> sender receiver") / n_steps
    return scores * EDGE_MASK.to(scores.device)


if MAIN:
    tests.test_eap_ig_scores(eap_ig_scores, eap_scores, model, make_ioi_task(model, 6, seed=5))

# %%

if MAIN:
    start = time.time()
    ioi_eap_ig = eap_ig_scores(model, ioi, n_steps=5)
    print(f"EAP-IG (5 steps): {time.time() - start:.1f}s\n")
    
    clean_value, corrupt_value = mean_metric(model, ioi, ioi.clean_tokens), mean_metric(model, ioi, ioi.corrupt_tokens)
    print("Completeness check: sum of scores of edges leaving the input embedding")
    print(f"  M(corrupt) - M(clean) = {corrupt_value - clean_value:+.3f}")
    print(f"  EAP                   = {ioi_eap[0].sum():+.3f}")
    for n_steps in [5, 20]:
        scores = ioi_eap_ig if n_steps == 5 else eap_ig_scores(model, ioi, n_steps=n_steps)
        print(f"  EAP-IG ({n_steps:>2} steps)     = {scores[0].sum():+.3f}")

# %%

if MAIN:
    ioi_eap_ig_curve = faithfulness_curve(model, ioi_heldout, ioi_eap_ig, KS)
    print(f"{'edges':>8} {'EAP':>7} {'EAP-IG':>7} {'random':>7}")
    for k, f_eap, f_ig, f_rand in zip(KS, ioi_eap_curve, ioi_eap_ig_curve, np.mean(ioi_random_curves, axis=0)):
        print(f"{k:>8} {f_eap:>7.2f} {f_ig:>7.2f} {f_rand:>7.2f}")
    
    utils.plot_faithfulness(
        {
            "IOI (held-out prompts)": {
                "ks": KS,
                "curves": {"EAP": ioi_eap_curve, "EAP-IG (5 steps)": ioi_eap_ig_curve},
                "random": ioi_random_curves,
                "n_edges": int(EDGE_MASK.sum()),
            }
        },
        title="How much of the IOI behaviour does the top-k circuit recover?",
    )

# %%

if MAIN:
    eap_circuit_1000, ig_circuit_1000 = top_k_circuit(ioi_eap, 1000), top_k_circuit(ioi_eap_ig, 1000)
    ig_only = ig_circuit_1000 & ~eap_circuit_1000
    print(f"Overlap of the two 1000-edge circuits: {(eap_circuit_1000 & ig_circuit_1000).sum().item()} edges\n")
    
    ig_only_scores = ioi_eap_ig.abs().masked_fill(~ig_only, -1).flatten()
    missing_edges = [divmod(idx, N_RECEIVERS) for idx in ig_only_scores.topk(10).indices.tolist()]
    eap_ranks = ioi_eap.abs().masked_fill(~EDGE_MASK.to(ioi_eap.device), -1).flatten().argsort(descending=True).argsort()
    missing_exact = patch_single_edges(model, ioi, missing_edges)
    
    display(pd.DataFrame([
        {
            "edge": f"{SENDER_NAMES[u]} -> {RECEIVER_NAMES[v]}",
            "EAP-IG score": round(ioi_eap_ig[u, v].item(), 3),
            "EAP score": round(ioi_eap[u, v].item(), 3),
            "EAP rank": eap_ranks[u * N_RECEIVERS + v].item() + 1,
            "exact single-edge effect": round(exact, 3),
        }
        for (u, v), exact in zip(missing_edges, missing_exact.tolist())
    ]))

# %%

def joint_vs_individual(
    model: HookedTransformer, task: Task, edges: list[tuple[int, int]], batch_size: int = 32
) -> tuple[Float[Tensor, " n_edges"], float]:
    """Returns (individual effects of corrupting each edge alone, effect of corrupting all of them together)."""
    individual = patch_single_edges(model, task, edges, batch_size)
    circuit = EDGE_MASK.clone()
    for u, v in edges:
        circuit[u, v] = False
    joint = run_with_circuit(model, task, circuit, batch_size) - mean_metric(model, task, task.clean_tokens)
    return individual, joint


if MAIN:
    tests.test_joint_vs_individual(joint_vs_individual, model, make_ioi_task(model, 6, seed=5))

# %%

if MAIN:
    edge_groups = {
        "input -> heads 0.1, 0.10 (q, k, v)": [(0, head_receiver(0, qkv, h)) for h in [1, 10] for qkv in "qkv"],
        "input -> all layer-0 heads (q, k, v)": [(0, head_receiver(0, qkv, h)) for h in range(N_HEADS) for qkv in "qkv"],
        "36 random edges": [tuple(valid_edges[i]) for i in t.randperm(len(valid_edges), generator=t.Generator().manual_seed(1))[:36].tolist()],
    }
    for name, edges in edge_groups.items():
        individual, joint = joint_vs_individual(model, ioi, edges)
        print(f"{name:>38}: sum of individual effects {individual.sum():+.3f}, joint effect {joint:+.3f}")

# %%

if MAIN:
    heldout_clean = mean_metric(model, ioi_heldout, ioi_heldout.clean_tokens)
    heldout_corrupt = mean_metric(model, ioi_heldout, ioi_heldout.corrupt_tokens)
    
    print(f"{'edges':>8} {'EAP':>6} {'EAP + 10 edges':>15} {'EAP-IG':>7}")
    for k in [1000, 2000, 5000]:
        circuit = top_k_circuit(ioi_eap, k)
        for u, v in missing_edges:
            circuit[u, v] = True
        rescued = (run_with_circuit(model, ioi_heldout, circuit) - heldout_corrupt) / (heldout_clean - heldout_corrupt)
        print(f"{k:>8} {ioi_eap_curve[KS.index(k)]:>6.2f} {rescued:>15.2f} {ioi_eap_ig_curve[KS.index(k)]:>7.2f}")

# %%

def make_greater_than_metric(model: HookedTransformer) -> Callable[[Tensor, Tensor], Tensor]:
    year_tokens = t.tensor([model.to_single_token(f"{year:02d}") for year in range(100)], device=model.cfg.device)

    def greater_than_metric(logits: Float[Tensor, "batch seq d_vocab"], start_years: Int[Tensor, " batch"]) -> Float[Tensor, " batch"]:
        probs = logits[:, -1].softmax(dim=-1)[:, year_tokens]  # (batch, 100)
        greater = t.arange(100, device=logits.device)[None, :] > start_years[:, None]
        return (probs * greater).sum(-1) - (probs * ~greater).sum(-1)

    return greater_than_metric


def make_greater_than_task(model: HookedTransformer, n: int, seed: int) -> Task:
    clean, corrupt, years = utils.make_greater_than_prompts(model, n, seed=seed)
    return Task(
        model.to_tokens(clean), model.to_tokens(corrupt), t.tensor(years, device=model.cfg.device), make_greater_than_metric(model)
    )

# %%

if MAIN:
    gt, gt_heldout = make_greater_than_task(model, 128, seed=0), make_greater_than_task(model, 128, seed=1)
    print(model.to_string(gt.clean_tokens[0, 1:]), "|", model.to_string(gt.corrupt_tokens[0, 1:]))
    print(f"Held-out: clean metric {mean_metric(model, gt_heldout, gt_heldout.clean_tokens):.3f}, "
          f"corrupt metric {mean_metric(model, gt_heldout, gt_heldout.corrupt_tokens):.3f}\n")
    
    gt_eap, gt_eap_ig = eap_scores(model, gt), eap_ig_scores(model, gt, n_steps=5)
    gt_curves = {"EAP": faithfulness_curve(model, gt_heldout, gt_eap, KS), "EAP-IG (5 steps)": faithfulness_curve(model, gt_heldout, gt_eap_ig, KS)}
    gt_random_curves = random_faithfulness(model, gt_heldout, KS)
    print(f"{'edges':>8} {'EAP':>7} {'EAP-IG':>7} {'random':>7}")
    for k, f_eap, f_ig, f_rand in zip(KS, gt_curves["EAP"], gt_curves["EAP-IG (5 steps)"], np.mean(gt_random_curves, axis=0)):
        print(f"{k:>8} {f_eap:>7.2f} {f_ig:>7.2f} {f_rand:>7.2f}")
    
    utils.plot_faithfulness(
        {
            "IOI (held-out prompts)": {"ks": KS, "curves": {"EAP": ioi_eap_curve, "EAP-IG (5 steps)": ioi_eap_ig_curve}, "random": ioi_random_curves, "n_edges": int(EDGE_MASK.sum())},
            "Greater-than (held-out prompts)": {"ks": KS, "curves": gt_curves, "random": gt_random_curves, "n_edges": int(EDGE_MASK.sum()), "legend_loc": "lower right"},
        },
        title="Faithfulness of top-k circuits found by EAP and EAP-IG, vs random edge sets of the same size",
    )

# %%

if MAIN:
    utils.plot_circuit_graph(
        ioi_eap_ig, EDGE_MASK, SENDER_NAMES, RECEIVER_NAMES, top_n=50,
        title="IOI: the 50 edges with the largest |EAP-IG score|", highlight=IOI_HEAD_ROLES,
    )

# %%

if MAIN:
    PLAY_CORRUPTION = "s2"
    PLAY_STEPS = 5
    
    play_task = make_ioi_task(model, 128, seed=0, corruption=PLAY_CORRUPTION)
    play_heldout = make_ioi_task(model, 128, seed=1, corruption=PLAY_CORRUPTION)
    print(f"Held-out: clean {mean_metric(model, play_heldout, play_heldout.clean_tokens):.3f}, "
          f"corrupt {mean_metric(model, play_heldout, play_heldout.corrupt_tokens):.3f}")
    
    play_curves = {
        "EAP": faithfulness_curve(model, play_heldout, eap_scores(model, play_task), KS),
        f"EAP-IG ({PLAY_STEPS} steps)": faithfulness_curve(model, play_heldout, eap_ig_scores(model, play_task, n_steps=PLAY_STEPS), KS),
    }
    play_random = random_faithfulness(model, play_heldout, KS, n_seeds=2)
    for name, curve in play_curves.items():
        print(f"{name:>16}: " + "  ".join(f"{k}: {f:.2f}" for k, f in zip(KS, curve)))
    
    utils.plot_faithfulness(
        {f"IOI, corruption = {PLAY_CORRUPTION!r}": {"ks": KS, "curves": play_curves, "random": play_random, "n_edges": int(EDGE_MASK.sum())}},
        title="Try it yourself: faithfulness with a different corruption",
    )

# %%
