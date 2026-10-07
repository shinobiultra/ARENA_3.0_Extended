# %%


import random
import sys
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch as t
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.utils import get_act_name

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter8_automated_circuits"
section = "part1_activation_patching_refresher"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_activation_patching_refresher.tests as tests
import part1_activation_patching_refresher.utils as utils

MAIN = __name__ == "__main__"

# %%

if MAIN:
    t.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    N_LAYERS, N_HEADS = model.cfg.n_layers, model.cfg.n_heads
    print(f"{N_LAYERS} layers, {N_HEADS} heads per layer, d_head={model.cfg.d_head}")

# %%

OPENERS = ["When", "After", "While", "Once"]
ACTIONS = ["went to", "were at", "arrived at", "got to"]
VERBS = ["gave", "handed", "passed", "sent"]
TRAIN_TEMPLATES = [
    f"{opener} {{A}} and {{B}} {action} the {{PLACE}}, {{C}} {verb} a {{OBJECT}} to"
    for opener in OPENERS
    for action in ACTIONS
    for verb in VERBS
]


@dataclass
class IOIExample:
    template: str  # e.g. "When {A} and {B} went to the store, {C} gave a drink to" (only the name slots are left)
    io: str  # indirect object, e.g. "Mary"
    s: str  # subject, e.g. "John"
    c: str  # a third name, used by the ABC corruption
    xyz: tuple[str, str, str]  # three more names, used by the XYZ corruption


def sample_ioi_examples(templates: list[str], n: int, seed: int = 0) -> list[IOIExample]:
    """Samples n IOI examples: a random template, place, object and six distinct names for each."""
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        io, s, c, x, y, z = rng.sample(utils.NAMES, 6)
        template = rng.choice(templates).format(
            A="{A}", B="{B}", C="{C}", PLACE=rng.choice(utils.PLACES), OBJECT=rng.choice(utils.OBJECTS)
        )
        examples.append(IOIExample(template, io, s, c, (x, y, z)))
    return examples


def resample_fillers(examples: list[IOIExample], templates: list[str], seed: int = 1) -> list[IOIExample]:
    """Same names as `examples`, but a freshly sampled template, place and object for each one."""
    fresh = sample_ioi_examples(templates, len(examples), seed=seed)
    return [replace(ex, template=new.template) for ex, new in zip(examples, fresh)]

# %%

def ioi_prompt(example: IOIExample, scheme: str = "ABB") -> str:
    """Fills the name slots of example.template according to the scheme ("ABB", "ABA", "ABC" or "XYZ")."""
    names = {
        "ABB": (example.io, example.s, example.s),
        "ABA": (example.io, example.s, example.io),
        "ABC": (example.io, example.s, example.c),
        "XYZ": example.xyz,
    }[scheme]
    return example.template.format(A=names[0], B=names[1], C=names[2])


if MAIN:
    tests.test_ioi_prompt(ioi_prompt, IOIExample)

# %%

if MAIN:
    N_PROMPTS = 100
    examples = sample_ioi_examples(TRAIN_TEMPLATES, N_PROMPTS, seed=0)
    placebo_examples = resample_fillers(examples, TRAIN_TEMPLATES, seed=1)
    
    for i in range(2):
        for scheme in ["ABB", "ABA", "ABC", "XYZ"]:
            print(f"{scheme:<8}{ioi_prompt(examples[i], scheme)}")
        print(f"{'placebo':<8}{ioi_prompt(placebo_examples[i], 'ABB')}\n")

# %%

def logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"], answer_tokens: Int[Tensor, "batch 2"]
) -> Float[Tensor, " batch"]:
    """Returns logit(correct) - logit(incorrect) at the final position, for each prompt."""
    final_logits = logits[:, -1]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens.to(final_logits.device))
    return answer_logits[:, 0] - answer_logits[:, 1]


if MAIN:
    tests.test_logit_diff(logit_diff)

# %%

@dataclass
class PatchingTask:
    """Matched clean/corrupted prompts sharing one token template, plus a per-prompt metric."""

    name: str
    clean_tokens: Int[Tensor, "batch seq"]
    corrupted_tokens: Int[Tensor, "batch seq"]
    metric: Callable[[Float[Tensor, "batch seq d_vocab"]], Float[Tensor, " batch"]]
    labels: list[str]

    def __post_init__(self):
        assert self.clean_tokens.shape == self.corrupted_tokens.shape, (
            f"Clean and corrupted tokens must have the same shape, got {tuple(self.clean_tokens.shape)} "
            f"and {tuple(self.corrupted_tokens.shape)}. Do all your prompts share one token template?"
        )
        assert len(self.labels) == self.clean_tokens.shape[1], "Need exactly one label per position"

    @property
    def n_prompts(self) -> int:
        return self.clean_tokens.shape[0]

    @property
    def seq_len(self) -> int:
        return self.clean_tokens.shape[1]


def make_ioi_task(
    model: HookedTransformer,
    examples: list[IOIExample],
    scheme: str = "ABC",
    corrupted_examples: list[IOIExample] | None = None,
    name: str | None = None,
) -> PatchingTask:
    """
    IOI task with clean prompts ioi_prompt(ex, "ABB"), and corrupted prompts ioi_prompt(ex', scheme) where ex'
    comes from `corrupted_examples` if given (e.g. for the placebo) and is the same example otherwise.
    """
    corrupted_examples = corrupted_examples or examples
    clean_tokens = model.to_tokens([ioi_prompt(ex, "ABB") for ex in examples])
    corrupted_tokens = model.to_tokens([ioi_prompt(ex, scheme) for ex in corrupted_examples])
    answer_tokens = t.tensor(
        [[model.to_single_token(" " + ex.io), model.to_single_token(" " + ex.s)] for ex in examples],
        device=clean_tokens.device,
    )
    # Label positions using the first prompt, with roles for the name positions
    roles = {" " + examples[0].io: "IO", " " + examples[0].s: "S"}
    labels = []
    for tok in clean_tokens[0]:
        tok_str = model.to_string(tok)
        label = roles.get(tok_str, tok_str.strip() or repr(tok_str))
        if label == "S":
            label = "S1" if "S1" not in labels else "S2"
        labels.append(label)
    labels[0], labels[-1] = "BOS", "END"
    return PatchingTask(
        name=name or f"IOI ({scheme})",
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        metric=partial(logit_diff, answer_tokens=answer_tokens),
        labels=labels,
    )

# %%

if MAIN:
    ioi_tasks = {scheme: make_ioi_task(model, examples, scheme) for scheme in ["ABA", "ABC", "XYZ"]}
    ioi_tasks["placebo"] = make_ioi_task(model, examples, "ABB", corrupted_examples=placebo_examples, name="IOI (placebo)")
    task = ioi_tasks["ABC"]  # our main task from now on
    answer_tokens = task.metric.keywords["answer_tokens"]
    
    print(task.labels)
    clean_logits = model(task.clean_tokens)
    clean_ld = task.metric(clean_logits)
    io_top1 = (clean_logits[:, -1].argmax(-1) == answer_tokens[:, 0]).float().mean()
    print(f"\nclean: logit diff {clean_ld.mean():.2f} ± {clean_ld.std():.2f}, IO is the top-1 prediction for {io_top1:.0%} of prompts")
    
    per_prompt_ld = {"clean (ABB)": clean_ld}
    for name, tsk in ioi_tasks.items():
        ld = tsk.metric(model(tsk.corrupted_tokens))
        per_prompt_ld[name if name == "placebo" else f"corrupted ({name})"] = ld
        print(f"{name:>8}: logit diff {ld.mean():5.2f} ± {ld.std():.2f}, fraction of prompts with logit diff > 0: {(ld > 0).float().mean():.0%}")
    
    del clean_logits  # full logits for 100 prompts are ~300MB; we only needed the metric

# %%

if MAIN:
    utils.plot_metric_strips(
        per_prompt_ld,
        title="IOI logit difference logit(IO) - logit(S), one point per prompt",
        xlabel="logit difference (measured w.r.t. the clean prompt's IO and S)",
    )

# %%

def patching_score(patched, base, source):
    """(patched - base) / (source - base). Works elementwise on floats or tensors."""
    return (patched - base) / (source - base)


if MAIN:
    tests.test_patching_score(patching_score)

# %%

def patch_run(
    model: HookedTransformer,
    base_tokens: Int[Tensor, "batch seq"],
    hook_name: str,
    source_activation: Tensor,
    index: tuple,
) -> Float[Tensor, "batch seq d_vocab"]:
    """Runs the model on base_tokens, with activation[index] at hook_name replaced by source_activation[index]."""

    def hook_fn(activation: Tensor, hook) -> Tensor:
        activation[index] = source_activation[index]
        return activation

    return model.run_with_hooks(base_tokens, fwd_hooks=[(hook_name, hook_fn)])


if MAIN:
    tests.test_patch_run(patch_run, model)

# %%

def layer_position_sweep(
    model: HookedTransformer,
    task: PatchingTask,
    act_name: str = "resid_pre",
    direction: str = "denoise",
    source_perm: Int[Tensor, " batch"] | None = None,
) -> Float[Tensor, "layer pos"]:
    """
    Patching score for patching act_name at every (layer, position), in the given direction. If source_perm is
    given, base prompt i is patched with the activation from source prompt source_perm[i].
    """
    assert direction in ("denoise", "noise")
    if direction == "denoise":
        base, source = task.corrupted_tokens, task.clean_tokens
    else:
        base, source = task.clean_tokens, task.corrupted_tokens
    base_metric = task.metric(model(base)).mean()
    source_logits, source_cache = model.run_with_cache(source, names_filter=lambda name: name.endswith(f"hook_{act_name}"))
    source_metric = task.metric(source_logits).mean()

    scores = t.zeros(model.cfg.n_layers, task.seq_len)
    for layer in range(model.cfg.n_layers):
        hook_name = get_act_name(act_name, layer)
        source_activation = source_cache[hook_name]
        if source_perm is not None:
            source_activation = source_activation[source_perm.to(source_activation.device)]
        for pos in range(task.seq_len):
            logits = patch_run(model, base, hook_name, source_activation, np.s_[:, pos])
            scores[layer, pos] = patching_score(task.metric(logits).mean(), base_metric, source_metric)
    return scores


if MAIN:
    tests.test_layer_position_sweep(layer_position_sweep, PatchingTask, model)

# %%

if MAIN:
    shift = t.roll(t.arange(task.n_prompts), shifts=1)  # prompt i receives prompt i-1's activation
    resid_maps = {
        "denoising (clean -> ABC)": layer_position_sweep(model, task, "resid_pre", "denoise"),
        "noising (ABC -> clean)": layer_position_sweep(model, task, "resid_pre", "noise"),
        "CONTROL: denoising, shuffled pairs": layer_position_sweep(model, task, "resid_pre", "denoise", source_perm=shift),
    }
    utils.plot_layer_pos_maps(resid_maps, task.labels, "IOI: patching the residual stream (resid_pre), ABC corruption", vmax=1.2)

# %%

if MAIN:
    S2, END = task.labels.index("S2"), task.seq_len - 1
    for layer, pos in [(0, S2), (4, S2), (7, S2), (8, END), (10, END), (11, END)]:
        matched = resid_maps["denoising (clean -> ABC)"][layer, pos]
        shuffled = resid_maps["CONTROL: denoising, shuffled pairs"][layer, pos]
        print(f"layer {layer:>2}, {task.labels[pos]:>3}: matched {matched:5.2f}   shuffled {shuffled:5.2f}")

# %%

if MAIN:
    toy_circuits = {
        # each function maps the values of (a, b) to y. In the serial circuit b just copies a, so y only depends on a
        "serial  a -> b -> y": lambda a, b: a,
        "AND     y = a and b": lambda a, b: a * b,
        "OR      y = a or b": lambda a, b: max(a, b),
    }
    for name, f in toy_circuits.items():
        y_clean, y_corr = f(1, 1), f(0, 0)
        denoise_a = patching_score(f(1, 0), y_corr, y_clean) + 0.0  # corrupted run, a restored to its clean value
        noise_a = patching_score(f(0, 1), y_clean, y_corr) + 0.0  # clean run, a set to its corrupted value (+ 0.0 avoids printing -0)
        print(f"{name:<22} denoising a: {denoise_a:.0f}   noising a: {noise_a:.0f}")

# %%

def head_sweep(
    model: HookedTransformer,
    base_tokens: Int[Tensor, "batch seq"],
    source_z: list[Float[Tensor, "batch seq n_heads d_head"]],
    metric: Callable[[Tensor], Float[Tensor, " batch"]],
) -> Float[Tensor, "layer head"]:
    """Mean metric when each head's output (hook_z, all positions) is replaced by the source, one head at a time."""
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            logits = patch_run(model, base_tokens, get_act_name("z", layer), source_z[layer], np.s_[:, :, head])
            results[layer, head] = metric(logits).mean()
    return results


if MAIN:
    tests.test_head_sweep(head_sweep, model)

# %%

def get_z(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> list[Float[Tensor, "batch seq n_heads d_head"]]:
    """Returns [cache["z", layer] for each layer] for a forward pass on tokens."""
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith("hook_z"))
    return [cache["z", layer] for layer in range(model.cfg.n_layers)]


def parse_head(node: str) -> tuple[int, int]:
    """'9.6' -> (9, 6)"""
    layer, head = node.split(".")
    return int(layer), int(head)


HEAD_LABELS = [f"{layer}.{head}" for layer in range(12) for head in range(12)]

IOI_HEAD_CLASSES = {  # from Wang et al. (2022), figure 2
    "name mover": ["9.9", "9.6", "10.0"],
    "backup name mover": ["10.10", "10.6", "10.2", "10.1", "11.2", "9.7", "9.0", "11.9"],
    "negative name mover": ["10.7", "11.10"],
    "S-inhibition": ["7.3", "7.9", "8.6", "8.10"],
    "induction": ["5.5", "5.8", "5.9", "6.9"],
    "duplicate token": ["0.1", "0.10", "3.0"],
    "previous token": ["2.2", "4.11"],
}

# %%

if MAIN:
    clean_z = get_z(model, task.clean_tokens)
    corrupted_z = get_z(model, task.corrupted_tokens)
    placebo_z = get_z(model, ioi_tasks["placebo"].corrupted_tokens)
    M_clean = task.metric(model(task.clean_tokens)).mean().item()
    M_corr = task.metric(model(task.corrupted_tokens)).mean().item()
    
    head_denoise = patching_score(head_sweep(model, task.corrupted_tokens, clean_z, task.metric), M_corr, M_clean)
    head_noise = patching_score(head_sweep(model, task.clean_tokens, corrupted_z, task.metric), M_clean, M_corr)
    head_placebo = patching_score(head_sweep(model, task.clean_tokens, placebo_z, task.metric), M_clean, M_corr)
    noise_floor = head_placebo.abs().max().item()
    
    print(f"M_clean = {M_clean:.2f}, M_corr = {M_corr:.2f}, placebo noise floor = {noise_floor:.3f}")
    print(f"Sum of denoising scores over all heads: {head_denoise.sum():.2f};  sum of noising scores: {head_noise.sum():.2f}")
    for name, grid in [("denoising", head_denoise), ("noising", head_noise), ("placebo", head_placebo)]:
        order = grid.flatten().abs().argsort(descending=True)[:8]
        print(f"{name:>9}, largest |score|: " + ", ".join(f"{HEAD_LABELS[i]} ({grid.flatten()[i]:+.2f})" for i in order))
    utils.plot_head_maps(
        {"denoising (clean -> ABC)": head_denoise, "noising (ABC -> clean)": head_noise, "CONTROL: noising from placebo": head_placebo},
        "IOI: patching each head's output (all positions), normalised patching score",
    )

# %%

if MAIN:
    for node in ["0.10", "3.0", "5.5", "5.9", "7.3", "7.9", "8.6", "8.10", "9.6", "9.9", "10.0", "10.7"]:
        layer, head = parse_head(node)
        noise, denoise = head_noise[layer, head].item(), head_denoise[layer, head].item()
        print(f"{node:>5}: noising {noise:+.2f}, denoising {denoise:+.2f}, denoising - noising {denoise - noise:+.2f}")
    utils.plot_labelled_scatter(
        head_noise.flatten(),
        head_denoise.flatten(),
        HEAD_LABELS,
        xlabel="noising score  (necessity: how much breaking this head hurts)",
        ylabel="denoising score  (sufficiency: how much restoring this head helps)",
        title="IOI heads: sufficiency vs necessity (ABC corruption)",
        n_labels=14,
        noise_floor=noise_floor,
        groups=IOI_HEAD_CLASSES,
    )

# %%

def head_position_sweep(
    model: HookedTransformer,
    base_tokens: Int[Tensor, "batch seq"],
    source_z: list[Float[Tensor, "batch seq n_heads d_head"]],
    metric: Callable[[Tensor], Float[Tensor, " batch"]],
    heads: list[str],
) -> Float[Tensor, "n_heads pos"]:
    """Mean metric when one head's output at one position is replaced by the source, for each head and position."""
    results = t.zeros(len(heads), base_tokens.shape[1])
    for i, node in enumerate(heads):
        layer, head = parse_head(node)
        for pos in range(base_tokens.shape[1]):
            logits = patch_run(model, base_tokens, get_act_name("z", layer), source_z[layer], np.s_[:, pos, head])
            results[i, pos] = metric(logits).mean()
    return results


if MAIN:
    tests.test_head_position_sweep(head_position_sweep, model)

# %%

if MAIN:
    key_heads = ["0.10", "3.0", "5.5", "5.9", "6.9", "7.3", "7.9", "8.6", "8.10", "9.6", "9.9", "10.0", "10.7", "11.10"]
    head_pos_noise = patching_score(head_position_sweep(model, task.clean_tokens, corrupted_z, task.metric, key_heads), M_clean, M_corr)
    utils.plot_layer_pos_maps(
        {"noising one head at one position": head_pos_noise},
        task.labels,
        "IOI: noising score for patching a head's output at a single position (ABC corruption)",
        ylabel="head",
        ylabels=key_heads,
        annotate_threshold=0.05,
    )
    print("Columns before S2 are all zero:", bool((head_pos_noise[:, : task.labels.index("S2")].abs() < 1e-4).all()))

# %%

def ablation_source(
    z: Float[Tensor, "batch seq n_heads d_head"], method: str, seed: int = 0
) -> Float[Tensor, "batch seq n_heads d_head"]:
    """Replacement activations for zero, mean or resample ablation."""
    if method == "zero":
        return t.zeros_like(z)
    if method == "mean":
        return z.mean(dim=0, keepdim=True).expand_as(z)
    if method == "resample":
        # A random cyclic order is a permutation with no fixed points: each prompt takes the next one's activation
        generator = t.Generator().manual_seed(seed)
        order = t.randperm(z.shape[0], generator=generator)
        perm = t.empty_like(order)
        perm[order] = order.roll(-1)
        return z[perm.to(z.device)]
    raise ValueError(f"Unknown method {method!r}")


if MAIN:
    tests.test_ablation_source(ablation_source)

# %%

if MAIN:
    sources = {
        "ABA (S2 -> IO)": get_z(model, ioi_tasks["ABA"].corrupted_tokens),
        "ABC (S2 -> new name)": corrupted_z,
        "XYZ (all names new)": get_z(model, ioi_tasks["XYZ"].corrupted_tokens),
        "zero ablation": [ablation_source(z, "zero") for z in clean_z],
        "mean ablation": [ablation_source(z, "mean") for z in clean_z],
        "resample ablation": [ablation_source(z, "resample", seed=layer) for layer, z in enumerate(clean_z)],
        "CONTROL: placebo": placebo_z,
    }
    noising_deltas = {name: head_sweep(model, task.clean_tokens, src, task.metric) - M_clean for name, src in tqdm(sources.items())}
    utils.plot_head_maps(
        noising_deltas, "Noising each head with different replacements: change in IOI logit difference", ncols=4, cbar_label="Δ logit diff"
    )

# %%

if MAIN:
    for name, delta in noising_deltas.items():
        order = delta.flatten().argsort()[:6]
        print(f"{name:<22} most negative: " + ", ".join(f"{HEAD_LABELS[i]} ({delta.flatten()[i]:+.2f})" for i in order))
    
    names = list(noising_deltas)
    corr = np.corrcoef(np.stack([noising_deltas[n].flatten().numpy() for n in names]))
    utils.plot_correlation_matrix(corr, names, "How similar are the head rankings? (correlation of Δ logit diff across heads)")

# %%

def all_nodes(model: HookedTransformer) -> set[str]:
    """Every head ("layer.head"), every MLP ("mlp{layer}"), and the final LayerNorm scale ("ln_final")."""
    heads = {f"{layer}.{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)}
    return heads | {f"mlp{layer}" for layer in range(model.cfg.n_layers)} | {"ln_final"}


def frozen_run_cache(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> ActivationCache:
    """Caches exactly the activations that run_with_frozen needs: hook_z, hook_mlp_out and ln_final.hook_scale."""
    names = ("hook_z", "hook_mlp_out", "ln_final.hook_scale")
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith(names))
    return cache


def run_with_frozen(
    model: HookedTransformer,
    base_tokens: Int[Tensor, "batch seq"],
    base_cache: ActivationCache,
    patches: dict[str, Float[Tensor, "batch seq d_head"]],
    recompute: set[str],
) -> Float[Tensor, "batch seq d_vocab"]:
    """
    Runs on base_tokens with the heads in `patches` set to the given z values, every node not in `recompute`
    frozen at its value in base_cache, and the nodes in `recompute` computed normally. Returns the logits.
    """

    def z_hook(z: Tensor, hook) -> Tensor:
        layer = hook.layer()
        for head in range(model.cfg.n_heads):
            node = f"{layer}.{head}"
            if node in patches:
                z[:, :, head] = patches[node]
            elif node not in recompute:
                z[:, :, head] = base_cache[hook.name][:, :, head]
        return z

    def freeze_hook(activation: Tensor, hook) -> Tensor:
        return base_cache[hook.name]

    fwd_hooks = [(get_act_name("z", layer), z_hook) for layer in range(model.cfg.n_layers)]
    fwd_hooks += [
        (get_act_name("mlp_out", layer), freeze_hook)
        for layer in range(model.cfg.n_layers)
        if f"mlp{layer}" not in recompute
    ]
    if "ln_final" not in recompute:
        fwd_hooks.append(("ln_final.hook_scale", freeze_hook))
    return model.run_with_hooks(base_tokens, fwd_hooks=fwd_hooks)


if MAIN:
    tests.test_run_with_frozen(run_with_frozen, all_nodes, frozen_run_cache, model)

# %%

def total_and_direct_effect(
    model: HookedTransformer,
    task: PatchingTask,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    heads: list[str],
) -> tuple[float, float]:
    """(TE, DE) of noising all of `heads` together: change in the mean metric relative to the clean run."""
    patches = {}
    for node in heads:
        layer, head = parse_head(node)
        patches[node] = corrupted_cache["z", layer][:, :, head]
    clean_metric = task.metric(model(task.clean_tokens)).mean()
    total = task.metric(run_with_frozen(model, task.clean_tokens, clean_cache, patches, all_nodes(model))).mean()
    direct = task.metric(run_with_frozen(model, task.clean_tokens, clean_cache, patches, set())).mean()
    return (total - clean_metric).item(), (direct - clean_metric).item()


if MAIN:
    tests.test_total_and_direct_effect(total_and_direct_effect, PatchingTask, model)

# %%

if MAIN:
    del sources  # free the ablation activations from part 3
    if t.cuda.is_available():
        t.cuda.empty_cache()
    clean_cache = frozen_run_cache(model, task.clean_tokens)
    corrupted_cache = frozen_run_cache(model, task.corrupted_tokens)
    
    head_te, head_de = t.zeros(N_LAYERS, N_HEADS), t.zeros(N_LAYERS, N_HEADS)
    for node in tqdm(HEAD_LABELS):
        layer, head = parse_head(node)
        head_te[layer, head], head_de[layer, head] = total_and_direct_effect(model, task, clean_cache, corrupted_cache, [node])
    head_ie = head_te - head_de
    
    utils.plot_labelled_scatter(
        head_de.flatten(),
        head_te.flatten(),
        HEAD_LABELS,
        xlabel="direct effect  (Δ logit diff, everything else frozen)",
        ylabel="total effect  (Δ logit diff, everything recomputes)",
        title="IOI: noising one head at a time (ABC corruption)",
        n_labels=14,
        groups=IOI_HEAD_CLASSES,
    )
    for name, grid in [("total", head_te), ("direct", head_de), ("indirect", head_ie)]:
        flat = grid.flatten()
        neg, pos = flat.argsort()[:5], flat.argsort(descending=True)[:4]
        print(f"{name:>8} effect, most negative: " + ", ".join(f"{HEAD_LABELS[i]} ({flat[i]:+.2f})" for i in neg))
        print(f"{'':>8}         most positive: " + ", ".join(f"{HEAD_LABELS[i]} ({flat[i]:+.2f})" for i in pos))

# %%

def path_restricted_effects(
    model: HookedTransformer,
    task: PatchingTask,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    sender: str,
    routes: dict[str, set[str]],
) -> dict[str, float]:
    """For each route, the effect of noising `sender` when only the nodes in that route recompute."""
    layer, head = parse_head(sender)
    patches = {sender: corrupted_cache["z", layer][:, :, head]}
    clean_metric = task.metric(model(task.clean_tokens)).mean()
    return {
        name: (task.metric(run_with_frozen(model, task.clean_tokens, clean_cache, patches, nodes)).mean() - clean_metric).item()
        for name, nodes in routes.items()
    }


if MAIN:
    tests.test_path_restricted_effects(path_restricted_effects, all_nodes, frozen_run_cache, PatchingTask, model)

# %%

if MAIN:
    NAME_MOVERS = IOI_HEAD_CLASSES["name mover"]
    late_heads = {f"{layer}.{head}" for layer in (9, 10, 11) for head in range(N_HEADS)}
    routes = {
        "total (all recompute)": all_nodes(model),
        "direct (none recompute)": set(),
        "via name movers only": set(NAME_MOVERS),
        "via NMs + negative NMs": set(NAME_MOVERS + IOI_HEAD_CLASSES["negative name mover"]),
        "via all heads in layers 9-11": late_heads,
        "via all heads": {node for node in all_nodes(model) if "." in node},
    }
    path_table = pd.DataFrame(
        {sender: path_restricted_effects(model, task, clean_cache, corrupted_cache, sender, routes) for sender in ["8.10", "7.9", "5.5", "3.0"]}
    ).T
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print(path_table.round(2))

# %%

if MAIN:
    nm_effect = total_and_direct_effect(model, task, clean_cache, corrupted_cache, NAME_MOVERS)
    s_inhibition_effect = total_and_direct_effect(model, task, clean_cache, corrupted_cache, IOI_HEAD_CLASSES["S-inhibition"])
    
    rng = random.Random(0)
    candidates = sorted(late_heads - set(NAME_MOVERS))
    random_sets = [rng.sample(candidates, 3) for _ in range(30)]
    random_effects = [total_and_direct_effect(model, task, clean_cache, corrupted_cache, heads) for heads in tqdm(random_sets)]
    
    print(f"Clean logit diff: {M_clean:.2f}")
    print(f"Name movers {NAME_MOVERS}: total effect {nm_effect[0]:+.2f}, direct effect {nm_effect[1]:+.2f}")
    print(f"S-inhibition heads: total effect {s_inhibition_effect[0]:+.2f}, direct effect {s_inhibition_effect[1]:+.2f}")
    gaps = t.tensor([te - de for te, de in random_effects])
    print(f"Random sets of 3 heads: |total - direct| has median {gaps.abs().median():.2f}, max {gaps.abs().max():.2f}")
    print(f"Name movers: total - direct = {nm_effect[0] - nm_effect[1]:+.2f}")
    
    utils.plot_total_vs_direct(
        {
            "CONTROL: 30 random sets of 3 heads from layers 9-11": random_effects,
            "name movers 9.9 + 9.6 + 10.0": [nm_effect],
            "S-inhibition heads 7.3 + 7.9 + 8.6 + 8.10": [s_inhibition_effect],
        },
        title="The Hydra effect: knocking out the name movers barely matters\n(noising sets of heads with ABC activations)",
    )

# %%

def head_direct_contributions(
    model: HookedTransformer, cache: ActivationCache, answer_tokens: Int[Tensor, "batch 2"]
) -> Float[Tensor, "layer head"]:
    """Mean (over prompts) direct logit attribution of each head at the final position, using cache's own LN scale."""
    z_end = t.stack([cache["z", layer][:, -1] for layer in range(model.cfg.n_layers)])  # [layer batch head d_head]
    head_out = t.einsum("lbhd,lhdm->lbhm", z_end, model.W_O)
    head_out = head_out / cache["ln_final.hook_scale"][:, -1][None, :, None, :]  # scale has shape [batch seq 1]
    answer_tokens = answer_tokens.to(model.W_U.device)
    direction = model.W_U[:, answer_tokens[:, 0]] - model.W_U[:, answer_tokens[:, 1]]  # [d_model batch]
    return t.einsum("lbhm,mb->lbh", head_out, direction).mean(dim=1).cpu()


if MAIN:
    tests.test_head_direct_contributions(head_direct_contributions, model)

# %%

if MAIN:
    def knockout_hook(z: Tensor, hook, heads: list[int]) -> Tensor:
        z[:, :, heads] = corrupted_cache[hook.name][:, :, heads]
        return z
    
    
    nm_by_layer = {9: [9, 6], 10: [0]}
    with model.hooks(fwd_hooks=[(get_act_name("z", layer), partial(knockout_hook, heads=heads)) for layer, heads in nm_by_layer.items()]):
        ko_logits, ko_cache = model.run_with_cache(task.clean_tokens, names_filter=lambda name: name.endswith(("hook_z", "ln_final.hook_scale")))
    
    dla_clean = head_direct_contributions(model, clean_cache, answer_tokens)
    dla_ko = head_direct_contributions(model, ko_cache, answer_tokens)
    change = dla_ko - dla_clean
    nm_change = sum(change[parse_head(n)] for n in NAME_MOVERS)
    
    ko_ld = task.metric(ko_logits).mean().item()
    del ko_logits
    print(f"Logit diff: clean {M_clean:.2f}, name movers knocked out {ko_ld:.2f}")
    print(f"Sum of the name movers' direct contributions in the clean run: {sum(dla_clean[parse_head(n)] for n in NAME_MOVERS):+.2f}")
    print(f"Change in the summed direct contributions of all 144 heads: {change.sum():+.2f}")
    print(f"  of which the name movers themselves: {nm_change:+.2f}")
    print(f"  of which all other heads:            {change.sum() - nm_change:+.2f}")
    
    order = change.flatten().argsort(descending=True)
    top_compensators = [HEAD_LABELS[i] for i in order[:10]]
    utils.plot_bars(
        {node: change[parse_head(node)].item() for node in top_compensators + NAME_MOVERS},
        title="Change in each head's direct contribution to the logit diff when the name movers are knocked out\n"
        "(10 largest increases, then the name movers themselves)",
        ylabel="Δ direct contribution",
        colors=["tab:green"] * 10 + ["tab:red"] * 3,
    )
    print("Direct contributions, clean -> knocked out:")
    print(", ".join(f"{n}: {dla_clean[parse_head(n)]:+.2f} -> {dla_ko[parse_head(n)]:+.2f}" for n in top_compensators[:6]))

# %%

def sample_greater_than_rows(model: HookedTransformer, n: int, seed: int = 0) -> list[tuple[str, int, int]]:
    """(noun, century, start year YY) triples whose years tokenize as [' XX', 'YY'] (and so does XX01)."""

    def splits_cleanly(century: int, yy: int) -> bool:
        tokens = model.to_tokens(f" {century}{yy:02d}", prepend_bos=False)[0]
        return [model.to_string(tok) for tok in tokens] == [f" {century}", f"{yy:02d}"]

    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        noun, century, yy = rng.choice(utils.GT_NOUNS), rng.randint(12, 17), rng.randint(2, 98)
        if splits_cleanly(century, yy) and splits_cleanly(century, 1):
            rows.append((noun, century, yy))
    return rows

# %%

def year_prob_diff(
    logits: Float[Tensor, "batch seq d_vocab"], start_years: Int[Tensor, " batch"], year_tokens: Int[Tensor, " 100"]
) -> Float[Tensor, " batch"]:
    """Sum of P(year > YY) minus sum of P(year <= YY), over the 100 two-digit year tokens, for each prompt."""
    probs = logits[:, -1].softmax(dim=-1)[:, year_tokens.to(logits.device)]  # [batch 100]
    greater = t.arange(100, device=logits.device)[None, :] > start_years.to(logits.device)[:, None]
    return (probs * greater).sum(-1) - (probs * ~greater).sum(-1)


if MAIN:
    tests.test_year_prob_diff(year_prob_diff)

# %%

def make_greater_than_task(model: HookedTransformer, n: int = 100, seed: int = 0, corrupt_year: int = 1) -> PatchingTask:
    """Greater-than task: clean 'from the year XXYY to the year XX', corrupted with YY -> corrupt_year, metric = PD."""
    rows = sample_greater_than_rows(model, n, seed)
    template = "The {noun} lasted from the year {century}{yy} to the year {century}"
    clean_tokens = model.to_tokens([template.format(noun=noun, century=c, yy=f"{yy:02d}") for noun, c, yy in rows])
    corrupted_tokens = model.to_tokens([template.format(noun=noun, century=c, yy=f"{corrupt_year:02d}") for noun, c, yy in rows])
    year_tokens = t.tensor([model.to_single_token(f"{y:02d}") for y in range(100)])
    start_years = t.tensor([yy for _, _, yy in rows])
    labels = ["BOS", "The", "noun", "lasted", "from", "the", "year", "XX", "YY", "to", "the", "year", "END (XX)"]
    return PatchingTask(
        name=f"greater-than (YY -> {corrupt_year:02d})",
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupted_tokens,
        metric=partial(year_prob_diff, start_years=start_years, year_tokens=year_tokens),
        labels=labels,
    )

# %%

if MAIN:
    gt_task = make_greater_than_task(model, n=100, seed=0)
    print(repr(model.to_string(gt_task.clean_tokens[0, 1:])), "|", repr(model.to_string(gt_task.corrupted_tokens[0, 1:])))
    gt_clean_logits, gt_corr_logits = model(gt_task.clean_tokens), model(gt_task.corrupted_tokens)
    gt_M_clean, gt_M_corr = gt_task.metric(gt_clean_logits).mean().item(), gt_task.metric(gt_corr_logits).mean().item()
    print(f"PD clean: {gt_M_clean:.3f},  PD corrupted (YY -> 01): {gt_M_corr:.3f}")
    
    year_tokens = gt_task.metric.keywords["year_tokens"].to(device)
    start_years = gt_task.metric.keywords["start_years"]
    gt_clean_year_probs = gt_clean_logits[:, -1].softmax(-1)[:, year_tokens]
    gt_corr_year_probs = gt_corr_logits[:, -1].softmax(-1)[:, year_tokens]
    del gt_clean_logits, gt_corr_logits  # ~260MB each; from now on we only need the metric values
    print(f"Mean probability mass on the 100 two-digit tokens (clean): {gt_clean_year_probs.sum(-1).mean():.3f}")
    utils.plot_year_probs(
        {
            "clean prompts (from the year XXYY ...)": gt_clean_year_probs,
            "corrupted prompts (from the year XX01 ...)": gt_corr_year_probs,
        },
        start_years,
        "Greater-than: GPT-2's distribution over the end year (rows sorted by the clean start year)",
    )

# %%

if MAIN:
    gt_maps = {f"{act} (denoising)": layer_position_sweep(model, gt_task, act, "denoise") for act in ["resid_pre", "attn_out", "mlp_out"]}
    utils.plot_layer_pos_maps(gt_maps, gt_task.labels, "Greater-than: denoising sweeps (clean -> YY=01)", vmax=1.0)
    yy, end = gt_task.labels.index("YY"), gt_task.seq_len - 1
    print("resid_pre at YY, by layer: ", [round(x, 2) for x in gt_maps["resid_pre (denoising)"][:, yy].tolist()])
    print("resid_pre at END, by layer:", [round(x, 2) for x in gt_maps["resid_pre (denoising)"][:, end].tolist()])

# %%

if MAIN:
    ioi_block_maps = {f"{act} (denoising)": layer_position_sweep(model, task, act, "denoise") for act in ["attn_out", "mlp_out"]}
    utils.plot_layer_pos_maps(ioi_block_maps, task.labels, "IOI for comparison: denoising attention and MLP outputs (clean -> ABC)", vmax=1.0)
    
    for name, maps, seq_len in [("greater-than", gt_maps, gt_task.seq_len), ("IOI", ioi_block_maps, task.seq_len)]:
        mlp_end, attn_end = maps["mlp_out (denoising)"][:, seq_len - 1], maps["attn_out (denoising)"][:, seq_len - 1]
        print(f"{name}, END position: sum over layers of MLP scores {mlp_end.sum():.2f}, of attention scores {attn_end.sum():.2f}")
        print(f"  MLP scores by layer:       {[round(x, 2) for x in mlp_end.tolist()]}")
        print(f"  attention scores by layer: {[round(x, 2) for x in attn_end.tolist()]}")

# %%

if MAIN:
    gt_head_denoise = patching_score(
        head_sweep(model, gt_task.corrupted_tokens, get_z(model, gt_task.clean_tokens), gt_task.metric), gt_M_corr, gt_M_clean
    )
    order = gt_head_denoise.flatten().argsort(descending=True)
    print("Top 10 heads (denoising score):", ", ".join(f"{HEAD_LABELS[i]} ({gt_head_denoise.flatten()[i]:.2f})" for i in order[:10]))
    print(f"Sum over all 144 heads: {gt_head_denoise.sum():.2f}")

# %%

def per_prompt_head_effects(
    model: HookedTransformer,
    task: PatchingTask,
    source_z: list[Float[Tensor, "batch seq n_heads d_head"]],
    heads: list[str],
) -> Float[Tensor, "n_heads batch"]:
    """Per-prompt change in the metric when each head is noised (on the clean prompts) with source_z."""
    clean_metric = task.metric(model(task.clean_tokens))
    effects = []
    for node in heads:
        layer, head = parse_head(node)
        logits = patch_run(model, task.clean_tokens, get_act_name("z", layer), source_z[layer], np.s_[:, :, head])
        effects.append(task.metric(logits) - clean_metric)
    return t.stack(effects).cpu()


def bootstrap_mean_ci(values: Float[Tensor, " n"], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap (1 - alpha) confidence interval for the mean of values."""
    generator = t.Generator().manual_seed(seed)
    values = values.float().cpu()
    idx = t.randint(0, len(values), (n_boot, len(values)), generator=generator)
    means = values[idx].mean(dim=-1)
    return means.quantile(alpha / 2).item(), means.quantile(1 - alpha / 2).item()


if MAIN:
    tests.test_per_prompt_head_effects(per_prompt_head_effects, PatchingTask, model)
    tests.test_bootstrap_mean_ci(bootstrap_mean_ci)

# %%

if MAIN:
    variance_heads = ["3.0", "5.5", "7.9", "8.6", "8.10", "9.6", "9.9", "10.0", "10.7", "11.10"]
    per_prompt = per_prompt_head_effects(model, task, corrupted_z, variance_heads)
    cis = [bootstrap_mean_ci(row) for row in per_prompt]
    
    rows = []
    for node, row, (lo, hi) in zip(variance_heads, per_prompt, cis):
        rows.append({"head": node, "mean": row.mean().item(), "95% CI": f"[{lo:+.2f}, {hi:+.2f}]", "std": row.std().item(),
                     "frac < 0": (row < 0).float().mean().item(), "min": row.min().item(), "max": row.max().item()})
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).set_index("head").round(2))
    utils.plot_per_prompt_effects(
        per_prompt, variance_heads, cis, "IOI: per-prompt effect of noising one head (ABC corruption)", "Δ logit diff (per prompt)"
    )

# %%

HELDOUT_TEMPLATES = {
    "BABA order": [tmpl.replace("{A}", "{X}").replace("{B}", "{A}").replace("{X}", "{B}") for tmpl in TRAIN_TEMPLATES],
    "argument": ["Then, {A} and {B} had a long argument. Afterwards {C} said to"],
    "found": ["Friends {A} and {B} found a {OBJECT} at the {PLACE}. {C} gave it to"],
    "working": ["While {A} and {B} were working at the {PLACE}, {C} decided to give a {OBJECT} to"],
}


def heldout_noising_scores(
    model: HookedTransformer, templates: dict[str, list[str]], n: int = 60, seed: int = 0
) -> dict[str, Float[Tensor, "layer head"]]:
    """Normalised head noising scores (ABC corruption) for each held-out template family."""
    scores = {}
    for name, family in templates.items():
        heldout_task = make_ioi_task(model, sample_ioi_examples(family, n, seed), "ABC", name=name)
        m_clean = heldout_task.metric(model(heldout_task.clean_tokens)).mean().item()
        m_corr = heldout_task.metric(model(heldout_task.corrupted_tokens)).mean().item()
        print(f"{name:>10}: {heldout_task.seq_len} tokens, clean logit diff {m_clean:.2f}, ABC logit diff {m_corr:.2f}")
        deltas = head_sweep(model, heldout_task.clean_tokens, get_z(model, heldout_task.corrupted_tokens), heldout_task.metric)
        scores[name] = patching_score(deltas, m_clean, m_corr)
    return scores

# %%

if MAIN:
    heldout_scores = heldout_noising_scores(model, HELDOUT_TEMPLATES)
    utils.plot_scatter_grid(
        head_noise.flatten(),
        {name: s.flatten() for name, s in heldout_scores.items()},
        HEAD_LABELS,
        xlabel="noising score, main (ABBA) templates",
        title="Do head noising scores transfer to held-out templates? (one point per head)",
    )
    for name, s in heldout_scores.items():
        order = s.flatten().argsort(descending=True)
        print(f"{name:>10} top 6: " + ", ".join(f"{HEAD_LABELS[i]} ({s.flatten()[i]:.2f})" for i in order[:6]))

# %%

def answer_probability(
    logits: Float[Tensor, "batch seq d_vocab"], answer_tokens: Int[Tensor, "batch 2"]
) -> Float[Tensor, " batch"]:
    """Probability of the correct answer token at the final position, for each prompt."""
    probs = logits[:, -1].softmax(dim=-1)
    return probs.gather(dim=-1, index=answer_tokens[:, :1].to(probs.device))[:, 0]


if MAIN:
    tests.test_answer_probability(answer_probability)

# %%

if MAIN:
    prob_metric = partial(answer_probability, answer_tokens=answer_tokens)
    P_clean = prob_metric(model(task.clean_tokens)).mean().item()
    delta_prob = head_sweep(model, task.clean_tokens, corrupted_z, prob_metric) - P_clean
    delta_ld = noising_deltas["ABC (S2 -> new name)"]
    r = np.corrcoef(delta_ld.flatten().numpy(), delta_prob.flatten().numpy())[0, 1]
    print(f"Clean P(IO) = {P_clean:.3f}. Correlation between the two metrics' head effects: r = {r:.2f}")
    disagree = [
        HEAD_LABELS[i] for i in range(144) if delta_ld.flatten()[i] * delta_prob.flatten()[i] < 0 and abs(delta_ld.flatten()[i]) > 0.2
    ]
    print(f"Heads with |Δ logit diff| > 0.2 whose two effects have opposite signs: {disagree}")
    utils.plot_labelled_scatter(
        delta_ld.flatten(),
        delta_prob.flatten(),
        HEAD_LABELS,
        xlabel="Δ logit diff  (logit IO - logit S)",
        ylabel="Δ P(IO)",
        title=f"Noising each head (ABC): two metrics, two rankings (r = {r:.2f})",
        diagonal=False,
        groups=IOI_HEAD_CLASSES,
    )

# %%

if MAIN:
    def name_stats(logits: Tensor) -> dict[str, float]:
        final = logits[:, -1]
        probs = final.softmax(-1)
        idx, io, s = t.arange(len(answer_tokens)), answer_tokens[:, 0], answer_tokens[:, 1]
        return {
            "logit IO": final[idx, io].mean().item(),
            "logit S": final[idx, s].mean().item(),
            "logit diff": (final[idx, io] - final[idx, s]).mean().item(),
            "P(IO)": probs[idx, io].mean().item(),
            "P(S)": probs[idx, s].mean().item(),
            "P(IO) + P(S)": (probs[idx, io] + probs[idx, s]).mean().item(),
        }
    
    
    clean_stats = name_stats(model(task.clean_tokens))
    breakdown = {"clean run": clean_stats}
    for node in disagree + ["9.9"]:
        layer, head = parse_head(node)
        patched = patch_run(model, task.clean_tokens, get_act_name("z", layer), corrupted_z[layer], np.s_[:, :, head])
        breakdown[f"noise {node} (change)"] = {k: v - clean_stats[k] for k, v in name_stats(patched).items()}
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(breakdown).T.round(3))

# %%

if MAIN:
    MY_TEMPLATES = ["At the {PLACE}, {A} and {B} talked for hours about the weather. Then {C} gave a {OBJECT} to"]
    MY_SCHEME = "ABC"
    
    my_task = make_ioi_task(model, sample_ioi_examples(MY_TEMPLATES, 60, seed=0), MY_SCHEME, name="my task")
    my_clean = my_task.metric(model(my_task.clean_tokens)).mean().item()
    my_corr = my_task.metric(model(my_task.corrupted_tokens)).mean().item()
    print(f"{my_task.seq_len} tokens. Clean logit diff {my_clean:.2f}, corrupted ({MY_SCHEME}) logit diff {my_corr:.2f}")
    
    my_resid = layer_position_sweep(model, my_task, "resid_pre", "denoise")
    utils.plot_layer_pos_maps({"denoising resid_pre": my_resid}, my_task.labels, f"My task: {MY_TEMPLATES[0]}", vmax=1.2)
    
    my_heads = patching_score(head_sweep(model, my_task.clean_tokens, get_z(model, my_task.corrupted_tokens), my_task.metric), my_clean, my_corr)
    order = my_heads.flatten().argsort(descending=True)
    print("Top heads by noising score:", ", ".join(f"{HEAD_LABELS[i]} ({my_heads.flatten()[i]:.2f})" for i in order[:8]))
    print(f"Correlation with the main task's noising scores: {np.corrcoef(my_heads.flatten(), head_noise.flatten())[0, 1]:.2f}")

# %%
