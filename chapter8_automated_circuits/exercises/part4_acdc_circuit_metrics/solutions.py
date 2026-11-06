# %%


import sys
import time
from collections.abc import Callable
from pathlib import Path

import einops
import matplotlib.pyplot as plt
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
section = "part4_acdc_circuit_metrics"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part4_acdc_circuit_metrics.tests as tests
import part4_acdc_circuit_metrics.utils as utils

# Part 4️⃣ reuses the GPT-2 edge graph and greater-than task from [8.2]
import part2_attribution_patching_eap.solutions as gpt2_graph

MAIN = __name__ == "__main__"

# %%

if MAIN:
    model = utils.load_attn_only_2l(device)
    print(f"{model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer, positional embeddings: {model.cfg.positional_embedding_type}")

# %%

if MAIN:
    HALF_LEN = 20
    INDUCTION_POS = slice(HALF_LEN + 1, 2 * HALF_LEN)  # positions whose next token is predictable by induction
    
    clean, corrupt = utils.make_induction_data(model, n_prompts=64, half_len=HALF_LEN, seed=0)
    heldout_clean, heldout_corrupt = utils.make_induction_data(model, n_prompts=64, half_len=HALF_LEN, seed=1)
    
    print("clean:  ", [model.to_string(tok) for tok in clean[0, :8]], "...", [model.to_string(tok) for tok in clean[0, HALF_LEN + 1 : HALF_LEN + 5]])
    print("corrupt:", [model.to_string(tok) for tok in corrupt[0, :8]], "...", [model.to_string(tok) for tok in corrupt[0, HALF_LEN + 1 : HALF_LEN + 5]])
    print(f"shapes: {tuple(clean.shape)}, induction positions {INDUCTION_POS.start}..{INDUCTION_POS.stop - 1}")

# %%

N_LAYERS, N_HEADS = 2, 12
SENDER_NAMES = ["embed"] + [f"{layer}.{head}" for layer in range(N_LAYERS) for head in range(N_HEADS)]
RECEIVER_NAMES = [f"{layer}.{head}<{c}>" for layer in range(N_LAYERS) for c in "qkv" for head in range(N_HEADS)] + ["logits"]
N_SENDERS, N_RECEIVERS = len(SENDER_NAMES), len(RECEIVER_NAMES)
LOGITS = N_RECEIVERS - 1


def head_sender(layer: int, head: int) -> int:
    return 1 + layer * N_HEADS + head


def head_receiver(layer: int, c: str, head: int) -> int:
    """Index of the receiver for the query ("q"), key ("k") or value ("v") input of a head."""
    return layer * 3 * N_HEADS + "qkv".index(c) * N_HEADS + head


assert SENDER_NAMES[head_sender(0, 7)] == "0.7" and RECEIVER_NAMES[head_receiver(1, "k", 4)] == "1.4<k>"
print(f"{N_SENDERS} senders, {N_RECEIVERS} receivers")

# %%

def build_edge_mask() -> Bool[Tensor, "sender receiver"]:
    mask = t.zeros(N_SENDERS, N_RECEIVERS, dtype=t.bool)
    mask[0, :] = True
    for head in range(N_HEADS):
        mask[head_sender(0, head), head_receiver(1, "q", 0) :] = True  # all layer-1 receivers, and the logits
        mask[head_sender(1, head), LOGITS] = True
    return mask


if MAIN:
    tests.test_build_edge_mask(build_edge_mask)

EDGE_MASK = build_edge_mask()

# %%

@t.no_grad()
def sender_outputs(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq sender d_model"]:
    names = ["hook_embed"] + [f"blocks.{layer}.attn.hook_result" for layer in range(N_LAYERS)]
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)
    return t.cat([cache["hook_embed"][:, :, None]] + [cache[f"blocks.{layer}.attn.hook_result"] for layer in range(N_LAYERS)], dim=2)


if MAIN:
    tests.test_sender_outputs(sender_outputs, model, clean[:4])

# %%

def run_graph(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    corrupt_out: Float[Tensor, "batch seq sender d_model"],
    in_circuit: Tensor,
) -> Float[Tensor, "batch seq d_vocab"]:
    """
    Runs the two-layer model on `tokens`, where each valid edge in `in_circuit` carries its sender's current output
    and each valid edge outside it carries the sender's corrupted output.
    """
    batch, seq = tokens.shape
    valid = EDGE_MASK.to(tokens.device).float()
    keep = in_circuit.to(tokens.device).float() * valid
    cut = valid - keep

    current = t.zeros_like(corrupt_out)
    current[:, :, 0] = model.W_E[tokens]
    pos = model.W_pos[:seq][:, None]  # (seq, 1, d_model), broadcasts over heads
    causal = t.triu(t.ones(seq, seq, dtype=t.bool, device=tokens.device), diagonal=1)
    bias = t.zeros(model.cfg.d_model, device=tokens.device)

    for layer in range(N_LAYERS):

        def receiver_input(c: str) -> Tensor:
            cols = slice(head_receiver(layer, c, 0), head_receiver(layer, c, 0) + N_HEADS)
            return (
                einops.einsum(current, keep[:, cols], "b p s d, s h -> b p h d")
                + einops.einsum(corrupt_out, cut[:, cols], "b p s d, s h -> b p h d")
                + bias
            )

        q = einops.einsum(receiver_input("q") + pos, model.W_Q[layer], "b p h d, h d e -> b p h e") + model.b_Q[layer]
        k = einops.einsum(receiver_input("k") + pos, model.W_K[layer], "b p h d, h d e -> b p h e") + model.b_K[layer]
        v = einops.einsum(receiver_input("v"), model.W_V[layer], "b p h d, h d e -> b p h e") + model.b_V[layer]
        scores = einops.einsum(q, k, "b q h e, b k h e -> b h q k") / model.cfg.d_head**0.5
        pattern = scores.masked_fill(causal, float("-inf")).softmax(dim=-1)
        z = einops.einsum(pattern, v, "b h q k, b k h e -> b q h e")
        start = head_sender(layer, 0)
        current[:, :, start : start + N_HEADS] = einops.einsum(z, model.W_O[layer], "b p h e, h e d -> b p h d")
        bias = bias + model.b_O[layer]

    final = (
        einops.einsum(current, keep[:, LOGITS], "b p s d, s -> b p d")
        + einops.einsum(corrupt_out, cut[:, LOGITS], "b p s d, s -> b p d")
        + bias
    )
    return final @ model.W_U + model.b_U


if MAIN:
    tests.test_run_graph(run_graph, sender_outputs, EDGE_MASK, model, clean, corrupt)

# %%

def kl_divergence(
    circuit_logits: Float[Tensor, "batch pos d_vocab"], full_logprobs: Float[Tensor, "batch pos d_vocab"]
) -> Float[Tensor, ""]:
    circuit_logprobs = circuit_logits.log_softmax(dim=-1)
    return (full_logprobs.exp() * (full_logprobs - circuit_logprobs)).sum(dim=-1).mean()


if MAIN:
    tests.test_kl_divergence(kl_divergence)

# %%

def induction_logprob(logits: Float[Tensor, "batch seq d_vocab"], tokens: Int[Tensor, "batch seq"]) -> float:
    """Mean log probability of the correct next token at the induction positions."""
    logprobs = logits[:, INDUCTION_POS].log_softmax(dim=-1)
    targets = tokens[:, INDUCTION_POS.start + 1 : INDUCTION_POS.stop + 1]
    return logprobs.gather(-1, targets[..., None]).mean().item()

# %%

if MAIN:
    corrupt_out = sender_outputs(model, corrupt)
    heldout_corrupt_out = sender_outputs(model, heldout_corrupt)
    with t.no_grad():
        full_logprobs = model(clean)[:, INDUCTION_POS].log_softmax(dim=-1)
    
    
    @t.inference_mode()
    def circuit_kl(circuit: Tensor) -> float:
        """KL divergence of the circuit from the full model, on the discovery prompts."""
        return kl_divergence(run_graph(model, clean, corrupt_out, circuit)[:, INDUCTION_POS], full_logprobs).item()
    
    
    @t.inference_mode()
    def circuit_logprob(circuit: Tensor) -> float:
        """Induction log probability of the circuit, on the held-out prompts."""
        return induction_logprob(run_graph(model, heldout_clean, heldout_corrupt_out, circuit), heldout_clean)
    
    
    EMPTY = t.zeros_like(EDGE_MASK)
    with t.inference_mode():
        print(f"Full model:    KL = {abs(circuit_kl(EDGE_MASK)):.3f}, induction log prob = {circuit_logprob(EDGE_MASK):.2f}")
        print(f"Empty circuit: KL = {circuit_kl(EMPTY):.3f}, induction log prob = {circuit_logprob(EMPTY):.2f}")
        lp_first = model(heldout_clean)[:, 1:HALF_LEN].log_softmax(-1).gather(-1, heldout_clean[:, 2 : HALF_LEN + 1, None]).mean().item()
        print(f"(For comparison, the full model's log prob on the unpredictable first half: {lp_first:.2f})")

# %%

def single_edge_effects(metric_fn: Callable[[Tensor], float], edge_mask: Tensor) -> Float[Tensor, "sender receiver"]:
    effects = t.zeros(edge_mask.shape)
    for u, v in tqdm(edge_mask.nonzero().tolist(), desc="single edges"):
        circuit = edge_mask.clone()
        circuit[u, v] = False
        effects[u, v] = metric_fn(circuit)
    return effects

# %%

if MAIN:
    tests.test_single_edge_effects(single_edge_effects)

# %%

if MAIN:
    single_kl = single_edge_effects(circuit_kl, EDGE_MASK)
    utils.plot_edge_heatmap(single_kl, EDGE_MASK, SENDER_NAMES, RECEIVER_NAMES, "KL divergence from the full model when one edge is cut")
    
    top = single_kl.flatten().topk(12)
    print("Edges with the largest single-edge effect:")
    for value, idx in zip(top.values.tolist(), top.indices.tolist()):
        u, v = divmod(idx, N_RECEIVERS)
        print(f"  {SENDER_NAMES[u]:>6} -> {RECEIVER_NAMES[v]:<8} KL = {value:.3f}")
    print(f"Edges with KL < 0.01: {int(((single_kl < 0.01) & EDGE_MASK).sum())} of {int(EDGE_MASK.sum())}")

# %%

# Receivers from the output backwards: logits, then layer-1 heads, then layer-0 heads (q, k, v of each head together)
RECEIVER_ORDER = [LOGITS] + [head_receiver(layer, c, head) for layer in reversed(range(N_LAYERS)) for head in reversed(range(N_HEADS)) for c in "qkv"]
# The node (sender) that each receiver belongs to: a head's q, k and v belong to that head; the logits to nobody
RECEIVER_OWNER = [head_sender(layer, head) for layer in range(N_LAYERS) for c in "qkv" for head in range(N_HEADS)] + [None]

assert sorted(RECEIVER_ORDER) == list(range(N_RECEIVERS))
print([RECEIVER_NAMES[v] for v in RECEIVER_ORDER[:7]], "...")

# %%

def acdc(
    metric_fn: Callable[[Tensor], float],
    edge_mask: Bool[Tensor, "sender receiver"],
    receiver_order: list[int],
    receiver_owner: list[int | None],
    threshold: float,
    verbose: bool = False,
) -> tuple[Bool[Tensor, "sender receiver"], int]:
    """
    Greedy edge removal from the output backwards. Returns (circuit, number of edges tested).
    """
    circuit = edge_mask.clone()
    current = metric_fn(circuit)
    n_tests = 0
    for v in tqdm(receiver_order, desc="receivers", disable=not verbose):
        owner = receiver_owner[v]
        if owner is not None and not circuit[owner].any():
            circuit[:, v] = False
            continue
        for u in reversed(edge_mask[:, v].nonzero().flatten().tolist()):
            circuit[u, v] = False
            new = metric_fn(circuit)
            n_tests += 1
            if new - current < threshold:
                current = new
            else:
                circuit[u, v] = True
    return circuit, n_tests


if MAIN:
    tests.test_acdc(acdc)

# %%

if MAIN:
    t0 = time.time()
    circuit, n_tests = acdc(circuit_kl, EDGE_MASK, RECEIVER_ORDER, RECEIVER_OWNER, threshold=0.3)
    print(f"ACDC tested {n_tests} of {int(EDGE_MASK.sum())} edges in {time.time() - t0:.1f}s and kept {int(circuit.sum())}")
    print(f"KL = {circuit_kl(circuit):.3f}, held-out induction log prob = {circuit_logprob(circuit):.2f}\n")
    for u, v in circuit.nonzero().tolist():
        print(f"  {SENDER_NAMES[u]:>6} -> {RECEIVER_NAMES[v]}")
    
    ROLES_FROM_1_2 = {"0.7": "prev token", "1.4": "induction", "1.10": "induction"}
    utils.draw_2l_circuit(circuit, SENDER_NAMES, RECEIVER_NAMES, f"The circuit ACDC finds for induction (τ = 0.3, {int(circuit.sum())} edges)", highlight=ROLES_FROM_1_2)

# %%

@t.no_grad()
def attention_by_offset(
    model: HookedTransformer, tokens: Int[Tensor, "batch seq"], layer: int, max_offset: int, start_pos: int = 8
) -> Float[Tensor, "n_heads offset"]:
    _, cache = model.run_with_cache(tokens, names_filter=f"blocks.{layer}.attn.hook_pattern")
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # (batch, head, query, key)
    # The diagonal at offset -d starts at query d. If d exceeds start_pos, keep the whole diagonal.
    return t.stack(
        [pattern.diagonal(offset=-d, dim1=-2, dim2=-1)[..., max(0, start_pos - d) :].mean(dim=(0, -1)) for d in range(max_offset + 1)],
        dim=-1,
    )

# %%

if MAIN:
    tests.test_attention_by_offset(attention_by_offset)

# %%

if MAIN:
    by_offset = attention_by_offset(model, clean, layer=0, max_offset=6).cpu()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(by_offset, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), [str(d) for d in range(7)])
    ax.set_yticks(range(N_HEADS), [f"0.{h}" for h in range(N_HEADS)])
    for h in range(N_HEADS):
        for d in range(7):
            if by_offset[h, d] > 0.15:
                ax.text(d, h, f"{by_offset[h, d]:.2f}", ha="center", va="center", fontsize=7, color="white" if by_offset[h, d] > 0.5 else "black")
    ax.set_xlabel("how many positions back the attended token is (0 = the current token)")
    ax.set_title("Layer-0 heads: mean attention by relative position (clean prompts)")
    fig.colorbar(im, ax=ax, label="mean attention", fraction=0.04)
    plt.tight_layout()
    plt.show()
    for h in [7, 4, 9]:
        print(f"0.{h}: " + ", ".join(f"{label} {v:.2f}" for label, v in zip(["self"] + [f"{d} back" for d in range(1, 7)], by_offset[h].tolist())))

# %%

def random_circuit(edge_mask: Tensor, n_edges: int, generator: t.Generator) -> Bool[Tensor, "sender receiver"]:
    valid = edge_mask.flatten().nonzero().flatten()
    chosen = valid[t.randperm(len(valid), generator=generator)[:n_edges]]
    circuit = t.zeros(edge_mask.numel(), dtype=t.bool)
    circuit[chosen] = True
    return circuit.reshape(edge_mask.shape)


def top_k_circuit(scores: Tensor, edge_mask: Tensor, k: int) -> Bool[Tensor, "sender receiver"]:
    flat = scores.float().masked_fill(~edge_mask, -float("inf")).flatten()
    circuit = t.zeros(edge_mask.numel(), dtype=t.bool)
    circuit[flat.topk(k).indices] = True
    return circuit.reshape(edge_mask.shape)


if MAIN:
    tests.test_random_circuit(random_circuit, EDGE_MASK)

# %%

if MAIN:
    tests.test_top_k_circuit(top_k_circuit)

# %%

if MAIN:
    THRESHOLDS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
    sweep = {}
    for tau in tqdm(THRESHOLDS, desc="thresholds"):
        found, n_tested = acdc(circuit_kl, EDGE_MASK, RECEIVER_ORDER, RECEIVER_OWNER, threshold=tau)
        sweep[tau] = {"circuit": found, "edges": int(found.sum()), "tested": n_tested, "kl": circuit_kl(found), "logprob": circuit_logprob(found)}
    
    generator = t.Generator().manual_seed(0)
    sizes = [sweep[tau]["edges"] for tau in THRESHOLDS]
    single_kls = [circuit_kl(top_k_circuit(single_kl, EDGE_MASK, k)) for k in sizes]
    random_kls = [[circuit_kl(random_circuit(EDGE_MASK, k, generator)) for _ in range(20)] for k in sizes]
    
    print(f"{'tau':>6} {'edges':>6} {'tested':>7} {'KL (ACDC)':>10} {'KL (single-edge top-k)':>23} {'KL (random, median)':>20}")
    for tau, s_kl, r_kls in zip(THRESHOLDS, single_kls, random_kls):
        row = sweep[tau]
        print(f"{tau:>6} {row['edges']:>6} {row['tested']:>7} {row['kl']:>10.3f} {s_kl:>23.3f} {np.median(r_kls):>20.3f}")
    
    utils.plot_size_vs_metric(
        {"ACDC (one point per threshold)": (sizes, [sweep[tau]["kl"] for tau in THRESHOLDS]), "top-k by single-edge effect": (sizes, single_kls)},
        title="How far is the circuit from the full model? (lower is better)",
        ylabel="KL divergence from the full model",
        random_band=(sizes, [np.percentile(r, 5) for r in random_kls], [np.median(r) for r in random_kls], [np.percentile(r, 95) for r in random_kls]),
        reference={"empty circuit": circuit_kl(EMPTY)},
    )

# %%

def faithfulness(value: float, full: float, empty: float) -> float:
    """Normalised faithfulness: 1 for the full model, 0 for the empty circuit."""
    return (value - empty) / (full - empty)


if MAIN:
    tests.test_faithfulness(faithfulness)

# %%

if MAIN:
    LP_FULL, LP_EMPTY = circuit_logprob(EDGE_MASK), circuit_logprob(EMPTY)
    
    
    def circuit_faithfulness(circuit: Tensor) -> float:
        return faithfulness(circuit_logprob(circuit), LP_FULL, LP_EMPTY)
    
    
    acdc_faith = [circuit_faithfulness(sweep[tau]["circuit"]) for tau in THRESHOLDS]
    single_faith = [circuit_faithfulness(top_k_circuit(single_kl, EDGE_MASK, k)) for k in sizes]
    generator = t.Generator().manual_seed(1)
    random_faith = [[circuit_faithfulness(random_circuit(EDGE_MASK, k, generator)) for _ in range(50)] for k in sizes]
    
    print(f"{'edges':>6} {'ACDC':>7} {'single-edge top-k':>18} {'random (median, 95th pct)':>27}")
    for k, a, s, r in zip(sizes, acdc_faith, single_faith, random_faith):
        print(f"{k:>6} {a:>7.2f} {s:>18.2f} {np.median(r):>14.2f}, {np.percentile(r, 95):.2f}")
    
    utils.plot_size_vs_metric(
        {"ACDC": (sizes, acdc_faith), "top-k by single-edge effect": (sizes, single_faith)},
        title="Faithfulness on held-out prompts (induction log prob)",
        ylabel="normalised faithfulness",
        random_band=(sizes, [np.percentile(r, 5) for r in random_faith], [np.median(r) for r in random_faith], [np.percentile(r, 95) for r in random_faith]),
        reference={"full model": 1.0, "empty circuit": 0.0},
    )

# %%

def completeness_scores(
    circuit: Tensor, edge_mask: Tensor, faith_fn: Callable[[Tensor], float], n_subsets: int, frac: float, generator: t.Generator
) -> tuple[Float[Tensor, " n"], Float[Tensor, " n"]]:
    circuit_scores, model_scores = [], []
    for _ in range(n_subsets):
        knockout = circuit & (t.rand(circuit.shape, generator=generator) < frac)
        circuit_scores.append(faith_fn(circuit & ~knockout))
        model_scores.append(faith_fn(edge_mask & ~knockout))
    return t.tensor(circuit_scores), t.tensor(model_scores)

# %%

if MAIN:
    tests.test_completeness_scores(completeness_scores)

# %%

if MAIN:
    TAU = 0.1
    chosen = sweep[TAU]["circuit"]
    generator = t.Generator().manual_seed(2)
    c_scores, m_scores = completeness_scores(chosen, EDGE_MASK, circuit_faithfulness, n_subsets=50, frac=0.25, generator=generator)
    rand = random_circuit(EDGE_MASK, int(chosen.sum()), generator)
    rc_scores, rm_scores = completeness_scores(rand, EDGE_MASK, circuit_faithfulness, n_subsets=50, frac=0.25, generator=generator)
    
    gap = (m_scores - c_scores)
    print(f"ACDC circuit (τ = {TAU}, {int(chosen.sum())} edges): mean F(M\\K) - F(C\\K) = {gap.mean():+.3f}, largest gap = {gap.max():+.3f}")
    print(f"Random circuit of the same size:   mean F(M\\K) - F(C\\K) = {(rm_scores - rc_scores).mean():+.3f}")
    utils.plot_completeness(c_scores, m_scores, (rc_scores, rm_scores), f"Completeness test: knock out a random quarter of the circuit (τ = {TAU})")

# %%

def edge_drops(circuit: Tensor, faith_fn: Callable[[Tensor], float]) -> Float[Tensor, " n_edges"]:
    base = faith_fn(circuit)
    drops = []
    for u, v in circuit.nonzero().tolist():
        smaller = circuit.clone()
        smaller[u, v] = False
        drops.append(base - faith_fn(smaller))
    return t.tensor(drops)

# %%

if MAIN:
    tests.test_edge_drops(edge_drops)

# %%

if MAIN:
    drops = {f"ACDC, τ = {tau}": edge_drops(sweep[tau]["circuit"], circuit_faithfulness) for tau in [0.3, 0.1, 0.03]}
    for name, d in drops.items():
        print(f"{name:<14}: {len(d):>3} edges, {int((d < 0.01).sum()):>3} with drop < 0.01, smallest drop {d.min():+.3f}, largest {d.max():+.3f}")
    utils.plot_minimality(drops, "Minimality: how much does each edge matter, given the rest of its circuit?")

# %%

REFERENCE_EDGES = [
    ("embed", "0.7<v>"),  # the previous-token head moves the previous token...
    ("0.7", "1.4<k>"), ("0.7", "1.10<k>"),  # ...into the induction heads' keys (K-composition)
    ("embed", "1.4<q>"), ("embed", "1.10<q>"),  # the induction heads query with the current token
    ("embed", "1.4<v>"), ("embed", "1.10<v>"),  # and copy the token they attend to
    ("1.4", "logits"), ("1.10", "logits"),
    ("embed", "logits"),  # the direct path (bigram statistics), which any faithful circuit needs
]
REFERENCE = t.zeros_like(EDGE_MASK)
for s, r in REFERENCE_EDGES:
    REFERENCE[SENDER_NAMES.index(s), RECEIVER_NAMES.index(r)] = True
print(f"Reference circuit: {int(REFERENCE.sum())} edges")

# %%

def roc_point(circuit: Tensor, reference: Tensor, edge_mask: Tensor) -> tuple[float, float]:
    circuit, reference = circuit & edge_mask, reference & edge_mask
    tpr = (circuit & reference).sum() / reference.sum()
    fpr = (circuit & ~reference).sum() / (edge_mask & ~reference).sum()
    return fpr.item(), tpr.item()


if MAIN:
    tests.test_roc_point(roc_point)

# %%

if MAIN:
    acdc_roc = [roc_point(sweep[tau]["circuit"], REFERENCE, EDGE_MASK) for tau in THRESHOLDS]
    ks = [1, 2, 3, 5, 8, 10, 13, 16, 20, 25, 33, 50, 75, 100, 150, 250, 529]
    single_roc = [roc_point(top_k_circuit(single_kl, EDGE_MASK, k), REFERENCE, EDGE_MASK) for k in ks]
    generator = t.Generator().manual_seed(3)
    random_scores = t.rand(EDGE_MASK.shape, generator=generator)
    random_roc = [roc_point(top_k_circuit(random_scores, EDGE_MASK, k), REFERENCE, EDGE_MASK) for k in ks]
    
    for tau, (fpr, tpr) in zip(THRESHOLDS, acdc_roc):
        print(f"ACDC τ = {tau:<6} edges = {sweep[tau]['edges']:>3}  TPR = {tpr:.2f}  FPR = {fpr:.3f}")
    utils.plot_roc(
        {"ACDC (thresholds)": tuple(zip(*acdc_roc)), "single-edge ranking": tuple(zip(*single_roc)), "random ranking": tuple(zip(*random_roc))},
        "Recovering the [1.2] induction circuit",
    )

# %%

if MAIN:
    generator = t.Generator().manual_seed(10)
    other = t.randint(1000, 10000, (clean.shape[0], HALF_LEN), generator=generator).to(device)
    corrupt_repeat = t.cat([clean[:, :1], other, other], dim=1)
    corrupt_repeat_out = sender_outputs(model, corrupt_repeat)
    
    
    @t.inference_mode()
    def circuit_kl_repeat(circuit: Tensor) -> float:
        return kl_divergence(run_graph(model, clean, corrupt_repeat_out, circuit)[:, INDUCTION_POS], full_logprobs).item()
    
    
    play_circuit, play_tested = acdc(circuit_kl_repeat, EDGE_MASK, RECEIVER_ORDER, RECEIVER_OWNER, threshold=0.3)
    print(f"{int(play_circuit.sum())} edges (tested {play_tested}); KL = {circuit_kl_repeat(play_circuit):.3f}")
    print("In this circuit but not the original:", [f"{SENDER_NAMES[u]}->{RECEIVER_NAMES[v]}" for u, v in (play_circuit & ~circuit).nonzero().tolist()])
    print("In the original but not this one:   ", [f"{SENDER_NAMES[u]}->{RECEIVER_NAMES[v]}" for u, v in (circuit & ~play_circuit).nonzero().tolist()])

# %%

if MAIN:
    gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
    gpt2.requires_grad_(False)
    gpt2.set_use_split_qkv_input(True)  # needed to cut edges into individual heads' q, k and v inputs
    gpt2.set_use_hook_mlp_in(True)
    GPT2_EDGES = gpt2_graph.EDGE_MASK
    
    gt = gpt2_graph.make_greater_than_task(gpt2, 64, seed=0)
    gt_heldout = gpt2_graph.make_greater_than_task(gpt2, 64, seed=1)
    print(gpt2.to_string(gt.clean_tokens[0, 1:]), "|", gpt2.to_string(gt.corrupt_tokens[0, 1:]))
    print(f"{gpt2_graph.N_SENDERS} senders, {gpt2_graph.N_RECEIVERS} receivers, {int(GPT2_EDGES.sum()):,} edges")

# %%

def gpt2_receiver_order() -> list[int]:
    order = [gpt2_graph.LOGITS_RECEIVER]
    for layer in reversed(range(gpt2_graph.N_LAYERS)):
        order.append(gpt2_graph.mlp_receiver(layer))
        for head in reversed(range(gpt2_graph.N_HEADS)):
            order += [gpt2_graph.head_receiver(layer, c, head) for c in "qkv"]
    return order


def gpt2_receiver_owner() -> list[int | None]:
    owner: list[int | None] = [None] * gpt2_graph.N_RECEIVERS
    for layer in range(gpt2_graph.N_LAYERS):
        owner[gpt2_graph.mlp_receiver(layer)] = gpt2_graph.mlp_sender(layer)
        for head in range(gpt2_graph.N_HEADS):
            for c in "qkv":
                owner[gpt2_graph.head_receiver(layer, c, head)] = gpt2_graph.head_sender(layer, head)
    return owner


if MAIN:
    tests.test_gpt2_order(gpt2_receiver_order, gpt2_receiver_owner, gpt2_graph)

# %%

if MAIN:
    gt_corrupt_out = gpt2_graph.sender_activations(gpt2, gt.corrupt_tokens)
    with t.no_grad():
        gt_full_logprobs = gpt2(gt.clean_tokens)[:, -1:].log_softmax(dim=-1)
    
    
    @t.inference_mode()
    def gpt2_kl(circuit: Tensor) -> float:
        logits = utils.gpt2_circuit_logits(gpt2, gt.clean_tokens, gt_corrupt_out, circuit, GPT2_EDGES, gpt2_graph)
        return kl_divergence(logits[:, None], gt_full_logprobs).item()
    
    
    print(f"KL of the full model: {abs(gpt2_kl(GPT2_EDGES)):.4f}, of the empty circuit: {gpt2_kl(t.zeros_like(GPT2_EDGES)):.3f}")

# %%

if MAIN:
    t0 = time.time()
    gt_circuit, gt_tested = acdc(gpt2_kl, GPT2_EDGES, gpt2_receiver_order(), gpt2_receiver_owner(), threshold=0.01, verbose=True)
    print(f"ACDC tested {gt_tested:,} of {int(GPT2_EDGES.sum()):,} edges in {(time.time() - t0) / 60:.1f} min, and kept {int(gt_circuit.sum())}")
    print(f"KL = {gpt2_kl(gt_circuit):.3f}")
    
    gt_nodes = [gpt2_graph.SENDER_NAMES[u] for u in gt_circuit.any(dim=1).nonzero().flatten().tolist()]
    print("Nodes with an edge in the circuit:", gt_nodes)

# %%

if MAIN:
    HANNA_HEADS = ["a5.1", "a5.5", "a6.1", "a6.9", "a7.10", "a8.8", "a8.11", "a9.1"]
    HANNA_MLPS = ["m8", "m9", "m10", "m11"]
    rows = [{"node": n, "Hanna et al.": "yes", "ACDC": "yes" if n in gt_nodes else "no"} for n in HANNA_HEADS + HANNA_MLPS]
    rows += [{"node": n, "Hanna et al.": "no", "ACDC": "yes"} for n in gt_nodes if n not in HANNA_HEADS + HANNA_MLPS]
    display(pd.DataFrame(rows).set_index("node"))
    
    utils.draw_gpt2_circuit(
        gt_circuit, gpt2_graph.SENDER_NAMES, gpt2_graph.RECEIVER_NAMES,
        title=f"Greater-than: the {int(gt_circuit.sum())}-edge circuit ACDC finds in GPT-2 Small (τ = 0.01)",
        highlight={h: "Hanna et al." for h in HANNA_HEADS + HANNA_MLPS},
    )

# %%

if MAIN:
    gt_heldout_corrupt_out = gpt2_graph.sender_activations(gpt2, gt_heldout.corrupt_tokens)
    
    
    @t.inference_mode()
    def gt_metric(circuit: Tensor) -> float:
        logits = utils.gpt2_circuit_logits(gpt2, gt_heldout.clean_tokens, gt_heldout_corrupt_out, circuit, GPT2_EDGES, gpt2_graph)
        return gt_heldout.metric(logits[:, None], gt_heldout.answers).mean().item()
    
    
    GT_FULL, GT_EMPTY = gt_metric(GPT2_EDGES), gt_metric(t.zeros_like(GPT2_EDGES))
    
    
    def node_circuit(nodes: list[str]) -> Tensor:
        """Every valid edge between the given nodes (plus the input embedding and the logits)."""
        keep = set(nodes) | {"input", "logits"}
        circuit = t.zeros_like(GPT2_EDGES)
        for u, v in GPT2_EDGES.nonzero().tolist():
            if gpt2_graph.SENDER_NAMES[u] in keep and gpt2_graph.RECEIVER_NAMES[v].split("<")[0] in keep:
                circuit[u, v] = True
        return circuit
    
    
    hanna_circuit = node_circuit(HANNA_HEADS + HANNA_MLPS)
    hanna_m0_circuit = node_circuit(HANNA_HEADS + HANNA_MLPS + ["m0"])
    
    generator = t.Generator().manual_seed(4)
    n_gt = int(gt_circuit.sum())
    gt_random = [faithfulness(gt_metric(random_circuit(GPT2_EDGES, n_gt, generator)), GT_FULL, GT_EMPTY) for _ in range(20)]
    gt_results = {
        f"ACDC ({n_gt} edges)": faithfulness(gt_metric(gt_circuit), GT_FULL, GT_EMPTY),
        f"Hanna et al. nodes ({int(hanna_circuit.sum())} edges)": faithfulness(gt_metric(hanna_circuit), GT_FULL, GT_EMPTY),
        f"Hanna et al. nodes + MLP0 ({int(hanna_m0_circuit.sum())} edges)": faithfulness(gt_metric(hanna_m0_circuit), GT_FULL, GT_EMPTY),
    }
    print(f"Probability difference: full model {GT_FULL:.3f}, empty circuit {GT_EMPTY:.3f}")
    for name, f in gt_results.items():
        print(f"  {name:<38} faithfulness {f:.2f}")
    print(f"  {'random circuits (' + str(n_gt) + ' edges)':<38} faithfulness median {np.median(gt_random):.2f}, max {max(gt_random):.2f}")
    
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(list(gt_results), list(gt_results.values()), color=["tab:blue", "tab:orange", "tab:green"])
    ax.scatter(gt_random, ["random circuits\n(same size as ACDC)"] * len(gt_random), color="gray", alpha=0.6, zorder=3)
    ax.axvline(1, color="black", ls="--", lw=1, label="full model")
    ax.axvline(0, color="tab:red", ls="--", lw=1, label="empty circuit")
    ax.set_xlabel("normalised faithfulness on held-out prompts (probability difference)")
    ax.set_title("Greater-than circuits in GPT-2 Small")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()

# %%
