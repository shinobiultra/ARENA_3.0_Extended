# %%


import sys
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from IPython.display import display
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter8_automated_circuits"
section = "part8_circuit_benchmarks"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part8_circuit_benchmarks.tests as tests
import part8_circuit_benchmarks.utils as utils

# Methods from earlier sections: EAP / EAP-IG and the GPT-2 graph ([8.2]), AtP and AtP* ([8.3]), ACDC ([8.4])
import part2_attribution_patching_eap.solutions as eap_8_2
import part3_atp_star.solutions as atp_8_3
import part4_acdc_circuit_metrics.solutions as acdc_8_4
import part4_acdc_circuit_metrics.utils as acdc_utils

MAIN = __name__ == "__main__"

# %%

VOCAB, HALF_LEN = 16, 8
SEQ = 2 * HALF_LEN + 1
N_LAYERS, N_HEADS, D_HEAD = 2, 3, 17

SUBSPACES: dict[str, slice] = {}
_start = 0
for _name, _size in [("tok", VOCAB), ("pos", SEQ), ("prev", VOCAB), ("out", VOCAB), ("junk", VOCAB)]:
    SUBSPACES[_name] = slice(_start, _start + _size)
    _start += _size
D_MODEL = _start

print(f"d_model = {D_MODEL}:", {name: (s.start, s.stop) for name, s in SUBSPACES.items()})


def make_empty_model() -> HookedTransformer:
    return utils.empty_attn_only_model(N_LAYERS, N_HEADS, D_MODEL, D_HEAD, VOCAB, SEQ, device)

# %%

def compile_select_aggregate(
    model: HookedTransformer,
    layer: int,
    head: int,
    query_space: slice,
    key_space: slice,
    predicate: Float[Tensor, "query_values key_values"],
    value_space: slice,
    out_space: slice,
    sharpness: float = 20.0,
    amplitude: float = 1.0,
) -> None:
    """Writes head (layer, head) so that it attends according to `predicate` and copies value_space into out_space."""
    attn = model.blocks[layer].attn
    n_query, n_key = predicate.shape
    n_value = value_space.stop - value_space.start
    with t.no_grad():
        attn.W_Q[head][query_space, :n_key] = predicate.to(attn.W_Q.device) * sharpness * model.cfg.d_head**0.5
        attn.W_K[head][key_space, :n_key] = t.eye(n_key, device=attn.W_K.device)
        attn.W_V[head][value_space, :n_value] = t.eye(n_value, device=attn.W_V.device)
        attn.W_O[head][:n_value, out_space] = t.eye(n_value, device=attn.W_O.device) * amplitude


if MAIN:
    tests.test_select_aggregate(compile_select_aggregate, lambda: make_empty_model().cpu(), SUBSPACES)

# %%

def compile_induction() -> HookedTransformer:
    model = make_empty_model()
    tok, pos, prev, out, junk = (SUBSPACES[k] for k in ["tok", "pos", "prev", "out", "junk"])
    with t.no_grad():
        model.embed.W_E[:, tok] = t.eye(VOCAB, device=device)
        model.pos_embed.W_pos[:, pos] = t.eye(SEQ, device=device)
        model.unembed.W_U[out, :] = 10 * t.eye(VOCAB, device=device)

    previous_position = t.zeros(SEQ, SEQ)
    for i in range(1, SEQ):
        previous_position[i, i - 1] = 1
    same_token = t.eye(VOCAB)
    # Layer 0: two redundant previous-token heads, and a distractor that averages all tokens into junk
    for head in [0, 1]:
        compile_select_aggregate(model, 0, head, pos, pos, previous_position, tok, prev)
    compile_select_aggregate(model, 0, 2, pos, pos, t.zeros(SEQ, SEQ), tok, junk)
    # Layer 1: the induction head, and two distractors writing to junk
    compile_select_aggregate(model, 1, 0, tok, prev, same_token, tok, out)
    compile_select_aggregate(model, 1, 1, junk, junk, same_token, junk, junk)
    compile_select_aggregate(model, 1, 2, pos, pos, t.eye(SEQ), tok, junk)
    return model


compiled = compile_induction()
data_generator = t.Generator().manual_seed(0)
clean, corrupt = utils.make_induction_data(64, VOCAB, HALF_LEN, data_generator, device)
POS = slice(HALF_LEN + 1, 2 * HALF_LEN)  # positions where the next token is predictable


def induction_logprob(logits: Tensor) -> Tensor:
    """Mean log probability of the correct next token at the induction positions (uses the global `clean`)."""
    logprobs = logits[:, POS].log_softmax(dim=-1)
    return logprobs.gather(-1, clean[:, POS.start + 1 : POS.stop + 1, None]).mean()


with t.no_grad():
    for name, toks in [("clean", clean), ("corrupted", corrupt)]:
        logits = compiled(toks)
        accuracy = (logits[:, POS].argmax(-1) == clean[:, POS.start + 1 : POS.stop + 1]).float().mean()
        print(f"{name:<10} prompts: log prob of the induction answer {induction_logprob(logits):.4f}, accuracy {accuracy:.1%}")

# %%

if MAIN:
    with t.no_grad():
        _, cache = compiled.run_with_cache(clean[:1])
    str_tokens = ["BOS"] + [str(tok) for tok in clean[0, 1:].tolist()]
    utils.plot_patterns(
        {
            "0.0 (previous token)": cache["blocks.0.attn.hook_pattern"][0, 0],
            "0.1 (backup previous token)": cache["blocks.0.attn.hook_pattern"][0, 1],
            "1.0 (induction)": cache["blocks.1.attn.hook_pattern"][0, 0],
            "1.2 (distractor)": cache["blocks.1.attn.hook_pattern"][0, 2],
        },
        str_tokens,
        "Attention patterns of the compiled model on one clean prompt",
    )

# %%

GROUND_TRUTH_EDGES = [
    ("input", "0.0<v>"), ("input", "0.1<v>"), ("0.0", "1.0<k>"), ("0.1", "1.0<k>"),
    ("input", "1.0<q>"), ("input", "1.0<v>"), ("1.0", "logits"),
]

# %%

def add_weight_noise(model: HookedTransformer, sigma: float, seed: int = 0) -> HookedTransformer:
    generator = t.Generator().manual_seed(seed)
    with t.no_grad():
        for name, param in model.named_parameters():
            if name.split(".")[-1] in ("W_Q", "W_K", "W_V", "W_O"):
                param.add_(t.randn(param.shape, generator=generator).to(param.device) * sigma)
    return model


with t.no_grad():
    for sigma in [0.0, 0.01, 0.03, 0.1]:
        noisy = add_weight_noise(compile_induction(), sigma)
        logits = noisy(clean)
        accuracy = (logits[:, POS].argmax(-1) == clean[:, POS.start + 1 : POS.stop + 1]).float().mean()
        print(f"sigma = {sigma:<5} log prob {induction_logprob(logits):8.4f}   accuracy {accuracy:.1%}")

# %%

def patching_scores(graph: utils.EdgeGraph, tokens: Tensor, corrupt_out: Tensor, metric: Callable[[Tensor], Tensor]) -> Tensor:
    """m(all edges) - m(all edges except (u, v)), for every valid edge."""
    scores = t.zeros(graph.edge_mask.shape)
    with t.no_grad():
        full = metric(graph.run(tokens, corrupt_out, graph.edge_mask)).item()
        for u, v in graph.edge_mask.nonzero().tolist():
            circuit = graph.edge_mask.clone()
            circuit[u, v] = False
            scores[u, v] = full - metric(graph.run(tokens, corrupt_out, circuit)).item()
    return scores


def eap_scores(graph: utils.EdgeGraph, tokens: Tensor, corrupt_out: Tensor, metric: Callable[[Tensor], Tensor]) -> Tensor:
    """d m / d keep[u, v] at keep = 1, for every valid edge."""
    keep = graph.edge_mask.float().to(tokens.device).requires_grad_(True)
    metric(graph.run(tokens, corrupt_out, keep)).backward()
    return keep.grad.cpu() * graph.edge_mask


def eap_ig_scores(
    graph: utils.EdgeGraph, tokens: Tensor, corrupt_out: Tensor, metric: Callable[[Tensor], Tensor], steps: int = 10
) -> Tensor:
    """Average of d m / d keep[u, v] over keep = alpha * 1, for alpha at the midpoints of `steps` intervals of [0, 1]."""
    total = t.zeros(graph.edge_mask.shape)
    for k in range(steps):
        keep = (graph.edge_mask.float() * (k + 0.5) / steps).to(tokens.device).requires_grad_(True)
        metric(graph.run(tokens, corrupt_out, keep)).backward()
        total += keep.grad.cpu()
    return total / steps * graph.edge_mask


if MAIN:
    _graph = utils.EdgeGraph(add_weight_noise(compile_induction(), 0.03))
    tests.test_edge_methods(patching_scores, eap_scores, eap_ig_scores, _graph, clean[:16], corrupt[:16],
                            lambda logits: logits[:, POS].log_softmax(-1).gather(-1, clean[:16, POS.start + 1 : POS.stop + 1, None]).mean())

# %%

def roc_auc(scores: Tensor, ground_truth: Tensor, edge_mask: Tensor) -> float:
    values = scores[edge_mask].abs().float()
    labels = ground_truth[edge_mask]
    pos, neg = values[labels], values[~labels]
    greater = (pos[:, None] > neg[None, :]).float()
    ties = (pos[:, None] == neg[None, :]).float()
    return (greater + 0.5 * ties).mean().item()


if MAIN:
    tests.test_roc_auc(roc_auc)

# %%

if MAIN:
    def benchmark(sigma: float, metric_kind: str = "logprob", acdc_thresholds: tuple = ()) -> dict:
        """Scores every edge of the compiled model with noise sigma; returns scores, AUCs and ACDC circuits."""
        model = add_weight_noise(compile_induction(), sigma)
        graph = utils.EdgeGraph(model)
        corrupt_out = graph.sender_outputs(corrupt)
        truth = graph.mask_from_edges(GROUND_TRUTH_EDGES)
        targets = clean[:, POS.start + 1 : POS.stop + 1, None]
    
        def metric(logits: Tensor) -> Tensor:
            if metric_kind == "logprob":
                return logits[:, POS].log_softmax(-1).gather(-1, targets).mean()
            logits = logits[:, POS]
            return (logits.gather(-1, targets)[..., 0] - logits.mean(-1)).mean()
    
        timings, scores = {}, {}
        for name, fn in [("exact patching", patching_scores), ("EAP", eap_scores), ("EAP-IG (10 steps)", eap_ig_scores)]:
            t0 = time.time()
            scores[name] = fn(graph, clean, corrupt_out, metric)
            timings[name] = time.time() - t0
        result = {"graph": graph, "truth": truth, "scores": scores, "timings": timings,
                  "auc": {name: roc_auc(s, truth, graph.edge_mask) for name, s in scores.items()}}
    
        with t.no_grad():
            full_lp = model(clean)[:, POS].log_softmax(-1)
    
        def kl(circuit: Tensor) -> float:
            with t.no_grad():
                lp = graph.run(clean, corrupt_out, circuit)[:, POS].log_softmax(-1)
            return (full_lp.exp() * (full_lp - lp)).sum(-1).mean().item()
    
        order = [graph.logits] + [graph.receiver(l, c, h) for l in reversed(range(N_LAYERS)) for h in reversed(range(N_HEADS)) for c in "qkv"]
        owner = [graph.sender(l, h) for l in range(N_LAYERS) for c in "qkv" for h in range(N_HEADS)] + [None]
        result["acdc"] = {tau: acdc_8_4.acdc(kl, graph.edge_mask, order, owner, tau)[0] for tau in acdc_thresholds}
        return result
    
    
    clean_bench = benchmark(0.0, acdc_thresholds=(0.001, 0.01, 0.1))
    truth, graph = clean_bench["truth"], clean_bench["graph"]
    for name, score in clean_bench["scores"].items():
        gt_values = [round(score[graph.index(s, r)].item(), 4) for s, r in GROUND_TRUTH_EDGES]
        print(f"{name:<18} AUC {clean_bench['auc'][name]:.3f}   ground-truth edge scores {gt_values}")
    for tau, circuit in clean_bench["acdc"].items():
        print(f"ACDC tau = {tau:<6} {int(circuit.sum())} edges, {int((circuit & truth).sum())}/7 ground-truth edges, {int((circuit & ~truth).sum())} others")

# %%

if MAIN:
    SIGMAS = [0.0, 0.01, 0.03, 0.1]
    noise_benches = {sigma: (clean_bench if sigma == 0.0 else benchmark(sigma, acdc_thresholds=(0.01,))) for sigma in SIGMAS}
    aucs = {name: [noise_benches[s]["auc"][name] for s in SIGMAS] for name in clean_bench["scores"]}
    costs = {"exact patching": "52 forward passes", "EAP": "1 forward + 1 backward", "EAP-IG (10 steps)": "10 forward + 10 backward"}
    utils.plot_auc_vs_noise(SIGMAS, aucs, "Recovering the compiled induction circuit as the weights get noisier", costs)
    
    print(f"{'sigma':>6} " + " ".join(f"{name:>18}" for name in aucs) + "   ACDC (tau=0.01): ground truth / others")
    for i, sigma in enumerate(SIGMAS):
        circuit = noise_benches[sigma]["acdc"][0.01]
        tr = noise_benches[sigma]["truth"]
        print(f"{sigma:>6} " + " ".join(f"{aucs[name][i]:>18.3f}" for name in aucs) + f"   {int((circuit & tr).sum())}/7 / {int((circuit & ~tr).sum())}")

# %%

if MAIN:
    node_model = add_weight_noise(compile_induction(), 0.03)
    node_model.set_use_split_qkv_input(False)
    targets = clean[:, POS.start + 1 : POS.stop + 1, None]
    
    
    def logprob_metric(logits: Tensor, answers: Tensor) -> Tensor:
        return logits[:, POS].log_softmax(-1).gather(-1, targets)[..., 0].mean(-1)
    
    
    def logit_metric(logits: Tensor, answers: Tensor) -> Tensor:
        logits = logits[:, POS]
        return (logits.gather(-1, targets)[..., 0] - logits.mean(-1)).mean(-1)
    
    
    clean_nodes, corrupt_nodes = atp_8_3.cache_nodes(node_model, clean), atp_8_3.cache_nodes(node_model, corrupt)
    node_results = {}
    for metric_name, metric_fn in [("log prob", logprob_metric), ("logit", logit_metric)]:
        task = eap_8_2.Task(clean, corrupt, clean, metric_fn)
        grads = atp_8_3.node_gradients(node_model, task)
        node_results[metric_name] = {
            "exact": atp_8_3.exact_node_effects(node_model, task, clean_nodes, corrupt_nodes).cpu(),
            "AtP": atp_8_3.atp_estimates(clean_nodes, corrupt_nodes, grads).mean(-1).cpu(),
            "AtP + QK fix": atp_8_3.atp_qk_fixed(clean_nodes, corrupt_nodes, grads).mean(-1).cpu(),
        }
    
    utils.plot_node_scores(
        {f"{method}, {metric_name} metric": node_results[metric_name][method] for metric_name in ["log prob", "logit"] for method in ["exact", "AtP", "AtP + QK fix"]},
        atp_8_3.KINDS,
        "Node effects in the compiled model (noise 0.03): exact patching vs first-order estimates",
    )

# %%

if MAIN:
    rows = []
    for kind, layer, head in [("v", 0, 0), ("v", 0, 1), ("q", 1, 0), ("k", 1, 0), ("v", 1, 0), ("z", 1, 0)]:
        k = atp_8_3.KINDS.index(kind)
        row = {"node": f"{kind}{layer}.{head}"}
        for metric_name in ["log prob", "logit"]:
            for method in ["exact", "AtP", "AtP + QK fix"]:
                row[f"{metric_name}: {method}"] = round(node_results[metric_name][method][k, layer, head].item(), 3)
        rows.append(row)
    display(pd.DataFrame(rows).set_index("node"))

# %%

if MAIN:
    logit_benches = {sigma: benchmark(sigma, metric_kind="logit") for sigma in SIGMAS}
    logit_aucs = {name: [logit_benches[s]["auc"][name] for s in SIGMAS] for name in clean_bench["scores"]}
    print(f"{'sigma':>6} " + " ".join(f"{name + ' (logit)':>26}" for name in logit_aucs))
    for i, sigma in enumerate(SIGMAS):
        print(f"{sigma:>6} " + " ".join(f"{logit_aucs[name][i]:>26.3f}" for name in logit_aucs))
    eap_gt = logit_benches[0.03]["scores"]["EAP"]
    print("\nEAP (logit metric, sigma = 0.03) on the ground-truth edges:",
          {f"{s}->{r}": round(eap_gt[graph.index(s, r)].item(), 3) for s, r in GROUND_TRUTH_EDGES})

# %%

if MAIN:
    gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
    gpt2.requires_grad_(False)
    gpt2.set_use_split_qkv_input(True)  # as in [8.2] and [8.4]: hooks on each head's q/k/v input
    gpt2.set_use_hook_mlp_in(True)
    gt = eap_8_2.make_greater_than_task(gpt2, 64, seed=0)
    gt_heldout = eap_8_2.make_greater_than_task(gpt2, 64, seed=1)
    EDGES = eap_8_2.EDGE_MASK
    print(f"{int(EDGES.sum()):,} edges; clean metric {eap_8_2.mean_metric(gpt2, gt, gt.clean_tokens):.3f}, corrupt metric {eap_8_2.mean_metric(gpt2, gt, gt.corrupt_tokens):.3f}")
    
    gpt2_timings, gpt2_scores = {}, {}
    t0 = time.time()
    gpt2_scores["EAP"] = eap_8_2.eap_scores(gpt2, gt).cpu()
    gpt2_timings["EAP"] = time.time() - t0
    t0 = time.time()
    gpt2_scores["EAP-IG"] = eap_8_2.eap_ig_scores(gpt2, gt, n_steps=5).cpu()
    gpt2_timings["EAP-IG"] = time.time() - t0
    
    gt_corrupt_out = eap_8_2.sender_activations(gpt2, gt.corrupt_tokens)
    with t.no_grad():
        gt_full_logprobs = gpt2(gt.clean_tokens)[:, -1:].log_softmax(-1)
    
    
    @t.inference_mode()
    def gpt2_kl(circuit: Tensor) -> float:
        logits = acdc_utils.gpt2_circuit_logits(gpt2, gt.clean_tokens, gt_corrupt_out, circuit, EDGES, eap_8_2)
        return acdc_8_4.kl_divergence(logits[:, None], gt_full_logprobs).item()
    
    
    t0 = time.time()
    acdc_circuit, acdc_tested = acdc_8_4.acdc(gpt2_kl, EDGES, acdc_8_4.gpt2_receiver_order(), acdc_8_4.gpt2_receiver_owner(), threshold=0.01)
    gpt2_timings["ACDC"] = time.time() - t0
    for name, seconds in gpt2_timings.items():
        print(f"{name:<7} {seconds:6.1f}s")
    print(f"ACDC kept {int(acdc_circuit.sum())} edges after testing {acdc_tested:,}")

# %%

if MAIN:
    def jaccard(a: Tensor, b: Tensor) -> float:
        return ((a & b).sum() / (a | b).sum()).item()
    
    
    k = int(acdc_circuit.sum())
    circuits = {
        "ACDC": acdc_circuit.cpu(),
        "EAP": eap_8_2.top_k_circuit(gpt2_scores["EAP"], k).cpu(),
        "EAP-IG": eap_8_2.top_k_circuit(gpt2_scores["EAP-IG"], k).cpu(),
    }
    generator = t.Generator().manual_seed(0)
    random_circuits = [acdc_8_4.random_circuit(EDGES, k, generator) for _ in range(10)]
    agreement = {f"{a} vs {b}": jaccard(circuits[a], circuits[b]) for a, b in [("ACDC", "EAP"), ("ACDC", "EAP-IG"), ("EAP", "EAP-IG")]}
    agreement["random vs random"] = jaccard(random_circuits[0], random_circuits[1])
    
    full = eap_8_2.mean_metric(gpt2, gt_heldout, gt_heldout.clean_tokens)
    empty = eap_8_2.mean_metric(gpt2, gt_heldout, gt_heldout.corrupt_tokens)
    faithful = {name: (eap_8_2.run_with_circuit(gpt2, gt_heldout, c) - empty) / (full - empty) for name, c in circuits.items()}
    faithful["random (median of 10)"] = float(np.median([(eap_8_2.run_with_circuit(gpt2, gt_heldout, c) - empty) / (full - empty) for c in random_circuits]))
    
    print(f"circuit size k = {k}")
    for name, value in agreement.items():
        print(f"  Jaccard {name:<18} {value:.2f}")
    for name, value in faithful.items():
        print(f"  faithfulness {name:<22} {value:.2f}")

# %%

if MAIN:
    def compile_soft_induction(sharpness: float = 2.0) -> HookedTransformer:
        """The induction program with soft attention everywhere."""
        model = make_empty_model()
        tok, pos, prev, out, junk = (SUBSPACES[k] for k in ["tok", "pos", "prev", "out", "junk"])
        with t.no_grad():
            model.embed.W_E[:, tok] = t.eye(VOCAB, device=device)
            model.pos_embed.W_pos[:, pos] = t.eye(SEQ, device=device)
            model.unembed.W_U[out, :] = 10 * t.eye(VOCAB, device=device)
        previous_position = t.diag(t.ones(SEQ - 1), diagonal=-1)
        for head in [0, 1]:
            compile_select_aggregate(model, 0, head, pos, pos, previous_position, tok, prev, sharpness=sharpness)
        compile_select_aggregate(model, 0, 2, pos, pos, t.zeros(SEQ, SEQ), tok, junk)
        compile_select_aggregate(model, 1, 0, tok, prev, t.eye(VOCAB), tok, out, sharpness=sharpness)
        compile_select_aggregate(model, 1, 1, junk, junk, t.eye(VOCAB), junk, junk)
        compile_select_aggregate(model, 1, 2, pos, pos, t.eye(SEQ), tok, junk)
        return model
    
    
    original_compile = compile_induction
    compile_induction = compile_soft_induction
    try:
        soft = benchmark(0.03)
    finally:
        compile_induction = original_compile
    with t.no_grad():
        print(f"soft attention: log prob of the answer {induction_logprob(add_weight_noise(compile_soft_induction(), 0.03)(clean)):.3f}")
    for name, auc in soft["auc"].items():
        print(f"  {name:<18} AUC {auc:.3f}")

# %%
