<!-- CELL: markdown -->
# [8.6] AtP* from Scratch

**By the end of this notebook, you will have shown that AtP* repairs two concrete false negatives in vanilla attribution patching - saturated attention and cancellation - and that its ranking better predicts exhaustive causal interventions in a pinned 70M transformer.**

```python
EXERCISE_ID = "8_6_atp_star_from_scratch"
GT_TIER = "GT-0"
DIFFICULTY = 5
IMPORTANCE = 5
EXPECTED_RUNTIME = "2-3 hours; the cached CPU reference run takes about 90 seconds"
REQUIRES_GPU = False
```

## Core Question

**Can two forward caches and a small number of backward passes find the components exhaustive activation patching would rank highest - without silently missing a saturated query or a cancellation path?**

This section continues original ARENA's circuit workflow: define a crisp behavior, make predictions, intervene exactly, use a cheaper exploratory tool, and then grade the approximation against the causal experiment. AtP* is a **screening method**. Its top candidates still go back through activation patching before they become a circuit claim.

## Learning Objectives

1. Cache aligned clean and corrupt activations at named component sites.
2. Build an exhaustive activation-patching oracle with component batching.
3. Derive vanilla AtP as `(corrupt - clean) * clean_gradient`.
4. Repair saturated Q/K sites by recomputing the attention softmax exactly.
5. Repair direct/indirect gradient cancellation with GradDrop.
6. Distinguish node-level AtP* from an EAP-style immediate-edge baseline.
7. Evaluate rankings with recovery@k, rank correlation, joint faithfulness, runtime, and tensor storage.
8. Test the same logic on 96 named Q/K sites and 24 clean/corrupt examples in a pinned TransformerLens model.

## Cold Open - two components with a score of zero

The toy graph has eight named patch sites. Its attention head is nearly saturated on a signal value.
Replacing the query flips attention and changes the metric by almost `-3`, but the clean softmax
derivative is almost zero. A second component writes `x` directly and `-x^2 / 2` through another
path. At `x=1`, those two gradients cancel exactly even though replacing `x` by zero changes the
metric by `-0.5`.

Before running anything, write down two predictions:

1. Which component should the QK correction rescue?
2. Which dropped backward path should reveal the cancellation source?

The point is not that gradients are useless. The value site is linear and vanilla AtP gets it exactly. The point is that a cheap approximation needs targeted diagnostics.

<!-- CELL: code -->
import json
import math
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from torch import Tensor
from typing import Any, Literal

chapter = "chapter8_automated_circuits"
section = "part6_atp_star"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
assets_dir = root_dir / chapter / "instructions" / "assets"
for path in (root_dir, exercises_dir):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import part6_atp_star.solutions as reference
import part6_atp_star.tests as tests
from part6_atp_star.solutions import (
    MethodEvaluation,
    TOY_CLEAN_VALUES,
    TOY_COMPONENT_INDEX,
    TOY_COMPONENTS,
    TOY_CORRUPT_VALUES,
    TOY_GRAD_PATHS,
    TOY_QK_COMPONENTS,
    ToyAtPStarModel,
    ToyRun,
    evaluate_toy_method,
    toy_site_tensor,
)

t.set_grad_enabled(True)
toy_model = ToyAtPStarModel()
clean_sites = toy_site_tensor(TOY_CLEAN_VALUES)
corrupt_sites = toy_site_tensor(TOY_CORRUPT_VALUES)
print(f"{len(TOY_COMPONENTS)} named sites; gradient controls: {TOY_GRAD_PATHS}")

<!-- CELL: markdown -->
### Exercise - cache the clean and corrupt worlds

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> Suggested time: 10 minutes
> ```

Implement `cache_clean_and_corrupt`. These two aligned runs are the reference points for every later approximation. Preserve the intermediate attention logits, pattern, branch outputs, and scalar metric; do not reduce the cache to the final answer.

<details><summary>Expected output</summary>

```text
All tests in `test_clean_corrupt_caches` passed!
clean metric = 2.900; corrupt metric = 0.000
clean attention pattern is approximately [1.0, 0.0]
```

</details>

<details><summary>Help - The model already returns its cache</summary>

Run both site tensors under `torch.no_grad()` and return the two `ToyRun` objects. Check the shapes before computing either endpoint.

</details>

<details><summary>Common bug</summary>

Detaching or copying only the metric discards `attention_pattern`, the activation whose gradient the QK correction needs.

</details>

<details><summary>Solution</summary>

{{solution:cache_clean_and_corrupt}}

</details>

<!-- CELL: code -->
# EXERCISE: cache_clean_and_corrupt

tests.test_clean_corrupt_caches(cache_clean_and_corrupt)
clean_run, corrupt_run = cache_clean_and_corrupt(toy_model, clean_sites, corrupt_sites)
print(f"clean metric = {clean_run.metric.item():.3f}; corrupt metric = {corrupt_run.metric.item():.3f}")
print("clean attention:", [round(x, 8) for x in clean_run.activations["attention_pattern"].tolist()])

<!-- CELL: markdown -->
### Exercise - build the exhaustive oracle in component batches

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> Suggested time: 20 minutes
> ```

Implement `batched_exact_patch_effects`. For each named site, replace its clean value by the corrupt value and measure `patched_metric - clean_metric`. Evaluate several independent interventions in one tensor batch. This is the causal ground truth AtP* will be judged against.

<details><summary>Expected output</summary>

```text
All tests in `test_batched_exact_patch_effects` passed!
All tests in `test_batched_exact_patch_validation` passed!
Q effect = -3.000; cancel effect = -0.500; null effect = 0.000
```

</details>

<details><summary>Help - Rows are interventions; columns are sites</summary>

Repeat the clean vector once per component in the current chunk. Advanced-index the diagonal `(row, component_index)` entries with corrupt values, run the batch once, then map rows back to names.

</details>

<details><summary>Common bug</summary>

Patching all sites in a chunk into every row measures one joint intervention, not one independent intervention per row.

</details>

<details><summary>Solution</summary>

{{solution:batched_exact_patch_effects}}

</details>

<!-- CELL: code -->
# EXERCISE: batched_exact_patch_effects

tests.test_batched_exact_patch_effects(batched_exact_patch_effects)
tests.test_batched_exact_patch_validation(batched_exact_patch_effects)
exact_effects = batched_exact_patch_effects(toy_model, clean_sites, corrupt_sites)
print({name: round(value, 6) for name, value in exact_effects.items()})

<!-- CELL: markdown -->
### Exercise - compute vanilla attribution patching

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
> Suggested time: 20 minutes
> ```

Implement `_toy_gradient_run` and `vanilla_attribution_patching`. Retain gradients on the site tensor and nonlinear intermediates, backpropagate the clean metric once, then compute the first-order estimate `(corrupt - clean) * grad`. Do not take absolute values yet: the sign is useful for debugging.

<details><summary>Expected output</summary>

```text
All tests in `test_vanilla_attribution_patching` passed!
Q: -0.0000016 despite exact -3.000
cancel source: 0.000 despite exact -0.500
V signal: -2.000, matching exact
```

</details>

<details><summary>Help - Make the site vector the leaf</summary>

Clone and detach the clean sites, call `requires_grad_(True)`, retain gradients on the intermediates needed by later exercises, and call `metric.backward()` exactly once.

</details>

<details><summary>Common bug</summary>

Using the corrupt gradient changes the Taylor expansion point. Summing gradients before multiplying by each site's own displacement also destroys component identity.

</details>

<details><summary>Solution</summary>

{{solution:_toy_gradient_run,vanilla_attribution_patching}}

</details>

<!-- CELL: code -->
# EXERCISE: _toy_gradient_run,vanilla_attribution_patching

tests.test_vanilla_attribution_patching(vanilla_attribution_patching)
vanilla_scores = vanilla_attribution_patching(toy_model, clean_sites, corrupt_sites)
print({name: f"{value:.7f}" for name, value in vanilla_scores.items()})

<!-- CELL: markdown -->
### Exercise - repair saturated attention with the QK correction

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> Suggested time: 25 minutes
> ```

Implement `qk_softmax_correction`. Patch a query or key in activation space, recompute the attention softmax exactly, subtract the clean pattern, and dot that pattern change with `d metric / d pattern`. This is Equations 7 and 10 of the AtP* paper in a two-key graph.

Notice what remains approximate: only computation *after* the attention pattern. The saturated softmax itself is no longer linearized.

<details><summary>Expected output</summary>

```text
All tests in `test_qk_softmax_correction` passed!
All tests in `test_qk_softmax_correction_requires_gradient` passed!
corrected Q = -3.000; each corrected K = -1.500
```

</details>

<details><summary>Help - Three counterfactual patterns, one downstream gradient</summary>

Construct a query-patched pattern and one pattern for each key patch. The downstream dot product is `(patched_pattern - clean_pattern) * pattern_grad`, summed over source positions.

</details>

<details><summary>Common bug</summary>

Multiplying a patched logit difference by `d metric / d logit` still linearizes through softmax. AtP* needs the exact *probability* change.

</details>

<details><summary>Solution</summary>

{{solution:qk_softmax_correction}}

</details>

<!-- CELL: code -->
# EXERCISE: qk_softmax_correction

tests.test_qk_softmax_correction(qk_softmax_correction)
tests.test_qk_softmax_correction_requires_gradient(qk_softmax_correction)
gradient_run, _ = _toy_gradient_run(toy_model, clean_sites)
print(qk_softmax_correction(gradient_run, clean_sites, corrupt_sites))

<!-- CELL: markdown -->
### Exercise - break cancellation with GradDrop

> ```yaml
> Difficulty: 🔴🔴🔴🔴🔴
> Importance: 🔵🔵🔵🔵🔵
> Suggested time: 30 minutes
> ```

Implement `atp_star_scores`. Repeat the backward pass once for each residual contribution, zeroing that contribution's gradient while preserving its forward value. Replace Q/K estimates with the corrected pattern-dot-gradient values for the same drop. Aggregate per-example magnitudes with the paper's `1 / (L - 1)` scaling.

For three toy paths, dropping `cancellation` removes the `-x` gradient and exposes the direct `+1` gradient. The other two drops still cancel to zero, so `(1 + 0 + 0) / 2 = 0.5`, exactly the intervention magnitude.

<details><summary>Expected output</summary>

```text
All tests in `test_atp_star_scores` passed!
All tests in `test_atp_star_grad_path_validation` passed!
AtP* exactly matches every toy intervention magnitude
drop cancellation -> cancel-source signed estimate = -1.000
```

</details>

<details><summary>Help - Forward values stay; only backward paths disappear</summary>

Call `_toy_gradient_run(..., drop_grad_paths=(path,))`. If dropping attention yields no pattern gradient, represent that missing gradient as zeros. Sum absolute per-drop estimates and divide by `len(paths) - 1`.

</details>

<details><summary>Common bug</summary>

Zeroing the forward activation is an ablation experiment, not GradDrop. It changes the clean behavior and invalidates the shared Taylor point.

</details>

<details><summary>Solution</summary>

{{solution:atp_star_scores}}

</details>

<!-- CELL: code -->
# EXERCISE: atp_star_scores

tests.test_atp_star_scores(atp_star_scores)
tests.test_atp_star_grad_path_validation(atp_star_scores)
atp_star_estimates, per_drop = atp_star_scores(toy_model, clean_sites, corrupt_sites)
print({name: round(value, 6) for name, value in atp_star_estimates.items()})
print("cancel source by dropped path:", {path: per_drop[path]["L0M0.cancel_source"] for path in TOY_GRAD_PATHS})

<!-- CELL: markdown -->
### Exercise - compare a distinct EAP-style edge baseline

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> Suggested time: 20 minutes
> ```

Implement `eap_style_edge_scores`. Attribute each immediate sender-to-receiver message, then sum absolute outgoing edge scores for a component. This avoids cancellation between the two outgoing edges of `cancel_source`, but it still linearizes at saturated attention logits. The comparison is useful because node AtP* and edge attribution answer different structural questions.

<details><summary>Expected output</summary>

```text
All tests in `test_eap_style_edge_scores` passed!
cancel source aggregated edge score = 1.000 (detected but overestimated)
query aggregated edge score < 0.000002 (still missed)
```

</details>

<details><summary>Help - Separate the two cancel-source receivers</summary>

Use the retained receiver gradients. For a Q-to-logit edge, the message displacement is `delta_q * clean_key`; for a value edge it is `clean_pattern * delta_value`.

</details>

<details><summary>Common bug</summary>

Using the total site gradient for both outgoing edges recreates the cancellation you are trying to diagnose.

</details>

<details><summary>Solution</summary>

{{solution:eap_style_edge_scores}}

</details>

<!-- CELL: code -->
# EXERCISE: eap_style_edge_scores

tests.test_eap_style_edge_scores(eap_style_edge_scores)
eap_edges, eap_scores = eap_style_edge_scores(toy_model, clean_sites, corrupt_sites)
print({name: round(value, 7) for name, value in eap_scores.items()})

<!-- CELL: markdown -->
### Exercise - grade rankings and intervene on their prefixes

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
> Suggested time: 30 minutes
> ```

Implement the evaluation stack: stable magnitude ranking, Spearman correlation with ties, cumulative exact-effect recovery@k, and joint top-k faithfulness. Recovery@k grades screening order against individual causal effects. Faithfulness actually patches the whole prefix and therefore exposes interaction, including overshoot and non-monotonicity.

<details><summary>Expected output</summary>

```text
All ranking and faithfulness tests passed!
Spearman: exact=1.000, AtP*=1.000, vanilla=0.627, EAP-style=0.563, random=-0.850
Recovery@4: exact/AtP*=0.899, vanilla=0.775, EAP-style=0.663, random=0.101
The exact top-1 faithfulness is 1.034 and the curve is not monotone
```

</details>

<details><summary>Help - Do not confuse effect mass with joint behavior</summary>

For recovery, sum absolute exact singleton effects selected by the proposed order. For faithfulness, patch selected sites together and divide the joint effect by the all-sites corrupt effect.

</details>

<details><summary>Common bug</summary>

Summing selected singleton effects and calling it faithfulness assumes additivity, exactly the assumption this toy graph violates.

</details>

<details><summary>Solution</summary>

{{solution:rank_components,_average_ranks,spearman_rank_correlation,recovery_at_k,topk_faithfulness_curve}}

</details>

<!-- CELL: code -->
# EXERCISE: rank_components,_average_ranks,spearman_rank_correlation,recovery_at_k,topk_faithfulness_curve

tests.test_rank_components(rank_components)
tests.test_spearman_rank_correlation(spearman_rank_correlation)
tests.test_recovery_at_k(recovery_at_k)
tests.test_topk_faithfulness_is_not_additive(topk_faithfulness_curve)

<!-- CELL: markdown -->
## Signature Result - exact effects, recovered ranking, and causal prefixes

The next cell recomputes every method from your functions. The left panel makes both false negatives
legible. The middle panels answer whether the ranking spends a limited verification budget well and
whether its selected prefix actually changes behavior. The final panel shows the measured speed/cost
tradeoff on this tiny graph; use the pass counts, not microsecond differences, when reasoning about
scale.

<!-- CELL: code -->
def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000


(exact_effects, exact_ms) = timed(lambda: batched_exact_patch_effects(toy_model, clean_sites, corrupt_sites, component_batch_size=4))
(vanilla_scores, vanilla_ms) = timed(lambda: vanilla_attribution_patching(toy_model, clean_sites, corrupt_sites))
((atp_star_estimates, per_drop), atp_star_ms) = timed(lambda: atp_star_scores(toy_model, clean_sites, corrupt_sites))
((eap_edges, eap_scores), eap_ms) = timed(lambda: eap_style_edge_scores(toy_model, clean_sites, corrupt_sites))
random_scores = {name: float(i) for i, name in enumerate(TOY_COMPONENTS)}

method_scores = {
    "Exact": exact_effects,
    "Vanilla AtP": vanilla_scores,
    "AtP*": atp_star_estimates,
    "EAP-style": eap_scores,
    "Random": random_scores,
}
method_times = {"Exact": exact_ms, "Vanilla AtP": vanilla_ms, "AtP*": atp_star_ms, "EAP-style": eap_ms, "Random": 0.0}
rankings = {name: rank_components(scores) for name, scores in method_scores.items()}
recoveries = {name: recovery_at_k(exact_effects, ranking) for name, ranking in rankings.items()}
faithfulness = {name: topk_faithfulness_curve(toy_model, clean_sites, corrupt_sites, ranking) for name, ranking in rankings.items()}
spearman = {name: spearman_rank_correlation(exact_effects, scores) for name, scores in method_scores.items()}

short = {
    "L0H0.Q@dest": "Q",
    "L0H0.K@signal": "K signal",
    "L0H0.K@decoy": "K decoy",
    "L0H0.V@signal": "V signal",
    "L0H0.V@decoy": "V decoy",
    "L0M0.cancel_source": "cancel",
    "L1M0.linear_source": "linear",
    "L1M1.null_source": "null",
}
order = rankings["Exact"]
x = np.arange(len(order))
width = 0.2
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
fig.patch.set_facecolor("#f7f7f4")

for offset, method in zip((-1.5, -0.5, 0.5, 1.5), ("Exact", "Vanilla AtP", "AtP*", "EAP-style"), strict=True):
    axes[0, 0].bar(x + offset * width, [abs(method_scores[method][name]) for name in order], width, label=method)
axes[0, 0].set_yscale("symlog", linthresh=1e-5)
axes[0, 0].set_xticks(x, [short[name] for name in order], rotation=25, ha="right")
axes[0, 0].set_ylabel("absolute component score")
axes[0, 0].set_title("Two false negatives, two targeted repairs", fontweight="bold")
axes[0, 0].legend(frameon=False, ncols=2, fontsize=8)

palette = {"Exact": "#333333", "Vanilla AtP": "#c45a4a", "AtP*": "#187a79", "EAP-style": "#d28e2c", "Random": "#9ea7ad"}
for method in method_scores:
    axes[0, 1].plot(range(1, 9), recoveries[method], marker="o", label=method, color=palette[method])
axes[0, 1].set(xlabel="components sent to exact verification", ylabel="recovery@k", ylim=(-0.03, 1.04))
axes[0, 1].set_title("AtP* spends the verification budget well", fontweight="bold")
axes[0, 1].legend(frameon=False, fontsize=8)

for method in method_scores:
    axes[1, 0].plot(range(1, 9), faithfulness[method], marker="o", label=method, color=palette[method])
axes[1, 0].axhline(1.0, color="#333333", linestyle="--", linewidth=1)
axes[1, 0].set(xlabel="jointly patched top-k", ylabel="normalized causal effect")
axes[1, 0].set_title("Joint faithfulness reveals interaction", fontweight="bold")

costs = {"Exact": (3, 0), "Vanilla AtP": (2, 1), "AtP*": (2, 3), "EAP-style": (2, 1)}
for method, (forward, backward) in costs.items():
    axes[1, 1].scatter(forward + backward, recoveries[method][3], s=130, color=palette[method])
    axes[1, 1].annotate(f"{method}\n{method_times[method]:.2f} ms", (forward + backward, recoveries[method][3]), xytext=(7, 2), textcoords="offset points", fontsize=8)
axes[1, 1].set(xlabel="toy forward + backward passes", ylabel="recovery@4", xlim=(0.5, 5.7), ylim=(0.58, 0.93))
axes[1, 1].set_title("Faithfulness / runtime tradeoff", fontweight="bold")

toy_asset = assets_dir / "atp_star_toy_signature.png"
fig.savefig(toy_asset, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()

print("Spearman:", {name: round(value, 3) for name, value in spearman.items()})
print("Recovery@4:", {name: round(values[3], 3) for name, values in recoveries.items()})

<!-- CELL: markdown -->
<details><summary>Interpretation - what AtP* fixed, and what it did not</summary>

- **QK fix:** the query's true magnitude is `3.0`; vanilla AtP and EAP-style local gradients assign about `1.6e-6`. Exact softmax recomputation restores the full pattern change.
- **GradDrop:** the cancel source's true magnitude is `0.5`; its full clean gradient is exactly zero. Dropping the nonlinear residual contribution exposes a signed estimate of `-1`, and the paper's aggregation returns `0.5`.
- **EAP-style edge attribution:** separating outgoing edges detects the cancellation source, but overestimates it as `1.0` and still misses saturated Q/K. Edge scores answer a different structural question from node scores.
- **Joint faithfulness:** the best singleton overshoots the full corrupt effect and adding another important component can reduce the curve. A good singleton ranking does not make a nonlinear circuit additive.
- **Cost:** exhaustive patching scales with component count; AtP* scales with model depth plus inexpensive QK recomputation. AtP* is still more expensive than vanilla AtP, so use it where false negatives matter.

</details>

<!-- CELL: markdown -->
### Exercise - define the real task metric and named Q/K sites

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
> Suggested time: 20 minutes
> ```

Implement `per_prompt_logit_diff`, `real_component_names`, and `parse_real_component`. The real study has 24 equal-length plural/singular pairs across three answer-token families. Each of six layers and eight heads contributes one final-position query and one final-position key site: 96 candidates total.

<details><summary>Expected output</summary>

```text
All tests in `test_per_prompt_logit_diff` passed!
All tests in `test_real_component_names_and_parser` passed!
24 examples; 96 named Q/K components; sequence length 3
```

</details>

<details><summary>Help - Keep task and component axes explicit</summary>

Gather positive and negative logits using one token id per batch row. Generate names in layer-major, head-major, Q-then-K order; reject malformed names rather than guessing.

</details>

<details><summary>Common bug</summary>

Averaging logits before applying per-example token ids mixes the `are/is`, `have/has`, and `run/runs` tasks.

</details>

<details><summary>Solution</summary>

{{solution:per_prompt_logit_diff,real_component_names,parse_real_component}}

</details>

<!-- CELL: code -->
# EXERCISE: per_prompt_logit_diff,real_component_names,parse_real_component

tests.test_per_prompt_logit_diff(per_prompt_logit_diff)
tests.test_real_component_names_and_parser(real_component_names, parse_real_component)
print(f"{len(reference.build_grammar_pairs())} examples; {len(real_component_names(6, 8))} named Q/K components")

<!-- CELL: markdown -->
## Real-model path - retain paths before ranking components

The next cells expose the same computation on 96 final-position rotary Q/K sites
in pinned Pythia-70M. Caching and batching are provided below; read their token
indices before implementing the real gradients. All scores and intervention
controls are computed here from the learner functions.

We freeze model parameters to avoid unnecessary gradients. That does **not** mean
intermediate activations should be detached: cutting every Q/K tensor silently
removes paths through later attention layers. GradDrop removes only one selected
layer's residual contributions per backward pass.

The paper's [GradDrop definition](https://arxiv.org/html/2403.00745v1#S3.SS1.SSS2)
zeros the selected residual contribution's gradient and scales the sum of absolute
estimates by `1 / (L - 1)`. This implementation reruns the clean forward pass for
each dropped layer; the paper can reuse it. Timings below include that overhead.

<!-- CELL: code -->
load_pinned_pythia70m = reference.load_pinned_pythia70m
prepare_grammar_task = reference.prepare_grammar_task
PYTHIA_MODEL_NAME = reference.PYTHIA_MODEL_NAME
PYTHIA_HF_ID = reference.PYTHIA_HF_ID
PYTHIA_REVISION = reference.PYTHIA_REVISION

# PROVIDED: _q_hook_name,_k_hook_name,_pattern_hook_name,cache_real_qk,_patch_component_chunk_hooks,exact_real_patch_effects

<!-- CELL: markdown -->
### Exercise - retain node gradients and drop exactly one layer

> Difficulty: 4/5 | Importance: 5/5 | Suggested time: 25 minutes

Implement vanilla node attribution for rotary queries and keys, then obtain
attention-pattern gradients with one layer's attention and MLP residual outputs
detached. Preserve all other edges in the graph. In the test chain, the first
query has derivative `2 * (1+3) * (1+5) = 48`; dropping the last layer changes the
first pattern derivative to `2 * (1+3) = 8`.

<details><summary>Expected output</summary>

The independent test below prints a success message. A detached intermediate
activation must fail the cross-layer derivative check.

</details>

<details><summary>Help</summary>

Follow tensor shapes explicitly. `requires_grad_(True)` starts a graph for frozen
inputs; `retain_grad()` stores intermediate derivatives without cutting existing
paths. Only the selected GradDrop residual contributions should detach.

</details>

<details><summary>Solution</summary>

{{solution:vanilla_real_atp_scores,_pattern_gradients}}

</details>

<details><summary>Interpretation</summary>

An approximation can correlate with exact patching despite a wrong derivative.
The hand-computable tests establish the operation before the real experiment
measures whether its ranking is useful.

</details>

<!-- CELL: code -->
# EXERCISE: vanilla_real_atp_scores,_pattern_gradients

tests.test_real_vanilla_gradients_preserve_later_qk_paths(vanilla_real_atp_scores)
tests.test_graddrop_only_cuts_the_requested_layer(_pattern_gradients)

<!-- CELL: markdown -->
### Exercise - apply the Q/K softmax correction to real cache tensors

> Difficulty: 4/5 | Importance: 5/5 | Suggested time: 25 minutes

For each final query, replace the query before recomputing its full softmax.
For each final key, change its one attention-logit entry while keeping other keys
fixed. Contract the resulting pattern change with the retained pattern gradient.
The final query can attend to every position, so no future-token mask is needed
for this restricted candidate set.

<details><summary>Expected output</summary>

The independent test below prints a success message. A detached intermediate
activation must fail the cross-layer derivative check.

</details>

<details><summary>Help</summary>

Follow tensor shapes explicitly. `requires_grad_(True)` starts a graph for frozen
inputs; `retain_grad()` stores intermediate derivatives without cutting existing
paths. Only the selected GradDrop residual contributions should detach.

</details>

<details><summary>Solution</summary>

{{solution:qk_corrected_real_estimates}}

</details>

<details><summary>Interpretation</summary>

An approximation can correlate with exact patching despite a wrong derivative.
The hand-computable tests establish the operation before the real experiment
measures whether its ranking is useful.

</details>

<!-- CELL: code -->
# EXERCISE: qk_corrected_real_estimates

tests.test_real_qk_correction_matches_two_token_oracle(qk_corrected_real_estimates)

<!-- CELL: markdown -->
### Exercise - aggregate absolute GradDrop estimates

> Difficulty: 4/5 | Importance: 5/5 | Suggested time: 25 minutes

Run the corrected estimator under every dropped layer. Take absolute values
before summing across drops and before averaging examples; divide by `L - 1`.
This preserves the direct-path scale while exposing cancelling indirect paths.

<details><summary>Expected output</summary>

The independent test below prints a success message. A detached intermediate
activation must fail the cross-layer derivative check.

</details>

<details><summary>Help</summary>

Follow tensor shapes explicitly. `requires_grad_(True)` starts a graph for frozen
inputs; `retain_grad()` stores intermediate derivatives without cutting existing
paths. Only the selected GradDrop residual contributions should detach.

</details>

<details><summary>Solution</summary>

{{solution:real_atp_star_scores}}

</details>

<details><summary>Interpretation</summary>

An approximation can correlate with exact patching despite a wrong derivative.
The hand-computable tests establish the operation before the real experiment
measures whether its ranking is useful.

</details>

<!-- CELL: code -->
# EXERCISE: real_atp_star_scores

tests.test_real_graddrop_aggregation(real_atp_star_scores)

<!-- CELL: markdown -->
### Evaluate rankings by intervening on their selected sites

These provided helpers compose the patching operations into joint top-k effects,
24 same-size random controls, and recovery curves. A sum of individual effects
and the effect of patching a set jointly are different quantities.

<!-- CELL: code -->
# PROVIDED: rank_named_scores,real_topk_patch_effect,real_component_set_effects,real_faithfulness_curve

<!-- CELL: code -->
# PROVIDED: same_size_random_circuit_control,_mapping_spearman,_cache_tensor_bytes

<!-- CELL: markdown -->
### Compose the experiment

The runner below uses the functions defined in this notebook. It measures exact,
vanilla, and corrected scores, then causally evaluates random and shuffled
rankings. The fixed grammar suite is reused for screening and verification, so
this comparison is descriptive; it is not held-out circuit generalization.

<!-- CELL: code -->
# PROVIDED: run_real_transformerlens_study

<!-- CELL: code -->
real = run_real_transformerlens_study(
    "cpu",
    component_batch_size=8,
    top_k=8,
    random_samples=24,
)
assert real["accepted"]
summary = {
    "examples": real["num_examples"],
    "components": real["num_components"],
    "clean_corrupt_gap": round(real["clean_corrupt_gap"], 3),
    "spearman": {name: round(value, 3) for name, value in real["spearman"].items()},
    "top_components": real["top_components"],
    "same_size_rank": f"{real['same_size_random']['rank']}/{real['same_size_random']['num_compared']}",
    "control_effects": {name: round(value, 3) for name, value in real["control_effects"].items()},
    "runtime_seconds": {name: round(value, 2) for name, value in real["runtime_seconds"].items()},
}
print(json.dumps(summary, indent=2))


(root_dir / chapter / "exercises" / section / "expected_outputs" / "reference_metrics.json").write_text(json.dumps(real, indent=2) + "\n")

<!-- CELL: markdown -->
<details><summary>Expected output</summary>

The exact floating-point values and CPU timings can vary slightly. The pinned reference should satisfy:

```text
examples = 24; components = 96
clean-corrupt logit-difference gap > 8.0
Spearman(exact, vanilla AtP) > 0.93
Spearman(exact, AtP*) > 0.99 and exceeds vanilla
AtP* top component = L5H5.K@end
AtP* eight-site causal effect > every one of 24 same-size random circuits
pair-shuffled and label-shuffled top-eight effects < the AtP* top-eight effect
```

</details>

<!-- CELL: code -->
exact = real["exact_effects"]
vanilla = real["vanilla_atp"]
star = real["atp_star"]
exact_order = real["exact_ranking"]
top_show = exact_order[:16]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
fig.patch.set_facecolor("#f7f7f4")

x = np.arange(len(top_show))
axes[0].bar(x - 0.2, [exact[name] for name in top_show], 0.2, label="Exact", color="#333333")
axes[0].bar(x, [vanilla[name] for name in top_show], 0.2, label="Vanilla AtP", color="#c45a4a")
axes[0].bar(x + 0.2, [star[name] for name in top_show], 0.2, label="AtP*", color="#187a79")
axes[0].set_xticks(x, top_show, rotation=70, ha="right", fontsize=7)
axes[0].set_ylabel("mean absolute effect / estimate")
axes[0].set_title("Top exact Q/K effects", fontweight="bold")
axes[0].legend(frameon=False, fontsize=8)

for method, color in (("Exact", "#333333"), ("Vanilla AtP", "#c45a4a"), ("AtP*", "#187a79")):
    values = real["recovery_at_k"][method]
    axes[1].plot(list(values), list(values.values()), marker="o", label=method, color=color)
axes[1].set(xlabel="components sent to exact verification", ylabel="recovery@k", ylim=(0, 0.86))
axes[1].set_title("Verified effect mass vs budget", fontweight="bold")
axes[1].legend(frameon=False)

random_effects = real["same_size_random"]["random_effects"]
axes[2].hist(random_effects, bins=10, color="#9ea7ad", edgecolor="white", label="same-size random")
axes[2].axvline(real["same_size_random"]["discovered_effect"], color="#187a79", linewidth=3, label="AtP* top 8")
axes[2].axvline(real["control_effects"]["pair_shuffled"], color="#d28e2c", linestyle="--", linewidth=2, label="pair shuffled")
axes[2].axvline(real["control_effects"]["label_shuffled"], color="#b23a48", linestyle=":", linewidth=2, label="label shuffled")
axes[2].set_xlabel("mean absolute causal effect")
axes[2].set_ylabel("circuits")
axes[2].set_title("Same-size and shuffled controls", fontweight="bold")
axes[2].legend(frameon=False, fontsize=8)

real_asset = assets_dir / "atp_star_transformerlens_signature.png"
fig.savefig(real_asset, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()

<!-- CELL: markdown -->
<details><summary>Interpretation - what the real result supports</summary>

- The behavior is real: the average clean-minus-corrupt logit-difference gap is about `8.40` across 24 examples.
- AtP* improves rank correlation with exhaustive Q/K patching from about `0.943` to `0.993`. It does not dominate vanilla AtP at every tiny `k`; read the computed recovery curve to identify the budgets at which each method wins.
- The top eight AtP* sites have causal effect about `0.489`, rank first against 24 reproducible same-size random circuits (random mean about `0.189`, best about `0.417`).
- A clean-to-clean pair-shuffled ranking and an independently label-shuffled ranking yield weaker true top-eight effects (about `0.266` and `0.395`). These are controls over ranking quality, not claims that those alternate metrics contain no grammatical information.
- Joint top-eight faithfulness is about `0.725` relative to patching all 96 candidate Q/K sites. This is useful screening, not a complete circuit.

</details>


The hook implementation now passes independent cross-layer derivative tests.
This corrected CPU run supersedes the earlier detached-node measurements.

<!-- CELL: markdown -->
## Try It Yourself - change the approximation budget

Choose **one** intervention before rerunning the cell:

1. Set `component_batch_size` to `1`, `4`, or `8`. Exact effects must stay fixed while wall time changes.
2. Change `top_k` from `8` to `4` or `16`. Compare recovery@k with joint faithfulness; they need not move together.
3. Remove `"cancellation"` from the toy GradDrop paths. Predict what happens to the cancel-source score before running.

<!-- CELL: code -->
PLAY_BATCH_SIZE = 4
PLAY_TOP_K = 4
PLAY_GRAD_PATHS = ("attention", "linear")

play_exact, play_ms = timed(
    lambda: batched_exact_patch_effects(
        toy_model, clean_sites, corrupt_sites,
        component_batch_size=PLAY_BATCH_SIZE,
    )
)
play_star, _ = atp_star_scores(
    toy_model, clean_sites, corrupt_sites,
    grad_paths=PLAY_GRAD_PATHS,
)
play_ranking = rank_components(play_star)
print("exact unchanged:", max(abs(play_exact[name] - exact_effects[name]) for name in TOY_COMPONENTS) < 1e-12)
print("wall time (ms):", round(play_ms, 3))
print("top-k:", play_ranking[:PLAY_TOP_K])
print("cancel score:", play_star["L0M0.cancel_source"])

plt.bar(["default GradDrop", "chosen paths"], [atp_star_estimates["L0M0.cancel_source"], play_star["L0M0.cancel_source"]])
plt.ylabel("Cancellation-source score")
plt.show()

<!-- CELL: markdown -->


<details><summary>Interpretation</summary>

Batch size is a systems choice and must not change the oracle. Removing the one gradient control that breaks the direct/indirect cancellation makes the cancel-source estimate collapse again. This is why GradDrop is a structured sweep, not ordinary stochastic dropout.

</details>

<!-- CELL: markdown -->
## Bonus - anomaly hunting

Choose one investigation and write a prediction before coding:

1. Sweep the clean Q/K magnitude from `0.5` to `5.0`. At what saturation level does vanilla AtP's query rank collapse while the QK correction remains stable?
2. Change the nonlinear branch coefficient from `-0.5` to `-0.45` or `-0.55`. Plot exact, AtP, and GradDrop estimates as cancellation becomes imperfect.
3. Find the first `k` where the real AtP* recovery curve trails vanilla AtP, then inspect the exact and approximate scores of the components exchanged at that boundary.
4. Replace the clean-to-clean pair control with singular donors from a held-out noun set. Does the control become stronger because number information is restored?

<details><summary>What counts as a useful anomaly?</summary>

A useful anomaly changes a causal metric, identifies the components responsible, and survives a matched comparison. “The bars moved” is not enough. Record the prompt pairs, candidate set, exact effects, ranking change, and a control which should not move.

</details>

<!-- CELL: markdown -->
## Limitations

- The exact graph is a deliberately small scalar model organism. It proves the two estimator failure modes but is not a transformer.
- The real study scores final-position rotary queries and keys only. Values, head outputs, MLP neurons, and every token position are outside the candidate set.
- The real result uses Pythia-70M-deduped on a course-generated grammatical agreement task, not the paper's Pythia-12B CITY or IOI experiments.
- AtP* improves the overall real ranking but does not beat vanilla AtP at every small verification budget.
- The same-size null has 24 reproducible draws rather than exhaustive enumeration over `96 choose 8` sets.
- GradDrop reduces important false negatives; it does not guarantee none remain. The paper's subset-sampling confidence bound is a worthwhile extension, not claimed here.
- CPU execution validates the notebook and real result. The CUDA entry point is implemented but deliberately not run in this task.

## Reading Links

- [AtP*: An efficient and scalable method for localizing LLM behaviour to components](https://arxiv.org/abs/2403.00745)
- [Attribution Patching: Activation Patching at Industrial Scale](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
- [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/abs/2310.10348)
- Original ARENA [Indirect Object Identification](../../../chapter1_transformer_interp/instructions/pages/21_[1.4.1]_Indirect_Object_Identification.md)

## Consolidating Understanding

1. Activation patching is the causal oracle; AtP and AtP* propose an order for spending that oracle budget.
2. Vanilla AtP is a first-order Taylor estimate around the clean activation.
3. QK correction recomputes the nonlinear attention-pattern change and linearizes only downstream computation.
4. GradDrop preserves the clean forward pass while cutting one residual contribution's backward path.
5. Node scores and edge scores have different candidate sets and can expose different failures.
6. Recovery@k grades singleton ranking; joint faithfulness tests a selected circuit under simultaneous intervention.
7. Random, pair-shuffled, and label-shuffled controls are part of the evidence, not decoration.
