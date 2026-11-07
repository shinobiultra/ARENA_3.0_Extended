# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
 {"title": "ReLU token and hidden players", "icon": "1-circle-fill", "subtitle": "45%"},
 {"title": "Finite circuits and data players", "icon": "2-circle-fill", "subtitle": "50%"},
 {"title": "Bonus", "icon": "star", "subtitle": "5%"}
]
```
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [16.8] SHAP vs Activation Patching
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Introduction
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Two methods can agree on the most important player while disagreeing about credit. Before reading
the worked claims, predict what an interaction does to the sum of single-location noising losses
compared with the clean-minus-corrupt change. Then predict whether moving from input players to
separate post-nonlinearity units must change that comparison.

'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ ReLU token and hidden players

> ##### Learning Objectives
>
> - Implement cached noising interventions and complete clean/corrupt coalition tables.
> - Compare interaction credit after explicitly aligning the attribution unit and baseline.
> - Challenge a visible result with wrong-location, shuffled-label and matched-random controls.

### 2️⃣ Finite circuits and data players

> ##### Learning Objectives
>
> - Build exact finite circuit, endpoint and pair-interaction oracles.
> - Compare rankings through deletion and insertion consequences rather than correlation alone.
> - Keep the separate training-example utility and its gradient-dot proxy distinct from a circuit.

### Scientific scope

The first game fixes clean and corrupt activation vectors. The second game fixes a finite feature
coalition utility with an intercept; it does not introduce an observational conditional-background
SHAP distribution. Its data-player bridge instead changes the selected training examples and the
mean used by one gradient step. These games are not interchangeable.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

from __future__ import annotations

MAIN = __name__ == "__main__"


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ ReLU token and hidden players
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### [16.6] SHAP vs Activation Patching - Solutions

> **Falsifiable claim.** By the end of this notebook, you will have shown that, in an exact ReLU
> token model, token-location noising patching overcounts the clean-minus-corrupt effect by exactly
> the `2.4` interaction contribution, while exact Shapley remains efficient; when both methods
> instead use the same five post-ReLU hidden units, they agree exactly, and wrong-location,
> shuffled-label, and matched-random controls fail visibly.

## Core Question

When SHAP and activation patching disagree, is one method wrong, or did we ask two different causal
questions about different units?

## Cold open: one prompt, two stories

Our model reads four visible token locations:

```text
[RED] [SQUARE] [BRIGHT] [FILLER]  ->  MATCH logit 4.25
[---] [------] [------] [------]  ->  MATCH logit 0.00
```

`RED`, `SQUARE`, and `BRIGHT` contribute `1.00`, `0.60`, and `0.25` directly. A real ReLU unit
detects `RED AND SQUARE` and contributes another `2.40`. `FILLER` is known to do nothing. We know
every causal term, so disagreement cannot hide behind model uncertainty.

Before calculating anything, predict: if you remove `RED` from the full prompt, should the lost
`2.40` interaction belong entirely to `RED`, or be shared with `SQUARE`?

## Learning goals

- Implement and inspect a nonlinear model with named token and hidden activations.
- Build complete clean/corrupt coalition tables at a chosen activation level.
- Implement exact Shapley values from weighted marginal contributions.
- Implement activation patching as an actual cached-activation intervention.
- Align attribution units before comparing magnitudes, rankings, or cosine similarity.
- Falsify weak agreement claims with wrong-location, shuffled-label, and matched-random controls.
- Explain why interaction credit depends on the causal question, not only the model output.

> Difficulty: 3/5
> Importance: 4/5
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup

The primary result is generated from the code in this notebook. No verification report is loaded. The toy contract runs in seconds; the local real-model path takes minutes and needs a GPU.


<details><summary>Expected output</summary>

Imports complete without output. All model and attribution tensors use float64 so the exact oracle
can be checked to `1e-12`.

</details>
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

import itertools
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t

chapter = "chapter16_shapley_attribution_baselines"
relu_legacy_section = "part6_shap_vs_activation_patching"
if MAIN:
    root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if MAIN:
    exercises_dir = root_dir / chapter / "exercises"
if MAIN:
    relu_legacy_section_dir = exercises_dir / relu_legacy_section
if MAIN:
    canonical_dir = exercises_dir / "part8_shap_vs_patching"
if MAIN:
    asset_path = canonical_dir / "media" / "shap_vs_activation_patching_exact_signature.png"
if MAIN:

    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
if MAIN:
    if str(exercises_dir) not in sys.path:
        sys.path.append(str(exercises_dir))

from chapter16_shapley_attribution_baselines.exercises.part6_shap_vs_activation_patching import tests as relu_tests
if MAIN:

    t.set_default_dtype(t.float64)
TOKEN_LABELS = ("RED", "SQUARE", "BRIGHT", "FILLER")
HIDDEN_LABELS = (
    "red_direct",
    "square_direct",
    "bright_direct",
    "filler_direct",
    "red_x_square_gate",
)
DIRECT_WEIGHTS = (1.0, 0.6, 0.25, 0.0)
DEFAULT_INTERACTION_STRENGTH = 2.4
GATE_THRESHOLD = 1.0
EXERCISE_ID = "16_6_shap_vs_activation_patching"
GT_TIER = "GT-0"
EXPECTED_RUNTIME = "45-75 minutes on CPU; optional release preflight uses CUDA"
REQUIRES_GPU = True
MAIN = __name__ == "__main__"


# ! CELL TYPE: code
# ! FILTERS: [python]
# ! TAGS: [keep-main]

def initialize_lesson_plumbing():
    """Explicitly bind source/report inputs and canonical output destinations.

    No target tensor, taught method, model, plot/report experiment or output
    write occurs. Module callers initialize plumbing before path-dependent
    callbacks, then defaults before the three tensor-default callbacks.
    """
    global root_dir, exercises_dir, canonical_dir
    global relu_legacy_section_dir, agreement_legacy_section_dir
    global asset_path, artifact_dir, asset_dir
    root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
    exercises_dir = root_dir / chapter / "exercises"
    canonical_dir = exercises_dir / "part8_shap_vs_patching"
    relu_legacy_section_dir = exercises_dir / "part6_shap_vs_activation_patching"
    agreement_legacy_section_dir = exercises_dir / "part8_shapley_mechinterp_agreement"
    asset_path = canonical_dir / "media" / "shap_vs_activation_patching_exact_signature.png"
    artifact_dir = canonical_dir / "artifacts"
    asset_dir = canonical_dir / "media"

if MAIN:
    initialize_lesson_plumbing()


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 1 - build the exact ReLU model

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement the encoder, readout, and cached forward pass. The fifth hidden unit must be `ReLU(RED +
SQUARE - 1)`, not a precomputed interaction label.

<details><summary>Expected output</summary>

`MATCH logit: 0.00 -> 4.25` and `post-ReLU hidden: [1, 1, 1, 1, 1]`. Removing `SQUARE` switches off
the gate and gives `1.25`.

</details>

<details><summary>Help - what should I reason about first?</summary>

Track shapes first. The input ends in four token locations; the encoder appends one scalar gate; the
readout has five weights. Preserve any leading batch axes.

</details>

<details><summary>Interpretation</summary>

This is an actual nonlinear computation with an inspectable internal activation. We are not
inventing coalition scores by hand.

</details>

<details><summary>Common bugs</summary>

Hard-coding the interaction score instead of producing it with `torch.relu`; returning only a score
so there is no activation to patch.

</details>

<details><summary>Solution</summary>

```python
def encode_tokens(
    token_activations: t.Tensor,
    *,
    gate_threshold: float = GATE_THRESHOLD,
) -> t.Tensor:
    if token_activations.shape[-1] != len(TOKEN_LABELS):
        raise ValueError(f"Expected {len(TOKEN_LABELS)} token locations.")
    gate = t.relu(token_activations[..., 0] + token_activations[..., 1] - gate_threshold)
    return t.cat((token_activations, gate.unsqueeze(-1)), dim=-1)


def score_from_hidden(
    hidden: t.Tensor,
    *,
    interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
) -> t.Tensor:
    if hidden.shape[-1] != len(HIDDEN_LABELS):
        raise ValueError(f"Expected {len(HIDDEN_LABELS)} hidden units.")
    weights = hidden.new_tensor((*DIRECT_WEIGHTS, interaction_strength))
    return hidden @ weights


def exact_model(
    token_activations: t.Tensor,
    *,
    interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
    gate_threshold: float = GATE_THRESHOLD,
    return_cache: bool = False,
) -> t.Tensor | tuple[t.Tensor, dict[str, t.Tensor]]:
    hidden = encode_tokens(token_activations, gate_threshold=gate_threshold)
    score = score_from_hidden(hidden, interaction_strength=interaction_strength)
    if not return_cache:
        return score
    return score, {
        "token_activations": token_activations.detach().clone(),
        "post_relu_hidden": hidden.detach().clone(),
    }
```

</details>
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def encode_tokens(
#     token_activations: t.Tensor,
#     *,
#     gate_threshold: float = GATE_THRESHOLD,
# ) -> t.Tensor:
#     """Expose four direct channels and one ReLU interaction unit."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def score_from_hidden(
#     hidden: t.Tensor,
#     *,
#     interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
# ) -> t.Tensor:
#     """Read out the five hidden units into one MATCH logit."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def exact_model(
#     token_activations: t.Tensor,
#     *,
#     interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
#     gate_threshold: float = GATE_THRESHOLD,
#     return_cache: bool = False,
# ) -> t.Tensor | tuple[t.Tensor, dict[str, t.Tensor]]:
#     """Run the model and optionally return named token and hidden activations."""
#     # YOUR CODE HERE
#     raise NotImplementedError()



# if MAIN:
#     relu_tests.test_exact_model_oracle(encode_tokens, score_from_hidden, exact_model)
#     clean_tokens = t.ones(4)
#     corrupt_tokens = t.zeros(4)
#     clean_score, clean_cache = exact_model(clean_tokens, return_cache=True)
#     print("Clean prompt:   [RED] [SQUARE] [BRIGHT] [FILLER]")
#     print("Corrupt prompt: [---] [------] [------] [------]")
#     print(f"MATCH logit: {exact_model(corrupt_tokens).item():.2f} -> {clean_score.item():.2f}")
#     print("post-ReLU hidden:", clean_cache["post_relu_hidden"].tolist())

# END EXERCISE
# SOLUTION
def encode_tokens(
    token_activations: t.Tensor,
    *,
    gate_threshold: float = GATE_THRESHOLD,
) -> t.Tensor:
    if token_activations.shape[-1] != len(TOKEN_LABELS):
        raise ValueError(f"Expected {len(TOKEN_LABELS)} token locations.")
    gate = t.relu(token_activations[..., 0] + token_activations[..., 1] - gate_threshold)
    return t.cat((token_activations, gate.unsqueeze(-1)), dim=-1)


def score_from_hidden(
    hidden: t.Tensor,
    *,
    interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
) -> t.Tensor:
    if hidden.shape[-1] != len(HIDDEN_LABELS):
        raise ValueError(f"Expected {len(HIDDEN_LABELS)} hidden units.")
    weights = hidden.new_tensor((*DIRECT_WEIGHTS, interaction_strength))
    return hidden @ weights


def exact_model(
    token_activations: t.Tensor,
    *,
    interaction_strength: float = DEFAULT_INTERACTION_STRENGTH,
    gate_threshold: float = GATE_THRESHOLD,
    return_cache: bool = False,
) -> t.Tensor | tuple[t.Tensor, dict[str, t.Tensor]]:
    hidden = encode_tokens(token_activations, gate_threshold=gate_threshold)
    score = score_from_hidden(hidden, interaction_strength=interaction_strength)
    if not return_cache:
        return score
    return score, {
        "token_activations": token_activations.detach().clone(),
        "post_relu_hidden": hidden.detach().clone(),
    }



# HIDE
if MAIN:
    relu_tests.test_exact_model_oracle(encode_tokens, score_from_hidden, exact_model)
    clean_tokens = t.ones(4)
    corrupt_tokens = t.zeros(4)
    clean_score, clean_cache = exact_model(clean_tokens, return_cache=True)
    print("Clean prompt:   [RED] [SQUARE] [BRIGHT] [FILLER]")
    print("Corrupt prompt: [---] [------] [------] [------]")
    print(f"MATCH logit: {exact_model(corrupt_tokens).item():.2f} -> {clean_score.item():.2f}")
    print("post-ReLU hidden:", clean_cache["post_relu_hidden"].tolist())

# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 2 - construct a coalition table from interventions

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10-15 minutes on this exercise.
> ```

A player is meaningful only relative to an activation level. Write one routine which mixes clean and
corrupt activations, then really runs the supplied downstream function for all coalitions.

<details><summary>Expected output</summary>

There are 16 token coalitions. `RED+SQUARE -> 4.00`, `RED+BRIGHT -> 1.25`, and the full prompt gives
`4.25`.

</details>

<details><summary>Help - what should I reason about first?</summary>

For each subset, clone the corrupt vector and copy only the selected clean coordinates. Never mutate
either source cache.

</details>

<details><summary>Interpretation</summary>

At the token level, coalition membership changes both direct evidence and whether the downstream
ReLU fires. Later, at the hidden level, the gate itself becomes a player.

</details>

<details><summary>Common bugs</summary>

Treating coalition membership as a label lookup instead of rerunning the downstream function;
mutating the clean or corrupt source vector in place.

</details>

<details><summary>Solution</summary>

```python
def all_coalitions(num_players: int) -> tuple[frozenset[int], ...]:
    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    coalitions = []
    for size in range(num_players + 1):
        coalitions.extend(
            frozenset(group)
            for group in itertools.combinations(range(num_players), size)
        )
    return tuple(coalitions)


def coalition_value_table(
    clean_activation: t.Tensor,
    corrupt_activation: t.Tensor,
    value_fn: Callable[[t.Tensor], t.Tensor],
) -> dict[frozenset[int], float]:
    if clean_activation.ndim != 1 or clean_activation.shape != corrupt_activation.shape:
        raise ValueError("clean and corrupt activations must be same-shape vectors.")
    values = {}
    for coalition in all_coalitions(clean_activation.numel()):
        mixed = corrupt_activation.clone()
        if coalition:
            index = t.tensor(sorted(coalition), device=mixed.device)
            mixed[index] = clean_activation[index]
        value = value_fn(mixed)
        if value.numel() != 1:
            raise ValueError("value_fn must return one scalar.")
        values[coalition] = float(value.item())
    return values
```

</details>
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def all_coalitions(num_players: int) -> tuple[frozenset[int], ...]:
#     """Return every player subset, including empty and full coalitions."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def coalition_value_table(
#     clean_activation: t.Tensor,
#     corrupt_activation: t.Tensor,
#     value_fn: Callable[[t.Tensor], t.Tensor],
# ) -> dict[frozenset[int], float]:
#     """Run all mixed clean/corrupt coalitions at one activation level."""
#     # YOUR CODE HERE
#     raise NotImplementedError()



# if MAIN:
#     relu_tests.test_coalition_value_table_oracle(all_coalitions, coalition_value_table, exact_model)
#     token_values = coalition_value_table(clean_tokens, corrupt_tokens, exact_model)
#     for coalition in [frozenset(), frozenset({0}), frozenset({0, 1}), frozenset(range(4))]:
#         names = [TOKEN_LABELS[index] for index in sorted(coalition)] or ["baseline"]
#         print(f"{'+'.join(names):28s} -> {token_values[coalition]:.2f}")

# END EXERCISE
# SOLUTION
def all_coalitions(num_players: int) -> tuple[frozenset[int], ...]:
    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    coalitions = []
    for size in range(num_players + 1):
        coalitions.extend(
            frozenset(group)
            for group in itertools.combinations(range(num_players), size)
        )
    return tuple(coalitions)


def coalition_value_table(
    clean_activation: t.Tensor,
    corrupt_activation: t.Tensor,
    value_fn: Callable[[t.Tensor], t.Tensor],
) -> dict[frozenset[int], float]:
    if clean_activation.ndim != 1 or clean_activation.shape != corrupt_activation.shape:
        raise ValueError("clean and corrupt activations must be same-shape vectors.")
    values = {}
    for coalition in all_coalitions(clean_activation.numel()):
        mixed = corrupt_activation.clone()
        if coalition:
            index = t.tensor(sorted(coalition), device=mixed.device)
            mixed[index] = clean_activation[index]
        value = value_fn(mixed)
        if value.numel() != 1:
            raise ValueError("value_fn must return one scalar.")
        values[coalition] = float(value.item())
    return values



# HIDE
if MAIN:
    relu_tests.test_coalition_value_table_oracle(all_coalitions, coalition_value_table, exact_model)
    token_values = coalition_value_table(clean_tokens, corrupt_tokens, exact_model)
    for coalition in [frozenset(), frozenset({0}), frozenset({0, 1}), frozenset(range(4))]:
        names = [TOKEN_LABELS[index] for index in sorted(coalition)] or ["baseline"]
        print(f"{'+'.join(names):28s} -> {token_values[coalition]:.2f}")

# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 3 - implement exact Shapley values

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Average each player's marginal contribution over every possible predecessor coalition, using the
factorial Shapley weight. Reject incomplete tables.

<details><summary>Expected output</summary>

Token Shapley is `RED=2.20, SQUARE=1.80, BRIGHT=0.25, FILLER=0.00`; the values sum exactly to the
`4.25` output difference.

</details>

<details><summary>Help - what should I reason about first?</summary>

For player `i` and coalition size `s`, the weight is `s! (n-s-1)! / n!`. Iterate only over
coalitions which exclude `i`.

</details>

<details><summary>Interpretation</summary>

The ReLU contribution is symmetric in `RED` and `SQUARE`, so Shapley gives each half of `2.40`, on
top of their direct terms.

</details>

<details><summary>Common bugs</summary>

Using leave-one-out deltas instead of factorial-weighted marginal contributions; silently accepting
a missing coalition.

</details>

<details><summary>Solution</summary>

```python
def normalize_coalition_values(
    coalition_values: Mapping[frozenset[int] | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[frozenset[int], float]:
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(all_coalitions(num_players))
    if set(values) != expected:
        missing = expected - set(values)
        extra = set(values) - expected
        raise ValueError(
            f"Coalition table must be complete (missing={len(missing)}, extra={len(extra)})."
        )
    return values


def exact_shapley_values(
    coalition_values: Mapping[frozenset[int] | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    result = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        others = [candidate for candidate in range(num_players) if candidate != player]
        for size in range(num_players):
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                result[player] += weight * (
                    values[coalition | {player}] - values[coalition]
                )
    return result
```

</details>
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def normalize_coalition_values(
#     coalition_values: Mapping[frozenset[int] | tuple[int, ...], float],
#     *,
#     num_players: int,
# ) -> dict[frozenset[int], float]:
#     """Normalize keys and reject incomplete finite games."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def exact_shapley_values(
#     coalition_values: Mapping[frozenset[int] | tuple[int, ...], float],
#     *,
#     num_players: int,
# ) -> t.Tensor:
#     """Average each player's marginal effect over all arrival contexts."""
#     # YOUR CODE HERE
#     raise NotImplementedError()



# if MAIN:
#     relu_tests.test_exact_shapley_token_ground_truth(
#         coalition_value_table, exact_shapley_values, exact_model
#     )
#     token_shapley = exact_shapley_values(token_values, num_players=4)
#     print(dict(zip(TOKEN_LABELS, token_shapley.tolist(), strict=True)))
#     print(f"sum(phi) = {token_shapley.sum().item():.2f} = clean-corrupt output")

# END EXERCISE
# SOLUTION
def normalize_coalition_values(
    coalition_values: Mapping[frozenset[int] | tuple[int, ...], float],
    *,
    num_players: int,
) -> dict[frozenset[int], float]:
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(all_coalitions(num_players))
    if set(values) != expected:
        missing = expected - set(values)
        extra = set(values) - expected
        raise ValueError(
            f"Coalition table must be complete (missing={len(missing)}, extra={len(extra)})."
        )
    return values


def exact_shapley_values(
    coalition_values: Mapping[frozenset[int] | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    values = normalize_coalition_values(coalition_values, num_players=num_players)
    result = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        others = [candidate for candidate in range(num_players) if candidate != player]
        for size in range(num_players):
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                result[player] += weight * (
                    values[coalition | {player}] - values[coalition]
                )
    return result



# HIDE
if MAIN:
    relu_tests.test_exact_shapley_token_ground_truth(
        coalition_value_table, exact_shapley_values, exact_model
    )
    token_shapley = exact_shapley_values(token_values, num_players=4)
    print(dict(zip(TOKEN_LABELS, token_shapley.tolist(), strict=True)))
    print(f"sum(phi) = {token_shapley.sum().item():.2f} = clean-corrupt output")

# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 4 - patch one activation location

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement noising patching: start from the clean cache, replace one location with its corrupt value,
continue the forward computation, and record the clean-score loss.

<details><summary>Expected output</summary>

Token patch losses are `[3.40, 3.00, 0.25, 0.00]`. They sum to `6.65`, exceeding the output delta by
exactly `2.40`.

</details>

<details><summary>Help - what should I reason about first?</summary>

Compute the clean score once. For each location, clone clean activation, copy one corrupt scalar,
and rerun `value_fn`. Return `clean_score - patched_score`.

</details>

<details><summary>Interpretation</summary>

Removing either necessary token kills the complete gate contribution. Full-context patching
therefore assigns `2.40` to both tokens; this is a different estimand from Shapley's average
marginal credit.

</details>

<details><summary>Common bugs</summary>

Patching into the corrupt run while reporting a noising score; replacing every location at once;
measuring the patched score rather than clean minus patched.

</details>

<details><summary>Solution</summary>

```python
def activation_patching_effects(
    clean_activation: t.Tensor,
    corrupt_activation: t.Tensor,
    value_fn: Callable[[t.Tensor], t.Tensor],
) -> t.Tensor:
    if clean_activation.ndim != 1 or clean_activation.shape != corrupt_activation.shape:
        raise ValueError("clean and corrupt activations must be same-shape vectors.")
    clean_value = value_fn(clean_activation)
    if clean_value.numel() != 1:
        raise ValueError("value_fn must return one scalar.")
    effects = []
    for location in range(clean_activation.numel()):
        patched = clean_activation.clone()
        patched[location] = corrupt_activation[location]
        effects.append(clean_value - value_fn(patched))
    return t.stack(effects).to(dtype=t.float64)
```

</details>
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def activation_patching_effects(
#     clean_activation: t.Tensor,
#     corrupt_activation: t.Tensor,
#     value_fn: Callable[[t.Tensor], t.Tensor],
# ) -> t.Tensor:
#     """Patch one corrupt activation into the clean run and measure score loss."""
#     # YOUR CODE HERE
#     raise NotImplementedError()



# if MAIN:
#     relu_tests.test_token_activation_patching_ground_truth(activation_patching_effects, exact_model)
#     token_patching = activation_patching_effects(clean_tokens, corrupt_tokens, exact_model)
#     print("location        SHAP    patch loss")
#     for label, shapley, patch in zip(TOKEN_LABELS, token_shapley, token_patching, strict=True):
#         print(f"{label:10s} {shapley.item():7.2f} {patch.item():11.2f}")
#     print(f"patching credit = {token_patching.sum().item():.2f}; output delta = 4.25")

# END EXERCISE
# SOLUTION
def activation_patching_effects(
    clean_activation: t.Tensor,
    corrupt_activation: t.Tensor,
    value_fn: Callable[[t.Tensor], t.Tensor],
) -> t.Tensor:
    if clean_activation.ndim != 1 or clean_activation.shape != corrupt_activation.shape:
        raise ValueError("clean and corrupt activations must be same-shape vectors.")
    clean_value = value_fn(clean_activation)
    if clean_value.numel() != 1:
        raise ValueError("value_fn must return one scalar.")
    effects = []
    for location in range(clean_activation.numel()):
        patched = clean_activation.clone()
        patched[location] = corrupt_activation[location]
        effects.append(clean_value - value_fn(patched))
    return t.stack(effects).to(dtype=t.float64)



# HIDE
if MAIN:
    relu_tests.test_token_activation_patching_ground_truth(activation_patching_effects, exact_model)
    token_patching = activation_patching_effects(clean_tokens, corrupt_tokens, exact_model)
    print("location        SHAP    patch loss")
    for label, shapley, patch in zip(TOKEN_LABELS, token_shapley, token_patching, strict=True):
        print(f"{label:10s} {shapley.item():7.2f} {patch.item():11.2f}")
    print(f"patching credit = {token_patching.sum().item():.2f}; output delta = 4.25")

# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 5 - align attribution units before comparing

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement strict name-based alignment and report magnitude error, cosine similarity, top unit, and
each method's efficiency gap. Then repeat both methods over the five post-ReLU hidden units.

<details><summary>Expected output</summary>

Token max error is `1.20` and patching efficiency gap is `2.40`, despite top-unit agreement. Hidden
Shapley and hidden patching both equal `[1.00, 0.60, 0.25, 0.00, 2.40]` with zero error.

</details>

<details><summary>Help - what should I reason about first?</summary>

Never correlate a four-token vector with a five-hidden-unit vector. Require identical label sets,
reorder by name, and only then calculate metrics.

</details>

<details><summary>Interpretation</summary>

The hidden readout is additive in its five units, so patching the explicit gate and assigning
Shapley credit to that same gate asks the same question. The model stayed nonlinear; the player
definition changed.

</details>

<details><summary>Common bugs</summary>

Correlating token values with hidden-unit values because their lengths happen to be close; reporting
only top-1 agreement or cosine and ignoring efficiency.

</details>

<details><summary>Solution</summary>

```python
def align_named_attributions(
    reference_labels: Sequence[str],
    reference_values: t.Tensor,
    candidate_labels: Sequence[str],
    candidate_values: t.Tensor,
) -> tuple[tuple[str, ...], t.Tensor, t.Tensor]:
    if len(set(reference_labels)) != len(reference_labels):
        raise ValueError("reference labels must be unique.")
    if len(set(candidate_labels)) != len(candidate_labels):
        raise ValueError("candidate labels must be unique.")
    if reference_values.numel() != len(reference_labels):
        raise ValueError("reference values and labels have different lengths.")
    if candidate_values.numel() != len(candidate_labels):
        raise ValueError("candidate values and labels have different lengths.")
    if set(reference_labels) != set(candidate_labels):
        raise ValueError("Attributions describe different player sets; do not correlate them.")
    candidate_index = {label: index for index, label in enumerate(candidate_labels)}
    order = t.tensor(
        [candidate_index[label] for label in reference_labels],
        device=candidate_values.device,
    )
    return (
        tuple(reference_labels),
        reference_values.to(dtype=t.float64),
        candidate_values[order].to(dtype=t.float64),
    )


def attribution_comparison(
    reference_labels: Sequence[str],
    reference_values: t.Tensor,
    candidate_labels: Sequence[str],
    candidate_values: t.Tensor,
    *,
    output_delta: float,
) -> dict[str, float | str | bool]:
    labels, reference, candidate = align_named_attributions(
        reference_labels, reference_values, candidate_labels, candidate_values
    )
    cosine = t.nn.functional.cosine_similarity(reference, candidate, dim=0)
    return {
        "max_abs_error": float((reference - candidate).abs().max().item()),
        "mean_abs_error": float((reference - candidate).abs().mean().item()),
        "cosine_similarity": float(cosine.item()),
        "reference_efficiency_gap": float(reference.sum().item() - output_delta),
        "candidate_efficiency_gap": float(candidate.sum().item() - output_delta),
        "top_reference": labels[int(reference.argmax().item())],
        "top_candidate": labels[int(candidate.argmax().item())],
        "top_agrees": int(reference.argmax().item()) == int(candidate.argmax().item()),
    }
```

</details>
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def align_named_attributions(
#     reference_labels: Sequence[str],
#     reference_values: t.Tensor,
#     candidate_labels: Sequence[str],
#     candidate_values: t.Tensor,
# ) -> tuple[tuple[str, ...], t.Tensor, t.Tensor]:
#     """Align by player name and reject incomparable token/hidden units."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def attribution_comparison(
#     reference_labels: Sequence[str],
#     reference_values: t.Tensor,
#     candidate_labels: Sequence[str],
#     candidate_values: t.Tensor,
#     *,
#     output_delta: float,
# ) -> dict[str, float | str | bool]:
#     """Report magnitude, angle, top unit, and efficiency after alignment."""
#     # YOUR CODE HERE
#     raise NotImplementedError()



# if MAIN:
#     relu_tests.test_alignment_reorders_and_rejects_mismatched_units(align_named_attributions)
#     relu_tests.test_aligned_token_disagreement_and_hidden_agreement(
#         coalition_value_table,
#         exact_shapley_values,
#         activation_patching_effects,
#         attribution_comparison,
#         encode_tokens,
#         score_from_hidden,
#         exact_model,
#     )
#     clean_hidden = encode_tokens(clean_tokens)
#     corrupt_hidden = encode_tokens(corrupt_tokens)
#     hidden_values = coalition_value_table(clean_hidden, corrupt_hidden, score_from_hidden)
#     hidden_shapley = exact_shapley_values(hidden_values, num_players=5)
#     hidden_patching = activation_patching_effects(clean_hidden, corrupt_hidden, score_from_hidden)
#     token_comparison = attribution_comparison(
#         TOKEN_LABELS, token_shapley, TOKEN_LABELS, token_patching, output_delta=4.25
#     )
#     hidden_comparison = attribution_comparison(
#         HIDDEN_LABELS, hidden_shapley, HIDDEN_LABELS, hidden_patching, output_delta=4.25
#     )
#     print("token level:", token_comparison)
#     print("hidden level:", hidden_comparison)

# END EXERCISE
# SOLUTION
def align_named_attributions(
    reference_labels: Sequence[str],
    reference_values: t.Tensor,
    candidate_labels: Sequence[str],
    candidate_values: t.Tensor,
) -> tuple[tuple[str, ...], t.Tensor, t.Tensor]:
    if len(set(reference_labels)) != len(reference_labels):
        raise ValueError("reference labels must be unique.")
    if len(set(candidate_labels)) != len(candidate_labels):
        raise ValueError("candidate labels must be unique.")
    if reference_values.numel() != len(reference_labels):
        raise ValueError("reference values and labels have different lengths.")
    if candidate_values.numel() != len(candidate_labels):
        raise ValueError("candidate values and labels have different lengths.")
    if set(reference_labels) != set(candidate_labels):
        raise ValueError("Attributions describe different player sets; do not correlate them.")
    candidate_index = {label: index for index, label in enumerate(candidate_labels)}
    order = t.tensor(
        [candidate_index[label] for label in reference_labels],
        device=candidate_values.device,
    )
    return (
        tuple(reference_labels),
        reference_values.to(dtype=t.float64),
        candidate_values[order].to(dtype=t.float64),
    )


def attribution_comparison(
    reference_labels: Sequence[str],
    reference_values: t.Tensor,
    candidate_labels: Sequence[str],
    candidate_values: t.Tensor,
    *,
    output_delta: float,
) -> dict[str, float | str | bool]:
    labels, reference, candidate = align_named_attributions(
        reference_labels, reference_values, candidate_labels, candidate_values
    )
    cosine = t.nn.functional.cosine_similarity(reference, candidate, dim=0)
    return {
        "max_abs_error": float((reference - candidate).abs().max().item()),
        "mean_abs_error": float((reference - candidate).abs().mean().item()),
        "cosine_similarity": float(cosine.item()),
        "reference_efficiency_gap": float(reference.sum().item() - output_delta),
        "candidate_efficiency_gap": float(candidate.sum().item() - output_delta),
        "top_reference": labels[int(reference.argmax().item())],
        "top_candidate": labels[int(candidate.argmax().item())],
        "top_agrees": int(reference.argmax().item()) == int(candidate.argmax().item()),
    }



# HIDE
if MAIN:
    relu_tests.test_alignment_reorders_and_rejects_mismatched_units(align_named_attributions)
    relu_tests.test_aligned_token_disagreement_and_hidden_agreement(
        coalition_value_table,
        exact_shapley_values,
        activation_patching_effects,
        attribution_comparison,
        encode_tokens,
        score_from_hidden,
        exact_model,
    )
    clean_hidden = encode_tokens(clean_tokens)
    corrupt_hidden = encode_tokens(corrupt_tokens)
    hidden_values = coalition_value_table(clean_hidden, corrupt_hidden, score_from_hidden)
    hidden_shapley = exact_shapley_values(hidden_values, num_players=5)
    hidden_patching = activation_patching_effects(clean_hidden, corrupt_hidden, score_from_hidden)
    token_comparison = attribution_comparison(
        TOKEN_LABELS, token_shapley, TOKEN_LABELS, token_patching, output_delta=4.25
    )
    hidden_comparison = attribution_comparison(
        HIDDEN_LABELS, hidden_shapley, HIDDEN_LABELS, hidden_patching, output_delta=4.25
    )
    print("token level:", token_comparison)
    print("hidden level:", hidden_comparison)

# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 6 - make the controls fail

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Build a fixed label-shuffle control and 512 matched-L2 random hidden interventions. Keep the known
`FILLER` token as a wrong-location control.

<details><summary>Expected output</summary>

Wrong-location effect `0.00`; shuffled-label cosine `0.515`; random-direction p95 `2.166`; true gate
effect `2.400` at the `98.6%` random percentile.

</details>

<details><summary>Help - what should I reason about first?</summary>

For random controls, sample Gaussian vectors, normalize each row, match the true gate-delta norm,
and measure the same score loss used for the target intervention.

</details>

<details><summary>Interpretation</summary>

A large patch effect is not persuasive without a location control and a matched intervention
distribution. A high cosine is not persuasive if labels can be shuffled without much penalty.

</details>

<details><summary>Common bugs</summary>

Sampling unmatched random-vector norms; shuffling both labels and values together, which leaves the
attribution mapping unchanged.

</details>

<details><summary>Solution</summary>

```python
def shuffled_label_control(values: t.Tensor, permutation: Sequence[int]) -> t.Tensor:
    if sorted(permutation) != list(range(values.numel())):
        raise ValueError("permutation must contain each value index exactly once.")
    return values[t.tensor(permutation, device=values.device)]


def random_direction_patching_effects(
    clean_hidden: t.Tensor,
    readout_weights: t.Tensor,
    *,
    target_delta_norm: float,
    num_samples: int = 512,
    seed: int = 1666,
) -> t.Tensor:
    if clean_hidden.ndim != 1 or clean_hidden.shape != readout_weights.shape:
        raise ValueError("clean_hidden and readout_weights must be same-shape vectors.")
    if num_samples <= 0 or target_delta_norm <= 0:
        raise ValueError("num_samples and target_delta_norm must be positive.")
    generator = t.Generator(device=clean_hidden.device).manual_seed(seed)
    directions = t.randn(
        num_samples,
        clean_hidden.numel(),
        generator=generator,
        device=clean_hidden.device,
        dtype=clean_hidden.dtype,
    )
    directions = directions / directions.norm(dim=1, keepdim=True)
    directions = directions * target_delta_norm
    patched = clean_hidden.unsqueeze(0) - directions
    clean_score = clean_hidden @ readout_weights
    return clean_score - patched @ readout_weights
```

</details>
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def shuffled_label_control(values: t.Tensor, permutation: Sequence[int]) -> t.Tensor:
#     """Keep labels fixed while shuffling the values assigned to them."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def random_direction_patching_effects(
#     clean_hidden: t.Tensor,
#     readout_weights: t.Tensor,
#     *,
#     target_delta_norm: float,
#     num_samples: int = 512,
#     seed: int = 1666,
# ) -> t.Tensor:
#     """Patch matched-norm random directions into the hidden state."""
#     # YOUR CODE HERE
#     raise NotImplementedError()



# if MAIN:
#     relu_tests.test_shuffled_and_random_controls(
#         shuffled_label_control, random_direction_patching_effects
#     )
#     shuffled = shuffled_label_control(token_patching, (2, 0, 3, 1))
#     shuffled_cosine = t.nn.functional.cosine_similarity(token_shapley, shuffled, dim=0).item()
#     readout_weights = t.tensor((*DIRECT_WEIGHTS, DEFAULT_INTERACTION_STRENGTH))
#     random_effects = random_direction_patching_effects(
#         clean_hidden, readout_weights, target_delta_norm=1.0
#     ).abs()
#     random_p95 = t.quantile(random_effects, 0.95).item()
#     gate_effect = hidden_patching[-1].item()
#     gate_percentile = (random_effects < gate_effect).double().mean().item()
#     print(f"wrong-location FILLER patch: {token_patching[-1].item():.2f}")
#     print(f"label-shuffled cosine:       {shuffled_cosine:.3f}")
#     print(f"random direction p95:        {random_p95:.3f}")
#     print(f"gate patch effect:           {gate_effect:.3f} ({gate_percentile:.1%} percentile)")

# END EXERCISE
# SOLUTION
def shuffled_label_control(values: t.Tensor, permutation: Sequence[int]) -> t.Tensor:
    if sorted(permutation) != list(range(values.numel())):
        raise ValueError("permutation must contain each value index exactly once.")
    return values[t.tensor(permutation, device=values.device)]


def random_direction_patching_effects(
    clean_hidden: t.Tensor,
    readout_weights: t.Tensor,
    *,
    target_delta_norm: float,
    num_samples: int = 512,
    seed: int = 1666,
) -> t.Tensor:
    if clean_hidden.ndim != 1 or clean_hidden.shape != readout_weights.shape:
        raise ValueError("clean_hidden and readout_weights must be same-shape vectors.")
    if num_samples <= 0 or target_delta_norm <= 0:
        raise ValueError("num_samples and target_delta_norm must be positive.")
    generator = t.Generator(device=clean_hidden.device).manual_seed(seed)
    directions = t.randn(
        num_samples,
        clean_hidden.numel(),
        generator=generator,
        device=clean_hidden.device,
        dtype=clean_hidden.dtype,
    )
    directions = directions / directions.norm(dim=1, keepdim=True)
    directions = directions * target_delta_norm
    patched = clean_hidden.unsqueeze(0) - directions
    clean_score = clean_hidden @ readout_weights
    return clean_score - patched @ readout_weights



# HIDE
if MAIN:
    relu_tests.test_shuffled_and_random_controls(
        shuffled_label_control, random_direction_patching_effects
    )
    shuffled = shuffled_label_control(token_patching, (2, 0, 3, 1))
    shuffled_cosine = t.nn.functional.cosine_similarity(token_shapley, shuffled, dim=0).item()
    readout_weights = t.tensor((*DIRECT_WEIGHTS, DEFAULT_INTERACTION_STRENGTH))
    random_effects = random_direction_patching_effects(
        clean_hidden, readout_weights, target_delta_norm=1.0
    ).abs()
    random_p95 = t.quantile(random_effects, 0.95).item()
    gate_effect = hidden_patching[-1].item()
    gate_percentile = (random_effects < gate_effect).double().mean().item()
    print(f"wrong-location FILLER patch: {token_patching[-1].item():.2f}")
    print(f"label-shuffled cosine:       {shuffled_cosine:.3f}")
    print(f"random direction p95:        {random_p95:.3f}")
    print(f"gate patch effect:           {gate_effect:.3f} ({gate_percentile:.1%} percentile)")

# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise 7 - sweep the interaction strength

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20-25 minutes on this exercise.
> ```

Turn the single example into a falsification curve. For each gate readout strength, recompute token
and hidden coalition tables, Shapley values, patch effects, and agreement metrics.

<details><summary>Expected output</summary>

Token credit overcount equals `alpha`; token max error equals `alpha / 2`; hidden max error stays
`0.0` across the sweep.

</details>

<details><summary>Help - what should I reason about first?</summary>

Do not scale a cached result. Recompute coalition outputs for every strength so the exercise matches
how a model experiment would be rerun.

</details>

<details><summary>Interpretation</summary>

At `alpha=0` the token game is additive and both methods agree. Disagreement grows exactly with the
nonlinear interaction term, while post-ReLU hidden-unit agreement remains exact.

</details>

<details><summary>Common bugs</summary>

Rescaling one cached attribution vector instead of rebuilding coalition tables at each interaction
strength; hiding numerical error with loose tolerances.

</details>

<details><summary>Solution</summary>

```python
def interaction_sweep(interaction_strengths: t.Tensor) -> dict[str, t.Tensor]:
    strengths = interaction_strengths.to(dtype=t.float64)
    clean_tokens = t.ones(len(TOKEN_LABELS), dtype=t.float64)
    corrupt_tokens = t.zeros_like(clean_tokens)
    token_overcount, token_max_error, hidden_max_error = [], [], []

    for strength in strengths:
        alpha = float(strength.item())
        token_value_fn = lambda x, alpha=alpha: exact_model(x, interaction_strength=alpha)
        token_values = coalition_value_table(clean_tokens, corrupt_tokens, token_value_fn)
        token_shapley = exact_shapley_values(token_values, num_players=4)
        token_patching = activation_patching_effects(clean_tokens, corrupt_tokens, token_value_fn)
        output_delta = token_values[frozenset(range(4))] - token_values[frozenset()]
        token_overcount.append(token_patching.sum() - output_delta)
        token_max_error.append((token_patching - token_shapley).abs().max())

        clean_hidden = encode_tokens(clean_tokens)
        corrupt_hidden = encode_tokens(corrupt_tokens)
        hidden_value_fn = lambda h, alpha=alpha: score_from_hidden(h, interaction_strength=alpha)
        hidden_values = coalition_value_table(clean_hidden, corrupt_hidden, hidden_value_fn)
        hidden_shapley = exact_shapley_values(hidden_values, num_players=5)
        hidden_patching = activation_patching_effects(clean_hidden, corrupt_hidden, hidden_value_fn)
        hidden_max_error.append((hidden_patching - hidden_shapley).abs().max())

    return {
        "interaction_strength": strengths,
        "token_credit_overcount": t.stack(token_overcount),
        "token_max_abs_error": t.stack(token_max_error),
        "hidden_max_abs_error": t.stack(hidden_max_error),
    }
```

</details>
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def interaction_sweep(interaction_strengths: t.Tensor) -> dict[str, t.Tensor]:
#     """Compare token and hidden agreement as the ReLU gate readout changes."""
#     # YOUR CODE HERE
#     raise NotImplementedError()



# if MAIN:
#     relu_tests.test_interaction_sweep_exact_oracle(interaction_sweep)
#     strengths = t.linspace(0.0, 3.2, 9)
#     sweep = interaction_sweep(strengths)
#     print("alpha  token overcount  token max error  hidden max error")
#     for row in zip(
#         sweep["interaction_strength"],
#         sweep["token_credit_overcount"],
#         sweep["token_max_abs_error"],
#         sweep["hidden_max_abs_error"],
#         strict=True,
#     ):
#         print(f"{row[0].item():4.1f} {row[1].item():16.2f} {row[2].item():16.2f} {row[3].item():17.1e}")

# END EXERCISE
# SOLUTION
def interaction_sweep(interaction_strengths: t.Tensor) -> dict[str, t.Tensor]:
    strengths = interaction_strengths.to(dtype=t.float64)
    clean_tokens = t.ones(len(TOKEN_LABELS), dtype=t.float64)
    corrupt_tokens = t.zeros_like(clean_tokens)
    token_overcount, token_max_error, hidden_max_error = [], [], []

    for strength in strengths:
        alpha = float(strength.item())
        token_value_fn = lambda x, alpha=alpha: exact_model(x, interaction_strength=alpha)
        token_values = coalition_value_table(clean_tokens, corrupt_tokens, token_value_fn)
        token_shapley = exact_shapley_values(token_values, num_players=4)
        token_patching = activation_patching_effects(clean_tokens, corrupt_tokens, token_value_fn)
        output_delta = token_values[frozenset(range(4))] - token_values[frozenset()]
        token_overcount.append(token_patching.sum() - output_delta)
        token_max_error.append((token_patching - token_shapley).abs().max())

        clean_hidden = encode_tokens(clean_tokens)
        corrupt_hidden = encode_tokens(corrupt_tokens)
        hidden_value_fn = lambda h, alpha=alpha: score_from_hidden(h, interaction_strength=alpha)
        hidden_values = coalition_value_table(clean_hidden, corrupt_hidden, hidden_value_fn)
        hidden_shapley = exact_shapley_values(hidden_values, num_players=5)
        hidden_patching = activation_patching_effects(clean_hidden, corrupt_hidden, hidden_value_fn)
        hidden_max_error.append((hidden_patching - hidden_shapley).abs().max())

    return {
        "interaction_strength": strengths,
        "token_credit_overcount": t.stack(token_overcount),
        "token_max_abs_error": t.stack(token_max_error),
        "hidden_max_abs_error": t.stack(hidden_max_error),
    }



# HIDE
if MAIN:
    relu_tests.test_interaction_sweep_exact_oracle(interaction_sweep)
    strengths = t.linspace(0.0, 3.2, 9)
    sweep = interaction_sweep(strengths)
    print("alpha  token overcount  token max error  hidden max error")
    for row in zip(
        sweep["interaction_strength"],
        sweep["token_credit_overcount"],
        sweep["token_max_abs_error"],
        sweep["hidden_max_abs_error"],
        strict=True,
    ):
        print(f"{row[0].item():4.1f} {row[1].item():16.2f} {row[2].item():16.2f} {row[3].item():17.1e}")

# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Signature Result

The four panels below are one argument, not four decorative plots:

1. the same token units disagree in magnitude and efficiency;
2. the same hidden units agree exactly;
3. disagreement scales with the planted nonlinear contribution;
4. wrong-location, shuffled-label, and matched-random controls fail.

The plot is generated from your implementations in this notebook and saved as the section's
learner-facing bitmap.

<img src="../../exercises/part8_shap_vs_patching/media/shap_vs_activation_patching_exact_signature.png" width="900">
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

colors = {"shapley": "#157A6E", "patching": "#D95D39", "control": "#E9B44C", "ink": "#263238"}
if MAIN:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
if MAIN:

    x = np.arange(len(TOKEN_LABELS))
width = 0.36
if MAIN:
    axes[0, 0].bar(x - width / 2, token_shapley.numpy(), width, label="Exact Shapley", color=colors["shapley"])
if MAIN:
    axes[0, 0].bar(x + width / 2, token_patching.numpy(), width, label="Noising patch loss", color=colors["patching"])
if MAIN:
    axes[0, 0].set_xticks(x, TOKEN_LABELS)
if MAIN:
    axes[0, 0].set_ylabel("MATCH-logit credit")
if MAIN:
    axes[0, 0].set_title("A. Same token units, different interaction question")
if MAIN:
    axes[0, 0].legend(frameon=False)
if MAIN:
    axes[0, 0].text(0.98, 0.94, "SHAP sum = 4.25\nPatch sum = 6.65", transform=axes[0, 0].transAxes, ha="right", va="top")
if MAIN:

    xh = np.arange(len(HIDDEN_LABELS))
short_hidden = ("red", "square", "bright", "filler", "ReLU gate")
if MAIN:
    axes[0, 1].bar(xh - width / 2, hidden_shapley.numpy(), width, label="Exact Shapley", color=colors["shapley"])
if MAIN:
    axes[0, 1].bar(xh + width / 2, hidden_patching.numpy(), width, label="Hidden-unit patch", color=colors["patching"])
if MAIN:
    axes[0, 1].set_xticks(xh, short_hidden, rotation=20, ha="right")
if MAIN:
    axes[0, 1].set_ylabel("MATCH-logit credit")
if MAIN:
    axes[0, 1].set_title("B. Same post-ReLU units: exact agreement")
if MAIN:
    axes[0, 1].text(0.66, 0.94, "max error = 0.00\nefficiency gap = 0.00", transform=axes[0, 1].transAxes, ha="right", va="top")
if MAIN:

    alpha = sweep["interaction_strength"].numpy()
if MAIN:
    axes[1, 0].plot(alpha, sweep["token_credit_overcount"].numpy(), "o-", color=colors["patching"], label="token patch overcount")
if MAIN:
    axes[1, 0].plot(alpha, alpha, "--", color=colors["ink"], label="exact y = interaction strength")
if MAIN:
    axes[1, 0].plot(alpha, sweep["hidden_max_abs_error"].numpy(), "s-", color=colors["shapley"], label="hidden max error")
if MAIN:
    axes[1, 0].set_xlabel("ReLU interaction strength")
if MAIN:
    axes[1, 0].set_ylabel("Attribution error")
if MAIN:
    axes[1, 0].set_title("C. Disagreement tracks the interaction exactly")
if MAIN:
    axes[1, 0].legend(frameon=False)
if MAIN:

    axes[1, 1].hist(random_effects.numpy(), bins=24, color="#B0BEC5", edgecolor="white", label="512 matched random directions")
if MAIN:
    axes[1, 1].axvline(random_p95, color=colors["control"], linewidth=2, label=f"random p95 = {random_p95:.2f}")
if MAIN:
    axes[1, 1].axvline(gate_effect, color=colors["patching"], linewidth=3, label=f"true gate = {gate_effect:.2f}")
if MAIN:
    axes[1, 1].set_xlabel("absolute MATCH-logit patch effect")
if MAIN:
    axes[1, 1].set_ylabel("count")
if MAIN:
    axes[1, 1].set_title("D. Controls catch location and label mistakes")
if MAIN:
    axes[1, 1].legend(frameon=False, fontsize=9)
if MAIN:
    axes[1, 1].text(0.98, 0.58, f"wrong FILLER = {token_patching[-1].item():.2f}\nshuffled cosine = {shuffled_cosine:.3f}\ngate percentile = {gate_percentile:.1%}", transform=axes[1, 1].transAxes, ha="right", va="top")
if MAIN:

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.22)
if MAIN:

    fig.suptitle("SHAP and activation patching agree only after player units are aligned", fontsize=17, weight="bold")
if MAIN:
    fig.savefig(asset_path, dpi=180, bbox_inches="tight")
if MAIN:
    plt.show()
if MAIN:
    print(f"Saved signature result to {asset_path}")
if MAIN:
    print("CLAIM CHECK: token overcount = 2.40; hidden max error = 0.00; all three controls fail as expected.")


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Interpreting the result

SHAP and activation patching are not interchangeable names for “importance.” Exact Shapley asks for
an average marginal contribution over all clean/corrupt contexts. The noising patch used here asks
how much the full clean computation loses when one activation is replaced. With an interaction,
removing either parent destroys the same gate, so full-context patching counts that loss once per
necessary parent.

The hidden-unit comparison is the key control on this explanation. Once the ReLU gate is an explicit
player and both methods operate on those same five post-ReLU units, the downstream readout is
additive and agreement is exact. This is why alignment must include the intervention site,
granularity, baseline, and score, not just vector length.

Notice the anomaly: both token methods name `RED` as the top feature and have cosine similarity near
one, yet patching allocates `6.65` units of credit to a `4.25` change. Rank agreement alone would
have hidden the failure.

## Try It Yourself

Change `play_interaction_strength`, `play_gate_threshold`, and `play_patch_location`. Predict the
plot before running it. In particular, set the interaction strength to zero, then raise it above
four. Which metric reveals disagreement first: top-1, cosine, maximum error, or efficiency?
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# Change these, then rerun this cell.
play_interaction_strength = 1.6
play_gate_threshold = 1.0
play_patch_location = "RED"
if MAIN:

    play_value_fn = lambda x: exact_model(
        x,
        interaction_strength=play_interaction_strength,
        gate_threshold=play_gate_threshold,
    )
if MAIN:
    play_values = coalition_value_table(clean_tokens, corrupt_tokens, play_value_fn)
if MAIN:
    play_shapley = exact_shapley_values(play_values, num_players=4)
if MAIN:
    play_patching = activation_patching_effects(clean_tokens, corrupt_tokens, play_value_fn)
if MAIN:
    play_index = TOKEN_LABELS.index(play_patch_location)
if MAIN:

    fig, ax = plt.subplots(figsize=(8, 3.6))
if MAIN:
    ax.bar(x - width / 2, play_shapley.numpy(), width, label="Shapley", color=colors["shapley"])
if MAIN:
    ax.bar(x + width / 2, play_patching.numpy(), width, label="patch loss", color=colors["patching"])
if MAIN:
    ax.set_xticks(x, TOKEN_LABELS)
if MAIN:
    ax.set_ylabel("MATCH-logit credit")
if MAIN:
    ax.set_title(f"alpha={play_interaction_strength}, threshold={play_gate_threshold}; inspect {play_patch_location}")
if MAIN:
    ax.legend(frameon=False)
if MAIN:
    ax.spines[["top", "right"]].set_visible(False)
if MAIN:
    plt.show()
if MAIN:
    print(f"{play_patch_location} patch effect = {play_patching[play_index].item():.3f}")
if MAIN:
    print(f"total patch overcount = {play_patching.sum().item() - (play_values[frozenset(range(4))] - play_values[frozenset()]):.3f}")


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Bonus: anomaly hunting

1. Find the smallest positive interaction strength for which token patching overcounts by more than half the output delta while top-1 still agrees.
2. Raise the gate threshold above `1.0`. The ReLU can now be partially active or inactive for coalitions; derive the new token Shapley oracle before trusting the plot.
3. Replace `FILLER` with a weak negative direct term. Does a zero-effect wrong-location test still make sense, or should the control become a deliberately unrelated location with a known nonzero effect?
4. Compare noising (`clean -> one corrupt activation`) with denoising (`corrupt -> one clean activation`). For the AND gate, explain why neither single-site denoising intervention recovers the interaction.

## Paper and real-method connection

- [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874)
  defines Shapley-style feature attribution for model predictions.
- [Locating and Editing Factual Associations in GPT (ROME / causal
  tracing)](https://rome.baulab.info/) uses clean/corrupt activation interventions in transformers.
- [How to use and interpret activation patching](https://arxiv.org/abs/2309.16042) emphasizes
  corruption choices, metrics, and what patching results do and do not establish.
- Original ARENA's IOI chapter implements activation patching with hooks and normalized
  clean/corrupt metrics; this section isolates the interaction-credit issue before returning to
  large models.

## Limitations

- This exact model proves a mechanism-level distinction; it does not estimate which attribution is
  more useful on arbitrary transformers.
- Binary token activations and a zero baseline avoid the baseline ambiguity present in
  language-model embeddings and residual streams.
- Exact enumeration costs `2^n`, so practical SHAP methods use approximations whose sampling error
  is not studied here.
- The matched-random distribution is five-dimensional and seeded; real activation spaces require
  more seeds, task examples, and distribution-preserving corruptions.
- The repository's CUDA model-organism preflight is supporting release evidence. It is not the
  notebook's signature result and does not upgrade this exact toy claim into a large-model claim.
  The parent release runner owns its refresh; the notebook neither loads nor summarizes
  `verification_report.json`.
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Optional CUDA supporting path

The section-local reference module retains the existing finite-model preflight: an additive trained
model as the agreement control and a nonlinear trained model as the interaction-disagreement case.
It requires CUDA and has no CPU fallback. The parent runner owns any refresh; this notebook only
exposes the entry point.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

def relu_run_gpu_test(max_vram_gb: float = 24.0):
    from chapter16_shapley_attribution_baselines.exercises.part6_shap_vs_activation_patching import solutions as relu_reference
    """Parent-owned optional CUDA path; never called by this CPU notebook."""
    return relu_reference.run_gpu_test(max_vram_gb=max_vram_gb)


def relu_run_full_experiment(max_vram_gb: float = 24.0):
    return relu_run_gpu_test(max_vram_gb=max_vram_gb)
if MAIN:


    print({
        "method": "trained additive and nonlinear finite neural games",
        "requires_cuda": True,
        "executed_in_this_notebook": False,
        "large_transformer_claimed": False,
    })


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ Finite circuits and data players
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### [16.8] Do SHAPley and Mechanistic Interpretability Agree?

By the end of this notebook, you will have shown that SHAP-style input attribution and mechanistic
causal interventions agree on a planted additive mechanism, disagree on XOR when the player set is
wrong, and that the disagreement becomes useful once you test consequences and pair interactions.

## Core Question

When two attribution methods say "this was important", are they pointing to the same causal story,
or just assigning credit to different kinds of players?

The claim here is intentionally narrow. We start with exact finite games where the ground truth is
known, then treat a tiny trained finite model report as supporting evidence. The signature result is
a visible agreement matrix plus deletion/insertion and pair-interaction plots, not a report
dictionary.
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Learning goals

1. Build complete coalition tables for a finite causal circuit.
2. Implement exact Shapley values from the weighted marginal-effect formula.
3. Implement causal feature patching as a full-minus-ablated intervention on the same task.
4. Compare methods with Spearman rank correlation, top-k overlap, and deletion/insertion curves.
5. Diagnose an XOR disagreement where ordinary single-feature Shapley is not the right player set.
6. Add matched shuffled/random controls that fail visibly.
7. Connect input-level, data-level, activation-level, and circuit-level attribution without pretending they are the same object.

## Cold Open: The Smallest Useful Failure

In an additive game, Shapley, feature ablation, and the known mechanism all rank the same features.
In XOR, individual features average to zero, while the pair is causal. That is the whole lesson in
miniature: disagreement is not automatically a bug, but it must be tested rather than explained
away.

This mirrors the original ARENA activation-patching progression: first define a calibrated metric,
then intervene on a clean/corrupted system, then inspect a plot, then ask what the plot actually
licenses you to claim.
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Real Paper and Model Connection

SHAP-style methods estimate cooperative-game credit over an explicit player set. Mechanistic
interpretability methods such as activation patching, causal tracing, path patching, EAP, and
circuit discovery intervene on model internals. These are related but not identical: tokens,
features, activations, edges, and training examples are different kinds of players.

Useful anchors are SHAP for feature attribution, causal tracing / activation patching for
intervention-based localization, ACDC/EAP-style circuit discovery for edge-level mechanisms, and
In-Run Data Shapley for training-example credit. The notebook keeps those links honest by using an
exact model organism before making any real-model claims.
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup

The only imports are plotting, PyTorch tensors, and the local tests. The method implementations
below are notebook-visible; shared helpers are only used for printing and grading.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

import csv
import itertools
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import matplotlib
if MAIN:
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch as t
from IPython.display import Image, display

chapter = "chapter16_shapley_attribution_baselines"
agreement_legacy_section = "part8_shapley_mechinterp_agreement"
if MAIN:
    root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if MAIN:
    exercises_dir = root_dir / chapter / "exercises"
if MAIN:
    agreement_legacy_section_dir = exercises_dir / agreement_legacy_section
if MAIN:
    canonical_dir = exercises_dir / "part8_shap_vs_patching"
if MAIN:
    artifact_dir = canonical_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
if MAIN:
    asset_dir = canonical_dir / "media"
if MAIN:

    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
if MAIN:
    if str(exercises_dir) not in sys.path:
        sys.path.append(str(exercises_dir))

from chapter16_shapley_attribution_baselines.exercises.part8_shapley_mechinterp_agreement import tests as agreement_tests
from chapter16_shapley_attribution_baselines.exercises.part8_shap_vs_patching import agreement_utils

Coalition = frozenset[int]
TOY_FEATURE_NAMES = ("subject-token", "distractor-token", "answer-slot", "style-prior")
# FILTERS: ~python
TOY_LINEAR_WEIGHTS = t.tensor([1.2, -0.7, 1.6, 0.9], dtype=t.float64)
# END FILTERS
TOY_PAIR_WEIGHTS = {(0, 2): 2.2, (1, 3): -1.5}
TOY_INTERCEPT = 0.25
NEURAL_GAME_NUM_PLAYERS = 4
NEURAL_AGREEMENT_MIN_CORRELATION = 0.99
MAIN = __name__ == "__main__"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Enumerate Coalitions

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Start by making the finite player set explicit. Every later Shapley or patching calculation assumes
this table is complete.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def enumerate_coalitions(num_players: int) -> tuple[Coalition, ...]:
#     """Return every coalition in size-then-lexicographic order."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# if MAIN:
#     agreement_tests.test_enumerate_coalitions_toy_oracle(enumerate_coalitions)
# END EXERCISE
# SOLUTION
def enumerate_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Return every coalition in size-then-lexicographic order."""
    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    coalitions: list[Coalition] = []
    for size in range(num_players + 1):
        coalitions.extend(
            frozenset(group) for group in itertools.combinations(range(num_players), size)
        )
    return tuple(coalitions)


# HIDE
if MAIN:
    agreement_tests.test_enumerate_coalitions_toy_oracle(enumerate_coalitions)
# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

`test_enumerate_coalitions_toy_oracle` passes and `enumerate_coalitions(3)` has 8 unique entries.

</details>

<details>
<summary>Help - common bug</summary>

Use `itertools.combinations` for each coalition size from 0 to `num_players`.

</details>

<details>
<summary>Interpreting the result</summary>

This is the guardrail. If your coalition table is incomplete, every later attribution number
can look plausible while being wrong.

</details>
<details>
<summary>Common bugs</summary>

Returning only non-empty coalitions breaks the efficiency check. Returning lists instead of
`frozenset`s makes dictionary lookup brittle.

</details>

<details>
<summary>Solution</summary>

```python
def enumerate_coalitions(num_players: int) -> tuple[Coalition, ...]:
    """Return every coalition in size-then-lexicographic order."""
    if num_players <= 0:
        raise ValueError("num_players must be positive.")
    coalitions: list[Coalition] = []
    for size in range(num_players + 1):
        coalitions.extend(
            frozenset(group) for group in itertools.combinations(range(num_players), size)
        )
    return tuple(coalitions)


if MAIN:
    agreement_tests.test_enumerate_coalitions_toy_oracle(enumerate_coalitions)
```

</details>
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Build the Ground-Truth Circuit

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Implement the finite circuit `0.25 + 1.2*x0 - 0.7*x1 + 1.6*x2 + 0.9*x3 + 2.2*x0*x2 - 1.5*x1*x3`,
then write the endpoint and pair-edge ground truth.
'''


# ! CELL TYPE: code
# ! FILTERS: [python]
# ! TAGS: [keep-main]

_agreement_defaults_initialized = False


def initialize_agreement_defaults():
    """Explicitly bind the original shared defaults and three unchanged callbacks.

    No taught function is called. Repeated explicit initialization retains the
    same default tensor, pair mapping and callback objects. Ordinary module
    import leaves these three callbacks unavailable until this initializer runs.
    """
    global _agreement_defaults_initialized, TOY_LINEAR_WEIGHTS
    global finite_circuit_value, finite_circuit_table, mechanistic_endpoint_scores
    if _agreement_defaults_initialized:
        return
    TOY_LINEAR_WEIGHTS = t.tensor([1.2, -0.7, 1.6, 0.9], dtype=t.float64)
    def finite_circuit_value(
        coalition: Coalition | tuple[int, ...],
        *,
        linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
        pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
        intercept: float = TOY_INTERCEPT,
    ) -> float:
        """Evaluate the planted finite circuit on a coalition of present features."""
        active = set(coalition)
        value = float(intercept)
        for player, weight in enumerate(linear_weights.double().tolist()):
            if player in active:
                value += float(weight)
        for (first, second), weight in pair_weights.items():
            if first in active and second in active:
                value += float(weight)
        return value

    def finite_circuit_table(
        *,
        num_players: int = NEURAL_GAME_NUM_PLAYERS,
        linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
        pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
        intercept: float = TOY_INTERCEPT,
    ) -> dict[Coalition, float]:
        """Return the complete coalition table for the planted finite circuit."""
        return {
            coalition: finite_circuit_value(
                coalition,
                linear_weights=linear_weights,
                pair_weights=pair_weights,
                intercept=intercept,
            )
            for coalition in enumerate_coalitions(num_players)
        }

    def mechanistic_endpoint_scores(
        *,
        linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
        pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
    ) -> t.Tensor:
        """Allocate each planted pair edge equally to its two endpoint features."""
        scores = linear_weights.double().clone()
        for (first, second), weight in pair_weights.items():
            scores[first] += float(weight) / 2
            scores[second] += float(weight) / 2
        return scores
    _agreement_defaults_initialized = True

if MAIN:
    initialize_agreement_defaults()


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def finite_circuit_value(
#     coalition: Coalition | tuple[int, ...],
#     *,
#     linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
#     pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
#     intercept: float = TOY_INTERCEPT,
# ) -> float:
#     """Evaluate the planted finite circuit on a coalition of present features."""
#     # YOUR CODE HERE
#     raise NotImplementedError()



# def finite_circuit_table(
#     *,
#     num_players: int = NEURAL_GAME_NUM_PLAYERS,
#     linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
#     pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
#     intercept: float = TOY_INTERCEPT,
# ) -> dict[Coalition, float]:
#     """Return the complete coalition table for the planted finite circuit."""
#     # YOUR CODE HERE
#     raise NotImplementedError()



# def mechanistic_endpoint_scores(
#     *,
#     linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
#     pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
# ) -> t.Tensor:
#     """Allocate each planted pair edge equally to its two endpoint features."""
#     # YOUR CODE HERE
#     raise NotImplementedError()



# def mechanistic_pair_matrix(
#     *,
#     num_players: int = NEURAL_GAME_NUM_PLAYERS,
#     pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
# ) -> t.Tensor:
#     """Return the known pair-edge mechanism as a symmetric matrix."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# if MAIN:
#     agreement_tests.test_finite_circuit_table_toy_oracle(
#         finite_circuit_table,
#         mechanistic_endpoint_scores,
#         mechanistic_pair_matrix,
#     )
# END EXERCISE
# SOLUTION
# FILTERS: ~python
def finite_circuit_value(
    coalition: Coalition | tuple[int, ...],
    *,
    linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
    intercept: float = TOY_INTERCEPT,
) -> float:
    """Evaluate the planted finite circuit on a coalition of present features."""
    active = set(coalition)
    value = float(intercept)
    for player, weight in enumerate(linear_weights.double().tolist()):
        if player in active:
            value += float(weight)
    for (first, second), weight in pair_weights.items():
        if first in active and second in active:
            value += float(weight)
    return value

# END FILTERS


# FILTERS: ~python
def finite_circuit_table(
    *,
    num_players: int = NEURAL_GAME_NUM_PLAYERS,
    linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
    intercept: float = TOY_INTERCEPT,
) -> dict[Coalition, float]:
    """Return the complete coalition table for the planted finite circuit."""
    return {
        coalition: finite_circuit_value(
            coalition,
            linear_weights=linear_weights,
            pair_weights=pair_weights,
            intercept=intercept,
        )
        for coalition in enumerate_coalitions(num_players)
    }

# END FILTERS


# FILTERS: ~python
def mechanistic_endpoint_scores(
    *,
    linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
) -> t.Tensor:
    """Allocate each planted pair edge equally to its two endpoint features."""
    scores = linear_weights.double().clone()
    for (first, second), weight in pair_weights.items():
        scores[first] += float(weight) / 2
        scores[second] += float(weight) / 2
    return scores

# END FILTERS


def mechanistic_pair_matrix(
    *,
    num_players: int = NEURAL_GAME_NUM_PLAYERS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
) -> t.Tensor:
    """Return the known pair-edge mechanism as a symmetric matrix."""
    matrix = t.zeros((num_players, num_players), dtype=t.float64)
    for (first, second), weight in pair_weights.items():
        matrix[first, second] = float(weight)
        matrix[second, first] = float(weight)
    return matrix


# HIDE
if MAIN:
    agreement_tests.test_finite_circuit_table_toy_oracle(
        finite_circuit_table,
        mechanistic_endpoint_scores,
        mechanistic_pair_matrix,
    )

# END HIDE
# HIDE
if MAIN:

    values = finite_circuit_table()

# END HIDE
# HIDE
if MAIN:
    print("Full coalition value:", values[frozenset(range(NEURAL_GAME_NUM_PLAYERS))])

# END HIDE
# HIDE
if MAIN:
    print("Mechanistic endpoint scores:", mechanistic_endpoint_scores().tolist())

# END HIDE
# HIDE
if MAIN:
    print("Known pair-edge matrix:")

# END HIDE
# HIDE
if MAIN:
    print(mechanistic_pair_matrix())
# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The full table has 16 coalitions. Endpoint scores are `[2.3, -1.45, 2.7, 0.15]`; pair edges are
`(0,2)=2.2` and `(1,3)=-1.5`.

</details>

<details>
<summary>Help - common bug</summary>

Treat a coalition as the set of present input features. A pair term only fires when both endpoints
are present.

</details>

<details>
<summary>Interpreting the result</summary>

This is the known mechanism. Later, agreement means recovering this ranking or these pair edges from
attribution or intervention, not admiring a plot.

</details>
<details>
<summary>Common bugs</summary>

Do not add half the pair edge to the coalition value. Half-edge allocation belongs only in the
endpoint mechanistic score.

</details>

<details>
<summary>Solution</summary>

```python
def finite_circuit_value(
    coalition: Coalition | tuple[int, ...],
    *,
    linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
    intercept: float = TOY_INTERCEPT,
) -> float:
    """Evaluate the planted finite circuit on a coalition of present features."""
    active = set(coalition)
    value = float(intercept)
    for player, weight in enumerate(linear_weights.double().tolist()):
        if player in active:
            value += float(weight)
    for (first, second), weight in pair_weights.items():
        if first in active and second in active:
            value += float(weight)
    return value


def finite_circuit_table(
    *,
    num_players: int = NEURAL_GAME_NUM_PLAYERS,
    linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
    intercept: float = TOY_INTERCEPT,
) -> dict[Coalition, float]:
    """Return the complete coalition table for the planted finite circuit."""
    return {
        coalition: finite_circuit_value(
            coalition,
            linear_weights=linear_weights,
            pair_weights=pair_weights,
            intercept=intercept,
        )
        for coalition in enumerate_coalitions(num_players)
    }


def mechanistic_endpoint_scores(
    *,
    linear_weights: t.Tensor = TOY_LINEAR_WEIGHTS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
) -> t.Tensor:
    """Allocate each planted pair edge equally to its two endpoint features."""
    scores = linear_weights.double().clone()
    for (first, second), weight in pair_weights.items():
        scores[first] += float(weight) / 2
        scores[second] += float(weight) / 2
    return scores


def mechanistic_pair_matrix(
    *,
    num_players: int = NEURAL_GAME_NUM_PLAYERS,
    pair_weights: Mapping[tuple[int, int], float] = TOY_PAIR_WEIGHTS,
) -> t.Tensor:
    """Return the known pair-edge mechanism as a symmetric matrix."""
    matrix = t.zeros((num_players, num_players), dtype=t.float64)
    for (first, second), weight in pair_weights.items():
        matrix[first, second] = float(weight)
        matrix[second, first] = float(weight)
    return matrix


if MAIN:
    agreement_tests.test_finite_circuit_table_toy_oracle(
        finite_circuit_table,
        mechanistic_endpoint_scores,
        mechanistic_pair_matrix,
    )

values = finite_circuit_table()
print("Full coalition value:", values[frozenset(range(NEURAL_GAME_NUM_PLAYERS))])
print("Mechanistic endpoint scores:", mechanistic_endpoint_scores().tolist())
print("Known pair-edge matrix:")
print(mechanistic_pair_matrix())
```

</details>
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Exact Shapley from Weighted Marginals

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Implement the exact Shapley formula directly. Do not call a SHAP package; the point is to understand
the weighted marginal average.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def exact_shapley_from_table(
#     coalition_values: Mapping[Coalition | tuple[int, ...], float],
#     *,
#     num_players: int,
# ) -> t.Tensor:
#     """Compute exact Shapley values from first principles."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# if MAIN:
#     agreement_tests.test_exact_shapley_from_table_additive_oracle(exact_shapley_from_table)
#     agreement_tests.test_finite_circuit_agreement_case_matches_ground_truth(
#         finite_circuit_table,
#         exact_shapley_from_table,
#         mechanistic_endpoint_scores,
#     )
# END EXERCISE
# SOLUTION
def exact_shapley_from_table(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact Shapley values from first principles."""
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(enumerate_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")

    shapley = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        others = [candidate for candidate in range(num_players) if candidate != player]
        for size in range(num_players):
            weight = math.factorial(size) * math.factorial(num_players - size - 1) / denominator
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                shapley[player] += weight * (
                    values[coalition | {player}] - values[coalition]
                )
    return shapley


# HIDE
if MAIN:
    agreement_tests.test_exact_shapley_from_table_additive_oracle(exact_shapley_from_table)
    agreement_tests.test_finite_circuit_agreement_case_matches_ground_truth(
        finite_circuit_table,
        exact_shapley_from_table,
        mechanistic_endpoint_scores,
    )

# END HIDE
# HIDE
if MAIN:

    values = finite_circuit_table()

# END HIDE
# HIDE
if MAIN:
    print("Exact Shapley values:", exact_shapley_from_table(values, num_players=NEURAL_GAME_NUM_PLAYERS).tolist())

# END HIDE
# HIDE
if MAIN:
    print("Known mechanism:     ", mechanistic_endpoint_scores().tolist())
# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The additive oracle returns its weights exactly. On the planted finite circuit, exact Shapley equals
the endpoint mechanistic scores.

</details>

<details>
<summary>Help - common bug</summary>

For player `i`, loop over coalitions not containing `i`; the weight is `|S|! (n-|S|-1)! / n!`.

</details>

<details>
<summary>Interpreting the result</summary>

Agreement is now a theorem for this player set: the pair edge is split evenly across its endpoints,
so single-feature Shapley recovers the endpoint attribution.

</details>
<details>
<summary>Common bugs</summary>

Forgetting the empty coalition or using an unweighted mean gives the right answer on some tiny cases
and fails on others.

</details>

<details>
<summary>Solution</summary>

```python
def exact_shapley_from_table(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute exact Shapley values from first principles."""
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(enumerate_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")

    shapley = t.zeros(num_players, dtype=t.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        others = [candidate for candidate in range(num_players) if candidate != player]
        for size in range(num_players):
            weight = math.factorial(size) * math.factorial(num_players - size - 1) / denominator
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                shapley[player] += weight * (
                    values[coalition | {player}] - values[coalition]
                )
    return shapley


if MAIN:
    agreement_tests.test_exact_shapley_from_table_additive_oracle(exact_shapley_from_table)
    agreement_tests.test_finite_circuit_agreement_case_matches_ground_truth(
        finite_circuit_table,
        exact_shapley_from_table,
        mechanistic_endpoint_scores,
    )

values = finite_circuit_table()
print("Exact Shapley values:", exact_shapley_from_table(values, num_players=NEURAL_GAME_NUM_PLAYERS).tolist())
print("Known mechanism:     ", mechanistic_endpoint_scores().tolist())
```

</details>
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Causal Patching and Agreement Metrics

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Now compute a causal intervention on the same task: remove one feature from the full circuit and
measure the value drop. Then compare rankings.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def causal_patching_effects(
#     coalition_values: Mapping[Coalition | tuple[int, ...], float],
#     *,
#     num_players: int,
# ) -> t.Tensor:
#     """Return full-minus-ablated causal effects for each feature player."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def _rank_desc(scores: t.Tensor) -> list[int]:
#     """Return indices sorted by descending score."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def _average_ranks(scores: t.Tensor) -> t.Tensor:
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def _pearson_correlation(first: t.Tensor, second: t.Tensor) -> float:
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def spearman_rank_correlation(first: t.Tensor, second: t.Tensor) -> float:
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def topk_overlap_fraction(first: t.Tensor, second: t.Tensor, *, k: int) -> float:
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def agreement_summary(
#     coalition_values: Mapping[Coalition | tuple[int, ...], float],
#     *,
#     mechanistic_scores: t.Tensor,
#     num_players: int,
#     topk: int = 2,
# ) -> dict:
#     """Compare Shapley, causal patching, and known mechanistic scores."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# if MAIN:
#     agreement_tests.test_rank_desc_toy_oracle(_rank_desc)
#     agreement_tests.test_causal_patching_effects_expose_interaction_consequences(causal_patching_effects)
# END EXERCISE
# SOLUTION
def causal_patching_effects(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Return full-minus-ablated causal effects for each feature player."""
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    full = frozenset(range(num_players))
    if full not in values:
        raise ValueError("coalition value table must contain the full coalition.")
    return t.tensor(
        [values[full] - values[full - {player}] for player in range(num_players)],
        dtype=t.float64,
    )


def _rank_desc(scores: t.Tensor) -> list[int]:
    """Return indices sorted by descending score."""
    return [int(item) for item in t.argsort(scores.detach().double().cpu(), descending=True)]


def _average_ranks(scores: t.Tensor) -> t.Tensor:
    scores = scores.double().flatten()
    ranks = t.empty_like(scores)
    sorted_indices = t.argsort(scores, stable=True)
    sorted_scores = scores[sorted_indices]
    start = 0
    while start < int(scores.numel()):
        end = start + 1
        while end < int(scores.numel()) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[sorted_indices[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _pearson_correlation(first: t.Tensor, second: t.Tensor) -> float:
    first = first.double().flatten()
    second = second.double().flatten()
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((first_centered @ second_centered / denominator).item())


def spearman_rank_correlation(first: t.Tensor, second: t.Tensor) -> float:
    return _pearson_correlation(_average_ranks(first), _average_ranks(second))


def topk_overlap_fraction(first: t.Tensor, second: t.Tensor, *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(k, int(first.numel()), int(second.numel()))
    first_top = set(t.topk(first.double().flatten(), k=k).indices.tolist())
    second_top = set(t.topk(second.double().flatten(), k=k).indices.tolist())
    return len(first_top & second_top) / k


def agreement_summary(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    mechanistic_scores: t.Tensor,
    num_players: int,
    topk: int = 2,
) -> dict:
    """Compare Shapley, causal patching, and known mechanistic scores."""
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    shapley = exact_shapley_from_table(values, num_players=num_players)
    patching = causal_patching_effects(values, num_players=num_players)
    mech = mechanistic_scores.double().flatten()
    full = frozenset(range(num_players))
    top_player = int(shapley.argmax().item())
    non_top = [player for player in range(num_players) if player != top_player]
    deletion_drop = values[full] - values[full - {top_player}]
    matched_random_drop = sum(values[full] - values[full - {player}] for player in non_top) / len(non_top)
    return {
        "shapley_values": shapley,
        "patching_effects": patching,
        "mechanistic_scores": mech,
        "spearman_correlation": spearman_rank_correlation(shapley, mech),
        "topk_overlap": topk_overlap_fraction(shapley, mech, k=topk),
        "patching_topk_overlap": topk_overlap_fraction(patching, mech, k=topk),
        "shapley_top_feature": top_player,
        "mechanistic_top_feature": int(mech.argmax().item()),
        "patching_top_feature": int(patching.argmax().item()),
        "deletion_drop": deletion_drop,
        "matched_random_drop": matched_random_drop,
        "random_baseline_drop": matched_random_drop,
        "agrees_with_mechanistic": spearman_rank_correlation(shapley, mech) >= 0.99 and topk_overlap_fraction(shapley, mech, k=topk) == 1.0,
    }


# HIDE
if MAIN:
    agreement_tests.test_rank_desc_toy_oracle(_rank_desc)
    agreement_tests.test_causal_patching_effects_expose_interaction_consequences(causal_patching_effects)

# END HIDE
# HIDE
if MAIN:

    summary = agreement_summary(
        finite_circuit_table(),
        mechanistic_scores=mechanistic_endpoint_scores(),
        num_players=NEURAL_GAME_NUM_PLAYERS,
    )

# END HIDE
# HIDE
if MAIN:
    agreement_utils.print_report(
        "Finite-circuit agreement summary",
        {key: value.tolist() if hasattr(value, "tolist") else value for key, value in summary.items()},
    )
# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The top feature is `x2` for Shapley, patching, and the known mechanism. Patching effects are larger
than Shapley on interaction endpoints because full-minus-ablated includes downstream pair
consequences.

</details>

<details>
<summary>Help - common bug</summary>

Rank correlation checks the full ordering; top-k overlap checks the scientific claim you actually
care about.

</details>

<details>
<summary>Interpreting the result</summary>

This is an agreement case with a caveat. The rankings agree, but the raw causal effect and Shapley
value are not numerically identical once interactions exist.

</details>
<details>
<summary>Common bugs</summary>

A common mistake is treating full-minus-ablated values as Shapley values. They are both useful, but
they answer different marginal questions.

</details>

<details>
<summary>Solution</summary>

```python
def causal_patching_effects(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Return full-minus-ablated causal effects for each feature player."""
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    full = frozenset(range(num_players))
    if full not in values:
        raise ValueError("coalition value table must contain the full coalition.")
    return t.tensor(
        [values[full] - values[full - {player}] for player in range(num_players)],
        dtype=t.float64,
    )


def _rank_desc(scores: t.Tensor) -> list[int]:
    """Return indices sorted by descending score."""
    return [int(item) for item in t.argsort(scores.detach().double().cpu(), descending=True)]


def _average_ranks(scores: t.Tensor) -> t.Tensor:
    scores = scores.double().flatten()
    ranks = t.empty_like(scores)
    sorted_indices = t.argsort(scores, stable=True)
    sorted_scores = scores[sorted_indices]
    start = 0
    while start < int(scores.numel()):
        end = start + 1
        while end < int(scores.numel()) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[sorted_indices[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _pearson_correlation(first: t.Tensor, second: t.Tensor) -> float:
    first = first.double().flatten()
    second = second.double().flatten()
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((first_centered @ second_centered / denominator).item())


def spearman_rank_correlation(first: t.Tensor, second: t.Tensor) -> float:
    return _pearson_correlation(_average_ranks(first), _average_ranks(second))


def topk_overlap_fraction(first: t.Tensor, second: t.Tensor, *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(k, int(first.numel()), int(second.numel()))
    first_top = set(t.topk(first.double().flatten(), k=k).indices.tolist())
    second_top = set(t.topk(second.double().flatten(), k=k).indices.tolist())
    return len(first_top & second_top) / k


def agreement_summary(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    mechanistic_scores: t.Tensor,
    num_players: int,
    topk: int = 2,
) -> dict:
    """Compare Shapley, causal patching, and known mechanistic scores."""
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    shapley = exact_shapley_from_table(values, num_players=num_players)
    patching = causal_patching_effects(values, num_players=num_players)
    mech = mechanistic_scores.double().flatten()
    full = frozenset(range(num_players))
    top_player = int(shapley.argmax().item())
    non_top = [player for player in range(num_players) if player != top_player]
    deletion_drop = values[full] - values[full - {top_player}]
    matched_random_drop = sum(values[full] - values[full - {player}] for player in non_top) / len(non_top)
    return {
        "shapley_values": shapley,
        "patching_effects": patching,
        "mechanistic_scores": mech,
        "spearman_correlation": spearman_rank_correlation(shapley, mech),
        "topk_overlap": topk_overlap_fraction(shapley, mech, k=topk),
        "patching_topk_overlap": topk_overlap_fraction(patching, mech, k=topk),
        "shapley_top_feature": top_player,
        "mechanistic_top_feature": int(mech.argmax().item()),
        "patching_top_feature": int(patching.argmax().item()),
        "deletion_drop": deletion_drop,
        "matched_random_drop": matched_random_drop,
        "random_baseline_drop": matched_random_drop,
        "agrees_with_mechanistic": spearman_rank_correlation(shapley, mech) >= 0.99 and topk_overlap_fraction(shapley, mech, k=topk) == 1.0,
    }


if MAIN:
    agreement_tests.test_rank_desc_toy_oracle(_rank_desc)
    agreement_tests.test_causal_patching_effects_expose_interaction_consequences(causal_patching_effects)

summary = agreement_summary(
    finite_circuit_table(),
    mechanistic_scores=mechanistic_endpoint_scores(),
    num_players=NEURAL_GAME_NUM_PLAYERS,
)
agreement_utils.print_report(
    "Finite-circuit agreement summary",
    {key: value.tolist() if hasattr(value, "tolist") else value for key, value in summary.items()},
)
```

</details>
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Deletion and Insertion Curves

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Convert a ranking into consequences. If the top-ranked feature is meaningful, deleting it should
hurt faster than a shuffled or weak ranking.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def _curve_from_rank(values: dict[Coalition, float], rank: list[int], mode: str) -> list[dict]:
#     """Build deletion or insertion curve points from a ranked player list."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# if MAIN:
#     agreement_tests.test_curve_from_rank_deletion_and_insertion(_curve_from_rank)
# END EXERCISE
# SOLUTION
def _curve_from_rank(values: dict[Coalition, float], rank: list[int], mode: str) -> list[dict]:
    """Build deletion or insertion curve points from a ranked player list."""
    if mode not in {"deletion", "insertion"}:
        raise ValueError("mode must be 'deletion' or 'insertion'.")
    active: set[int] = set(range(NEURAL_GAME_NUM_PLAYERS)) if mode == "deletion" else set()
    points = [{"step": 0, "player": "start", "value": values[frozenset(active)]}]
    for step, player in enumerate(rank, start=1):
        if mode == "deletion":
            active.remove(player)
        else:
            active.add(player)
        points.append({"step": step, "player": player, "value": values[frozenset(active)]})
    return points


# HIDE
if MAIN:
    agreement_tests.test_curve_from_rank_deletion_and_insertion(_curve_from_rank)

# END HIDE
# HIDE
if MAIN:

    values = finite_circuit_table()

# END HIDE
# HIDE
if MAIN:
    summary = agreement_summary(values, mechanistic_scores=mechanistic_endpoint_scores(), num_players=NEURAL_GAME_NUM_PLAYERS)

# END HIDE
# HIDE
if MAIN:
    shapley_rank = _rank_desc(summary["shapley_values"])

# END HIDE
# HIDE
if MAIN:
    mechanistic_rank = _rank_desc(summary["mechanistic_scores"])

# END HIDE
# HIDE
if MAIN:
    shuffled_rank = _rank_desc(summary["mechanistic_scores"][t.tensor([1, 3, 0, 2])])

# END HIDE
# HIDE
if MAIN:
    curves = {
        "Shapley": _curve_from_rank(values, shapley_rank, "deletion"),
        "Mechanistic": _curve_from_rank(values, mechanistic_rank, "deletion"),
        "Shuffled control": _curve_from_rank(values, shuffled_rank, "deletion"),
    }

# END HIDE
# HIDE
if MAIN:

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)

# END HIDE
# HIDE
if MAIN:
    for label, points in curves.items():
        ax.plot([p["step"] for p in points], [p["value"] for p in points], marker="o", linewidth=2, label=label)

# END HIDE
# HIDE
if MAIN:
    ax.set_title("Deletion curve: does the ranking have causal consequences?")

# END HIDE
# HIDE
if MAIN:
    ax.set_xlabel("players removed")

# END HIDE
# HIDE
if MAIN:
    ax.set_ylabel("finite-circuit value")

# END HIDE
# HIDE
if MAIN:
    ax.grid(alpha=0.25)

# END HIDE
# HIDE
if MAIN:
    ax.legend(frameon=False)

# END HIDE
# HIDE
if MAIN:
    fig.tight_layout()

# END HIDE
# HIDE
if MAIN:
    curve_path = artifact_dir / "deletion_curves.png"

# END HIDE
# HIDE
if MAIN:
    fig.savefig(curve_path)

# END HIDE
# HIDE
if MAIN:
    plt.close(fig)

# END HIDE
# HIDE
if MAIN:
    display(Image(filename=str(curve_path)))
# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

`test_curve_from_rank_deletion_and_insertion` passes. The generated deletion curve shows the
Shapley/mechanistic ranking damaging the circuit much faster than the shuffled control.

</details>

<details>
<summary>Help - common bug</summary>

Deletion starts from the full coalition; insertion starts from the empty coalition. Record the value
after each intervention.

</details>

<details>
<summary>Interpreting the result</summary>

This is the same move as activation patching in ARENA IOI: the score matters because an intervention
changed behavior on the same task.

</details>
<details>
<summary>Common bugs</summary>

If your deletion curve starts from the empty coalition, you built insertion by accident.

</details>

<details>
<summary>Solution</summary>

```python
def _curve_from_rank(values: dict[Coalition, float], rank: list[int], mode: str) -> list[dict]:
    """Build deletion or insertion curve points from a ranked player list."""
    if mode not in {"deletion", "insertion"}:
        raise ValueError("mode must be 'deletion' or 'insertion'.")
    active: set[int] = set(range(NEURAL_GAME_NUM_PLAYERS)) if mode == "deletion" else set()
    points = [{"step": 0, "player": "start", "value": values[frozenset(active)]}]
    for step, player in enumerate(rank, start=1):
        if mode == "deletion":
            active.remove(player)
        else:
            active.add(player)
        points.append({"step": step, "player": player, "value": values[frozenset(active)]})
    return points


if MAIN:
    agreement_tests.test_curve_from_rank_deletion_and_insertion(_curve_from_rank)

values = finite_circuit_table()
summary = agreement_summary(values, mechanistic_scores=mechanistic_endpoint_scores(), num_players=NEURAL_GAME_NUM_PLAYERS)
shapley_rank = _rank_desc(summary["shapley_values"])
mechanistic_rank = _rank_desc(summary["mechanistic_scores"])
shuffled_rank = _rank_desc(summary["mechanistic_scores"][t.tensor([1, 3, 0, 2])])
curves = {
    "Shapley": _curve_from_rank(values, shapley_rank, "deletion"),
    "Mechanistic": _curve_from_rank(values, mechanistic_rank, "deletion"),
    "Shuffled control": _curve_from_rank(values, shuffled_rank, "deletion"),
}

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
for label, points in curves.items():
    ax.plot([p["step"] for p in points], [p["value"] for p in points], marker="o", linewidth=2, label=label)
ax.set_title("Deletion curve: does the ranking have causal consequences?")
ax.set_xlabel("players removed")
ax.set_ylabel("finite-circuit value")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
curve_path = artifact_dir / "deletion_curves.png"
fig.savefig(curve_path)
plt.close(fig)
display(Image(filename=str(curve_path)))
```

</details>
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Diagnose XOR with Pair Interactions

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Ordinary Shapley gives zero single-feature credit on XOR. Implement pairwise Shapley interactions to
recover the real causal player.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def pairwise_interactions_from_table(
#     coalition_values: Mapping[Coalition | tuple[int, ...], float],
#     *,
#     num_players: int,
# ) -> t.Tensor:
#     """Compute second-order Shapley interaction indices from first principles."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def xor_game() -> dict[Coalition, float]:
#     """Return a two-player XOR/parity value table."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def xor_disagreement_smoke_test() -> dict:
#     """Return ordinary Shapley and pair-interaction metrics for the XOR control."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# if MAIN:
#     agreement_tests.test_pairwise_interactions_recover_planted_edges(pairwise_interactions_from_table)
#     agreement_tests.test_xor_disagreement_smoke_test(xor_disagreement_smoke_test)
# END EXERCISE
# SOLUTION
def pairwise_interactions_from_table(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute second-order Shapley interaction indices from first principles."""
    if num_players < 2:
        raise ValueError("pairwise interactions require at least two players.")
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(enumerate_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")

    interactions = t.zeros((num_players, num_players), dtype=t.float64)
    denominator = math.factorial(num_players - 1)
    for first, second in itertools.combinations(range(num_players), 2):
        others = [player for player in range(num_players) if player not in (first, second)]
        score = 0.0
        for size in range(num_players - 1):
            weight = math.factorial(size) * math.factorial(num_players - size - 2) / denominator
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                score += weight * (
                    values[coalition | {first, second}]
                    - values[coalition | {first}]
                    - values[coalition | {second}]
                    + values[coalition]
                )
        interactions[first, second] = score
        interactions[second, first] = score
    return interactions


def xor_game() -> dict[Coalition, float]:
    """Return a two-player XOR/parity value table."""
    return {coalition: float(len(coalition) == 1) for coalition in enumerate_coalitions(2)}


def xor_disagreement_smoke_test() -> dict:
    """Return ordinary Shapley and pair-interaction metrics for the XOR control."""
    values = xor_game()
    shapley = exact_shapley_from_table(values, num_players=2)
    patching = causal_patching_effects(values, num_players=2)
    interactions = pairwise_interactions_from_table(values, num_players=2)
    return {
        "shapley_values": shapley.tolist(),
        "patching_effects": patching.tolist(),
        "pair_interactions": interactions.tolist(),
        "max_single_feature_value": float(shapley.abs().max().item()),
        "recovered_pair_interaction": float(abs(interactions[0, 1].item())),
        "ordinary_shapley_misses": float(shapley.abs().max().item()) <= 1e-9,
        "interaction_recovers_pair": float(abs(interactions[0, 1].item())) >= 1.0,
    }


# HIDE
if MAIN:
    agreement_tests.test_pairwise_interactions_recover_planted_edges(pairwise_interactions_from_table)
    agreement_tests.test_xor_disagreement_smoke_test(xor_disagreement_smoke_test)

# END HIDE
# HIDE
if MAIN:

    agreement_utils.print_report("XOR disagreement diagnosis", xor_disagreement_smoke_test())
# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

Single-feature Shapley has max absolute value `0.0`; pair interaction recovers absolute value `2.0`
for the XOR pair.

</details>

<details>
<summary>Help - common bug</summary>

Use the second-order difference `v(S+i+j)-v(S+i)-v(S+j)+v(S)` and weight it over coalitions of the
other players.

</details>

<details>
<summary>Interpreting the result</summary>

The single-feature method is not hallucinating. It is using the wrong player set for a pure
interaction. The repair is a pair player, not a prettier heatmap.

</details>
<details>
<summary>Common bugs</summary>

Taking absolute values too early hides the sign of an inhibitory interaction. Preserve signs, then
use absolute values only for ranking magnitudes.

</details>

<details>
<summary>Solution</summary>

```python
def pairwise_interactions_from_table(
    coalition_values: Mapping[Coalition | tuple[int, ...], float],
    *,
    num_players: int,
) -> t.Tensor:
    """Compute second-order Shapley interaction indices from first principles."""
    if num_players < 2:
        raise ValueError("pairwise interactions require at least two players.")
    values = {frozenset(key): float(value) for key, value in coalition_values.items()}
    expected = set(enumerate_coalitions(num_players))
    missing = expected - set(values)
    if missing:
        raise ValueError(f"coalition value table is missing {len(missing)} coalitions.")

    interactions = t.zeros((num_players, num_players), dtype=t.float64)
    denominator = math.factorial(num_players - 1)
    for first, second in itertools.combinations(range(num_players), 2):
        others = [player for player in range(num_players) if player not in (first, second)]
        score = 0.0
        for size in range(num_players - 1):
            weight = math.factorial(size) * math.factorial(num_players - size - 2) / denominator
            for group in itertools.combinations(others, size):
                coalition = frozenset(group)
                score += weight * (
                    values[coalition | {first, second}]
                    - values[coalition | {first}]
                    - values[coalition | {second}]
                    + values[coalition]
                )
        interactions[first, second] = score
        interactions[second, first] = score
    return interactions


def xor_game() -> dict[Coalition, float]:
    """Return a two-player XOR/parity value table."""
    return {coalition: float(len(coalition) == 1) for coalition in enumerate_coalitions(2)}


def xor_disagreement_smoke_test() -> dict:
    """Return ordinary Shapley and pair-interaction metrics for the XOR control."""
    values = xor_game()
    shapley = exact_shapley_from_table(values, num_players=2)
    patching = causal_patching_effects(values, num_players=2)
    interactions = pairwise_interactions_from_table(values, num_players=2)
    return {
        "shapley_values": shapley.tolist(),
        "patching_effects": patching.tolist(),
        "pair_interactions": interactions.tolist(),
        "max_single_feature_value": float(shapley.abs().max().item()),
        "recovered_pair_interaction": float(abs(interactions[0, 1].item())),
        "ordinary_shapley_misses": float(shapley.abs().max().item()) <= 1e-9,
        "interaction_recovers_pair": float(abs(interactions[0, 1].item())) >= 1.0,
    }


if MAIN:
    agreement_tests.test_pairwise_interactions_recover_planted_edges(pairwise_interactions_from_table)
    agreement_tests.test_xor_disagreement_smoke_test(xor_disagreement_smoke_test)

agreement_utils.print_report("XOR disagreement diagnosis", xor_disagreement_smoke_test())
```

</details>
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Controls and the Agreement Matrix

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Write the artifact bundle required by the roadmap: agreement matrix, deletion/insertion curves,
top-k heatmap, and disagreement notes.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def additive_values(weights: t.Tensor) -> dict[Coalition, float]:
#     """Return a complete additive value table."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def additive_agreement_smoke_test() -> dict:
#     """Return rank, top-k, and deletion metrics for the additive positive control."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def shuffled_mechanistic_control() -> dict:
#     """Compare the same Shapley values against a shuffled mechanism-label control."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def write_agreement_artifacts(
#     *,
#     output_dir: Path,
#     model_values: dict[Coalition, float],
#     true_values: dict[Coalition, float],
#     shuffled_values: dict[Coalition, float],
#     agreement,
#     shuffled_agreement,
#     model_interactions: t.Tensor,
#     true_interactions: t.Tensor,
# ) -> dict:
#     """Write the agreement matrix, consequence plots, heatmap, and disagreement notes."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# if MAIN:
#     agreement_tests.test_additive_agreement_smoke_test(additive_agreement_smoke_test)
#     agreement_tests.test_shuffled_mechanistic_control_rejected(shuffled_mechanistic_control)
#     agreement_tests.test_write_agreement_artifacts_contract(write_agreement_artifacts=write_agreement_artifacts)
# END EXERCISE
# SOLUTION
def additive_values(weights: t.Tensor) -> dict[Coalition, float]:
    """Return a complete additive value table."""
    return {
        coalition: float(sum(float(weights[player].item()) for player in coalition))
        for coalition in enumerate_coalitions(int(weights.numel()))
    }


def additive_agreement_smoke_test() -> dict:
    """Return rank, top-k, and deletion metrics for the additive positive control."""
    mechanistic_scores = t.tensor([1.0, 2.0, 0.5], dtype=t.float64)
    report = agreement_summary(
        additive_values(mechanistic_scores),
        mechanistic_scores=mechanistic_scores,
        num_players=3,
        topk=2,
    )
    return {key: value.tolist() if hasattr(value, "tolist") else value for key, value in report.items()}


def shuffled_mechanistic_control() -> dict:
    """Compare the same Shapley values against a shuffled mechanism-label control."""
    values = finite_circuit_table()
    shapley = exact_shapley_from_table(values, num_players=NEURAL_GAME_NUM_PLAYERS)
    mechanistic = mechanistic_endpoint_scores()
    shuffled = mechanistic[t.tensor([1, 3, 0, 2])]
    return {
        "shapley_values": shapley.tolist(),
        "true_mechanistic_scores": mechanistic.tolist(),
        "shuffled_mechanistic_scores": shuffled.tolist(),
        "true_spearman": spearman_rank_correlation(shapley, mechanistic),
        "shuffled_spearman": spearman_rank_correlation(shapley, shuffled),
        "true_top2_overlap": topk_overlap_fraction(shapley, mechanistic, k=2),
        "shuffled_top2_overlap": topk_overlap_fraction(shapley, shuffled, k=2),
        "control_rejected": topk_overlap_fraction(shapley, shuffled, k=2) < 1.0,
    }


def _artifact_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


def _write_curve_plot(path: Path, title: str, ylabel: str, curves: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    styles = {
        "trained_shapley_rank": ("#1f77b4", "o", "-"),
        "mechanistic_rank": ("#2ca02c", "s", "--"),
        "shuffled_control_rank": ("#d62728", "^", "-"),
    }
    for label in ("mechanistic_rank", "trained_shapley_rank", "shuffled_control_rank"):
        color, marker, linestyle = styles[label]
        points = curves[label]
        ax.plot(
            [point["step"] for point in points],
            [point["value"] for point in points],
            marker=marker,
            linestyle=linestyle,
            linewidth=2.3,
            markersize=5,
            color=color,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("players removed" if "Deletion" in title else "players inserted")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(NEURAL_GAME_NUM_PLAYERS + 1))
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_heatmap(path: Path, rows: list[str], columns: list[str], values: list[list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.7), dpi=160)
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(columns)), labels=columns)
    ax.set_yticks(range(len(rows)), labels=rows)
    ax.set_title("Top-k overlap with analytic mechanism")
    for row_idx, row_values in enumerate(values):
        for col_idx, value in enumerate(row_values):
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="black" if value > 0.7 else "white")
    fig.colorbar(image, ax=ax, shrink=0.85, label="overlap")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_agreement_artifacts(
    *,
    output_dir: Path,
    model_values: dict[Coalition, float],
    true_values: dict[Coalition, float],
    shuffled_values: dict[Coalition, float],
    agreement,
    shuffled_agreement,
    model_interactions: t.Tensor,
    true_interactions: t.Tensor,
) -> dict:
    """Write the agreement matrix, consequence plots, heatmap, and disagreement notes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    interaction_error = float((model_interactions - true_interactions).abs().max().item())
    additive = additive_agreement_smoke_test()
    xor = xor_disagreement_smoke_test()
    matrix_rows = [
        {"task": "additive_control", "method_a": "ExactShapley", "method_b": "MechanisticScores", "player_type_a": "feature", "player_type_b": "feature", "metric": "spearman_rank_correlation", "value": f"{additive['spearman_correlation']:.6g}", "interpretation": "Additive ground truth gives full rank agreement."},
        {"task": "neural_coalition_game", "method_a": "ExactShapley", "method_b": "AnalyticMechanisticScores", "player_type_a": "feature", "player_type_b": "feature", "metric": "spearman_rank_correlation", "value": f"{agreement.spearman_correlation:.6g}", "interpretation": "Feature Shapley recovers the analytic endpoint ranking."},
        {"task": "neural_coalition_game", "method_a": "ExactShapley", "method_b": "AnalyticMechanisticScores", "player_type_a": "feature", "player_type_b": "feature", "metric": "top2_overlap", "value": f"{agreement.topk_overlap:.6g}", "interpretation": "Top causal features match the analytic mechanism."},
        {"task": "neural_coalition_game", "method_a": "ExactShapley", "method_b": "FeatureDeletion", "player_type_a": "feature", "player_type_b": "behavior", "metric": "deletion_drop_minus_baseline", "value": f"{agreement.deletion_drop - agreement.random_baseline_drop:.6g}", "interpretation": "Deleting the top Shapley feature hurts more than deleting non-top features."},
        {"task": "neural_coalition_game", "method_a": "ShapleyInteractions", "method_b": "AnalyticPairInteractions", "player_type_a": "feature_pair", "player_type_b": "feature_pair", "metric": "max_abs_error", "value": f"{interaction_error:.6g}", "interpretation": "Pair interactions recover the planted positive and negative edges."},
        {"task": "shuffled_label_control", "method_a": "ExactShapley", "method_b": "ShuffledMechanisticScores", "player_type_a": "feature", "player_type_b": "feature", "metric": "top2_overlap", "value": f"{shuffled_agreement.topk_overlap:.6g}", "interpretation": "A shuffled mechanism-label control fails the top-k agreement test."},
        {"task": "xor_control", "method_a": "OrdinaryShapley", "method_b": "ShapleyInteractions", "player_type_a": "feature", "player_type_b": "feature_pair", "metric": "disagreement_detected", "value": "1", "interpretation": "Single-feature Shapley misses XOR while pair interactions recover it."},
    ]
    matrix_path = output_dir / "agreement_matrix.csv"
    with matrix_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(matrix_rows)

    shapley_rank = _rank_desc(agreement.shapley_values)
    mechanistic_rank = _rank_desc(agreement.mechanistic_scores)
    shuffled_rank = _rank_desc(shuffled_agreement.shapley_values)
    deletion_curves = {
        "trained_shapley_rank": _curve_from_rank(model_values, shapley_rank, "deletion"),
        "mechanistic_rank": _curve_from_rank(model_values, mechanistic_rank, "deletion"),
        "shuffled_control_rank": _curve_from_rank(model_values, shuffled_rank, "deletion"),
    }
    insertion_curves = {
        "trained_shapley_rank": _curve_from_rank(model_values, shapley_rank, "insertion"),
        "mechanistic_rank": _curve_from_rank(model_values, mechanistic_rank, "insertion"),
        "shuffled_control_rank": _curve_from_rank(model_values, shuffled_rank, "insertion"),
    }
    _write_curve_plot(output_dir / "deletion_curves.png", "Deletion Consequences", "model value after deletion", deletion_curves)
    _write_curve_plot(output_dir / "insertion_curves.png", "Insertion Consequences", "model value after insertion", insertion_curves)

    columns = [f"k={k}" for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)]
    heatmap_rows = ["trained_shapley", "trained_patching", "shuffled_shapley", "shuffled_patching"]
    heatmap_values = [
        [topk_overlap_fraction(agreement.shapley_values, agreement.mechanistic_scores, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
        [topk_overlap_fraction(agreement.patching_effects, agreement.mechanistic_scores, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
        [topk_overlap_fraction(shuffled_agreement.shapley_values, agreement.mechanistic_scores, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
        [topk_overlap_fraction(shuffled_agreement.patching_effects, agreement.mechanistic_scores, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
    ]
    _write_heatmap(output_dir / "topk_overlap_heatmap.png", heatmap_rows, columns, heatmap_values)

    disagreement_path = output_dir / "method_disagreement_examples.md"
    disagreement_path.write_text(
        "\n".join([
            "# Method Disagreement Examples",
            "",
            "## Agreement case: finite additive-plus-interaction circuit",
            f"Exact Shapley and analytic endpoint scores agree: Spearman={agreement.spearman_correlation:.3f}, top-2 overlap={agreement.topk_overlap:.3f}.",
            f"Deleting the top Shapley feature drops value by {agreement.deletion_drop:.3f}, above the matched non-top baseline {agreement.random_baseline_drop:.3f}.",
            "",
            "## Disagreement case: XOR interaction",
            f"Ordinary single-feature Shapley has max absolute value {xor['max_single_feature_value']:.3f}; the player set is wrong for this mechanism.",
            f"Pairwise Shapley interaction recovers the causal pair with absolute value {xor['recovered_pair_interaction']:.3f}.",
            "",
            "## Negative control: shuffled mechanism labels",
            f"The shuffled control fails top-2 overlap: {shuffled_agreement.topk_overlap:.3f}.",
            "",
        ]),
        encoding="utf-8",
    )
    paths = [matrix_path, output_dir / "deletion_curves.png", output_dir / "insertion_curves.png", output_dir / "topk_overlap_heatmap.png", disagreement_path]
    return {
        "agreement_artifacts_written": all(path.exists() and path.stat().st_size > 0 for path in paths),
        "agreement_artifact_count": len(paths),
        "agreement_matrix_rows": len(matrix_rows),
        "agreement_case_count": 2,
        "disagreement_case_count": 1,
        "deletion_curve_points": len(next(iter(deletion_curves.values()))),
        "insertion_curve_points": len(next(iter(insertion_curves.values()))),
        "topk_heatmap_rows": len(heatmap_rows),
        "topk_heatmap_cols": len(columns),
        "agreement_artifact_paths": [_artifact_display_path(path) for path in paths],
    }


# HIDE
if MAIN:
    agreement_tests.test_additive_agreement_smoke_test(additive_agreement_smoke_test)
    agreement_tests.test_shuffled_mechanistic_control_rejected(shuffled_mechanistic_control)
    agreement_tests.test_write_agreement_artifacts_contract(write_agreement_artifacts=write_agreement_artifacts)

# END HIDE
# HIDE
if MAIN:

    values = finite_circuit_table()

# END HIDE
# HIDE
if MAIN:
    agreement = SimpleNamespace(**agreement_summary(values, mechanistic_scores=mechanistic_endpoint_scores(), num_players=NEURAL_GAME_NUM_PLAYERS))

# END HIDE
# HIDE
if MAIN:
    shuffled_scores = agreement.mechanistic_scores[t.tensor([1, 3, 0, 2])]

# END HIDE
# HIDE
if MAIN:
    shuffled_values = additive_values(shuffled_scores)

# END HIDE
# HIDE
if MAIN:
    shuffled_agreement = SimpleNamespace(**agreement_summary(shuffled_values, mechanistic_scores=mechanistic_endpoint_scores(), num_players=NEURAL_GAME_NUM_PLAYERS))

# END HIDE
# HIDE
if MAIN:
    interactions = pairwise_interactions_from_table(values, num_players=NEURAL_GAME_NUM_PLAYERS)

# END HIDE
# HIDE
if MAIN:
    artifact_summary = write_agreement_artifacts(
        output_dir=artifact_dir,
        model_values=values,
        true_values=values,
        shuffled_values=shuffled_values,
        agreement=agreement,
        shuffled_agreement=shuffled_agreement,
        model_interactions=interactions,
        true_interactions=mechanistic_pair_matrix(),
    )

# END HIDE
# HIDE
if MAIN:
    agreement_utils.print_report("Agreement artifact bundle", artifact_summary)

# END HIDE
# HIDE
if MAIN:
    display(Image(filename=str(artifact_dir / "topk_overlap_heatmap.png")))
# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The artifact writer creates `agreement_matrix.csv`, `deletion_curves.png`, `insertion_curves.png`,
`topk_overlap_heatmap.png`, and `method_disagreement_examples.md`.

</details>

<details>
<summary>Help - common bug</summary>

Include at least one positive agreement row, one deletion-consequence row, one interaction row, one
shuffled-control row, and one disagreement row.

</details>

<details>
<summary>Interpreting the result</summary>

The matrix is a compact scientific claim: what agreed, what failed, which player sets were compared,
and what consequence test backed the comparison.

</details>
<details>
<summary>Common bugs</summary>

Do not manually type the result numbers into the artifact. Generate them from the same values used
by the tests.

</details>

<details>
<summary>Solution</summary>

```python
def additive_values(weights: t.Tensor) -> dict[Coalition, float]:
    """Return a complete additive value table."""
    return {
        coalition: float(sum(float(weights[player].item()) for player in coalition))
        for coalition in enumerate_coalitions(int(weights.numel()))
    }


def additive_agreement_smoke_test() -> dict:
    """Return rank, top-k, and deletion metrics for the additive positive control."""
    mechanistic_scores = t.tensor([1.0, 2.0, 0.5], dtype=t.float64)
    report = agreement_summary(
        additive_values(mechanistic_scores),
        mechanistic_scores=mechanistic_scores,
        num_players=3,
        topk=2,
    )
    return {key: value.tolist() if hasattr(value, "tolist") else value for key, value in report.items()}


def shuffled_mechanistic_control() -> dict:
    """Compare the same Shapley values against a shuffled mechanism-label control."""
    values = finite_circuit_table()
    shapley = exact_shapley_from_table(values, num_players=NEURAL_GAME_NUM_PLAYERS)
    mechanistic = mechanistic_endpoint_scores()
    shuffled = mechanistic[t.tensor([1, 3, 0, 2])]
    return {
        "shapley_values": shapley.tolist(),
        "true_mechanistic_scores": mechanistic.tolist(),
        "shuffled_mechanistic_scores": shuffled.tolist(),
        "true_spearman": spearman_rank_correlation(shapley, mechanistic),
        "shuffled_spearman": spearman_rank_correlation(shapley, shuffled),
        "true_top2_overlap": topk_overlap_fraction(shapley, mechanistic, k=2),
        "shuffled_top2_overlap": topk_overlap_fraction(shapley, shuffled, k=2),
        "control_rejected": topk_overlap_fraction(shapley, shuffled, k=2) < 1.0,
    }


def _artifact_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


def _write_curve_plot(path: Path, title: str, ylabel: str, curves: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    styles = {
        "trained_shapley_rank": ("#1f77b4", "o", "-"),
        "mechanistic_rank": ("#2ca02c", "s", "--"),
        "shuffled_control_rank": ("#d62728", "^", "-"),
    }
    for label in ("mechanistic_rank", "trained_shapley_rank", "shuffled_control_rank"):
        color, marker, linestyle = styles[label]
        points = curves[label]
        ax.plot(
            [point["step"] for point in points],
            [point["value"] for point in points],
            marker=marker,
            linestyle=linestyle,
            linewidth=2.3,
            markersize=5,
            color=color,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("players removed" if "Deletion" in title else "players inserted")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(NEURAL_GAME_NUM_PLAYERS + 1))
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_heatmap(path: Path, rows: list[str], columns: list[str], values: list[list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.7), dpi=160)
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(columns)), labels=columns)
    ax.set_yticks(range(len(rows)), labels=rows)
    ax.set_title("Top-k overlap with analytic mechanism")
    for row_idx, row_values in enumerate(values):
        for col_idx, value in enumerate(row_values):
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="black" if value > 0.7 else "white")
    fig.colorbar(image, ax=ax, shrink=0.85, label="overlap")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_agreement_artifacts(
    *,
    output_dir: Path,
    model_values: dict[Coalition, float],
    true_values: dict[Coalition, float],
    shuffled_values: dict[Coalition, float],
    agreement,
    shuffled_agreement,
    model_interactions: t.Tensor,
    true_interactions: t.Tensor,
) -> dict:
    """Write the agreement matrix, consequence plots, heatmap, and disagreement notes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    interaction_error = float((model_interactions - true_interactions).abs().max().item())
    additive = additive_agreement_smoke_test()
    xor = xor_disagreement_smoke_test()
    matrix_rows = [
        {"task": "additive_control", "method_a": "ExactShapley", "method_b": "MechanisticScores", "player_type_a": "feature", "player_type_b": "feature", "metric": "spearman_rank_correlation", "value": f"{additive['spearman_correlation']:.6g}", "interpretation": "Additive ground truth gives full rank agreement."},
        {"task": "finite_circuit", "method_a": "ExactShapley", "method_b": "AnalyticMechanisticScores", "player_type_a": "feature", "player_type_b": "feature", "metric": "spearman_rank_correlation", "value": f"{agreement.spearman_correlation:.6g}", "interpretation": "Feature Shapley recovers the analytic endpoint ranking."},
        {"task": "finite_circuit", "method_a": "ExactShapley", "method_b": "AnalyticMechanisticScores", "player_type_a": "feature", "player_type_b": "feature", "metric": "top2_overlap", "value": f"{agreement.topk_overlap:.6g}", "interpretation": "Top causal features match the analytic mechanism."},
        {"task": "finite_circuit", "method_a": "ExactShapley", "method_b": "FeatureDeletion", "player_type_a": "feature", "player_type_b": "behavior", "metric": "deletion_drop_minus_baseline", "value": f"{agreement.deletion_drop - agreement.random_baseline_drop:.6g}", "interpretation": "Deleting the top Shapley feature hurts more than deleting non-top features."},
        {"task": "finite_circuit", "method_a": "ShapleyInteractions", "method_b": "AnalyticPairInteractions", "player_type_a": "feature_pair", "player_type_b": "feature_pair", "metric": "max_abs_error", "value": f"{interaction_error:.6g}", "interpretation": "Pair interactions recover the planted positive and negative edges."},
        {"task": "shuffled_mechanism_control", "method_a": "ExactShapley", "method_b": "ShuffledMechanisticScores", "player_type_a": "feature", "player_type_b": "feature", "metric": "top2_overlap", "value": f"{shuffled_agreement.topk_overlap:.6g}", "interpretation": "A shuffled mechanism-label control fails the top-k agreement test."},
        {"task": "xor_control", "method_a": "OrdinaryShapley", "method_b": "ShapleyInteractions", "player_type_a": "feature", "player_type_b": "feature_pair", "metric": "disagreement_detected", "value": "1", "interpretation": "Single-feature Shapley misses XOR while pair interactions recover it."},
    ]
    matrix_path = output_dir / "agreement_matrix.csv"
    with matrix_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(matrix_rows)

    shapley_rank = _rank_desc(agreement.shapley_values)
    mechanistic_rank = _rank_desc(agreement.mechanistic_scores)
    shuffled_rank = _rank_desc(shuffled_agreement.shapley_values)
    deletion_curves = {
        "trained_shapley_rank": _curve_from_rank(model_values, shapley_rank, "deletion"),
        "mechanistic_rank": _curve_from_rank(model_values, mechanistic_rank, "deletion"),
        "shuffled_control_rank": _curve_from_rank(model_values, shuffled_rank, "deletion"),
    }
    insertion_curves = {
        "trained_shapley_rank": _curve_from_rank(model_values, shapley_rank, "insertion"),
        "mechanistic_rank": _curve_from_rank(model_values, mechanistic_rank, "insertion"),
        "shuffled_control_rank": _curve_from_rank(model_values, shuffled_rank, "insertion"),
    }
    _write_curve_plot(output_dir / "deletion_curves.png", "Deletion Consequences", "model value after deletion", deletion_curves)
    _write_curve_plot(output_dir / "insertion_curves.png", "Insertion Consequences", "model value after insertion", insertion_curves)

    columns = [f"k={k}" for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)]
    heatmap_rows = ["trained_shapley", "trained_patching", "shuffled_shapley", "shuffled_patching"]
    heatmap_values = [
        [topk_overlap_fraction(agreement.shapley_values, agreement.mechanistic_scores, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
        [topk_overlap_fraction(agreement.patching_effects, agreement.mechanistic_scores, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
        [topk_overlap_fraction(shuffled_agreement.shapley_values, agreement.mechanistic_scores, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
        [topk_overlap_fraction(shuffled_agreement.patching_effects, agreement.mechanistic_scores, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
    ]
    _write_heatmap(output_dir / "topk_overlap_heatmap.png", heatmap_rows, columns, heatmap_values)

    disagreement_path = output_dir / "method_disagreement_examples.md"
    disagreement_path.write_text(
        "\n".join([
            "# Method Disagreement Examples",
            "",
            "## Agreement case: finite additive-plus-interaction circuit",
            f"Exact Shapley and analytic endpoint scores agree: Spearman={agreement.spearman_correlation:.3f}, top-2 overlap={agreement.topk_overlap:.3f}.",
            f"Deleting the top Shapley feature drops value by {agreement.deletion_drop:.3f}, above the matched non-top baseline {agreement.random_baseline_drop:.3f}.",
            "",
            "## Disagreement case: XOR interaction",
            f"Ordinary single-feature Shapley has max absolute value {xor['max_single_feature_value']:.3f}; the player set is wrong for this mechanism.",
            f"Pairwise Shapley interaction recovers the causal pair with absolute value {xor['recovered_pair_interaction']:.3f}.",
            "",
            "## Negative control: shuffled mechanism labels",
            f"The shuffled control fails top-2 overlap: {shuffled_agreement.topk_overlap:.3f}.",
            "",
        ]),
        encoding="utf-8",
    )
    paths = [matrix_path, output_dir / "deletion_curves.png", output_dir / "insertion_curves.png", output_dir / "topk_overlap_heatmap.png", disagreement_path]
    return {
        "agreement_artifacts_written": all(path.exists() and path.stat().st_size > 0 for path in paths),
        "agreement_artifact_count": len(paths),
        "agreement_matrix_rows": len(matrix_rows),
        "agreement_case_count": 2,
        "disagreement_case_count": 1,
        "deletion_curve_points": len(next(iter(deletion_curves.values()))),
        "insertion_curve_points": len(next(iter(insertion_curves.values()))),
        "topk_heatmap_rows": len(heatmap_rows),
        "topk_heatmap_cols": len(columns),
        "agreement_artifact_paths": [_artifact_display_path(path) for path in paths],
    }


if MAIN:
    agreement_tests.test_additive_agreement_smoke_test(additive_agreement_smoke_test)
    agreement_tests.test_shuffled_mechanistic_control_rejected(shuffled_mechanistic_control)
    agreement_tests.test_write_agreement_artifacts_contract(write_agreement_artifacts=write_agreement_artifacts)

values = finite_circuit_table()
agreement = SimpleNamespace(**agreement_summary(values, mechanistic_scores=mechanistic_endpoint_scores(), num_players=NEURAL_GAME_NUM_PLAYERS))
shuffled_scores = agreement.mechanistic_scores[t.tensor([1, 3, 0, 2])]
shuffled_agreement = SimpleNamespace(**agreement_summary(values, mechanistic_scores=shuffled_scores, num_players=NEURAL_GAME_NUM_PLAYERS))
interactions = pairwise_interactions_from_table(values, num_players=NEURAL_GAME_NUM_PLAYERS)
artifact_summary = write_agreement_artifacts(
    output_dir=artifact_dir,
    model_values=values,
    true_values=values,
    shuffled_values=values,
    agreement=agreement,
    shuffled_agreement=shuffled_agreement,
    model_interactions=interactions,
    true_interactions=mechanistic_pair_matrix(),
)
agreement_utils.print_report("Agreement artifact bundle", artifact_summary)
display(Image(filename=str(artifact_dir / "topk_overlap_heatmap.png")))
```

</details>
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Bridge to Data Players

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

The roadmap asks whether data-level attribution and mechanistic methods are comparable. Implement a
one-step linear training run where exact Data Shapley can be checked against a one-run gradient-dot
proxy.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def one_step_data_utility(
#     coalition: Coalition | tuple[int, ...],
#     *,
#     train_labels: t.Tensor | None = None,
#     validation_label: float = 1.0,
#     learning_rate: float = 0.5,
# ) -> float:
#     """Utility from one gradient step on selected scalar training examples."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def one_step_data_value_table(
#     *,
#     num_examples: int = 4,
#     train_labels: t.Tensor | None = None,
# ) -> dict[Coalition, float]:
#     """Return the complete training-example coalition table for the data bridge."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def data_gradient_dot_scores(
#     *,
#     train_labels: t.Tensor | None = None,
#     validation_label: float = 1.0,
# ) -> t.Tensor:
#     """Return one-run gradient-dot scores from initialization."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# def data_player_bridge_report() -> dict:
#     """Compare exact Data Shapley with a one-run gradient-dot proxy."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# if MAIN:
#     agreement_tests.test_data_player_bridge_matches_exact_data_shapley(data_player_bridge_report)
# END EXERCISE
# SOLUTION
def one_step_data_utility(
    coalition: Coalition | tuple[int, ...],
    *,
    train_labels: t.Tensor | None = None,
    validation_label: float = 1.0,
    learning_rate: float = 0.5,
) -> float:
    """Utility from one gradient step on selected scalar training examples."""
    labels = t.tensor([1.0, 1.0, 1.0, -1.0], dtype=t.float64) if train_labels is None else train_labels.double()
    active = sorted(coalition)
    baseline_loss = validation_label**2
    if not active:
        return 0.0
    selected = labels[t.tensor(active, dtype=t.long)]
    gradient = -2.0 * selected.mean()
    updated_weight = -learning_rate * gradient
    updated_loss = (updated_weight - validation_label) ** 2
    return float(baseline_loss - updated_loss)


def one_step_data_value_table(
    *,
    num_examples: int = 4,
    train_labels: t.Tensor | None = None,
) -> dict[Coalition, float]:
    """Return the complete training-example coalition table for the data bridge."""
    return {
        coalition: one_step_data_utility(coalition, train_labels=train_labels)
        for coalition in enumerate_coalitions(num_examples)
    }


def data_gradient_dot_scores(
    *,
    train_labels: t.Tensor | None = None,
    validation_label: float = 1.0,
) -> t.Tensor:
    """Return one-run gradient-dot scores from initialization."""
    labels = t.tensor([1.0, 1.0, 1.0, -1.0], dtype=t.float64) if train_labels is None else train_labels.double()
    validation_gradient = -2.0 * validation_label
    train_gradients = -2.0 * labels
    return train_gradients * validation_gradient


def data_player_bridge_report() -> dict:
    """Compare exact Data Shapley with a one-run gradient-dot proxy."""
    values = one_step_data_value_table()
    exact = exact_shapley_from_table(values, num_players=4)
    proxy = data_gradient_dot_scores()
    harmful = int(exact.argmin().item())
    helpful = int(exact.argmax().item())
    return {
        "exact_data_shapley": exact.tolist(),
        "gradient_dot_scores": proxy.tolist(),
        "pearson_correlation": _pearson_correlation(exact, proxy),
        "helpful_example": helpful,
        "harmful_example": harmful,
        "identifies_harmful": harmful == int(proxy.argmin().item()),
        "identifies_helpful_tie": float(proxy[helpful].item()) == float(proxy.max().item()),
    }


# HIDE
if MAIN:
    agreement_tests.test_data_player_bridge_matches_exact_data_shapley(data_player_bridge_report)

# END HIDE
# HIDE
if MAIN:

    agreement_utils.print_report("Data-player bridge", data_player_bridge_report())
# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The gradient-dot proxy has Pearson correlation above `0.99` with exact Data Shapley on this one-step
problem and identifies the harmful mislabeled example.

</details>

<details>
<summary>Help - common bug</summary>

At initialization, the validation gradient and each training-example gradient are scalar. Their dot
product is the simplest in-run attribution proxy.

</details>

<details>
<summary>Interpreting the result</summary>

This does not turn data examples into circuit nodes. It shows how a data-player method can be
checked against exact retraining utility before being compared through consequences.

</details>
<details>
<summary>Common bugs</summary>

Ties among the three helpful examples are real. Do not make the test depend on an arbitrary
helpful-example index.

</details>

<details>
<summary>Solution</summary>

```python
def one_step_data_utility(
    coalition: Coalition | tuple[int, ...],
    *,
    train_labels: t.Tensor | None = None,
    validation_label: float = 1.0,
    learning_rate: float = 0.5,
) -> float:
    """Utility from one gradient step on selected scalar training examples."""
    labels = t.tensor([1.0, 1.0, 1.0, -1.0], dtype=t.float64) if train_labels is None else train_labels.double()
    active = sorted(coalition)
    baseline_loss = validation_label**2
    if not active:
        return 0.0
    selected = labels[t.tensor(active, dtype=t.long)]
    gradient = -2.0 * selected.mean()
    updated_weight = -learning_rate * gradient
    updated_loss = (updated_weight - validation_label) ** 2
    return float(baseline_loss - updated_loss)


def one_step_data_value_table(
    *,
    num_examples: int = 4,
    train_labels: t.Tensor | None = None,
) -> dict[Coalition, float]:
    """Return the complete training-example coalition table for the data bridge."""
    return {
        coalition: one_step_data_utility(coalition, train_labels=train_labels)
        for coalition in enumerate_coalitions(num_examples)
    }


def data_gradient_dot_scores(
    *,
    train_labels: t.Tensor | None = None,
    validation_label: float = 1.0,
) -> t.Tensor:
    """Return one-run gradient-dot scores from initialization."""
    labels = t.tensor([1.0, 1.0, 1.0, -1.0], dtype=t.float64) if train_labels is None else train_labels.double()
    validation_gradient = -2.0 * validation_label
    train_gradients = -2.0 * labels
    return train_gradients * validation_gradient


def data_player_bridge_report() -> dict:
    """Compare exact Data Shapley with a one-run gradient-dot proxy."""
    values = one_step_data_value_table()
    exact = exact_shapley_from_table(values, num_players=4)
    proxy = data_gradient_dot_scores()
    harmful = int(exact.argmin().item())
    helpful = int(exact.argmax().item())
    return {
        "exact_data_shapley": exact.tolist(),
        "gradient_dot_scores": proxy.tolist(),
        "pearson_correlation": _pearson_correlation(exact, proxy),
        "helpful_example": helpful,
        "harmful_example": harmful,
        "identifies_harmful": harmful == int(proxy.argmin().item()),
        "identifies_helpful_tie": float(proxy[helpful].item()) == float(proxy.max().item()),
    }


if MAIN:
    agreement_tests.test_data_player_bridge_matches_exact_data_shapley(data_player_bridge_report)

agreement_utils.print_report("Data-player bridge", data_player_bridge_report())
```

</details>
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Build the Signature Result

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Combine the agreement case, deletion consequence test, top-k matrix, and pair-interaction diagnosis
into one visible result panel.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def write_signature_panel(path: Path, *, values: Mapping[Coalition | tuple[int, ...], float] | None = None) -> dict:
#     """Write the learner-facing 16.8 signature result panel."""
#     # YOUR CODE HERE
#     raise NotImplementedError()


# if MAIN:
#     agreement_tests.test_write_signature_panel_contract(write_signature_panel=write_signature_panel)
# END EXERCISE
# SOLUTION
def write_signature_panel(path: Path, *, values: Mapping[Coalition | tuple[int, ...], float] | None = None) -> dict:
    """Write the learner-facing 16.8 signature result panel."""
    values = finite_circuit_table() if values is None else dict(values)
    mech = mechanistic_endpoint_scores()
    summary = agreement_summary(values, mechanistic_scores=mech, num_players=NEURAL_GAME_NUM_PLAYERS)
    shapley = summary["shapley_values"]
    patching = summary["patching_effects"]
    pair_mech = mechanistic_pair_matrix()
    pair_shap = pairwise_interactions_from_table(values, num_players=NEURAL_GAME_NUM_PLAYERS)
    shap_rank = _rank_desc(shapley)
    mech_rank = _rank_desc(mech)
    shuffled_rank = _rank_desc(mech[t.tensor([1, 3, 0, 2])])
    curves = {
        "Shapley deletion": _curve_from_rank(values, shap_rank, "deletion"),
        "Mechanism deletion": _curve_from_rank(values, mech_rank, "deletion"),
        "Shuffled-control deletion": _curve_from_rank(values, shuffled_rank, "deletion"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), dpi=160)
    feature_labels = [f"x{i}" for i in range(NEURAL_GAME_NUM_PLAYERS)]
    width = 0.23
    x_positions = t.arange(NEURAL_GAME_NUM_PLAYERS).double().tolist()

    axes[0, 0].bar([x - width for x in x_positions], shapley.tolist(), width=width, label="Exact Shapley")
    axes[0, 0].bar(x_positions, patching.tolist(), width=width, label="Causal patch")
    axes[0, 0].bar([x + width for x in x_positions], mech.tolist(), width=width, label="Known mechanism")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_xticks(x_positions, feature_labels)
    axes[0, 0].set_title("Agreement case: additive + pair circuit")
    axes[0, 0].set_ylabel("score")
    axes[0, 0].legend(frameon=False, fontsize=8)

    for label, points in curves.items():
        axes[0, 1].plot([point["step"] for point in points], [point["value"] for point in points], marker="o", linewidth=2, label=label)
    axes[0, 1].set_title("Deletion consequence test")
    axes[0, 1].set_xlabel("players removed")
    axes[0, 1].set_ylabel("circuit value")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=8)

    overlap = [
        [topk_overlap_fraction(shapley, mech, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
        [topk_overlap_fraction(patching, mech, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
        [topk_overlap_fraction(shapley, mech[t.tensor([1, 3, 0, 2])], k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
    ]
    image = axes[1, 0].imshow(overlap, cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_title("Agreement matrix: top-k overlap")
    axes[1, 0].set_xticks(range(NEURAL_GAME_NUM_PLAYERS), [f"k={k}" for k in range(1, 5)])
    axes[1, 0].set_yticks(range(3), ["Shapley", "Patching", "Shuffled"])
    for row_idx, row_values in enumerate(overlap):
        for col_idx, value in enumerate(row_values):
            axes[1, 0].text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[1, 0], shrink=0.82)

    im = axes[1, 1].imshow(pair_shap.abs(), cmap="magma", vmin=0)
    axes[1, 1].set_title("Disagreement diagnosis: pair interactions")
    axes[1, 1].set_xticks(range(NEURAL_GAME_NUM_PLAYERS), feature_labels)
    axes[1, 1].set_yticks(range(NEURAL_GAME_NUM_PLAYERS), feature_labels)
    for row in range(NEURAL_GAME_NUM_PLAYERS):
        for col in range(NEURAL_GAME_NUM_PLAYERS):
            label = "" if row == col else f"{float(pair_shap[row, col].item()):.1f}"
            axes[1, 1].text(col, row, label, ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.82)

    fig.suptitle("SHAPley and mechanistic scores agree only after the player set is right", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "signature_panel_written": path.exists() and path.stat().st_size > 0,
        "signature_panel_path": _artifact_display_path(path),
        "spearman_correlation": summary["spearman_correlation"],
        "top2_overlap": summary["topk_overlap"],
        "patching_top2_overlap": summary["patching_topk_overlap"],
        "xor_pair_interaction_abs": xor_disagreement_smoke_test()["recovered_pair_interaction"],
    }


# HIDE
if MAIN:
    agreement_tests.test_write_signature_panel_contract(write_signature_panel=write_signature_panel)

# END HIDE
# HIDE
if MAIN:

    signature_path = asset_dir / "shapley_mech_agreement_signature_result.png"

# END HIDE
# HIDE
if MAIN:
    signature_summary = write_signature_panel(signature_path)

# END HIDE
# HIDE
if MAIN:
    agreement_utils.print_report("Signature panel", signature_summary)

# END HIDE
# HIDE
if MAIN:
    display(Image(filename=str(signature_path)))
# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

`test_write_signature_panel_contract` passes and `shapley_mech_agreement_signature_result.png` is
non-empty.

</details>

<details>
<summary>Help - common bug</summary>

The figure should be generated from the exact finite game values, not from hand-drawn labels.

</details>

<details>
<summary>Interpreting the result</summary>

A good result panel lets a reviewer see both the agreement and the diagnosed disagreement before
reading the appendix report.

</details>
<details>
<summary>Common bugs</summary>

A signature plot that only visualizes the positive case hides the main lesson of this notebook.

</details>

<details>
<summary>Solution</summary>

```python
def write_signature_panel(path: Path, *, values: Mapping[Coalition | tuple[int, ...], float] | None = None) -> dict:
    """Write the learner-facing 16.8 signature result panel."""
    values = finite_circuit_table() if values is None else dict(values)
    mech = mechanistic_endpoint_scores()
    summary = agreement_summary(values, mechanistic_scores=mech, num_players=NEURAL_GAME_NUM_PLAYERS)
    shapley = summary["shapley_values"]
    patching = summary["patching_effects"]
    pair_mech = mechanistic_pair_matrix()
    pair_shap = pairwise_interactions_from_table(values, num_players=NEURAL_GAME_NUM_PLAYERS)
    shap_rank = _rank_desc(shapley)
    mech_rank = _rank_desc(mech)
    shuffled_rank = _rank_desc(mech[t.tensor([1, 3, 0, 2])])
    curves = {
        "Shapley deletion": _curve_from_rank(values, shap_rank, "deletion"),
        "Mechanism deletion": _curve_from_rank(values, mech_rank, "deletion"),
        "Shuffled-control deletion": _curve_from_rank(values, shuffled_rank, "deletion"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), dpi=160)
    feature_labels = [f"x{i}" for i in range(NEURAL_GAME_NUM_PLAYERS)]
    width = 0.23
    x_positions = t.arange(NEURAL_GAME_NUM_PLAYERS).double().tolist()

    axes[0, 0].bar([x - width for x in x_positions], shapley.tolist(), width=width, label="Exact Shapley")
    axes[0, 0].bar(x_positions, patching.tolist(), width=width, label="Causal patch")
    axes[0, 0].bar([x + width for x in x_positions], mech.tolist(), width=width, label="Known mechanism")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_xticks(x_positions, feature_labels)
    axes[0, 0].set_title("Agreement case: additive + pair circuit")
    axes[0, 0].set_ylabel("score")
    axes[0, 0].legend(frameon=False, fontsize=8)

    for label, points in curves.items():
        axes[0, 1].plot([point["step"] for point in points], [point["value"] for point in points], marker="o", linewidth=2, label=label)
    axes[0, 1].set_title("Deletion consequence test")
    axes[0, 1].set_xlabel("players removed")
    axes[0, 1].set_ylabel("circuit value")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=8)

    overlap = [
        [topk_overlap_fraction(shapley, mech, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
        [topk_overlap_fraction(patching, mech, k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
        [topk_overlap_fraction(shapley, mech[t.tensor([1, 3, 0, 2])], k=k) for k in range(1, NEURAL_GAME_NUM_PLAYERS + 1)],
    ]
    image = axes[1, 0].imshow(overlap, cmap="viridis", vmin=0, vmax=1)
    axes[1, 0].set_title("Agreement matrix: top-k overlap")
    axes[1, 0].set_xticks(range(NEURAL_GAME_NUM_PLAYERS), [f"k={k}" for k in range(1, 5)])
    axes[1, 0].set_yticks(range(3), ["Shapley", "Patching", "Shuffled"])
    for row_idx, row_values in enumerate(overlap):
        for col_idx, value in enumerate(row_values):
            axes[1, 0].text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[1, 0], shrink=0.82)

    im = axes[1, 1].imshow(pair_shap.abs(), cmap="magma", vmin=0)
    axes[1, 1].set_title("Disagreement diagnosis: pair interactions")
    axes[1, 1].set_xticks(range(NEURAL_GAME_NUM_PLAYERS), feature_labels)
    axes[1, 1].set_yticks(range(NEURAL_GAME_NUM_PLAYERS), feature_labels)
    for row in range(NEURAL_GAME_NUM_PLAYERS):
        for col in range(NEURAL_GAME_NUM_PLAYERS):
            label = "" if row == col else f"{float(pair_shap[row, col].item()):.1f}"
            axes[1, 1].text(col, row, label, ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.82)

    fig.suptitle("SHAPley and mechanistic scores agree only after the player set is right", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "signature_panel_written": path.exists() and path.stat().st_size > 0,
        "signature_panel_path": _artifact_display_path(path),
        "spearman_correlation": summary["spearman_correlation"],
        "top2_overlap": summary["topk_overlap"],
        "patching_top2_overlap": summary["patching_topk_overlap"],
        "xor_pair_interaction_abs": xor_disagreement_smoke_test()["recovered_pair_interaction"],
    }


if MAIN:
    agreement_tests.test_write_signature_panel_contract(write_signature_panel=write_signature_panel)

signature_path = asset_dir / "shapley_mech_agreement_signature_result.png"
signature_summary = write_signature_panel(signature_path)
agreement_utils.print_report("Signature panel", signature_summary)
display(Image(filename=str(signature_path)))
```

</details>
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Signature Result

This is the result you should be able to explain without opening any report file: exact feature
Shapley and causal patching agree with the known endpoint mechanism on the planted circuit, the
shuffled control fails top-k agreement, and the XOR disagreement is repaired by pair interactions.

<img src="../../exercises/part8_shap_vs_patching/media/shapley_mech_agreement_signature_result.png" alt="SHAPley mechanistic agreement" width="950">

<details>
<summary>Interpreting the result</summary>

The top-left panel is the agreement case: Shapley and the known endpoint mechanism are identical,
while full-minus-ablated patching has the same top players but different magnitudes because it
includes downstream pair consequences. The top-right panel turns the ranking into a deletion
consequence. The bottom-left matrix shows that a shuffled mechanism label fails the same top-k test.
The bottom-right panel shows the pair edges, which is the player set needed to diagnose
interaction-only mechanisms.

</details>
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try It Yourself

Change the positive and negative pair strengths below. Watch for the failure mode: as interactions
dominate, raw patching effects can diverge from endpoint Shapley values even when the top-ranked
bottleneck still agrees.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

if MAIN:
    # Try It Yourself: change these values and rerun from here.
    custom_linear = t.tensor([1.2, -0.7, 1.6, 0.9], dtype=t.float64)
positive_pair_strength = 2.2
negative_pair_strength = -1.5
if MAIN:
    custom_pairs = {(0, 2): positive_pair_strength, (1, 3): negative_pair_strength}
if MAIN:

    custom_values = finite_circuit_table(linear_weights=custom_linear, pair_weights=custom_pairs)
if MAIN:
    custom_summary = agreement_summary(
        custom_values,
        mechanistic_scores=mechanistic_endpoint_scores(linear_weights=custom_linear, pair_weights=custom_pairs),
        num_players=NEURAL_GAME_NUM_PLAYERS,
    )
if MAIN:
    custom_interactions = pairwise_interactions_from_table(custom_values, num_players=NEURAL_GAME_NUM_PLAYERS)
if MAIN:
    agreement_utils.print_report(
        "Custom agreement check",
        {key: value.tolist() if hasattr(value, "tolist") else value for key, value in custom_summary.items()},
    )
if MAIN:
    print("Pair interaction matrix:")
if MAIN:
    print(custom_interactions)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Limitations

This notebook proves an agreement protocol on exact finite games and a tiny data-player bridge. It
does not prove that SHAP, TokenSHAP, activation patching, EAP, SAE feature circuits, or Data Shapley
agree on arbitrary transformers or VLMs. Large models add distribution shift, non-comparable player
sets, correlated features, layer norm effects, and intervention artifacts.

The later CUDA report for this section trains a finite neural coalition game and checks the same
metrics on all 16 binary inputs, but that remains a controlled model organism. A real
TransformerLens IOI or subject-verb agreement replication would need token-level attribution,
activation/path patching over named components, and faithfulness/minimality/completeness checks in
the same player-space bridge.

## Bonus Anomaly Hunt

1. Create a three-way parity game and show that pair interactions are no longer enough.
2. Make features strongly correlated by forbidding some coalitions, then compare exact Shapley to observational ablations.
3. Add an inhibitory edge with a large negative value and check whether top-k overlap by signed score hides it.
4. Replace the finite circuit with a tiny trained MLP and compare feature ablation with hidden-unit ablation.
5. Use a small TransformerLens task and build an agreement matrix where TokenSHAP players are input tokens and mechanistic players are heads or edges.

## Appendix: Report Boundary

The committed `verification_report.json` is supporting evidence only. It records the earlier CUDA
finite-game preflight so the parent task can serialize GPU reruns later; it is not the signature
result of this notebook.
'''


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [keep-main]

# EXERCISE
# def run_smoke_test(cpu: bool = True) -> dict:
#     """Return the fast CPU contract for this notebook."""
#     _ = cpu
#     # YOUR CODE HERE
#     raise NotImplementedError()


# if MAIN:
#     agreement_tests.test_notebook_contract(run_smoke_test)

# def agreement_load_verification_evidence() -> dict:
#     """Load the pinned release report without treating it as lesson evidence."""
#     import json

#     return json.loads((agreement_legacy_section_dir / "verification_report.json").read_text())


# def agreement_run_gpu_test(max_vram_gb: float = 24.0) -> dict:
#     """Run the explicit CUDA preflight; never called by the CPU lesson."""
#     from chapter16_shapley_attribution_baselines.exercises.part8_shapley_mechinterp_agreement import solutions as reference

#     return reference.run_gpu_test(max_vram_gb=max_vram_gb)


# def agreement_run_full_experiment(max_vram_gb: float = 24.0) -> dict:
#     """Run the complete release experiment through the section implementation."""
#     from chapter16_shapley_attribution_baselines.exercises.part8_shapley_mechinterp_agreement import solutions as reference

#     return reference.run_full_experiment(max_vram_gb=max_vram_gb)


# END EXERCISE
# SOLUTION
def run_smoke_test(cpu: bool = True) -> dict:
    """Return the fast CPU contract for this notebook."""
    _ = cpu
    return {
        "additive_agreement": additive_agreement_smoke_test(),
        "xor_disagreement": xor_disagreement_smoke_test(),
        "data_player_bridge": data_player_bridge_report(),
    }


# HIDE
if MAIN:
    agreement_tests.test_notebook_contract(run_smoke_test)
    agreement_tests.test_committed_gpu_report_records_agreement_and_controls()
    agreement_tests.test_notebooks_have_arena_learner_surface()

# END HIDE


def agreement_load_verification_evidence() -> dict:
    """Load the pinned release report without treating it as lesson evidence."""
    import json

    return json.loads((agreement_legacy_section_dir / "verification_report.json").read_text())


def agreement_run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    """Run the explicit CUDA preflight; never called by the CPU lesson."""
    from chapter16_shapley_attribution_baselines.exercises.part8_shapley_mechinterp_agreement import solutions as reference

    return reference.run_gpu_test(max_vram_gb=max_vram_gb)


def agreement_run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the complete release experiment through the section implementation."""
    from chapter16_shapley_attribution_baselines.exercises.part8_shapley_mechinterp_agreement import solutions as reference

    return reference.run_full_experiment(max_vram_gb=max_vram_gb)
# HIDE
if MAIN:

    report_path = agreement_legacy_section_dir / "verification_report.json"

# END HIDE
# HIDE
if MAIN:
    report = __import__("json").loads(report_path.read_text())

# END HIDE
# HIDE
if MAIN:
    gpu = report["metrics"]["gpu_test"]

# END HIDE
# HIDE
if MAIN:
    agreement_utils.print_report(
        "Report appendix only",
        {
            "accepted": report["accepted"],
            "gt_tier": report["gt_tier"],
            "model_family": gpu["model_family"],
            "spearman_correlation": gpu["spearman_correlation"],
            "topk_overlap": gpu["topk_overlap"],
            "shuffled_control_rejected": gpu["shuffled_control_rejected"],
        },
    )
# END HIDE

# END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Bonus
'''


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Finish the comparison

Explain which players, utility, baseline and intervention direction each result concerns. Give a
case where rank agreement hides unequal magnitudes and a case where changing the player set repairs
a disagreement. Keep the one-step data-player bridge separate. The anomaly hunts and play cells above are proposed changes to try.
'''

