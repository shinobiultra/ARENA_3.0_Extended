<!-- CELL: markdown -->
# [14.4] Maze Transformers: Policies, Probes, and State Patching

> **Notebooks:** paired exercises and solutions are in this directory.

**Train a next-action maze policy, compare its paths with exact BFS, then test what its residual stream represents and uses.** The experiment measures behavior and action-state interventions. Its stronger internal-search and route-specificity hypotheses must survive controls.

## Core Question

Does a transformer that predicts the next move use the maze's walls, and do its probes and activation patches reveal more than geometry and next-action information?

You will require four kinds of evidence:

> **behavior -> state decoding -> counterfactual dependence -> causal intervention**

This follows the research rhythm of ARENA's OthelloGPT chapter: first understand exact game state, then decode it, then intervene and ask whether downstream behavior changes as predicted.

<!-- CELL: code -->
import copy
import json
import math
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from IPython.display import display
from torch import Tensor, nn
from torch.nn import functional as F

GT_TIER = "GT-3"
EXERCISE_ID = "14_4_maze_transformers_and_latent_planning"
DIFFICULTY = 5
IMPORTANCE = 5
EXPECTED_RUNTIME = "2-3 hours; about 15-60 seconds for the CPU signature run"
REQUIRES_GPU = True

chapter = "chapter14_jepa_world_models"
section = "part4_maze_transformer_world_model"
root_dir = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / chapter).exists())
section_dir = root_dir / chapter / "exercises" / section
assets_dir = root_dir / chapter / "instructions" / "assets"
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from chapter14_jepa_world_models.exercises.part4_maze_transformer_world_model import tests
from chapter14_jepa_world_models.exercises.part4_maze_transformer_world_model.solutions import (
    ACTION_DELTAS,
    ACTION_NAMES,
    BOS,
    CURRENT,
    CURRENT_GOAL,
    ExperimentArtifacts,
    GOAL,
    MAX_MAZE_SIZE,
    Maze,
    MazeDataset,
    OPEN,
    PAD,
    PatchingResult,
    ProbeResult,
    QUERY,
    QUERY_POSITION,
    Rollout,
    RolloutMetrics,
    SEQUENCE_LENGTH,
    SIZE_5,
    SIZE_6,
    SIZE_7,
    StateExample,
    TrainingResult,
    VOCAB_SIZE,
    WALL,
)

t.set_num_threads(min(8, os.cpu_count() or 1))
plt.style.use("seaborn-v0_8-whitegrid")

<!-- CELL: markdown -->
## Learning Objectives

You will be able to:

- generate deterministic mazes with exact walls, legal moves, BFS distances, route cells, and shortest actions;
- serialize maze states into fixed-length causal sequences;
- implement a decoder-only transformer with an inspectable layer/position cache;
- train a shortest-path policy and preserve immutable checkpoints;
- score path legality, goal reach, and optimality separately;
- train layerwise distance, legal_degree, and action probes on held-out wall layouts;
- compare against random labels, shuffled mazes, and a wall-blind template policy;
- patch route-state residuals and reject wrong-position, matched-norm random, and unrelated-route controls.

## Why This Is a World-Model Question

The training target is only the next action. BFS distance and local legal degree are never supplied as
targets. Here legal degree means the number of open cardinal neighbors; it is not a BFS search frontier. A
distance probe may recover geometry, and a residual patch may transfer an action preference. We compare
against Manhattan distance and same-action donors from different mazes before interpreting these as evidence
of internal search.

Each decision receives the full maze and updated current cell. There is no learned transition model, latent rollout, or search over future latent states in this lab.

<!-- CELL: markdown -->
## Cold Open: A Plausible Path Can Still Be Wrong

Consider three outputs on the same maze:

1. every move is legal, but the path loops;
2. the path reaches the goal, but takes four extra moves;
3. the path reaches the goal in exactly the BFS distance.

These satisfy different claims. Throughout this notebook you will report **legal path**, **goal reach**, and **exact optimality** separately. A rendered line that merely looks path-like is not evidence.

The decoder is evaluated twice: raw argmax exposes illegal moves; legal-action-constrained decoding asks whether its ranking contains a usable plan when exact environment legality is supplied.

<!-- CELL: markdown -->
### Exercise 1: Build the Exact BFS Oracle

> **Difficulty:** 3/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 25 minutes

Implement legal neighbors, shortest path, a full distance-to-go map, and action labels. Start from the hand-computed maze; do not begin with model training.

<details>
<summary>Expected output</summary>

```text
All tests in `test_bfs_shortest_path_hand_oracle` passed!
All tests in `test_distance_map_exact` passed!
```

</details>

<details>
<summary>Help - getting started</summary>

Use a FIFO queue and record a parent the first time a cell is reached. For the distance map, run the same search backward from the goal.

</details>

<details>
<summary>Common bugs</summary>

- using a stack and accidentally implementing DFS;
- revisiting cells and destroying shortest-path guarantees;
- treating walls as merely high-cost cells;
- swapping row and column deltas.

</details>

<details>
<summary>Interpretation</summary>

BFS is the trusted oracle for every later label and metric. The model never gets to define what counts as a good path.

</details>

<details>
<summary>Solution</summary>

{{solution:legal_neighbors, bfs_shortest_path, distance_map, action_between}}

</details>

<!-- CELL: code -->
# EXERCISE: legal_neighbors

# EXERCISE: bfs_shortest_path

# EXERCISE: distance_map

# EXERCISE: action_between

<!-- CELL: code -->
tests.test_bfs_shortest_path_hand_oracle(bfs_shortest_path)
tests.test_distance_map_exact(distance_map)
tests.test_action_between_and_legality(action_between)

<!-- CELL: markdown -->
### Exercise 2: Generate Disjoint Maze Templates

> **Difficulty:** 3/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 20 minutes

Sample solvable mazes, optionally require a route longer than Manhattan distance, and reject wall/start/goal templates already present in another split.

<details>
<summary>Expected output</summary>

```text
All tests in `test_generate_maze_deterministic` passed!
```

</details>

<details>
<summary>Help - getting started</summary>

Seed a local Python RNG, clear start and goal cells, call your BFS oracle, and hash the complete wall grid plus endpoints.

</details>

<details>
<summary>Common bugs</summary>

- comparing only random seeds rather than layouts;
- allowing start or goal to remain walls;
- calling a path a detour when its length equals Manhattan distance.

</details>

<details>
<summary>Interpretation</summary>

A zero-overlap template split blocks the weakest memorization story. It does not by itself prove algorithmic generalization.

</details>

<details>
<summary>Solution</summary>

{{solution:_maze_key, generate_maze, generate_maze_collection}}

</details>

<!-- CELL: code -->
# EXERCISE: _maze_key

# EXERCISE: generate_maze

# EXERCISE: generate_maze_collection

<!-- CELL: code -->
tests.test_generate_maze_deterministic(generate_maze)

<!-- CELL: code -->
def plot_maze(axis, maze, *, paths=(), labels=(), title=""):
    image = np.ones((maze.size, maze.size, 3))
    image[maze.walls.numpy()] = np.array([0.12, 0.14, 0.16])
    axis.imshow(image, interpolation="nearest")
    colours = ["#2364aa", "#e76f51", "#2a9d8f"]
    for path, label, colour in zip(paths, labels, colours):
        if path:
            rows = [cell[0] for cell in path]
            columns = [cell[1] for cell in path]
            axis.plot(columns, rows, marker="o", linewidth=2.2, markersize=4, label=label, color=colour)
    axis.scatter([maze.start[1]], [maze.start[0]], s=110, marker="s", color="#2a9d8f", edgecolor="black", label="start")
    axis.scatter([maze.goal[1]], [maze.goal[0]], s=140, marker="*", color="#f4a261", edgecolor="black", label="goal")
    axis.set_xticks(range(maze.size)); axis.set_yticks(range(maze.size))
    axis.set_xticks(np.arange(-0.5, maze.size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, maze.size, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axis.set_title(title)

preview_maze = generate_maze(19, require_detour=True)
figure, axis = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True)
plot_maze(axis, preview_maze, paths=(preview_maze.path,), labels=("exact BFS",), title=f"Exact detour: distance {preview_maze.distance}")
axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
plt.show()

<!-- CELL: markdown -->
### Exercise 3: Tokenize the Entire Maze State

> **Difficulty:** 2/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 15 minutes

Serialize `[BOS, SIZE, padded row-major grid, QUERY]`. Mark the current and goal cells in the grid itself so the query token can causally attend to the complete state.

<details>
<summary>Expected output</summary>

```text
All tests in `test_tokenize_maze_state` passed!
```

</details>

<details>
<summary>Help - getting started</summary>

Pad smaller mazes to 7 by 7 with wall tokens. Grid token position `2 + 7*row + column` must remain stable across sizes.

</details>

<details>
<summary>Common bugs</summary>

- padding with OPEN;
- moving QUERY before the grid;
- flattening column-major;
- overwriting CURRENT_GOAL incorrectly.

</details>

<details>
<summary>Interpretation</summary>

Stable token geometry lets a positional embedding refer to a physical cell. The causal query is the only position read by the action head.

</details>

<details>
<summary>Solution</summary>

{{solution:tokenize_maze_state}}

</details>

<!-- CELL: code -->
# EXERCISE: tokenize_maze_state

<!-- CELL: code -->
tests.test_tokenize_maze_state(tokenize_maze_state)

<!-- CELL: markdown -->
### Exercise 4: Turn BFS Into Dense Policy Supervision

> **Difficulty:** 3/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 20 minutes

For every reachable non-goal cell, select the first legal neighbor whose exact distance is one smaller. Store
distance and local legal-degree labels for evaluation only. Also record whether the wall-blind template move
is itself shortest. Two optimal moves with different tie-breaking rules must not become a false detour
example.

<details>
<summary>Expected output</summary>

```text
All tests in `test_build_state_dataset_bfs_policy` passed!
```

</details>

<details>
<summary>Help - getting started</summary>

Compute one reverse distance map per maze. Iterate over every open cell, not only the canonical route, so recovery states are in the training distribution.

</details>

<details>
<summary>Common bugs</summary>

- supervising only route states;
- using Manhattan distance through walls;
- leaking distance/local legal-degree labels into the token sequence.

</details>

<details>
<summary>Interpretation</summary>

Dense dynamic-programming supervision fixes a real distribution-shift problem: one rollout mistake should not move the policy into an entirely unseen state type.

</details>

<details>
<summary>Solution</summary>

{{solution:template_action, build_state_dataset}}

</details>

<!-- CELL: code -->
# PROVIDED: template_action

# EXERCISE: build_state_dataset

<!-- CELL: code -->
tests.test_build_state_dataset_bfs_policy(build_state_dataset)
tests.test_detour_labels_exclude_equally_short_ties(build_state_dataset)

<!-- CELL: markdown -->
### Exercise 5: Implement Causal Attention and an Inspectable Block

> **Difficulty:** 4/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 30 minutes

Implement multi-head self-attention with an explicit upper-triangular mask. Return attention patterns and per-head values, then cache residual, attention, and MLP contributions in each block.

<details>
<summary>Expected output</summary>

```text
All tests in `test_attention_is_causal` passed!
```

</details>

<details>
<summary>Help - getting started</summary>

Reshape QKV to `[batch, heads, sequence, head_dim]`; set strictly future scores to negative infinity before softmax.

</details>

<details>
<summary>Common bugs</summary>

- masking the diagonal;
- applying softmax before the mask;
- transposing head and sequence axes;
- caching tensors after they have been overwritten.

</details>

<details>
<summary>Interpretation</summary>

The causality test changes future tokens and demands bitwise-near-identical earlier outputs. A triangular-looking heatmap alone is weaker evidence.

</details>

<details>
<summary>Solution</summary>

{{solution:CausalSelfAttention, TransformerBlock}}

</details>

<!-- CELL: code -->
# EXERCISE: CausalSelfAttention

# EXERCISE: TransformerBlock

<!-- CELL: code -->
tests.test_attention_is_causal(CausalSelfAttention)

<!-- CELL: markdown -->
### Exercise 6: Assemble the Decoder and Residual Cache

> **Difficulty:** 4/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 25 minutes

Add token/position embeddings, causal blocks, final normalization, and a four-action head at QUERY. Support replacing one layer/position residual with an externally supplied vector.

<details>
<summary>Expected output</summary>

```text
All tests in `test_model_cache_and_residual_patch` passed!
```

</details>

<details>
<summary>Help - getting started</summary>

Apply the patch after the selected block and before the next block. Updating the cached `resid_post` keeps cache semantics honest.

</details>

<details>
<summary>Common bugs</summary>

- unembedding every grid position instead of QUERY;
- adding patches rather than replacing them;
- patching after final logits;
- returning detached training logits.

</details>

<details>
<summary>Interpretation</summary>

Replacing the final query residual with another run's residual must exactly reproduce that run's logits. This is the local oracle for the patching machinery, not the scientific result.

</details>

<details>
<summary>Solution</summary>

{{solution:TinyMazeTransformer}}

</details>

<!-- CELL: code -->
# EXERCISE: TinyMazeTransformer

<!-- CELL: code -->
tests.test_model_cache_and_residual_patch(TinyMazeTransformer)

<!-- CELL: markdown -->
### Exercise 7: Train and Preserve Checkpoints

> **Difficulty:** 4/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 30 minutes

Train with cross-entropy on exact next actions. Record train loss and independent-layout accuracy every epoch, and clone CPU state dictionaries at early, middle, and final checkpoints.

<details>
<summary>Expected output</summary>

```text
All tests in `test_training_and_checkpointing` passed!
```

</details>

<details>
<summary>Help - getting started</summary>

Use a seeded permutation each epoch. A checkpoint needs `detach().cpu().clone()`; a shallow state-dict reference mutates as training continues.

</details>

<details>
<summary>Common bugs</summary>

- evaluating on training layouts;
- saving views of live parameters;
- omitting optimizer zeroing;
- reporting only final loss.

</details>

<details>
<summary>Interpretation</summary>

Like the grokking chapter, checkpoint curves let you ask when capabilities and representations appear, rather than treating the final network as timeless.

</details>

<details>
<summary>Solution</summary>

{{solution:action_accuracy, train_maze_transformer}}

</details>

<!-- CELL: code -->
# EXERCISE: action_accuracy

# EXERCISE: train_maze_transformer

<!-- CELL: code -->
tests.test_training_and_checkpointing(train_maze_transformer)

<!-- CELL: markdown -->
### Exercise 8: Roll Out Paths and Score Three Different Claims

> **Difficulty:** 3/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 25 minutes

Autoregress by updating the current-cell token after each action. Evaluate raw argmax and legal-action-constrained decoding, plus a wall-blind axis-first template baseline.

<details>
<summary>Expected output</summary>

```text
All tests in `test_template_rollout_scores_validity_separately` passed!
```

</details>

<details>
<summary>Help - getting started</summary>

Terminate on illegal moves, loops, goal reach, or the step budget. Optimal means reached goal and action count equals exact BFS distance.

</details>

<details>
<summary>Common bugs</summary>

- treating every legal prefix as a valid solution;
- counting goal reach as optimality;
- silently masking illegal raw outputs;
- allowing loops until timeout without marking failure.

</details>

<details>
<summary>Interpretation</summary>

Constrained decoding measures whether model rankings contain a usable policy when environment legality is supplied. Raw illegal rate remains an independent limitation.

</details>

<details>
<summary>Solution</summary>

{{solution:apply_action, rollout_model, template_action, template_rollout, score_rollouts}}

</details>

<!-- CELL: code -->
# EXERCISE: apply_action

# EXERCISE: rollout_model

# EXERCISE: template_action

# EXERCISE: template_rollout

# EXERCISE: score_rollouts

<!-- CELL: code -->
tests.test_template_rollout_scores_validity_separately(template_rollout, score_rollouts)
tests.test_rollout_scores_replay_actions(score_rollouts)
tests.test_raw_and_constrained_rollout_use_learner_policy(rollout_model)

<!-- CELL: markdown -->
### Exercise 9: Decode Distance, Legal Degree, and Next Action Across Layers

> **Difficulty:** 4/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 30 minutes

Cache QUERY representations before any block and after every block. Fit held-out ridge probes for near/medium/far BFS distance, legal-neighbor count, and exact next action.

<details>
<summary>Expected output</summary>

```text
All tests in `test_ridge_probe_toy_oracle` passed!
```

</details>

<details>
<summary>Help - getting started</summary>

Fit preprocessing on train features only. Use independent held-out layouts and compare distance probes against separately shuffled train and test labels.

</details>

<details>
<summary>Common bugs</summary>

- fitting and testing on states from the same mazes;
- leaking held-out normalization statistics;
- reporting training accuracy;
- treating a probe as causal evidence.

</details>

<details>
<summary>Interpretation</summary>

A layer sweep asks *when* state becomes linearly available. Randomized labels test finite-sample probe capacity, but intervention is still needed to test use.

</details>

<details>
<summary>Solution</summary>

{{solution:extract_layer_features, ridge_probe_accuracy, layerwise_probe_metrics}}

</details>

<!-- CELL: code -->
# EXERCISE: extract_layer_features

# EXERCISE: ridge_probe_accuracy

# EXERCISE: layerwise_probe_metrics

<!-- CELL: code -->
tests.test_ridge_probe_toy_oracle(ridge_probe_accuracy)

<!-- CELL: markdown -->
### Exercise 10: Break the Maze-Action Relationship

> **Difficulty:** 3/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 20 minutes

Shuffle wall layouts across examples while preserving each current and goal marker. Evaluate all states and those where the wall-blind move is blocked or not shortest.

<details>
<summary>Expected output</summary>

```text
detour accuracy on intact mazes > shuffled-maze detour accuracy
```

</details>

<details>
<summary>Help - getting started</summary>

Remove CURRENT/GOAL markers from donor grids before overlaying the original markers. Keep labels and split fixed.

</details>

<details>
<summary>Common bugs</summary>

- shuffling labels instead of walls;
- leaving donor start/goal markers;
- evaluating an easier random subset rather than detour states.

</details>

<details>
<summary>Interpretation</summary>

The gap can be modest because many shortest actions point toward the goal regardless of walls. Here detour states require that the template's next move be blocked or fail to reduce exact BFS distance. Disagreement with the canonical action alone is insufficient: multiple next actions can be optimal.

</details>

<details>
<summary>Solution</summary>

{{solution:shuffled_maze_accuracy, detour_action_accuracy}}

</details>

<!-- CELL: code -->
# EXERCISE: shuffled_maze_accuracy

# EXERCISE: detour_action_accuracy

<!-- CELL: code -->
tests.test_shuffle_preserves_markers_and_input(shuffled_maze_accuracy)

<!-- CELL: markdown -->
### Exercise 11: Patch Counterfactual Route State

> **Difficulty:** 5/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 40 minutes

Choose two correctly solved states from the same maze that require different next actions. Patch the donor run's residual into the recipient across layers and positions. Compare a middle-layer query patch against wrong-position, random-direction, unrelated-state, and same-action other-maze donors.

<details>
<summary>Expected output</summary>

```text
final query replacement reproduces donor logits exactly
middle-layer specificity is measured, not assumed
```

</details>

<details>
<summary>Help - getting started</summary>

Normalize the corrupt-vs-clean action margin so clean is 0 and corrupt is 1. Exclude the final-layer exact-recovery cell when making a mechanistic claim.

</details>

<details>
<summary>Common bugs</summary>

- patching mazes with the same target action;
- dividing by a near-zero clean/corrupt gap;
- comparing additive targeted patches to replacement controls;
- claiming final-query recovery as a discovered circuit.

</details>

<details>
<summary>Interpretation</summary>

The intervention asks whether downstream action computation uses information in this residual. A same-action
donor from another maze tests the stronger route-specific interpretation. Compare its effect with targeted
patches on exactly the same eligible pairs, and display the pair count. Similar effects support transferable
action information without identifying a search algorithm.

</details>

<details>
<summary>Solution</summary>

{{solution:_state_tokens, _action_margin, residual_patching_heatmap, _counterfactual_pairs, route_state_intervention_metrics}}

</details>

<!-- CELL: code -->
# PROVIDED: _state_tokens

# PROVIDED: _action_margin

# EXERCISE: residual_patching_heatmap

# EXERCISE: _counterfactual_pairs

# EXERCISE: route_state_intervention_metrics

<!-- CELL: code -->
tests.test_residual_patching_heatmap_has_exact_final_recovery(residual_patching_heatmap)
tests.test_same_action_donor_exposes_action_only_state(route_state_intervention_metrics)

<!-- CELL: markdown -->
### Exercise 12: Run the Complete Controlled Experiment

> **Difficulty:** 4/5 &nbsp; | &nbsp; **Importance:** 5/5 &nbsp; | &nbsp; **Time:** 20 minutes

Combine disjoint datasets, training, checkpoints, rollouts, probes, shuffled mazes, patch controls, and two OOD conditions. Fail loudly when any core claim misses its threshold.

<details>
<summary>Expected output</summary>

```text
held-out action >= 0.72; constrained goal reach >= 0.58; distance probe margin >= 0.25; targeted patch effect >= 0.15 and is positive on >= 85% of pairs
```

</details>

<details>
<summary>Help - getting started</summary>

Keep raw and constrained rollout metrics side by side. Record size-6 transfer even if it fails; it is an OOD diagnostic, not a release gate.

</details>

<details>
<summary>Common bugs</summary>

- selecting successful mazes before computing aggregate rates;
- hiding raw illegal moves;
- reusing training templates;
- averaging targeted and control effects together.

</details>

<details>
<summary>Interpretation</summary>

The experiment supports a narrow controlled claim: usable internal maze state and a causal route-state signal. It does not recover a complete planning algorithm from weights.

</details>

<details>
<summary>Solution</summary>

{{solution:_rollout_suite, run_signature_experiment, validate_experiment}}

</details>

<!-- CELL: code -->
# EXERCISE: _rollout_suite

# EXERCISE: run_signature_experiment

# EXERCISE: validate_experiment

<!-- CELL: code -->
print('Full experiment assembled; run the signature cell once below.')

<!-- CELL: markdown -->
## Signature Result

Run the complete experiment once. Read the panels in order:

The CPU run trains on 260 layouts and holds out 70. Path metrics use the first 60
held-out layouts in generator order; the generator deliberately puts required
detour mazes first. Probes use the first 1,200 state examples per split. These
are fixed diagnostic cohorts, not uniform samples of all possible mazes.

1. inspect one actual held-out maze and compare the predicted route with exact BFS;
2. verify learning on independent wall templates rather than training loss alone;
3. locate linearly decodable distance/legal-degree/action state across layers;
4. compare legal, goal-reaching, and optimal paths across decoding controls;
5. inspect where counterfactual route-state patches matter;
6. compare targeted patches with generic next-action donors on the same eligible pairs.

<!-- CELL: code -->
artifacts = run_signature_experiment(
    device="cpu",
    seed=0,
    train_maze_count=260,
    heldout_maze_count=70,
    epochs=10,
    d_model=56,
)
try:
    validate_experiment(artifacts)
    print("The original numerical thresholds passed; inspect the stronger controls below.")
except AssertionError as error:
    print(f"A numerical threshold failed: {error}. Keep the result visible and inspect the failure.")
metrics = artifacts.metrics

example_index = 0  # Fixed before inspecting outcomes; do not select a successful route.
example_maze = artifacts.heldout_mazes[example_index]
example_rollout = artifacts.model_rollouts[example_index]

figure = plt.figure(figsize=(16, 9), constrained_layout=True)
grid = figure.add_gridspec(2, 3)
path_axis = figure.add_subplot(grid[0, 0])
train_axis = figure.add_subplot(grid[0, 1])
probe_axis = figure.add_subplot(grid[0, 2])
path_metric_axis = figure.add_subplot(grid[1, 0])
patch_axis = figure.add_subplot(grid[1, 1])
control_axis = figure.add_subplot(grid[1, 2])

plot_maze(
    path_axis,
    example_maze,
    paths=(example_maze.path, example_rollout.cells),
    labels=("exact BFS", "model rollout"),
    title=f"Held-out maze: legal={example_rollout.legal}, optimal={example_rollout.optimal}",
)
path_axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))

epochs = np.arange(1, len(artifacts.training.train_loss) + 1)
train_axis.plot(epochs, artifacts.training.train_loss, marker="o", color="#2364aa", label="train CE")
train_axis.set_xlabel("epoch")
train_axis.set_ylabel("cross-entropy", color="#2364aa")
train_axis.tick_params(axis="y", labelcolor="#2364aa")
accuracy_axis = train_axis.twinx()
accuracy_axis.plot(epochs, artifacts.training.heldout_action_accuracy, marker="s", color="#e76f51", label="held-out action")
accuracy_axis.axhline(0.25, linestyle=":", color="#7a7a7a", label="uniform random action")
accuracy_axis.set_ylim(0, 1)
accuracy_axis.legend(fontsize=7, loc="lower right")
accuracy_axis.set_ylabel("held-out accuracy", color="#e76f51")
accuracy_axis.tick_params(axis="y", labelcolor="#e76f51")
train_axis.set_title(f"Learning on disjoint templates (overlap={metrics['train_heldout_template_overlap']})")

layer_names = ["embed", *[f"L{layer}" for layer in range(1, len(artifacts.probes.distance_accuracy))]]
probe_axis.plot(layer_names, artifacts.probes.distance_accuracy, marker="o", label="BFS distance", color="#2a9d8f")
probe_axis.plot(layer_names, artifacts.probes.legal_degree_accuracy, marker="s", label="local legal degree", color="#e9c46a")
probe_axis.plot(layer_names, artifacts.probes.action_accuracy, marker="^", label="next action", color="#2364aa")
probe_axis.plot(layer_names, artifacts.probes.random_label_distance_accuracy, marker="x", linestyle="--", label="random distance labels", color="#7a7a7a")
probe_axis.axhline(artifacts.probes.manhattan_distance_accuracy, color="#8f3d8f", linestyle=":", label="Manhattan distance bins")
probe_axis.set_ylim(0, 1.02)
probe_axis.set_ylabel("held-out probe accuracy")
probe_axis.set_title("State probes and a geometry baseline")
probe_axis.legend(fontsize=8)

conditions = ["constrained", "raw", "wall-blind\ntemplate"]
legal = [metrics["legal_path_rate"], 1 - metrics["raw_illegal_move_rate"], 1 - metrics["template_illegal_move_rate"]]
reach = [metrics["goal_reach_rate"], metrics["raw_goal_reach_rate"], metrics["template_goal_reach_rate"]]
optimal = [metrics["optimal_path_rate"], metrics["raw_optimal_path_rate"], metrics["template_optimal_path_rate"]]
x = np.arange(3); width = 0.25
path_metric_axis.bar(x - width, legal, width, label="all moves legal", color="#2a9d8f")
path_metric_axis.bar(x, reach, width, label="reaches goal", color="#e9c46a")
path_metric_axis.bar(x + width, optimal, width, label="exactly optimal", color="#e76f51")
path_metric_axis.set_xticks(x, conditions)
path_metric_axis.set_ylim(0, 1.05)
path_metric_axis.set_title("A path image is not a metric")
path_metric_axis.legend(fontsize=8)

image = patch_axis.imshow(
    artifacts.patching.heatmap,
    aspect="auto",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
patch_axis.axvline(QUERY_POSITION, color="#f4a261", linewidth=2)
patch_axis.set_yticks(range(artifacts.model.n_layers), [f"after L{index + 1}" for index in range(artifacts.model.n_layers)])
patch_axis.set_xticks([2, 14, 26, 38, 51], ["grid 0", "grid 12", "grid 24", "grid 36", "QUERY"], rotation=35, ha="right")
patch_axis.set_title("Counterfactual patch recovery")
figure.colorbar(image, ax=patch_axis, label="normalized corrupt-action recovery")

control_names = ["targeted\nquery", "wrong\nposition", "norm-matched\nrandom", "unrelated\nstate", "paired\ntargeted", "same action\nother maze"]
control_values = [
    metrics["targeted_patch_effect"], metrics["wrong_position_patch_effect"],
    metrics["matched_norm_random_patch_effect"], metrics["unrelated_route_patch_effect"],
    metrics["same_action_targeted_patch_effect"], metrics["same_action_other_maze_patch_effect"],
]
control_axis.bar(control_names, control_values, color=["#2364aa", "#7a7a7a", "#d1495b", "#e9c46a", "#2364aa", "#8f3d8f"])
control_axis.tick_params(axis="x", labelsize=7)
control_axis.axhline(0, color="black", linewidth=1)
control_axis.set_ylabel("median effect (wrong/random use absolute value)")
control_axis.set_title(f"Middle-layer patches: n={metrics['patch_pair_count']}; paired n={metrics['same_action_patch_pair_count']}")

figure.suptitle("Maze transformer: solve, decode state, then intervene", fontsize=16)
signature_path = assets_dir / "maze_transformer_signature_result.png"
figure.savefig(signature_path, dpi=180, bbox_inches="tight")
plt.show()

summary_rows = [
    ("held-out action", metrics["heldout_action_accuracy"]),
    ("detour action", metrics["detour_action_accuracy"]),
    ("shuffled-maze detour", metrics["shuffled_maze_detour_accuracy"]),
    ("constrained goal reach", metrics["goal_reach_rate"]),
    ("raw goal reach", metrics["raw_goal_reach_rate"]),
    ("wall-blind template goal reach", metrics["template_goal_reach_rate"]),
    ("density OOD goal reach", metrics["density_ood_goal_reach_rate"]),
    ("size-6 OOD goal reach", metrics["size_ood_goal_reach_rate"]),
]
display(pd.DataFrame(summary_rows, columns=["metric", "value"]).style.format({"value": "{:.3f}"}))
print(f"best distance probe: {metrics['best_distance_probe_accuracy']:.3f} vs random labels {metrics['random_label_distance_probe_accuracy']:.3f}")
print(f"targeted patch: {metrics['targeted_patch_effect']:.3f} vs matched random {metrics['matched_norm_random_patch_effect']:.3f}")
print(f"same-action donor: {metrics['same_action_other_maze_patch_effect']:.3f} vs paired targeted {metrics['same_action_targeted_patch_effect']:.3f}")
print(f"Manhattan distance bins: {metrics['manhattan_distance_accuracy']:.3f}")
print(f"saved: {signature_path}")

<!-- CELL: markdown -->
<details>
<summary>Interpretation - how strong is the world-state claim?</summary>

The model is trained only to choose a BFS-downhill action. Held-out action accuracy and optimal constrained
rollouts establish behavior on unseen wall layouts. Distance and legal-degree probes show that useful state is
linearly available inside the query stream. Same-maze route-state patches then change the relative next-action
logits, while wrong-position and matched-random controls are much smaller.

Compare the paired same-action control before calling the intervention route-specific. If an activation from
another maze with the same next action works similarly, the result is compatible with transferring an action
preference. Also compare BFS distance probes with the explicit Manhattan baseline: easy distance bins need not
reveal a search computation.

In the checked CPU run, the policy reaches the goal on 41/60 mazes with legal-action constraints, versus 16/60
with raw argmax and 12/60 for the wall-blind template. All reached paths in this run happen to be shortest;
the scorer tests nonoptimal successes separately. The true-detour accuracy gap after shuffling walls is 0.096.

The stronger interpretation fails its controls: Manhattan distance bins score 0.853 versus the best learned
distance probe at 0.647. On 23 matched pairs, the same-action other-maze donor effect is 0.253 versus 0.189
for the targeted donor. These are descriptive results from one seed, with interventions selected on correctly
classified route states. They support usable next-action information, not a discovered internal search or a
route-specific representation.

</details>

<details>
<summary>Limitations - what remains unresolved?</summary>

Raw argmax still makes many illegal moves; constrained decoding supplies legality from the environment. Size-6
OOD transfer is weak because row/column positions that were padded walls during training become active. The
shuffled-maze detour gap is modest, and colorless 5 by 5 mazes are far simpler than natural planning or
language tasks. The final-query patch is exact by construction, so the scientific comparison uses a middle
layer.

The larger CUDA configuration (900 training layouts, 180 held-out layouts,
18 epochs, width 72) is evaluated separately against the original thresholds.
Before CUDA training, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in a fresh process;
the training function enables deterministic algorithms and highest float32
matrix precision. See PyTorch's [reproducibility guidance](https://docs.pytorch.org/docs/2.12/notes/randomness.html).
Its first 60 rollout layouts have a different
detour composition from the CPU cohort, so do not interpret the two configurations
as a controlled scaling comparison. The thresholds have not been weakened.

</details>

<!-- CELL: markdown -->
## Try It Yourself: Change the Maze, Decoder, and Intervention

Change `PLAY_SEED`, `PLAY_DENSITY`, and `CONSTRAIN_LEGAL`. Compare the rendered route with BFS before looking at the metrics. Then change `PATCH_LAYER` and `PATCH_POSITION`: patching the query should usually matter more than patching an early grid token at the same layer.

<!-- CELL: code -->
PLAY_SEED = 314
PLAY_DENSITY = 0.22
CONSTRAIN_LEGAL = True

play_maze = generate_maze(PLAY_SEED, obstacle_density=PLAY_DENSITY, require_detour=True)
play_rollout = rollout_model(artifacts.model, play_maze, constrain_legal=CONSTRAIN_LEGAL)
figure, axis = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
plot_maze(
    axis,
    play_maze,
    paths=(play_maze.path, play_rollout.cells),
    labels=("exact BFS", "model"),
    title=f"legal={play_rollout.legal}, goal={play_rollout.reached_goal}, optimal={play_rollout.optimal}",
)
axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.show()
print("actions:", [ACTION_NAMES[action] for action in play_rollout.actions])

<!-- CELL: code -->
PATCH_LAYER = max(0, artifacts.model.n_layers - 2)
PATCH_POSITION = QUERY_POSITION
play_pair = _counterfactual_pairs(artifacts.model, artifacts.heldout_mazes, max_pairs=1)[0]
pair_maze, clean_index, corrupt_index = play_pair
device = next(artifacts.model.parameters()).device
clean_tokens = _state_tokens(pair_maze, pair_maze.path[clean_index], device)
corrupt_tokens = _state_tokens(pair_maze, pair_maze.path[corrupt_index], device)
clean_logits, _ = artifacts.model(clean_tokens, return_cache=True)
corrupt_logits, corrupt_cache = artifacts.model(corrupt_tokens, return_cache=True)
replacement = corrupt_cache["blocks"][PATCH_LAYER]["resid_post"][:, PATCH_POSITION]
patched_logits = artifacts.model(
    clean_tokens,
    residual_patch=(PATCH_LAYER, PATCH_POSITION, replacement),
)
probabilities = t.stack((clean_logits.softmax(-1)[0], corrupt_logits.softmax(-1)[0], patched_logits.softmax(-1)[0])).detach().cpu()
display(pd.DataFrame(probabilities.numpy(), index=["clean", "corrupt", "patched"], columns=ACTION_NAMES).style.format("{:.3f}"))

<!-- CELL: markdown -->
## Bonus: Anomaly Hunting

Choose one result that weakens the simple story and investigate it:

1. Why does raw decoding make illegal moves even when constrained optimality is strong?
2. Find a loop made only of legal moves. Which wrong action begins it?
3. Restrict evaluation to detour states. Does shuffling walls hurt more?
4. Compare distance probes at checkpoints. Does the probe rise before or after action accuracy?
5. Explain why patching the final QUERY residual recovers the corrupt logits exactly but says little about where planning happened.
6. Inspect size-6 failures. Which formerly padded positions became active?
7. Replace an unrelated route state with the same target action. Does its control effect increase?

Do not reroll seeds until the anomaly disappears. A failed generalization or control is information about the method's boundary.

## Reading and Research Directions

- [OthelloGPT: Emergent World Representations](https://arxiv.org/abs/2210.13382)
- [Structured World Representations in Maze-Solving Transformers](https://arxiv.org/abs/2312.02566). The paper
studies topology reconstruction and spatial structure. This lab uses dense next-action supervision and local
state probes; it does not reproduce the paper's full-maze reconstruction or adjacency-head analysis.
- Repeat with recurrent state-space models and compare where distance becomes decodable.
- Train on sizes 5 and 6, then test whether size 7 transfer becomes systematic.
- Replace linear probes with causal distance-direction interventions and test path changes.
