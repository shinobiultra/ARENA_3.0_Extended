from __future__ import annotations

import ast
import json
import copy
import re
import textwrap
from pathlib import Path

PART = Path(__file__).resolve().parent
ROOT = PART.parents[2]
SOURCE_PATH = PART / "solutions.py"
EXERCISE_PATH = PART / "5.8_Mixture_of_Experts_from_Scratch_exercises.ipynb"
SOLUTION_PATH = PART / "5.8_Mixture_of_Experts_from_Scratch_solutions.ipynb"
PAGE_PATH = (
    ROOT
    / "chapter5_modern_architectures"
    / "instructions"
    / "pages"
    / "08_[5.8]_Mixture_of_Experts_from_Scratch.md"
)


source_text = SOURCE_PATH.read_text()
source_lines = source_text.splitlines(keepends=True)
tree = ast.parse(source_text)
definitions = {
    node.name: node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
}


def definition_source(name: str) -> str:
    node = definitions[name]
    first_line = min([node.lineno, *(item.lineno for item in node.decorator_list)])
    return "".join(source_lines[first_line - 1 : node.end_lineno]).rstrip()


def definitions_source(*names: str) -> str:
    return "\n\n\n".join(definition_source(name) for name in names)


def set_source(notebook: dict, cell_index: int, text: str) -> None:
    notebook["cells"][cell_index]["source"] = text.rstrip().splitlines(keepends=True)


def wrap_long_markdown_prose(source: str) -> str:
    """Keep regenerated paragraphs readable while leaving fenced code untouched."""

    lines = []
    fenced = False
    for line in source.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and len(line.rstrip("\n")) > 400:
            prefix = re.match(r"\s*(?:> )?", line).group()
            line = textwrap.fill(
                line[len(prefix):].rstrip("\n"), width=100,
                initial_indent=prefix, subsequent_indent=prefix,
                break_long_words=False, break_on_hyphens=False,
            ) + "\n"
        lines.append(line)
    return "".join(lines)


SETUP = """import copy
import itertools
import math
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import Image, display

chapter = "chapter5_modern_architectures"
section = "part8_mixture_of_experts"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part8_mixture_of_experts.tests as tests
from part8_mixture_of_experts.solutions import (
    BOS_TOKEN,
    DATA_TOKEN_OFFSET,
    REGIME_NAMES,
    REGIME_TOKEN_OFFSET,
    ROUTER_PROBE_CHANCE_ACCURACY,
    TASK_CHANCE_ACCURACY,
    TASK_EVALUATOR_SUPPORT,
    VALUE_COUNT,
    VOCAB_SIZE,
    CapacityState,
    CausalInterventionResult,
    CausalSelfAttention,
    MixedSequenceStreams,
    MoEConfig,
    MoEForwardOutput,
    ModelOutput,
    RouterOutput,
    SignatureResult,
    TrainingConfig,
    TrainingTrace,
    sinusoidal_position_table,
)

EXERCISE_ID = "5_8_mixture_of_experts_from_scratch"
GT_TIER = "GT-3"
DIFFICULTY = 4
IMPORTANCE = 5
EXPECTED_RUNTIME = "2-3 hours learner time; measured CPU reference runtime is recorded with the result"
REQUIRES_GPU = False

t.set_num_threads(min(4, t.get_num_threads()))
device = t.device("cpu")
print(f"PyTorch {t.__version__}; learner device = {device}; CUDA is not used")"""


MARKDOWN = {
    0: "# [5.8] Mixture of Experts from Scratch\n"
    "\n"
    "## Core Question\n"
    "\n"
    "Can a sparse transformer solve a mixed instruction task while its router has collapsed, and what "
    "measurements distinguish task accuracy, balanced use, and causal expert dependence?\n"
    "\n"
    "**By the end of this notebook, you will have shown that high task accuracy can coexist with collapsed "
    "routing, and that route preference does not by itself establish causal expert importance.**\n"
    "\n"
    " You will implement deterministic top-k routing, finite capacity, dispatch, combination, and router "
    "losses, then train paired `aux+z`, `aux-only`, `z-only`, and `neither` conditions from the same "
    "initialization for each of three independent seeds. The CPU result separates load-balancing effects "
    "from z-loss effects, hides the current operation token from the router probe, chooses preferred experts "
    "on dev data, and evaluates every expert intervention on disjoint final data.\n"
    "\n"
    '<img src="../assets/mixture_of_experts_signature.png" width="1100">\n'
    "\n"
    "## Learning Objectives\n"
    "\n"
    "You will:\n"
    "\n"
    "- define a portable lower-index tie rule and preserve a task-loss gradient through a top-1 gate;\n"
    "- admit finite-capacity assignments left-to-right within each sequence;\n"
    "- implement differentiable dispatch, combination, independent experts, load balancing, and z-loss;\n"
    "- split decoder-block composition from cache validation and greedy generation;\n"
    "- construct unique mixed train, dev, final-test, and longer-sequence streams;\n"
    "- measure context-only router decodability after hiding the current operation token;\n"
    "- compute signed within-seed main effects and interactions for the paired 2x2 aux/z design;\n"
    "- distinguish soft probability entropy, hard assignment-load entropy, and absolute router-logit scale;\n"
    "- count every position evaluated by the router linear layer separately from FFN slots;\n"
    "- select preferred experts on dev and exhaustively ablate/force all experts on final data.\n"
    "\n"
    "## Roadmap\n"
    "\n"
    "The section follows the compositional rhythm of ARENA 1.1: derive one operation, implement it, test it "
    "immediately, and use it in the next operation. The supplied attention core is ordinary plumbing from "
    "the transformer chapter. Every MoE mechanism and every CPU signature-result training/evaluation "
    "function remains visible below.",
    2: "## Cold Open: a top-1 gate must still train its router\n"
    "\n"
    "For logits `[log(0.6), log(0.3), log(0.1)]`, top-2 selects experts 0 and 1.\n"
    "Our top-k convention renormalizes their gates to `[2/3, 1/3]`. Switch top-1\n"
    "instead keeps the selected full-softmax probability `0.6`: the expert's\n"
    "contribution is `0.6 * expert_0(x)`. Its task-loss derivative reaches the router\n"
    "through an ordinary softmax derivative.\n"
    "\n"
    "Normalizing a single retained probability to one would erase this gradient.\n"
    "We do not use a straight-through estimator. Expert selection itself remains\n"
    "discrete; the gradient differentiates the selected probability, not argmax.\n"
    "Equal logits choose the lower expert index, so ties have a portable meaning.\n",
    3: "### Exercise 1 - implement deterministic `topk_router`\n"
    "\n"
    "> Difficulty: 3/5 | Importance: 5/5 | Suggested time: 18 minutes\n"
    "\n"
    "Return full probabilities, deterministic top-k indices, selected affinities,\n"
    "and retained gates. Use the selected probability for top-1; normalize only\n"
    "when selecting more than one expert.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "For probabilities `[0.6, 0.3, 0.1]`: top-1 gate is `0.6`, top-2 gates are\n"
    "`[2/3, 1/3]`, and the task-loss router gradient is nonzero. Ties resolve to the\n"
    "lower expert index.\n"
    "\n"
    "</details>\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "Use stable descending `argsort` for indices and `gather` for probabilities.\n"
    "Do not detach selected probabilities or normalize the single gate to one.\n"
    "\n"
    "</details>\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "Sparse selection and a task-loss gradient can coexist. The Switch convention\n"
    "makes the selected expert's contribution depend on router confidence.\n"
    "\n"
    "</details>\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    5: "### Exercise 2 - allocate a causal capacity ledger\n"
    "\n"
    "> Difficulty: 4/5 | Importance: 5/5 | Suggested time: 22 minutes\n"
    "\n"
    "For each sequence, allocate `ceil(capacity_factor * active_assignments /\n"
    "number_of_experts)` slots per expert. Visit positions and selected experts\n"
    "left-to-right, dropping assignments once that expert's quota is full. Keep\n"
    "requested counts, retained counts, slots, and drop fraction explicit.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "The hand-computed ledger has capacity 2, requested counts `[3,3,2]`, retained\n"
    "counts `[2,2,2]`, and two drops. Its slots are\n"
    "`[[0,0],[1,0],[-1,1],[1,-1]]`. Altering a later affinity cannot evict an\n"
    "already admitted earlier assignment. Empty and masked rows remain valid.\n"
    "\n"
    "</details>\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "Maintain a counter per expert and sequence. Affinities are part of the router\n"
    "record but do not rank admission here. A global affinity sort would allow\n"
    "future token identity to change an earlier decoder output.\n"
    "\n"
    "</details>\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "This preserves earlier outputs when later identities change within a fixed\n"
    "structural mask and sequence length. Extending that length changes the quota;\n"
    "finite-capacity chunked and full-sequence execution need not agree. Use\n"
    "unbounded capacity for exact cached generation. This per-sequence teaching\n"
    "policy differs from production batch-level expert dispatch.\n"
    "\n"
    "</details>\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    7: "### Exercise 3 - dispatch tokens and combine expert outputs\n"
    "\n"
    "> ```yaml\n"
    "> Difficulty: 4/5\n"
    "> Importance: 5/5\n"
    "> Suggested time: 25 minutes\n"
    "> ```\n"
    "\n"
    "Pack retained token copies into `(experts, capacity, d_model)`, then scatter weighted outputs back. Do "
    "not renormalize after overflow.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "```text\n"
    "manual combined outputs match exactly\n"
    "activation gradient norm > 0 and gate gradient norm > 0\n"
    "All dispatch/combine tests passed!\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "Recover token IDs from flat assignment IDs with integer division by `top_k`. `index_add` combines "
    "multiple retained routes without overwriting them.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "Dispatch is not bookkeeping around the method; it is the sparse computation. The gradient test rules "
    "out an implementation that gets the forward values right by detaching the token or gate path.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    9: "### Exercise 4 - implement independent expert MLPs\n"
    "\n"
    "> ```yaml\n"
    "> Difficulty: 2/5\n"
    "> Importance: 5/5\n"
    "> Suggested time: 12 minutes\n"
    "> ```\n"
    "\n"
    "Build one two-layer GELU MLP per expert and apply expert `e` only to dispatch buffer `e`.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "```text\n"
    "output shape equals input shape\n"
    "expert parameters have different storage\n"
    "input gradients are nonzero\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "Use a `ModuleList` of separately constructed `Sequential` modules. Repeating one module reference would "
    "share parameters.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "Sparse activation does not imply shared weights. Independent expert parameters are the source of the "
    "MoE's larger total parameter count.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    11: "### Exercise 5 - implement load balancing, z-loss, and paired effects\n"
    "\n"
    "> ```yaml\n"
    "> Difficulty: 3/5\n"
    "> Importance: 5/5\n"
    "> Suggested time: 24 minutes\n"
    "> ```\n"
    "\n"
    "Implement the Switch-style soft-importance times detached hard-load loss, squared log-partition "
    "z-loss, and signed effects for the paired 2x2 experiment. A main effect is `mean(with factor) - "
    "mean(without factor)` within one shared initialization.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "```text\n"
    "balanced aux loss = 1; collapsed aux loss > 2.8\n"
    "aux gradient lowers the overloaded expert and raises unused experts\n"
    "common +10 logit shift strongly increases z-loss\n"
    "one z-loss gradient step lowers log-partition RMS without changing a uniform softmax\n"
    "paired aux effect = 5, z effect = 3, interaction = 2\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "Detach the hard assignment fraction but not soft importance. For the factorial contrast, average the "
    "two within-z and within-no-z differences. Do not compare unrelated runs.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "Load balancing changes relative expert use. z-loss penalizes absolute `logsumexp(router_logits)` scale "
    "even when a common logit shift leaves softmax routing unchanged. The signature reports both paired "
    "effects directly.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    13: "### Exercise 6 - assemble the sparse MoE layer\n"
    "\n"
    "> ```yaml\n"
    "> Difficulty: 4/5\n"
    "> Importance: 5/5\n"
    "> Suggested time: 30 minutes\n"
    "> ```\n"
    "\n"
    "Compose router, capacity, dispatch, expert computation, combination, auxiliary losses, forced routing, "
    "and ablation. Return the full debug surface.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "```text\n"
    "unbounded top-2 output matches a token-by-token oracle\n"
    "finite biased top-1 exposes dropped tokens\n"
    "top-1 task path gives nonzero router-weight gradients\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "Pass `route.topk_affinities` to capacity and `route.topk_gates` to combine. Flatten only inside the "
    "MoE layer.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "The brute-force oracle protects routing semantics; the task-gradient test proves the top-1 router can "
    "learn even when both explicit router regularizers are zero.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "After your implementation passes, the code cell prints one complete overflow ledger: hard routes, "
    "requested and kept counts, dropped assignments, and the per-token combined weight.",
    15: "### Exercise 7 - compose one pre-norm MoE decoder block\n"
    "\n"
    "> ```yaml\n"
    "> Difficulty: 3/5\n"
    "> Importance: 5/5\n"
    "> Suggested time: 15 minutes\n"
    "> ```\n"
    "\n"
    "Apply supplied causal attention with its residual, then your normalized sparse MoE with a second "
    "residual. Return the attention cache and MoE diagnostics.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "```text\n"
    "block output and MoE output preserve `(batch, sequence, d_model)`\n"
    "cache shape is `(batch, heads, sequence, head_dim)`\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "Keep attention and MoE layer norms separate. The residual path belongs to the block, not `SparseMoE`.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "This deliberately stops before cache orchestration. Splitting the old oversized exercise gives "
    "immediate feedback on block composition before model-level validation.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    17: "### Exercise 8 - build and validate the cached MoE transformer\n"
    "\n"
    "> ```yaml\n"
    "> Difficulty: 4/5\n"
    "> Importance: 5/5\n"
    "> Suggested time: 30 minutes\n"
    "> ```\n"
    "\n"
    "Stack blocks, tie token embeddings, validate every cache, and implement greedy cached generation. "
    "Full/chunk parity is required only with unbounded capacity.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "```text\n"
    "unbounded full and tokenwise cached logits agree\n"
    "finite-capacity full pass keeps 4 assignments; token chunks keep 6\n"
    "greedy cache output matches full-prefix argmax\n"
    "malformed caches raise ValueError\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "All layer caches must be simultaneously empty or populated, have matching `(batch, heads, time, "
    "head_dim)` shapes, and share past length. Use `capacity_factor=inf` in greedy generation.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "A finite-capacity router allocates slots within each forward call. Changing chunk boundaries changes "
    "the competition set and can change kept assignments, so finite-capacity cache parity is not a valid "
    "invariant.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    19: "### Exercise 9 - construct the mixed-operation language task\n"
    "\n"
    "> ```yaml\n"
    "> Difficulty: 3/5\n"
    "> Importance: 5/5\n"
    "> Suggested time: 22 minutes\n"
    "> ```\n"
    "\n"
    "Generate `BOS, value, operation, next_value, ...` streams where operations can change at every "
    "transition. Grade only predictions after operation tokens.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "```text\n"
    "copy/successor/predecessor exact examples match\n"
    "perfect operation-position logits give loss < 1e-6\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "For a mixed row, `regimes` has shape `(batch, n_data_tokens - 1)`. Interleave each operation "
    "immediately before the value it determines.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "Every target has exact ground truth. The fixed-position objective grades the same routed positions for "
    "every row while allowing operation identities to be genuinely mixed.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    21: "### Exercise 10 - fit a current-token-delexicalized router probe\n"
    "\n"
    "> ```yaml\n"
    "> Difficulty: 4/5\n"
    "> Importance: 5/5\n"
    "> Suggested time: 25 minutes\n"
    "> ```\n"
    "\n"
    "Build one row per operation event, preserve its causal prefix, replace the current operation token "
    "with `BOS_TOKEN`, and extract the router probabilities at that position. Then fit a closed-form ridge "
    "probe with a bias column on dev and evaluate on disjoint final data.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "```text\n"
    "matched prefixes with copy/successor/predecessor labels have identical delexicalized features\n"
    "one-hot features recover all labels in the probe-mechanics test\n"
    "signature context-only accuracy is near 1/3, not the leaked current-token score\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "Use `nonzero(operation_mask, as_tuple=True)` to gather event rows and positions. Clone each source "
    "row, replace only its current operation token, and rely on causal attention to exclude future tokens. "
    "Solve the ridge normal equation with `torch.linalg.solve`.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "The operation token is a legitimate input to the model but an invalid feature for a probe whose target "
    "is that token's identity. This probe asks only whether the preceding context predicts the "
    "independently assigned current operation. Direct routes by operation are reported separately as "
    "descriptive behavior.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    23: "### Exercise 11 - test expert-label symmetry\n"
    "\n"
    "> ```yaml\n"
    "> Difficulty: 3/5\n"
    "> Importance: 5/5\n"
    "> Suggested time: 12 minutes\n"
    "> ```\n"
    "\n"
    "Deep-copy an MoE and apply the same permutation to router output rows and expert modules.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "```text\n"
    "coherently permuted and original outputs agree to floating precision\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "If new expert `j` is old expert `permutation[j]`, new router row `j` must be the same old row.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "Expert numbers have no intrinsic meaning. The signature heatmap is one seed with arbitrary expert "
    "labels, not a canonical cross-seed alignment.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    25: "### Exercise 12 - run exhaustive fixed-model interventions\n"
    "\n"
    "> ```yaml\n"
    "> Difficulty: 4/5\n"
    "> Importance: 5/5\n"
    "> Suggested time: 20 minutes\n"
    "> ```\n"
    "\n"
    "For one trained model and one fixed final stream, score every operation after ablating each expert and "
    "after forcing each expert. Preferred experts are selected separately from dev routing; the final "
    "stream is used only for effect estimation.\n"
    "\n"
    "<details><summary>Expected output</summary>\n"
    "\n"
    "```text\n"
    "intact `(3,)`, ablated `(3, 3)`, forced `(3, 3)`\n"
    "all nine ablations and all nine forcing cells match direct evaluations\n"
    "```\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Help</summary>\n"
    "\n"
    "Compute the intact vector once, then stack one evaluation per expert along the expert axis. There are "
    "no random intervention “seeds”: the three trained model seeds are the independent repetitions, while "
    "each fixed model uses the exhaustive table.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Interpretation</summary>\n"
    "\n"
    "These interventions test functional dependence within each independently trained toy model; a "
    "near-zero ablation effect is a failed importance claim, even when forcing an expert causes damage. "
    "Expert numbers remain permutation-symmetric, and preferred-expert selection is protected from "
    "final-test reuse.\n"
    "\n"
    "</details>\n"
    "\n"
    "<details><summary>Solution</summary>\n"
    "\n"
    "```python\n"
    "{{solution}}\n"
    "```\n"
    "\n"
    "</details>",
    27: "## Visible experiment code: streams and dense baseline\n"
    "\n"
    "The next cells define the unique mixed streams and equal-width dense baseline. Train, dev, and final "
    "rows are sampled without replacement; the longer evaluation independently samples unique 17-data-token "
    "rows.\n"
    "\n"
    "The dense block gathers only graded operation positions before its FFN. Sparse experts execute their "
    "padded `(experts, capacity)` buffer. Ceiling each sequence's capacity pads the default sparse buffer "
    "to 1,620 slots versus 1,440 dense slots; equal width is not equal executed work. The router linear "
    "layer still evaluates **all input positions**, including positions excluded from expert capacity; its "
    "FLOPs are counted from the full router-logit tensor.",
    30: "## Visible experiment code: paired training, probing, interventions, and accounting\n"
    "\n"
    "For each seed, one sparse initialization is deep-copied into the four aux/z conditions and its shared "
    "weights are copied into the dense baseline. All five conditions receive identical ordered batches. "
    "Seeds 7, 19, and 31 are independently initialized, trained, and evaluated repetitions.\n"
    "\n"
    "The 2x2 conditions produce signed paired main effects and interactions for hard load, overflow, "
    "router-logit RMS, and log-partition RMS. The context-only probe is fit on dev and scored on final "
    "after hiding each current operation token. Preferred experts are selected from dev routes; every "
    "ablation and force intervention is then evaluated on final. The CPU function is the learner contract. "
    "`run_gpu_test` and `run_full_experiment` are strict CUDA release hooks and never fall back to CPU.",
    32: "## Signature Result: task accuracy cannot certify routing health\n"
    "\n"
    "Run the complete CPU experiment from the definitions above. This is live training over three "
    "independent seeds, not a fixture or serialized report. The output prints routing, capacity ledgers, "
    "factorial effects, and exhaustive intervention tables before rendering the six-panel result.\n"
    "\n"
    "Before running, write down two distinct hypotheses: (1) load-balancing loss\n"
    "changes routing health; (2) the dev-preferred expert is more necessary than\n"
    "other experts. Keep the original positive-specialization decision rule below.\n"
    "A failed rule is a result to explain, not a reason to change its threshold.\n",
    34: "## Interpreting the Result\n"
    "\n"
    "- **Panel A:** all models are scored over the evaluator's full 15-logit prediction support, so uniform "
    "chance is `1/15`. The five data tokens are the only targets, but the evaluator does not suppress the "
    "other ten logits.\n"
    "- **Panel B:** final hard routes are descriptive because the operation token legitimately enters the "
    "model. Stars use preferred experts selected on disjoint dev data. The separately reported "
    "delexicalized context probe stays near its three-class `1/3` chance.\n"
    "- **Panel C:** each dot is one independently initialized/trained seed. Positive aux effects on "
    "hard-load entropy and negative aux effects on drop fraction directly test load balancing; a negative z "
    "effect on log-partition RMS directly tests scale control.\n"
    "- **Panel D:** the first seed's fixed model is evaluated in every expert-by-operation ablation cell. "
    "No resampled choices are treated as repetitions.\n"
    "- **Panel E:** sparse FFNs execute 12.5% more padded slots than the dense control on this stream. "
    "Router FLOPs use every routed input position, not only operation positions admitted to experts.\n"
    "- **Panel F:** raw log-partition RMS by condition makes the z-loss scale effect visible alongside the "
    "paired contrast.\n"
    "\n"
    "The preferred-expert ablation hypothesis is rejected on this run: the mean preferred drop is zero, "
    "while forcing nonpreferred experts is disruptive. These are different interventions. Preserved argmax "
    "accuracy does not prove zero logit effect, and forcing damage does not establish that a normally used "
    "expert is necessary. The long-stream accuracy near 0.66 is another visible limitation.",
    35: "## Try It Yourself\n"
    "\n"
    "Change `play_aux`, `play_z`, or `play_capacity`, then inspect the training trace, requested/kept "
    "expert counts, dev and final route tables, and every final intervention. This 60-step run is "
    "exploratory and is not used by signature acceptance checks.",
    37: "## Limitations\n"
    "\n"
    "- The task has five target values, three operations, one small decoder layer, and deliberately biased "
    "router initialization. It is a controlled model organism.\n"
    "- The full-vocabulary uniform chance line is `1/15`; a classifier restricted to the five known target "
    "tokens would instead have `1/5` chance. This evaluator applies no such restriction.\n"
    "- Current-token delexicalization removes direct operation-token identity but does not erase "
    "information in the causal prefix. Near-chance accuracy is therefore evidence about this generated "
    "stream, not a universal guarantee.\n"
    "- The three seeds are genuine independent training repetitions, but three is still too few for "
    "population-level uncertainty claims. Expert identities are not aligned across seeds.\n"
    "- Exhaustive interventions establish fixed-model dependence. They do not identify a unique mechanism "
    "when expert outputs can be redundant or compensatory.\n"
    "- Padded expert buffers overstate optimized sparse-kernel work; accounting includes FFN and router "
    "matmuls, not GELU, bias, attention, communication, memory traffic, or wall-time parity.\n"
    "- Finite capacity is allocated per forward call. Full-sequence and chunked decoding can retain "
    "different assignments; greedy cache tests use unbounded capacity.\n"
    "- Switch top-1 uses its selected softmax probability; top-k with k greater than one uses a "
    "renormalized convention. The discrete selection is not differentiated.\n"
    "- The complete learner path is CPU-only and uses no external checkpoint. CUDA is required only by "
    "strict parent release wrappers and is intentionally not exercised here.\n"
    "\n"
    "## Further Research / Anomaly Hunting\n"
    "\n"
    "- Increase `top_k` to 2 and compare Switch probability scaling with normalized multi-expert gates.\n"
    "- Add independent seeds and align experts by behavior before aggregating route heatmaps.\n"
    "- Replace padded expert loops with a segmented kernel and extend accounting to communication, memory "
    "traffic, and measured latency.\n"
    "- Repeat the context-only assay on a stream with deliberately predictable operation schedules to see "
    "the control rise for a valid reason.\n"
    "\n"
    "## Reading\n"
    "\n"
    "- Shazeer et al., [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)\n"
    "- Fedus et al., [Switch Transformers](https://arxiv.org/abs/2101.03961)\n"
    "- Zoph et al., [ST-MoE](https://arxiv.org/abs/2202.08906)\n"
    "\n"
    "## What You Established\n"
    "\n"
    "You established exact routing and capacity semantics, direct aux/z interpretation tests, "
    "current-token-delexicalized probing, all-position router accounting, dev-selected exhaustive "
    "interventions, and repeated paired evidence from three independently trained models.",
}


OVERFLOW_DEMO = """overflow_demo = SparseMoE(
    MoEConfig(d_model=8, n_heads=2, d_expert=12, router_bias_init=20)
).eval()
overflow_output = overflow_demo(t.zeros(2, 6, 8), capacity_factor=1.0)
print({
    "top1_experts": overflow_output.router.topk_indices[..., 0].tolist(),
    "requested_counts": overflow_output.capacity.requested_counts.tolist(),
    "kept_counts": overflow_output.capacity.kept_counts.tolist(),
    "dropped_count": overflow_output.capacity.dropped_count,
    "combined_weight": overflow_output.combined_weight.tolist(),
})"""


DELEX_DEMO = """delex_demo_operations = t.tensor([
    [0, 1, 2, 0],
    [1, 1, 2, 0],
    [2, 1, 2, 0],
])
delex_demo_sequences = make_regime_sequences(
    delex_demo_operations,
    t.tensor([2, 2, 2]),
    n_data_tokens=5,
)
delex_demo_model = SparseMoETransformer(
    MoEConfig(d_model=12, n_heads=3, d_expert=18, max_seq_len=20)
).eval()
delex_demo_features, delex_demo_labels = delexicalized_router_event_features(
    delex_demo_model,
    delex_demo_sequences,
)
delex_first_events = t.arange(3) * delex_demo_operations.shape[1]
print("first-event labels:", delex_demo_labels[delex_first_events].tolist())
print("matched-prefix delexicalized features:\\n", delex_demo_features[delex_first_events])"""


EXERCISE_CODE = {
    4: """def topk_router(router_logits: t.Tensor, top_k: int) -> RouterOutput:
    raise NotImplementedError()

tests.test_topk_router_exact(topk_router)
tests.test_topk_router_ties_are_portable_and_switch_top1_has_task_gradient(topk_router)""",
    6: """def compute_capacity_state(
    topk_indices, topk_affinities, n_experts, capacity_factor, token_mask=None
):
    raise NotImplementedError()

tests.test_capacity_ledger_exact(compute_capacity_state)
tests.test_capacity_is_left_to_right_and_handles_partial_masked_and_empty_batches(compute_capacity_state)
tests.test_capacity_is_prefix_and_batch_invariant(compute_capacity_state)""",
    8: """def dispatch_tokens(token_activations, topk_indices, capacity, n_experts):
    raise NotImplementedError()

def combine_expert_outputs(expert_outputs, topk_indices, topk_gates, capacity):
    raise NotImplementedError()

tests.test_dispatch_and_combine_manual(dispatch_tokens, combine_expert_outputs)
tests.test_dispatch_and_combine_preserve_activation_and_gate_gradients(dispatch_tokens, combine_expert_outputs)""",
    10: """class ExpertMLPs(nn.Module):
    def __init__(self, n_experts, d_model, d_expert):
        super().__init__()
        raise NotImplementedError()

    def forward(self, dispatched):
        raise NotImplementedError()

tests.test_expert_mlps_are_independent_and_differentiable(ExpertMLPs)""",
    12: """def load_balancing_loss(router_probabilities, topk_indices, token_mask=None):
    raise NotImplementedError()

def router_z_loss(router_logits, token_mask=None):
    raise NotImplementedError()

def paired_factorial_effects(condition_values):
    raise NotImplementedError()

tests.test_router_losses_have_semantic_controls(load_balancing_loss, router_z_loss)
tests.test_paired_factorial_effects_are_signed_with_minus_without(paired_factorial_effects)""",
    14: """class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError()

    def forward(
        self,
        hidden,
        *,
        token_mask=None,
        capacity_factor=None,
        force_expert=None,
        ablate_experts=(),
    ):
        raise NotImplementedError()

tests.test_sparse_moe_matches_unbounded_bruteforce(SparseMoE)
tests.test_capacity_drop_accounting_is_visible(SparseMoE)
tests.test_top1_sparse_moe_language_path_reaches_router(SparseMoE)"""
    + "\n\n"
    + OVERFLOW_DEMO,
    16: """class MoETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError()

    def forward(self, hidden, *, past_kv=None, token_mask=None, capacity_factor=None, force_expert=None, ablate_experts=()):
        raise NotImplementedError()

tests.test_moe_transformer_block_shapes(MoETransformerBlock)""",
    18: """class SparseMoETransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError()

    def forward(self, input_ids, *, caches=None, use_cache=False, capacity_factor=None, force_expert=None, ablate_experts=(), moe_token_mask=None):
        raise NotImplementedError()

    def greedy_generate(self, input_ids, max_new_tokens):
        raise NotImplementedError()

tests.test_causal_attention_has_no_future_leakage(CausalSelfAttention)
tests.test_complete_model_and_cache_parity(SparseMoETransformer)
tests.test_finite_capacity_resets_per_chunk_and_unbounded_greedy_is_exact(SparseMoETransformer)
tests.test_finite_capacity_decoder_has_prefix_and_batch_invariance(SparseMoETransformer)
tests.test_malformed_caches_are_rejected(SparseMoETransformer)""",
    20: """def regime_transition(regime, value, modulus=5):
    raise NotImplementedError()

def make_regime_sequences(regimes, starts, n_data_tokens, modulus=5):
    raise NotImplementedError()

def next_token_loss(logits, sequences):
    raise NotImplementedError()

tests.test_regime_dataset_is_exact_and_causal(make_regime_sequences)
tests.test_next_token_objective_grades_only_operations(next_token_loss)
tests.test_chance_baselines_match_evaluator_support()""",
    22: """@t.inference_mode()
def delexicalized_router_event_features(model, sequences):
    raise NotImplementedError()

def fit_router_probe(features, labels, n_classes=3, ridge=1e-3):
    raise NotImplementedError()

def predict_router_probe(features, weights):
    raise NotImplementedError()

tests.test_router_probe_recovers_known_histograms(fit_router_probe, predict_router_probe)
tests.test_delexicalized_router_probe_hides_current_operation_token(delexicalized_router_event_features)"""
    + "\n\n"
    + DELEX_DEMO,
    24: """def permute_moe_experts(moe: SparseMoE, permutation: Sequence[int]) -> SparseMoE:
    raise NotImplementedError()

tests.test_coherent_expert_permutation_is_invariant(permute_moe_experts)""",
    26: """def next_token_accuracy_by_regime(logits, sequences):
    raise NotImplementedError()

@t.inference_mode()
def evaluate_sparse_by_regime(model, sequences, *, force_expert=None, ablate_experts=()):
    raise NotImplementedError()

@t.inference_mode()
def causal_expert_intervention_table(model, sequences):
    raise NotImplementedError()

tests.test_causal_intervention_table_matches_direct_runs(causal_expert_intervention_table)""",
}


SOLUTION_GROUPS = {
    4: ("topk_router",),
    6: ("compute_capacity_state",),
    8: ("dispatch_tokens", "combine_expert_outputs"),
    10: ("ExpertMLPs",),
    12: ("load_balancing_loss", "router_z_loss", "paired_factorial_effects"),
    14: ("SparseMoE",),
    16: ("MoETransformerBlock",),
    18: ("SparseMoETransformer",),
    20: ("regime_transition", "make_regime_sequences", "next_token_loss"),
    22: ("delexicalized_router_event_features", "fit_router_probe", "predict_router_probe"),
    24: ("permute_moe_experts",),
    26: (
        "next_token_accuracy_by_regime",
        "evaluate_sparse_by_regime",
        "causal_expert_intervention_table",
    ),
    28: (
        "next_token_accuracy",
        "normalized_router_entropy",
        "normalized_load_entropy",
        "router_logit_rms",
        "router_log_partition_rms",
        "_operation_rows",
        "_sequence_signatures",
        "_enumerate_short_mixed_examples",
        "_sample_unique_long_mixed_examples",
        "build_disjoint_mixed_streams",
    ),
    29: ("DenseTransformerBlock", "DenseTransformer"),
    31: (
        "train_sparse_model",
        "train_dense_model",
        "routing_by_regime",
        "evaluate_sparse",
        "count_parameters",
        "clone_factorial_models",
        "measured_compute_accounting",
        "_context_probe_accuracy_with_control",
        "_run_training_experiment",
        "run_cpu_experiment",
        "render_signature_result",
        "result_to_json",
        "_signature_result_is_accepted",
        "_require_cuda",
        "run_gpu_test",
        "run_full_experiment",
    ),
}


TEST_CALLS = {
    4: "tests.test_topk_router_exact(topk_router)\ntests.test_topk_router_ties_are_portable_and_switch_top1_has_task_gradient(topk_router)",
    6: (
        "tests.test_capacity_ledger_exact(compute_capacity_state)\n"
        "tests.test_capacity_is_left_to_right_and_handles_partial_masked_and_empty_batc"
        "hes(compute_capacity_state)\n"
        "tests.test_capacity_is_prefix_and_batch_invariant(compute_capacity_state)"
    ),
    8: "tests.test_dispatch_and_combine_manual(dispatch_tokens, combine_expert_outputs)\ntests.test_dispatch_and_combine_preserve_activation_and_gate_gradients(dispatch_tokens, combine_expert_outputs)",
    10: "tests.test_expert_mlps_are_independent_and_differentiable(ExpertMLPs)",
    12: "tests.test_router_losses_have_semantic_controls(load_balancing_loss, router_z_loss)\ntests.test_paired_factorial_effects_are_signed_with_minus_without(paired_factorial_effects)",
    14: "tests.test_sparse_moe_matches_unbounded_bruteforce(SparseMoE)\ntests.test_capacity_drop_accounting_is_visible(SparseMoE)\ntests.test_top1_sparse_moe_language_path_reaches_router(SparseMoE)",
    16: "tests.test_moe_transformer_block_shapes(MoETransformerBlock)",
    18: (
        "tests.test_causal_attention_has_no_future_leakage(CausalSelfAttention)\n"
        "tests.test_complete_model_and_cache_parity(SparseMoETransformer)\n"
        "tests.test_finite_capacity_resets_per_chunk_and_unbounded_greedy_is_exact(Spar"
        "seMoETransformer)\n"
        "tests.test_finite_capacity_decoder_has_prefix_and_batch_invariance(SparseMoETr"
        "ansformer)\n"
        "tests.test_malformed_caches_are_rejected(SparseMoETransformer)"
    ),
    20: "tests.test_regime_dataset_is_exact_and_causal(make_regime_sequences)\ntests.test_next_token_objective_grades_only_operations(next_token_loss)\ntests.test_chance_baselines_match_evaluator_support()",
    22: "tests.test_router_probe_recovers_known_histograms(fit_router_probe, predict_router_probe)\ntests.test_delexicalized_router_probe_hides_current_operation_token(delexicalized_router_event_features)",
    24: "tests.test_coherent_expert_permutation_is_invariant(permute_moe_experts)",
    26: "tests.test_causal_intervention_table_matches_direct_runs(causal_expert_intervention_table)",
    28: "tests.test_disjoint_streams_are_unique_and_genuinely_mixed()",
    29: "tests.test_factorial_models_clone_identical_sparse_initialization_and_align_dense()\ntests.test_measured_ffn_accounting_matches_slots_and_keeps_router_overhead_separate()",
}


CELL_DEMOS = {14: OVERFLOW_DEMO, 22: DELEX_DEMO}


SIGNATURE_CELL = """figure_path = root_dir / chapter / "instructions" / "assets" / "mixture_of_experts_signature.png"
signature_started = time.perf_counter()
signature = run_cpu_experiment(steps=180, seeds=(7, 19, 31), save_path=figure_path)
signature_seconds = time.perf_counter() - signature_started

headline_keys = (
    "aux_z_final_test_accuracy",
    "neither_final_test_accuracy",
    "aux_z_hard_load_entropy",
    "neither_hard_load_entropy",
    "neither_dropped_fraction",
    "context_only_router_probe_accuracy",
    "context_only_shuffled_label_probe_accuracy",
    "factorial_aux_main_effect_hard_load_entropy",
    "factorial_aux_main_effect_dropped_fraction",
    "factorial_z_main_effect_router_log_partition_rms",
    "preferred_expert_ablation_drop",
    "mean_nonpreferred_expert_ablation_drop",
    "preferred_expert_forcing_drop",
    "mean_nonpreferred_expert_forcing_drop",
    "router_evaluated_positions",
    "sparse_router_matmul_flops",
)
for name in headline_keys:
    print(f"{name:55s} mean={signature.metrics[name]:.6f} seeds={signature.seed_metrics[name]}")
print(f"CPU wall time: {signature_seconds:.3f}s")
print("dev routing by operation:\\n", signature.dev_routing_by_regime)
print("dev-selected preferred experts:", signature.preferred_experts.tolist())
print("final routing by operation:\\n", signature.routing_by_regime)
print("requested / kept counts:", signature.requested_counts.tolist(), signature.kept_counts.tolist())
print("intact accuracy:", signature.intervention.intact_accuracy.tolist())
print("all-expert ablated accuracy:\\n", signature.intervention.ablated_accuracy)
print("all-expert forced accuracy:\\n", signature.intervention.forced_accuracy)
display(Image(filename=str(figure_path)))

assert signature.metrics["aux_z_final_test_accuracy"] >= 0.98
assert signature.metrics["dense_equal_width_final_test_accuracy"] >= 0.98
assert signature.metrics["aux_z_long_mixed_accuracy"] >= 0.60
assert signature.metrics["aux_z_hard_load_entropy"] >= 0.95
assert signature.metrics["neither_hard_load_entropy"] <= 0.05
assert signature.metrics["neither_dropped_fraction"] >= 0.60
assert signature.metrics["context_only_router_probe_accuracy"] <= ROUTER_PROBE_CHANCE_ACCURACY + 0.12
assert signature.metrics["context_only_shuffled_label_probe_accuracy"] <= ROUTER_PROBE_CHANCE_ACCURACY + 0.12
assert signature.metrics["factorial_aux_main_effect_hard_load_entropy"] > 0.20
assert signature.metrics["factorial_aux_main_effect_dropped_fraction"] < -0.20
assert signature.metrics["factorial_z_main_effect_router_log_partition_rms"] < 0
specialization_supported = signature.metrics["preferred_expert_ablation_drop"] > signature.metrics["mean_nonpreferred_expert_ablation_drop"]
print("Preferred-expert ablation hypothesis:", "SUPPORTED" if specialization_supported else "REJECTED")
print("Original positive-specialization release contract:", _signature_result_is_accepted(signature))
assert signature.metrics["mean_nonpreferred_expert_forcing_drop"] > signature.metrics["preferred_expert_forcing_drop"]
assert signature.metrics["coherent_expert_permutation_max_diff"] <= 3e-6
assert signature.metrics["sparse_executed_ffn_slots"] == 1620, "Account for per-sequence capacity rounding."
assert signature.metrics["sparse_ffn_matmul_flops"] / signature.metrics["dense_ffn_matmul_flops"] == 1.125, "Padded sparse work exceeds this equal-width dense control."
assert signature.metrics["router_evaluated_positions"] > signature.metrics["active_operation_tokens"]
print("Mechanics and regularizer checks passed; inspect the specialization verdict above.")"""


PLAY_CELL = """play_seed = 43
play_aux = 0.2
play_z = 0.0
play_capacity = 0.75
play_training = TrainingConfig(steps=60, batch_size=63, train_data_tokens=9, eval_every=20, seed=play_seed)
play_streams = build_disjoint_mixed_streams(play_training, seed=play_seed, device=device)
play_models, _ = clone_factorial_models(MoEConfig(capacity_factor=play_capacity), seed=play_seed)
play_model = play_models["aux-only"]
play_trace = train_sparse_model(
    play_model,
    play_training,
    aux_coefficient=play_aux,
    z_coefficient=play_z,
    device=device,
    train_batches=play_streams.train_batches,
    dev_sequences=play_streams.dev_sequences,
)
play_accuracy, play_moe = evaluate_sparse(
    play_model,
    play_streams.final_test_sequences,
    capacity_factor=play_capacity,
)
play_inputs = play_streams.final_test_sequences[:, :-1]
play_mask = play_inputs.ge(REGIME_TOKEN_OFFSET) & play_inputs.lt(DATA_TOKEN_OFFSET)
play_dev_routing = routing_by_regime(play_model, play_streams.dev_sequences)
play_final_routing = routing_by_regime(play_model, play_streams.final_test_sequences)
play_preferred = play_dev_routing.argmax(dim=-1)
play_intervention = causal_expert_intervention_table(play_model, play_streams.final_test_sequences)
print("trace steps:", play_trace.steps)
print("trace hard-load entropy:", play_trace.hard_load_entropies)
print("trace drop fraction:", play_trace.dropped_fractions)
print("trace log-partition RMS:", play_trace.router_log_partition_rms)
print({
    "final_test_accuracy": play_accuracy,
    "soft_probability_entropy": normalized_router_entropy(play_moe.router.probabilities, play_mask),
    "hard_load_entropy": normalized_load_entropy(play_moe.capacity.requested_counts),
    "dropped_fraction": play_moe.capacity.dropped_fraction,
    "requested_counts": play_moe.capacity.requested_counts.tolist(),
    "kept_counts": play_moe.capacity.kept_counts.tolist(),
})
print("dev routing:\\n", play_dev_routing)
print("dev-selected preferred experts:", play_preferred.tolist())
print("final routing:\\n", play_final_routing)
print("all-expert ablated accuracy:\\n", play_intervention.ablated_accuracy)
print("all-expert forced accuracy:\\n", play_intervention.forced_accuracy)"""


def build() -> None:
    cells = []
    code_indices = {1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29, 31, 33, 36}
    for index in range(38):
        kind = "code" if index in code_indices else "markdown"
        cell = {
            "id": f"moe-{index:02d}",
            "cell_type": kind,
            "metadata": {},
            "source": MARKDOWN.get(index, "").splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    exercise = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    solution = copy.deepcopy(exercise)
    for notebook in (exercise, solution):
        set_source(notebook, 1, SETUP)
        for index in EXERCISE_CODE:
            answer = definitions_source(*SOLUTION_GROUPS[index])
            ast.parse(answer)
            narrative = MARKDOWN[index - 1].replace("{{solution}}", answer)
            set_source(notebook, index - 1, narrative)
    for index, code in EXERCISE_CODE.items():
        set_source(exercise, index, code)
        exercise["cells"][index]["metadata"]["tags"] = ["exercise"]
    for index in SOLUTION_GROUPS:
        source = definitions_source(*SOLUTION_GROUPS[index])
        if index in TEST_CALLS:
            source += "\n\n" + TEST_CALLS[index]
        if index in CELL_DEMOS:
            source += "\n\n" + CELL_DEMOS[index]
        set_source(solution, index, source)
    for notebook in (exercise, solution):
        for index in (28, 29):
            set_source(
                notebook,
                index,
                definitions_source(*SOLUTION_GROUPS[index]) + "\n\n" + TEST_CALLS[index],
            )
        set_source(notebook, 33, SIGNATURE_CELL)
        set_source(notebook, 36, PLAY_CELL)
        # Break the supplied experiment into readable, inspectable stages.
        stages = [
            (
                "Matched optimization",
                "Train all conditions on the same ordered batches. Locate the task loss, aux loss, z-loss, zero_grad, backward, and optimizer step before proceeding.",
                ("train_sparse_model", "train_dense_model"),
            ),
            (
                "Evaluation and compute accounting",
                "Fit nothing on final data. Count router positions separately from the padded expert buffer.",
                (
                    "routing_by_regime",
                    "evaluate_sparse",
                    "count_parameters",
                    "clone_factorial_models",
                    "measured_compute_accounting",
                    "_context_probe_accuracy_with_control",
                ),
            ),
            (
                "Compose the repeated experiment",
                "The seed loop creates new models and streams; dev selects experts and final data measures interventions. Trace which quantities are averaged and which belong only to the representative seed.",
                ("_run_training_experiment", "run_cpu_experiment"),
            ),
            (
                "Draw the six-panel result",
                "The plot consumes live arrays. Inspect every seed dot and compare forced and ablated effects separately.",
                ("render_signature_result",),
            ),
            (
                "Preserve the original decision rule",
                "The positive-specialization predicate remains unchanged and may return False. A rejected hypothesis remains visible; the CUDA release hooks reject it rather than certifying the old claim.",
                (
                    "result_to_json",
                    "_signature_result_is_accepted",
                    "_require_cuda",
                    "run_gpu_test",
                    "run_full_experiment",
                ),
            ),
        ]
        staged_cells = []
        for index, (title, explanation, names) in enumerate(stages):
            staged_cells.append(
                {
                    "id": f"moe-stage-{index}-text",
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"### {title}\n\n{explanation}\n"],
                }
            )
            staged_cells.append(
                {
                    "id": f"moe-stage-{index}-code",
                    "cell_type": "code",
                    "metadata": {},
                    "source": [definitions_source(*names)],
                    "execution_count": None,
                    "outputs": [],
                }
            )
        notebook["cells"][31:32] = staged_cells
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
    for notebook in (exercise, solution):
        for cell in notebook["cells"]:
            cell_source = cell["source"]
            if isinstance(cell_source, list):
                cell_source = "".join(cell_source)
            if cell["cell_type"] == "markdown":
                cell_source = wrap_long_markdown_prose(cell_source)
            cell["source"] = cell_source.splitlines(keepends=True)
    EXERCISE_PATH.write_text(json.dumps(exercise, indent=1) + "\n")
    SOLUTION_PATH.write_text(json.dumps(solution, indent=1) + "\n")
    chunks = []
    for cell in exercise["cells"]:
        text = "".join(cell["source"]).rstrip()
        chunks.append(text if cell["cell_type"] == "markdown" else f"```python\n{text}\n```")
    PAGE_PATH.write_text("\n\n".join(chunks) + "\n")


if __name__ == "__main__":
    build()
