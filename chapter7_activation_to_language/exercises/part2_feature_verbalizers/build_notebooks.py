"""Build the learner and solution notebooks plus the mirrored instruction page."""

from __future__ import annotations

import ast
import inspect
import pprint
import sys
import textwrap
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[3]
SECTION_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from chapter7_activation_to_language.exercises.part2_feature_verbalizers import solutions

EXERCISE_PATH = SECTION_DIR / "7.2_Feature_Verbalizers_exercises.ipynb"
SOLUTION_PATH = SECTION_DIR / "7.2_Feature_Verbalizers_solutions.ipynb"
PAGE_PATH = ROOT / "chapter7_activation_to_language/instructions/pages/02_[7.2]_Feature_Verbalizers.md"


def source(*names: str) -> str:
    return "\n\n".join(
        textwrap.dedent(inspect.getsource(getattr(solutions, name))).rstrip()
        for name in names
    )


def stub(signatures: str) -> str:
    return textwrap.dedent(signatures).strip()


def solution_details(code: str) -> str:
    ast.parse(code)
    return f"""
<details>
<summary>Solution</summary>

```python
{code}
```

</details>
""".strip()


def exercise_markdown(
    title: str,
    body: str,
    expected: str,
    help_text: str,
    solution_code: str,
    interpretation: str,
    *,
    difficulty: str = "medium",
) -> str:
    return f"""### Exercise - {title}

> ```yaml
> Difficulty: {difficulty}
> Importance: high
> ```

{body}

<details>
<summary>Expected output</summary>

{expected}

</details>

<details>
<summary>Help - reveal a hint</summary>

{help_text}

</details>

{solution_details(solution_code)}

<details>
<summary>Interpretation</summary>

{interpretation}

</details>
"""


HEADER = r"""# [7.2] Feature Verbalizers

By the end of this notebook, you will have shown that a training-only
subject-number direction changes Pythia's `are` versus `is` preference in both
directions beyond matched within-pair control directions, while perfect
decoding by itself is explained by lexical input structure.

```yaml
exercise_id: 7_2_feature_verbalizers
gt_tier: GT-1 toy oracles plus a pinned real-model CUDA result
expected_runtime: 60-90 minutes
requires_gpu: true
```

## Core Question

When does the phrase **"the subject noun is plural"** explain a residual-stream
direction, rather than merely redescribe a lexical correlation?

We will fit one transparent direction on real cached activations from a pinned
`EleutherAI/pythia-70m` checkpoint without touching held-out labels. Then we do
the work that a verbalizer needs: inspect top, bottom, random, and near-threshold
examples; operationalize candidate descriptions; mine counterexamples; revise
the description; compare accuracy with brevity; and finally intervene through a
real forward hook. The decisive causal null is not isotropic noise. It consists
of 256 decoder directions induced by label swaps within the same training noun
pairs, with the target's exact per-example replacement coefficients.

## Learning Objectives

By the end, you will be able to:

1. implement a mean-difference activation decoder and exact projection replacement;
2. cache final-token residual activations from a pinned causal language model;
3. construct an evidence table and revise an explanation using counterexamples;
4. expose embedding and orthographic shortcuts with a full layer sweep;
5. design a randomization null that preserves matched-pair structure; and
6. perform a real residual-stream intervention and measure both directions.

## The Research Loop

Let $h_i^{(\ell)}$ be the final-token residual for prompt $i$ after layer
$\ell$. A mean-difference direction fit on the training split is

$$d = \frac{\mathbb{E}[h\mid y=1]-\mathbb{E}[h\mid y=0]}
{\lVert\mathbb{E}[h\mid y=1]-\mathbb{E}[h\mid y=0]\rVert_2}.$$

This equation gives a direction, not a meaning. A verbalizer is a hypothesis
$v(x)$ whose predictions can disagree with the feature side
$\mathbb{1}[h(x)\cdot d\geq\tau]$. Those disagreements are useful evidence:
they tell us how to revise the text. Only after freezing the revision do we look
at the held-out vocabulary.

For a causal replacement, we preserve every component orthogonal to $d$:

$$h' = h + (s_{target} - h\cdot d)d.$$

The matched causal null replaces $d$ with directions $d_b$ fit after swapping
labels *within* training pairs. Each null receives the same coefficient
$s_{target}-h\cdot d$ for every held-out row and runs through the same hook. This
asks whether our direction is more behaviorally effective than directions the
paired design itself can manufacture.

### Visual Intuition

Think of every prompt as a point in residual space. A decoder draws one axis
through those points. Top and bottom examples suggest a story; near-threshold
examples and counterexamples attack it. The layer sweep asks whether the story
was present before any transformer computation. The intervention asks a
different question: if we move only along that axis, does downstream behavior
move with it?

### Common Bugs

- Permuting individual labels even though the dataset was collected in matched noun pairs.
- Combining irregular and nonce nouns, which hides that the suffix rule is perfect on nonce plurals.
- Calling a string dictionary a tokenizer-ID baseline.
- Treating a perfect embedding-layer decoder as evidence of computation in layer 3.
- Showing orthogonal controls while omitting pair-preserving decoder directions.
- Printing custom predictions without actually running the promised activation intervention.
- Ranking candidate prose on the held-out labels, then calling the split held out.
"""

SETUP = f"""from __future__ import annotations

import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
from IPython.display import display

chapter = "chapter7_activation_to_language"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / "part2_feature_verbalizers"
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_feature_verbalizers.tests as tests

MODEL_ID = {solutions.MODEL_ID!r}
MODEL_REVISION = {solutions.MODEL_REVISION!r}
LAYER = {solutions.LAYER}
IS_TOKEN = {solutions.IS_TOKEN!r}
ARE_TOKEN = {solutions.ARE_TOKEN!r}
PAIR_CAUSAL_NULL_SEED = {solutions.PAIR_CAUSAL_NULL_SEED}
PAIR_CAUSAL_NULL_COUNT = {solutions.PAIR_CAUSAL_NULL_COUNT}
Number = Literal["singular", "plural"]
Split = Literal["train", "validation", "heldout"]
Morphology = Literal["regular", "irregular", "nonce"]

{source("DirectionDecoder", "ActivationCache", "NumberExample", "Hypothesis", "BinaryEvaluation", "HookIntervention", "LayerSweepCache")}
"""

DATA_SETUP = f"""HYPOTHESES = {pprint.pformat(solutions.HYPOTHESES, width=88)}

TRAIN_PAIRS = {pprint.pformat(solutions.TRAIN_PAIRS, width=88)}

VALIDATION_PAIRS = {pprint.pformat(solutions.VALIDATION_PAIRS, width=88)}

HELDOUT_PAIRS = {pprint.pformat(solutions.HELDOUT_PAIRS, width=88)}

{source("_pair_examples", "make_number_dataset", "split_number_dataset", "validate_number_dataset", "number_labels", "_single_token_id", "load_pinned_model")}

dataset = make_number_dataset()
train_examples, validation_examples, heldout_examples = split_number_dataset(dataset)
print(f"train={{len(train_examples)}}, validation={{len(validation_examples)}}, heldout={{len(heldout_examples)}}")
display(pd.DataFrame([asdict(example) for example in train_examples[:8]]))
"""

TOY_DECODER_SOLUTION = source(
    "fit_mean_difference_decoder",
    "direction_scores",
    "direction_predictions",
)
TOY_DECODER_STUB = stub(
    """
    def fit_mean_difference_decoder(activations: t.Tensor, labels: t.Tensor) -> DirectionDecoder:
        raise NotImplementedError()


    def direction_scores(activations: t.Tensor, decoder: DirectionDecoder) -> t.Tensor:
        raise NotImplementedError()


    def direction_predictions(activations: t.Tensor, decoder: DirectionDecoder) -> t.Tensor:
        raise NotImplementedError()
    """
)

REPLACEMENT_SOLUTION = source("replace_direction_component")
REPLACEMENT_STUB = stub(
    """
    def replace_direction_component(
        activations: t.Tensor,
        direction: t.Tensor,
        target_scores: t.Tensor | float,
    ) -> t.Tensor:
        raise NotImplementedError()
    """
)

CACHE_SOLUTION = source(
    "cache_final_token_activations",
    "cache_layer_sweep",
    "layer_sweep_accuracies",
)
CACHE_STUB = stub(
    """
    def cache_final_token_activations(
        model,
        tokenizer,
        prompts: Sequence[str],
        *,
        layer: int = LAYER,
    ) -> ActivationCache:
        raise NotImplementedError()


    def cache_layer_sweep(
        model,
        tokenizer,
        prompts: Sequence[str],
    ) -> LayerSweepCache:
        raise NotImplementedError()


    def layer_sweep_accuracies(
        train_cache: LayerSweepCache,
        train_labels: t.Tensor,
        test_cache: LayerSweepCache,
        test_labels: t.Tensor,
    ) -> tuple[float, ...]:
        raise NotImplementedError()
    """
)

HYPOTHESIS_SOLUTION = source(
    "hypothesis_predictions",
    "score_hypotheses",
    "select_hypothesis",
    "rank_feature_examples",
    "mine_counterexamples",
    "brevity_frontier",
)
HYPOTHESIS_STUB = stub(
    """
    def hypothesis_predictions(
        examples: Sequence[NumberExample],
        hypothesis: Hypothesis,
    ) -> t.Tensor:
        raise NotImplementedError()


    def score_hypotheses(
        examples: Sequence[NumberExample],
        activation_predictions: t.Tensor,
        candidates: Sequence[Hypothesis] = HYPOTHESES,
        *,
        seed: int = 0,
    ) -> tuple[dict[str, object], ...]:
        raise NotImplementedError()


    def select_hypothesis(rows: Sequence[dict[str, object]]) -> str:
        raise NotImplementedError()


    def rank_feature_examples(
        examples: Sequence[NumberExample],
        scores: t.Tensor,
        threshold: float,
        *,
        n: int = 2,
        seed: int = 0,
    ) -> tuple[dict[str, object], ...]:
        raise NotImplementedError()


    def mine_counterexamples(
        examples: Sequence[NumberExample],
        activation_predictions: t.Tensor,
        hypothesis: Hypothesis,
    ) -> tuple[dict[str, object], ...]:
        raise NotImplementedError()


    def brevity_frontier(rows: Sequence[dict[str, object]]) -> tuple[str, ...]:
        raise NotImplementedError()
    """
)

CONTROL_SOLUTION = source(
    "evaluate_binary_predictions",
    "final_noun_token_ids",
    "seen_plural_lookup_predictions",
    "orthographic_feature_matrix",
    "orthographic_ridge_predictions",
    "paired_swap_decoder_accuracies",
)
CONTROL_STUB = stub(
    """
    def evaluate_binary_predictions(
        predictions: t.Tensor,
        labels: t.Tensor,
        baseline_predictions: t.Tensor,
        subset_mask: t.Tensor,
    ) -> BinaryEvaluation:
        raise NotImplementedError()


    def final_noun_token_ids(
        tokenizer,
        examples: Sequence[NumberExample],
    ) -> tuple[int, ...]:
        raise NotImplementedError()


    def seen_plural_lookup_predictions(
        tokenizer,
        train_examples: Sequence[NumberExample],
        test_examples: Sequence[NumberExample],
    ) -> t.Tensor:
        raise NotImplementedError()


    def orthographic_feature_matrix(
        examples: Sequence[NumberExample],
    ) -> t.Tensor:
        raise NotImplementedError()


    def orthographic_ridge_predictions(
        train_examples: Sequence[NumberExample],
        train_labels: t.Tensor,
        test_examples: Sequence[NumberExample],
        *,
        ridge: float = 1e-2,
    ) -> t.Tensor:
        raise NotImplementedError()


    def paired_swap_decoder_accuracies(
        train_activations: t.Tensor,
        train_labels: t.Tensor,
        test_activations: t.Tensor,
        test_labels: t.Tensor,
    ) -> tuple[float, ...]:
        raise NotImplementedError()
    """
)

HOOK_SOLUTION = source(
    "orthogonal_random_direction",
    "paired_swap_decoder_directions",
    "orthogonal_random_directions",
    "run_residual_edit",
    "run_orthogonal_control_distribution",
    "run_direction_control_distribution",
)
HOOK_STUB = stub(
    """
    def orthogonal_random_direction(direction: t.Tensor, *, seed: int = 0) -> t.Tensor:
        raise NotImplementedError()


    def paired_swap_decoder_directions(
        train_activations: t.Tensor,
        train_labels: t.Tensor,
        *,
        count: int = PAIR_CAUSAL_NULL_COUNT,
        seed: int = PAIR_CAUSAL_NULL_SEED,
    ) -> tuple[tuple[int, ...], t.Tensor]:
        raise NotImplementedError()


    def orthogonal_random_directions(
        direction: t.Tensor, *, seeds: Sequence[int] = tuple(range(64))
    ) -> t.Tensor:
        raise NotImplementedError()


    def run_residual_edit(
        model,
        tokenizer,
        prompts: Sequence[str],
        edit_vectors: t.Tensor,
        *,
        layer: int = LAYER,
    ) -> HookIntervention:
        raise NotImplementedError()


    def run_orthogonal_control_distribution(
        model,
        tokenizer,
        prompts: Sequence[str],
        replacement_coefficients: t.Tensor,
        target_direction: t.Tensor,
        baseline_margin: float,
        *,
        seeds: Sequence[int] = tuple(range(64)),
    ) -> dict[str, object]:
        raise NotImplementedError()


    def run_direction_control_distribution(
        model,
        tokenizer,
        prompts: Sequence[str],
        replacement_coefficients: t.Tensor,
        control_directions: t.Tensor,
        baseline_margin: float,
        *,
        effect_sign: float,
        chunk_size: int = 32,
        layer: int = LAYER,
    ) -> dict[str, object]:
        raise NotImplementedError()
    """
)

RUNNER_SOLUTION = source("_accuracy", "run_real_model_signature")

EXERCISES = [
    exercise_markdown(
        "fit a direction on exact toy ground truth",
        """Start with eight toy residual vectors. Their first coordinate is the
only class signal; the other coordinates cancel within each class. Compute the
positive-minus-negative mean direction, normalize it, and put the decision
threshold halfway between projected class means.""",
        """The exact oracle prints `Mean-difference decoder: exact toy oracle passed.`
The direction is `[1, 0, 0]`, the threshold is `0`, and every score equals the
first activation coordinate.""",
        """Compute two mean vectors, subtract negative from positive, and normalize
once. Orienting the difference this way makes larger scores mean plural.""",
        TOY_DECODER_SOLUTION,
        """This decoder is transparent: its only learned object is one direction
and one midpoint. The toy oracle proves the algebra before any model result can
look persuasive.""",
    ),
    exercise_markdown(
        "replace one activation component exactly",
        """Implement the operation used by the causal experiment. For each row,
change only its coordinate along a unit direction so the new projection equals
the requested target score. Every orthogonal component must remain unchanged.""",
        """The test prints two passed oracles: the new dot products equal the target
scores and all coordinates orthogonal to the toy direction are unchanged.""",
        """If `s = h @ d`, the needed edit is `(target - s) * d`. Avoid changing
the whole vector or constructing a new score without editing the activation.""",
        REPLACEMENT_SOLUTION,
        """This is an activation replacement, not `new_score = old_score + alpha`.
Later the same edit vector is inserted into Pythia's residual stream and the
remaining transformer layers produce new logits.""",
    ),
    exercise_markdown(
        "cache residuals and audit every layer",
        """Load the pinned 70M-parameter checkpoint from the local Hugging Face
cache. Return layer-3 final-token activations and the model's downstream
`logit(" are") - logit(" is")` margin. Then cache the embedding residual plus
all six block outputs in one forward pass. Labels must not be accepted by either
cache function.""",
        """The caches have shapes `(24, 512)`, `(12, 512)`, and `(24, 512)`.
Both candidate continuations tokenize to one token, and the cache API has no
label argument. The layer sweep has seven entries from `embedding` to `block 5`.""",
        """For block index `layer`, Hugging Face stores the post-block residual at
`hidden_states[layer + 1]`. With left padding, index `-1` is the final prompt
token for every row.""",
        CACHE_SOLUTION,
        """These are model-produced residuals. The train labels are introduced
only after caching, when fitting the decoder; validation and held-out labels are
not inputs to either activation extraction or direction fitting.""",
        difficulty="hard",
    ),
    exercise_markdown(
        "inspect evidence, mine counterexamples, and revise the explanation",
        """Build top, bottom, random, and near-threshold validation examples before
choosing any prose. Turn each candidate explanation into an executable predicate,
shuffle display order, score agreement, mine disagreements with the tempting
`ends in s` story, and compute the agreement-versus-word-count frontier. The
held-out labels remain untouched.""",
        """The evidence table contains all four sampling groups. `The mice` and
`The geese` falsify the suffix story. Candidate order changes across seeds but
semantic scores do not. The revised `subject_number_plural` explanation reaches
agreement `1.00` and remains on the brevity frontier.""",
        """Sort scores for top and bottom rows, absolute distance from the threshold
for near misses, and use a seeded sample for random rows. A counterexample is a
row where the executable predicate disagrees with the feature side.""",
        HYPOTHESIS_SOLUTION,
        """A candidate phrase has become falsifiable: it predicts which examples
should lie above the direction threshold. The revision loop is operationalized,
not generated by another language model, so open-ended automatic explanation
generation remains explicitly preflight.""",
    ),
    exercise_markdown(
        "expose lexical shortcuts before held-out interpretation",
        """Evaluate the activation decoder on 24 unseen noun tokens. Compare it
with the suffix verbalizer, a train-only orthographic ridge model, a final-token-ID
lookup, a decoder fit at the embedding output and every later layer, and the exact `2**12`
directions obtained by independently swapping singular/plural labels within
each matched training pair. Report regular, irregular, and nonce rows
separately, plus the complete null distribution and exact p-value.""",
        """The matched-subset oracle demonstrates that a method scoring `0.50` on
a subset does not survive when the baseline scores `1.00` on the same subset.
On Pythia, the target direction reaches `1.00` held-out accuracy. The suffix
rule scores `1.00/0.50/1.00` on regular/irregular/nonce groups. The embedding
decoder is already `1.00`, so later-layer decoding cannot be credited to
transformer computation. The exhaustive paired null has mean `0.500`, p95 `1.000`, and
exact `p=0.134`; 548 of 4,096 null fits also score `1.00`. This is a negative
significance result and the plot must show it.""",
        """Index predictions, labels, and baseline predictions with the same
`subset_mask` before taking either subset mean. For the null, keep each
singular/plural lexeme pair intact and enumerate its two possible orientations.""",
        CONTROL_SOLUTION,
        """Only the irregular group falsifies the suffix story; nonce plurals all
end in `s`. The target does not exceed the paired-null tail, so held-out
decoding alone is not statistically persuasive. That failure makes the later
causal controls, rather than accuracy, the signature evidence.""",
        difficulty="hard",
    ),
    exercise_markdown(
        "intervene against a pair-preserving causal null",
        """Register a hook on Pythia block 3. Add one precomputed edit vector to
the final token of each batch row, let layers 4-5 and the unembedding run, then
return the new `are - is` margins. Test singular-to-plural and plural-to-singular
replacement. Fit 256 control directions from precommitted within-pair training
label swaps, apply the target's exact per-example coefficients to each control,
and run every edit through the same hook. Keep orthogonal controls only as a
secondary diagnostic.""",
        """A zero edit reproduces cached logits. Replacing singular examples'
number component with the plural training mean moves the mean margin from about
`-4.81` to `+3.67`; reverse replacement moves `+4.66` to `-4.95`. Each target
delta is compared with its own 64-direction coefficient-matched orthogonal
distribution. The stronger paired causal p-values are about `0.016` forward and
`0.008` reverse. An exact two-dimensional model proves that batched controls use
the supplied coefficients and execute the registered hook.""",
        """A Hugging Face block can return a tensor or a tuple. Clone the hidden
tensor, edit `[:, -1, :]`, reconstruct the tuple if needed, and keep a hook-call
counter so silent non-execution cannot pass.""",
        HOOK_SOLUTION,
        """Bidirectional edits establish causal sensitivity rather than a one-way
ceiling artifact. The paired null is difficult because it preserves the lexical
geometry that made decoding look perfect. Clearing it supports a narrow causal
claim, not uniqueness, completeness, or a general feature-explanation system.""",
        difficulty="hard",
    ),
]


def code_for(mode: str, stub_code: str, solution_code: str, suffix: str) -> str:
    implementation = solution_code if mode == "solutions" else stub_code
    return implementation + "\n\n" + suffix.strip() + "\n"


def build_cells(mode: str) -> list:
    """Build the repaired learner path; the legacy builder above is not invoked."""

    cells = [
        nbformat.v4.new_markdown_cell(HEADER),
        nbformat.v4.new_markdown_cell(
            """## Setup

The default device is CUDA. Maintainers can explicitly set `ARENA_DEVICE=cpu`
for deterministic CPU execution; this is an explicit mode, not a runtime
fallback. The checkpoint is pinned and loaded from the local cache."""
        ),
        nbformat.v4.new_code_cell(SETUP),
        nbformat.v4.new_markdown_cell(
            """## Cold Open

Pythia prefers `are` after `The pilots` and `is` after `The pilot`. A perfect
decoder would still be uninteresting if the embedding table already separates
those words. Write down three different claims before running anything:

1. **decodable:** one axis predicts a label;
2. **verbalizable:** a short rule predicts which examples activate that axis;
3. **causal:** editing that axis changes model behavior more than a matched null.

This notebook will reject the first as evidence of layer-3 computation, revise
the second using counterexamples, and retain only a narrow version of the third."""
        ),
        nbformat.v4.new_code_cell(DATA_SETUP),
        nbformat.v4.new_markdown_cell(
            """## 1. Toy Ground Truth

Before touching a language model, prove the direction and replacement algebra
on vectors where every answer is known exactly."""
        ),
        nbformat.v4.new_markdown_cell(EXERCISES[0]),
        nbformat.v4.new_code_cell(
            code_for(
                mode,
                TOY_DECODER_STUB,
                TOY_DECODER_SOLUTION,
                """tests.test_mean_difference_decoder(
    fit_mean_difference_decoder,
    direction_scores,
)""",
            )
        ),
        nbformat.v4.new_markdown_cell(EXERCISES[1]),
        nbformat.v4.new_code_cell(
            code_for(
                mode,
                REPLACEMENT_STUB,
                REPLACEMENT_SOLUTION,
                "tests.test_component_replacement(replace_direction_component)",
            )
        ),
        nbformat.v4.new_markdown_cell(
            """## 2. Cache Before Labels

`output.hidden_states[0]` is the embedding residual. Entry `k + 1` is the
post-block residual for block `k`. Caching every depth in one pass gives the
control that the old experiment missed: did the transformer compute the feature,
or did token identity already expose it?"""
        ),
        nbformat.v4.new_code_cell(
            """DEVICE = os.environ.get("ARENA_DEVICE", "cuda")
if DEVICE not in {"cpu", "cuda"}:
    raise ValueError("ARENA_DEVICE must be exactly 'cpu' or 'cuda'.")
if DEVICE == "cuda" and not t.cuda.is_available():
    raise RuntimeError("CUDA was explicitly selected but is unavailable; no fallback is used.")
model, tokenizer = load_pinned_model(DEVICE)
print(
    f"Loaded {MODEL_ID}@{MODEL_REVISION[:8]} on {next(model.parameters()).device}; "
    f"Torch {t.__version__}, d_model={model.config.hidden_size}"
)"""
        ),
        nbformat.v4.new_markdown_cell(EXERCISES[2]),
        nbformat.v4.new_code_cell(
            code_for(
                mode,
                CACHE_STUB,
                CACHE_SOLUTION,
                """train_cache = cache_final_token_activations(
    model, tokenizer, [example.prompt for example in train_examples]
)
validation_cache = cache_final_token_activations(
    model, tokenizer, [example.prompt for example in validation_examples]
)
heldout_cache = cache_final_token_activations(
    model, tokenizer, [example.prompt for example in heldout_examples]
)
tests.test_layer_sweep_exact_oracle(layer_sweep_accuracies)
train_sweep = cache_layer_sweep(model, tokenizer, [x.prompt for x in train_examples])
heldout_sweep = cache_layer_sweep(model, tokenizer, [x.prompt for x in heldout_examples])
assert train_cache.activations.shape == (24, 512)
assert train_sweep.layer_names == (
    "embedding", "block 0", "block 1", "block 2", "block 3", "block 4", "block 5"
)
assert "labels" not in __import__("inspect").signature(cache_layer_sweep).parameters
print("Cached:", train_cache.activations.shape, train_sweep.layer_names)""",
            )
        ),
        nbformat.v4.new_markdown_cell(
            """## 3. From Examples to a Falsifiable Explanation

The direction is fit on the 24 training rows only. Validation rows may be used
to inspect and revise prose; held-out rows may not. First examine evidence from
all four sampling regimes. Near-threshold examples are especially valuable
because a small miss can separate two plausible descriptions."""
        ),
        nbformat.v4.new_markdown_cell(EXERCISES[3]),
        nbformat.v4.new_code_cell(
            code_for(
                mode,
                HYPOTHESIS_STUB,
                HYPOTHESIS_SOLUTION,
                """train_labels = number_labels(train_examples)
validation_scores = direction_scores(
    validation_cache.activations,
    decoder := fit_mean_difference_decoder(train_cache.activations, train_labels),
)
validation_predictions = direction_predictions(validation_cache.activations, decoder)
tests.test_feature_evidence_loop(
    rank_feature_examples,
    mine_counterexamples,
    brevity_frontier,
)
feature_rows = rank_feature_examples(
    validation_examples, validation_scores, decoder.threshold, n=2, seed=0
)
display(pd.DataFrame(feature_rows))

hypothesis_rows = score_hypotheses(
    validation_examples, validation_predictions, HYPOTHESES, seed=7
)
suffix_hypothesis = next(h for h in HYPOTHESES if h.hypothesis_id == "surface_suffix_s")
suffix_counterexamples = mine_counterexamples(
    validation_examples, validation_predictions, suffix_hypothesis
)
selected_hypothesis = select_hypothesis(hypothesis_rows)
display(pd.DataFrame(hypothesis_rows).sort_values(["agreement", "word_count"], ascending=[False, True]))
display(pd.DataFrame(suffix_counterexamples))
print("Revision: surface_suffix_s ->", selected_hypothesis)
print("Brevity frontier:", brevity_frontier(hypothesis_rows))""",
            )
        ),
        nbformat.v4.new_markdown_cell(
            """### Stop and Predict

The revised description may fit validation perfectly. Before revealing the
layer sweep, predict whether accuracy rises gradually with depth. If embedding
accuracy is already perfect, what causal claim remains possible?"""
        ),
        nbformat.v4.new_markdown_cell(
            """## 4. Held-Out Shortcut Audit

The final noun vocabulary is disjoint. Every baseline uses the identical 24
rows. The exact decoding null enumerates all `2**12` within-pair orientations;
it is a negative result if many null directions also classify perfectly."""
        ),
        nbformat.v4.new_markdown_cell(EXERCISES[4]),
        nbformat.v4.new_code_cell(
            code_for(
                mode,
                CONTROL_STUB,
                CONTROL_SOLUTION,
                """tests.test_matched_subset_comparison(evaluate_binary_predictions)
tests.test_token_id_controls(tokenizer, final_noun_token_ids, seen_plural_lookup_predictions)
tests.test_pair_preserving_null_exact_oracle(paired_swap_decoder_accuracies)

heldout_labels = number_labels(heldout_examples)
heldout_predictions = direction_predictions(heldout_cache.activations, decoder)
suffix_predictions = hypothesis_predictions(heldout_examples, suffix_hypothesis)
lookup_predictions = seen_plural_lookup_predictions(tokenizer, train_examples, heldout_examples)
orthographic_predictions = orthographic_ridge_predictions(
    train_examples, train_labels, heldout_examples
)
sweep_accuracies = layer_sweep_accuracies(
    train_sweep, train_labels, heldout_sweep, heldout_labels
)
paired_null_accuracies = paired_swap_decoder_accuracies(
    train_cache.activations, train_labels, heldout_cache.activations, heldout_labels
)
activation_accuracy = float(heldout_predictions.eq(heldout_labels).float().mean())
paired_null_tensor = t.tensor(paired_null_accuracies)
paired_null_p = float((paired_null_tensor >= activation_accuracy).float().mean())
shortcut_table = pd.DataFrame({
    "method": ["layer-3 direction", "embedding direction", "suffix rule", "orthographic ridge", "token-id lookup"],
    "held-out accuracy": [
        activation_accuracy,
        sweep_accuracies[0],
        float(suffix_predictions.eq(heldout_labels).float().mean()),
        float(orthographic_predictions.eq(heldout_labels).float().mean()),
        float(lookup_predictions.eq(heldout_labels).float().mean()),
    ],
})
display(shortcut_table)
display(pd.DataFrame({"layer": train_sweep.layer_names, "held-out accuracy": sweep_accuracies}))
print(
    f"Exact decoding null: n={len(paired_null_accuracies)}, "
    f"p95={t.quantile(paired_null_tensor, 0.95):.3f}, p={paired_null_p:.4f}"
)""",
            )
        ),
        nbformat.v4.new_markdown_cell(
            """## 5. Causal Test

Orthogonal Gaussian directions are easy controls because they deliberately
remove the shared structure. Our primary null is harder: refit directions after
within-pair label swaps, then reuse the target replacement coefficient for each
example. The target must clear this distribution in both directions."""
        ),
        nbformat.v4.new_markdown_cell(EXERCISES[5]),
        nbformat.v4.new_code_cell(
            code_for(
                mode,
                HOOK_STUB,
                HOOK_SOLUTION,
                """tests.test_orthogonal_control_construction(
    orthogonal_random_direction, orthogonal_random_directions
)
tests.test_paired_causal_direction_bank_exact_oracle(paired_swap_decoder_directions)
tests.test_batched_control_intervention_exact_hook_oracle(
    run_direction_control_distribution
)
zero_test = run_residual_edit(
    model,
    tokenizer,
    [example.prompt for example in heldout_examples[:2]],
    t.zeros((2, model.config.hidden_size)),
)
t.testing.assert_close(
    zero_test.next_token_margins,
    heldout_cache.next_token_margins[:2],
    rtol=1e-4,
    atol=5e-4,
)
assert zero_test.hook_calls == 1
print("Real zero edit reproduced cached logits and executed one hook.")""",
            )
        ),
        nbformat.v4.new_markdown_cell(
            """## 6. Signature Result

The final orchestration is supplied as readable plumbing, but every scientific
operation above is learner-written. A runtime trace executes the real model and
checks the code objects actually called; dead-code names cannot satisfy it, and
the entrypoint is forbidden from naming a reference module."""
        ),
        nbformat.v4.new_code_cell(RUNNER_SOLUTION),
        nbformat.v4.new_code_cell(
            """required_learner_functions = (
    cache_final_token_activations,
    cache_layer_sweep,
    fit_mean_difference_decoder,
    rank_feature_examples,
    mine_counterexamples,
    orthographic_ridge_predictions,
    paired_swap_decoder_directions,
    run_residual_edit,
    run_direction_control_distribution,
)
result = tests.test_runtime_dependency_provenance(
    run_real_model_signature,
    required_learner_functions,
    model,
    tokenizer,
)
metrics = result["metrics"]
intervention = result["intervention"]
print(result["claim"])
print(
    f"embedding/layer-3 accuracy={metrics['embedding_accuracy']:.3f}/"
    f"{metrics['heldout_activation_accuracy']:.3f}; decoding p={metrics['paired_null_exact_p']:.4f}"
)
print(
    f"forward target/null95/p={intervention['target_delta']:.3f}/"
    f"{metrics['forward_paired_causal_p95']:.3f}/{metrics['forward_paired_causal_p']:.4f}"
)
print(
    f"reverse target/null95/p={intervention['reverse_target_delta']:.3f}/"
    f"{metrics['reverse_paired_causal_p95']:.3f}/{metrics['reverse_paired_causal_p']:.4f}"
)"""
        ),
        nbformat.v4.new_markdown_cell(
            """The six panels distinguish the negative decoding result from the
positive paired causal result. They also show the evidence that changed the
explanation, rather than presenting only a final phrase.

![Expected real-model signature](../../instructions/assets/feature_verbalizers_number_signature.png)"""
        ),
        nbformat.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

axes[0, 0].plot(
    range(len(result["layer_sweep"]["layer_names"])),
    result["layer_sweep"]["heldout_accuracies"],
    marker="o",
    color="#00798C",
)
axes[0, 0].set_xticks(range(7), result["layer_sweep"]["layer_names"], rotation=35, ha="right")
axes[0, 0].set_ylim(0, 1.05)
axes[0, 0].set(title="Embedding already decodes perfectly", ylabel="held-out accuracy")

explanation_frame = pd.DataFrame(result["hypothesis_rows"]).sort_values("agreement")
axes[0, 1].barh(explanation_frame["hypothesis_id"], explanation_frame["agreement"], color="#EDAE49")
for y, (_, row) in enumerate(explanation_frame.iterrows()):
    axes[0, 1].text(row["agreement"] - 0.02, y, f"{row['word_count']} words", va="center", ha="right", fontsize=8)
axes[0, 1].set_xlim(0, 1.05)
axes[0, 1].set(title="Counterexamples revise the explanation", xlabel="validation agreement")

decode_null = t.tensor(result["paired_null_accuracies"])
axes[0, 2].hist(decode_null, bins=24, color="#8D6A9F", alpha=0.85)
axes[0, 2].axvline(metrics["heldout_activation_accuracy"], color="#00798C", linewidth=2)
axes[0, 2].set(title=f"Decoding fails paired null: p={metrics['paired_null_exact_p']:.3f}", xlabel="held-out accuracy")

evidence_frame = pd.DataFrame(result["feature_examples"])
group_colors = {"top": "#00798C", "bottom": "#D1495B", "random": "#7A7A7A", "near miss": "#EDAE49"}
axes[1, 0].barh(
    evidence_frame["prompt"],
    evidence_frame["score"] - result["decoder"]["threshold"],
    color=[group_colors[group] for group in evidence_frame["group"]],
)
axes[1, 0].axvline(0, color="#333333", linewidth=1)
axes[1, 0].set(title="Top, bottom, random, near-miss evidence", xlabel="score minus threshold")

forward_null = intervention["paired_causal_distribution"]["effects"]
axes[1, 1].hist(forward_null, bins=20, color="#A7A7A7", alpha=0.8)
axes[1, 1].axvline(intervention["target_delta"], color="#00798C", linewidth=2, label="target")
axes[1, 1].set(title=f"Forward paired causal p={metrics['forward_paired_causal_p']:.3f}", xlabel="effect toward plural")
axes[1, 1].legend()

reverse_null = intervention["reverse_paired_causal_distribution"]["effects"]
axes[1, 2].hist(reverse_null, bins=20, color="#A7A7A7", alpha=0.8)
axes[1, 2].axvline(intervention["reverse_target_delta"], color="#D1495B", linewidth=2, label="target")
axes[1, 2].set(title=f"Reverse paired causal p={metrics['reverse_paired_causal_p']:.3f}", xlabel="effect toward singular")
axes[1, 2].legend()

fig.suptitle("Feature verbalizer: lexical decoding fails; matched causal test survives", fontsize=15, fontweight="bold")
asset_path = root_dir / chapter / "instructions/assets/feature_verbalizers_number_signature.png"
fig.savefig(asset_path, dpi=170, bbox_inches="tight")
plt.show()
print("SIGNATURE PASSED:", result["signature_passed"])"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Try It Yourself

Change `CUSTOM_PAIRS`, predict the result first, then run both the direction
score and the real hook. Try one regular pair and one collective or irregular
noun. The cell changes only singular rows toward the learned positive mean."""
        ),
        nbformat.v4.new_code_cell(
            """CUSTOM_PAIRS = [("captain", "captains"), ("mouse", "mice")]
custom_prompts = [f"The {noun}" for pair in CUSTOM_PAIRS for noun in pair]
custom_cache = cache_final_token_activations(model, tokenizer, custom_prompts)
custom_predictions = direction_predictions(custom_cache.activations, decoder)
custom_indices = t.tensor([0, 2])
custom_edits = replace_direction_component(
    custom_cache.activations[custom_indices], decoder.direction, decoder.positive_mean
) - custom_cache.activations[custom_indices]
custom_intervention = run_residual_edit(
    model, tokenizer, [custom_prompts[i] for i in custom_indices], custom_edits
)
display(pd.DataFrame({
    "prompt": [custom_prompts[i] for i in custom_indices],
    "direction side before": ["plural" if custom_predictions[i] else "singular" for i in custom_indices],
    "are-is before": custom_cache.next_token_margins[custom_indices].tolist(),
    "are-is after": custom_intervention.next_token_margins.tolist(),
}))
assert custom_intervention.hook_calls == 1"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Hunt an Anomaly

Probe `news`, `police`, `series`, and `scissors`. Decide whether each failure
attacks morphology, grammatical agreement, tokenizer structure, or the claim's
scope. Then add your strongest counterexample to `CUSTOM_PAIRS`.

## Interpreting the Result

The embedding and block-3 decoders both score `1.00`; 548 of 4,096 exact paired
decoding nulls also score `1.00`. Decodability is therefore a negative result,
and the layer sweep attributes it to lexical input structure rather than a
computation performed at block 3. On validation data, irregular plurals falsify
the shorter suffix story, so the operationalized loop revises it to grammatical
number before held-out evaluation.

The causal result is narrower and positive. Replacing the learned component
changes the real downstream `are-is` margin in both directions. The target
effects exceed the 95th percentiles of 256 precommitted pair-swap decoder
directions, using identical per-example coefficients and the same hook. This is
evidence of causal sensitivity in this prompt family, not a unique feature or a
complete agreement mechanism.

## Limitations

This is one 70M model, one prompt template, one preselected layer, and only 24
held-out rows. Candidate explanations are human-authored executable predicates,
not generated language; automatic generation, revision across free-form
counterexamples, brevity benchmarking against model-written explanations, and
neuron/SAE/transcoder feature selection remain **PREFLIGHT**. The spelling ridge
is intentionally small and does not exhaust lexical baselines. The paired
causal test is Monte Carlo (`n=256`), and the target is not the most extreme null
draw in either direction even though its plus-one p-values are below `0.05`.

## Reproducing the recorded evidence

The supporting `verification_report.json` is generated after the complete CUDA
experiment. It binds these notebook, page, code, and test versions to the
measurements. The figures and tables above use live model outputs.

## Reading Links

- [Bills et al., *Language models can explain neurons in language models*](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
- [Bricken et al., *Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Pythia model suite](https://arxiv.org/abs/2304.01373)
"""
        ),
    ]
    cells.append(nbformat.v4.new_markdown_cell(
        "## Maintainer verification\n\nThe functions below rerun the same learner experiment "
        "on an explicit device and enforce the 24 GiB budget."
    ))
    cells.append(nbformat.v4.new_code_cell(source("run_smoke_test", "run_gpu_test", "run_full_experiment")))
    return cells


def build_notebook(mode: str, path: Path) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=build_cells(mode),
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.14"},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell.id = f"verbalizers-{index:02d}"
    nbformat.write(notebook, path)


def build_page() -> None:
    blocks = []
    for cell in build_cells("exercises"):
        if cell.cell_type == "markdown":
            blocks.append(cell.source)
        else:
            blocks.append(f"```python\n{cell.source}\n```")
    page = "\n\n".join(blocks).replace(
        "../../instructions/assets/feature_verbalizers_number_signature.png",
        "../assets/feature_verbalizers_number_signature.png",
    )
    PAGE_PATH.write_text(page.rstrip() + "\n")


if __name__ == "__main__":
    build_notebook("exercises", EXERCISE_PATH)
    build_notebook("solutions", SOLUTION_PATH)
    build_page()
