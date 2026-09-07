# Native DAS development and final-evaluation protocol

This protocol covers the existing name-role causal-abstraction requirement.
The native path has been checked on pinned GPT-2; no final prompt has been
evaluated by the model. Final prompt strings and pairings were defined in
`das_ioi_data.py` before the first model run. Structural tests may inspect them
without executing a model or using outcomes for selection.

The initial layer screen used the following grid, set before that run:
block inputs 3, 6, 9 × repeated-subject / final-token positions; rank one;
120 Adam updates, learning rate .01, minibatch 16, initialization seed 0.
The results already observed when this protocol was written are retained in
`~/.cache/arena_extended/das/native-screen-v1`. This is a development protocol,
not a claim of preregistration before every development observation.

## Data and high-level model

Each prompt has two distinct first-occurrence names A and B, and a repeated-name
role R: 0 means A repeats, 1 means B repeats. The high-level answer is B when
R=0 and A when R=1. An interchange copies source R, preserving base A and B.
Thus a cross-name source's own answer is deliberately a wrong target.

All rows are retained. Train has 72 prompts / 288 intervention pairs;
development has 48 / 192; final has 96 / 384. Each base has four sources:
same-name or disjoint-name pair, each with R=0 or R=1. Cross-name donors use
the next disjoint pair in that split, with the same template and orientation.
Every base's explicit high-level counterfactual prompt is also present.
Final uses three new templates and different name pairings; half of final
name pairs include a name absent from alignment training. The model may have
seen these names during pretraining; only the alignment data are held out.

## Selection

1. Complete every layer/position candidate in the first screen. Select the
   largest **unconditional cross-name, changed-role full-vocabulary IIA**.
   Break ties by larger target probability on that subgroup, then smaller
   layer index, then subject before end.
2. At the selected site, fit ranks 1, 2, 4, 8 for 240 updates using seed 0,
   learning rate .01 and minibatch 16. Use the same IIA/probability rule;
   break a remaining tie by smaller rank. Preserve all candidates.
3. Freeze the site, rank, training recipe, intervention distribution and
   evaluation code before any final-model execution. Train seeds 0, 1, 2
   independently; use every seed, with no best-seed selection.

These are finite development comparisons, not an exhaustive layer/hyperparameter
search. The final set is not used to choose a layer, rank, seed, checkpoint,
prompt family, metric or a favorable displayed example.

## Controls and measurements

- Exact identity, no intervention, whole-position source replacement, a
  training-only observational ridge role probe, and an observational mean
  direction. A probe can accurately classify R without a useful intervention.
- Three alignment fits with permuted counterfactual targets (fixed seeds
  400, 401, 402), trained with the identical recipe.
- All 32 independent random subspaces of the selected rank (seeds 700–731).
- Explicit counterfactual prompt outputs, exposing errors of the native model
  separately from failures of the alignment. No correctness-based filtering
  of the primary metric.
- Full-vocabulary IIA and target probability on all pairs, changed-role pairs,
  cross-name changed/unchanged pairs and same-name changed pairs. Show counts.
- Separately show the conditional subset on which both the base and explicit
  counterfactual are natively correct. This is a diagnostic denominator, not
  a replacement for the unconditional result.
- Cross-name donor-answer probability measures identity leakage. Its opposite
  is not by itself proof that all base information is preserved. Also show
  native full-vocabulary distributions and output tokens on fixed examples.

Every seed and every control remains in the displayed distributions. Accuracy
is a finite-task approximate agreement claim, not a unique feature identity,
proof about arbitrary interventions, or a complete causal-abstraction theorem.
Do not promote Interchange Intervention Training: all GPT-2 weights stay frozen.

## Execution and evidence

Use GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU FP32,
eager attention, four CPU threads. The installed native block implementation
and library versions are recorded. Cache complete recipient block inputs and
the actual native attention mask. Resume native blocks with gradients enabled;
only the width × rank QR parameter is optimized. Check weight hashes and the
absence of model gradients after each run.

The first native check compares 12 no-op/donor conditions at all six sites,
variable-length padding, full-forward gradients, independent finite differences
using a non-QR projector, and hook cleanup. FP32 matrix-multiply differences
are retained numerically; double-precision derivatives agree to about 2e-15
between the native hook and cached paths. Final release still requires full
own-answer execution, a separate fresh native reproduction, exact input hashes,
archived actual outcomes and visual inspection of every figure.


## Frozen selection and first final execution

The six-site screen completed in 93.052 seconds, selecting block 9 input / end.
The rank screen completed in 60.540 seconds, selecting rank eight (38/48
development cross-name role changes). All three controlled development seeds
then completed in 101.324 seconds. The final recipe is `das_protocol.py`.

The first complete own-answer learner, `full-own-answer-v1`, ran the final model
only after that selection. It passed in 292.081 seconds. Its three seeds obtain
71/96, 73/96, 72/96 unconditional final cross-name changed-role IIA. Whole-position
donor replacement also gets 72/96; no superiority claim is made. Every control
and final error is retained. Fresh-native-v1 then reproduced all 16,512 compared
outputs across 43 conditions, with zero full-vocabulary logit error and zero
top-token mismatches. The 17,232 native forward rows include 96 baseline rows,
16,128 tail intervention rows and 1,008 full-hook validation rows.

The next own-answer candidate corrects two heatmap layouts, displays actual
winning tokens and adds a descriptive template/base-name breakdown. It changes
no data, selection, fitting recipe or final examples. The final set is now
observed; these presentation refinements are not new held-out model selection.
