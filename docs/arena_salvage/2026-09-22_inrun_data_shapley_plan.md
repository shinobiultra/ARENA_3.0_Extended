# Extend 16.7 from retraining values to a training-run accountant

The user explicitly requested continued course writing while the independent
SPEX experiments run. Root owns this existing selected lesson; the sole Astra
agent owns the capstone. Original ARENA remains immutable. The current notebook
pair, page and original 1.1 pair/page were inspected before editing.

## Current learner failure

The eight-example exact-retraining game is useful, but the chapter's checkpoint
gradient proxy does not teach the local coalition game underlying In-Run Data
Shapley. It omits ghost dot products, minibatch normalization, nonlinear training
and the distinction between exact trajectory accounting and deletion/retraining.

## Target section claim

Students can allocate a fixed SGD trajectory's validation-loss reduction, check
first-order errors against exact local coalition values, and compute the same
gradient similarities without materializing per-example parameter gradients.
Retain the existing exact and Monte Carlo retraining exercises. Do not identify
the trajectory game with the independently retrained-subset game.

## Signature result

Training-step attribution heatmaps, exact versus first-order accounting curves,
and a real handwritten-digit gallery with attribution-selected and matched-random
deletion/retraining controls. Report measured timing without claiming a speedup
from the name of the algorithm.

## Student-implemented functions

An explicit two-layer tanh classifier, its per-example gradients, ghost layer
dot products, local coalition utility, exact local Shapley, first/second-order
allocations, SGD accumulation and a fixed-schedule deletion/retraining control.
Parameter updates and accumulation stay visible in the notebook.

## Controls and baselines

Brute-force local coalitions, autograd gradient/Hessian references, duplicate and
null rows, frozen versus recomputed checkpoints, wrong batch normalization,
first/second-order approximations, shuffled ranking labels, random attribution
orders and matched deletion counts. A failure to rank corrupted examples or to
improve final loss remains an outcome, not grounds to change the dataset.

## Play cells

Change the step size on a fixed local game and compare approximation error;
change the validation target or remove a selected training example and retrain.

## Files and validation

Own `part7_data_shapley_in_one_training_run/`, its corresponding page/assets and
focused tests. Prepare outside the canonical notebook pair until full own-answer
execution and figure review. Shared registration changes follow accepted results.
Use primary paper sections 3–4 and the already pinned UCI digit acquisition
helper from 16.9; it is data plumbing only, never an imported attribution method.

Check exact local Shapley by independent permutations, gradient identities by
autograd, normalization and counterfactual semantics with adversarial tests.
Execute completed student answers in a clean CPU kernel, compare a fresh run,
inspect every figure and play, then run relevant course gates and preservation.

## Prospective small real-data comparison

Use the existing pinned 16.9 UCI 1-versus-7 train/ranking/final splits, including
their twelve fixed corrupted labels. Strip the helper's appended bias coordinate;
the learner network has explicit biases. Train a float64 64→16→1 tanh network
with plain SGD, batch size 4, four epochs and learning rate 0.1. Use seeds 16701,
16702 and 16703 for initialization and recorded batch permutations. Only the
48 ranking images define attribution; the 64 official-test images are evaluated
after attribution/deletion choices are fixed. For the first seed, check every
one of each batch's 16 local coalitions. All three seeds retain full ghost scores,
loss trajectories and step allocations. Compare deletion of the twelve lowest
scores with twelve highest, twelve seeded-random rows and the known corrupted
rows; use the same initialization and original batch schedule, masking removed
rows with a fixed denominator of four so training cost and step count remain
matched. Include shuffled ranking-label attribution using a fixed seed. Keep all
outcomes. This is a compact nonlinear classifier study, not a foundation-model
pretraining experiment or a claim of equivalence with retraining Data Shapley.
