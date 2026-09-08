# Prospective digit attribution protocol — version 1

Recorded before any digit-model fit or digit-query-loss evaluation for section
16.9. Five-row mathematical fixtures are separate foundation checks. The raw
public data and their metadata have been inspected; no digit-model outcome has been used
to choose examples, a method, a checkpoint or a hyperparameter. Keep all results,
including failed rankings. Implementation errors must be recorded and corrected;
changes motivated by observed outcomes require a new, explicitly exploratory
protocol rather than overwriting this one.

## Question and data

Which training images reduce a query's loss? Compare an infinitesimal objective
perturbation, contributions along an actual optimization trajectory, checkpoint
approximations and independently retrained models. These are different estimands.
No method is claimed to compute Data Shapley.

Use the original UCI Optical Recognition of Handwritten Digits files, created by
E. Alpaydin and C. Kaynak (1998), DOI 10.24432/C50P49, under CC BY 4.0.
The documented training and test sets use different groups of writers. Use the
original training file for training/ranking and the original test file for final
queries. This improves on the preliminary proposal to resplit sklearn's bundled
test-only subset. Individual writer IDs are not available in these feature files;
do not claim a new per-writer analysis. The UCI test file matches sklearn's data.

- Source: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- Archive SHA-256: `0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0`
- `optdigits.tra`: `e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd`
- `optdigits.tes`: `6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8`

Digit 1 is target zero and digit 7 target one. Divide the 64 pixel counts by
sixteen and append a constant one coordinate. There is no fitted normalization.
`attribution_data.py` hashes source row IDs to select 64 examples per class for
training, 24 per class for ranking and 32 per class for final. The same method
selects six training rows per true class for flipped labels. Preserve original
labels, modified labels and IDs. Both clean and 12/128-flipped conditions run;
no corruption is introduced into ranking or final labels. `data_manifest.json`
records all exact selected IDs and array hashes before training.

## Objectives and signs

Use a 65-parameter binary logistic model in CPU float64, with
`R(theta) = 0.02 * ||theta||^2 / 2`, including the bias coordinate. Query loss is
unregularized binary cross entropy. Train with the normalized weighted mean of
per-example cross entropy plus the same fixed R after every deletion.

Write `q_i = loss_i + R` and `F = mean(q_i)`. Upweight influence is the derivative
of query loss at the optimum of `F + epsilon*q_i`. It is
`-g_query @ solve(H, g_i)`, with the regularizer included in g_i and H, but not
in the query gradient. Positive means that increasing that example's weight
increases query loss. Negating and dividing by n estimates the change on removal;
positive removal change means the example was helpful.

Why include R in q_i? At epsilon = -1/n the objective is
`sum_{j != i}(loss_j + R)/n`, proportional to
`mean_{j != i}(loss_j) + R`. Upweighting only loss_i instead changes the relative
penalty after deletion. Show this alternative as a labelled control, never
compare the two derivatives as though they represented the same perturbation.

## Fits and trajectories

Converged reference models and all counterfactual refits start independently at
zero. Batched Newton updates use each fit's own Armijo line search (coefficient
1e-4, at most 30 halvings, objective roundoff allowance of eight machine epsilons
times max(1, absolute objective)), a maximum of 60 updates and maximum-coordinate
objective-gradient tolerance 1e-10. Every fit must meet the criterion. Compare
several foundation fits against independent SciPy root solving; native batched
fits must also match individually executed refits for prescribed first cases.
Batching may share arithmetic, never parameters, normalization or convergence.

Before native training, the five-row mathematical fixture exposed a stalled
3.56e-10 gradient when rounded objective values rejected a useful final step.
The roundoff allowance above corrects that implementation issue. The required
1e-10 stationarity threshold and the objective remain unchanged.

For each clean/corrupted condition, retain three single-example SGD trajectories
with order seeds 0, 1 and 2. Start at zero, visit every training row once per
epoch in its seeded permutation and record all pre/post states and row IDs.
Train for 40 epochs: learning rate .05 in epochs 1–20, then .01 in epochs 21–40.
Each update uses the gradient of q_i. No momentum, adaptive scaling or hidden
optimizer is used. Record the final gradient norm and distance to the converged
reference; finite SGD is not silently described as that optimum.

For exact path accounting, assign each observed query-loss reduction to the
single training row used in that update. Verify that summing across rows equals
initial minus final query loss. Compare against the first-order gradient product
at each actual pre-update state. Since each update includes regularization,
its gradient contribution is included and explained. A raw-data-gradient-only
control exposes the omitted contribution. No unique exact allocation is claimed
for a minibatch update.

Record all forty epoch-end checkpoints. Compute TracInCP with their actual epoch
learning rates for four fixed choices: all epochs, epochs 1–20, epochs 21–40 and
only epoch 40. Keep all three order seeds and all choices. Subsets are partial
checkpoint scores, not estimates that are rescaled to sum to total loss. Compare
rankings with exact path accounting and with converged-model leave-one-out effects;
the latter comparison crosses estimands and does not make TracIn a deletion
identity. No method, seed or checkpoint is selected on final outcomes.

## Counterfactuals and controls

For each clean/corrupted converged model:

1. Refit all 128 single-row deletions. Preserve query losses and probabilities
   for all 48 ranking and all 64 final queries, including every sign error.
2. Compute upweight influence with added solver damping 0, .01, .1 and 1. Damping
   is an approximation control; it does not redefine the refitted objective.
3. Compare derivatives at Newton states 0, 1, 3 and the final converged state.
   The convergence residual is shown beside the derivative error. If convergence
   precedes state 3, use the converged state and explicitly record that fact.
4. Refit the twelve predeclared weight-perturbation examples at `n*epsilon` in
   [-1, -.5, -.1, -.01, .01, .1, .5, 1]. The examples are the first three retained
   source rows in each true-label/flipped-status group. All probes and perturbation
   sizes remain visible. This is 96 independent perturbation fits per condition.
5. Show gradient similarity without H inverse, signed pixel-cosine similarity,
   and all 32 random rankings with seeds 1000–1031. Pixel similarity uses the
   image coordinates before the bias append, multiplied by target-label agreement.
   It is a labelled, loss-agnostic retrieval baseline.

Deletion policies use the average score over the 48 ranking queries only. Remove
low helpfulness scores first, with stable source-row tie breaking, at budgets
2, 4, 8, 16 and 32. Match every policy to exactly half the budget in each observed
training class. Policies are undamped converged influence, converged gradient
similarity, signed pixel similarity, actual ranking-query leave-one-out effects,
twelve TracInCP policies (three orders times four choices), and all 32 random
rankings: 48 policies total. Refit every selected deletion set from zero and
retain final mean loss/accuracy and per-query effects. The ranking-query exact
leave-one-out policy is a labelled expensive baseline, not a final-data oracle.
Identical deletion sets may share a fit only if exact membership and all fitted
source inputs match; retain each policy's identity and result separately.

Final-query labels may be used to describe each fixed final query's training-image
attributions, but never to select a deletion policy or its hyperparameters.
The fixed final gallery uses the first two source-row IDs per class. Play cells
use ranking queries by default and recompute scores/refits when changed.

## Independent validation and limitations

Before final access, test mathematical primitives and develop the learner using
ranking queries. Then freeze every method and run the complete own-answer
notebook, including its predetermined final stage. A separate fresh execution
must reload pinned source bytes, retrain all paths, refit all counterfactuals
and compare all actual outputs. Render and inspect every figure, equation and
dropdown before publishing. Prototype source files are not release evidence.

This is a small strictly convex classifier on real low-resolution digit images.
It does not reproduce either paper's large neural experiments, establish robust
influence in nonconvex models, estimate causal data value in general, or validate
writer-level generalization beyond the documented source split. Three SGD orders
share data and are not three independent datasets. A ghost affine-layer identity
may teach the gradient-product mechanism, but is not automatically evidence of
GhostSuite integration, tied-token gradients or multi-step In-Run Data Shapley.

## Primary method sources

- Koh and Liang (2017), https://proceedings.mlr.press/v70/koh17a/koh17a.pdf
  and https://proceedings.mlr.press/v70/koh17a/koh17a-supp.pdf
- Pruthi et al. (2020), https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf
  and https://proceedings.neurips.cc/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Supplemental.pdf

The original methods, normalization conventions, path/checkpoint distinction,
convergence assumptions and affine gradient-product derivation have been read.
The implementation and planned teaching examples are new for this course.
