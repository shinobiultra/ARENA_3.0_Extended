# Frozen TreeSHAP experiment

This protocol is written before the first native tree fit. The UCI Concrete
Compressive Strength data and recipe-group split are reused from section 16.12;
that section's neural results are known. No new unseen-dataset claim is made.
All 1030 rows are retained, including replicate and conflicting outcomes.
There are 758 training, 153 explanation and 119 audit rows, in 298/64/65
non-overlapping seven-ingredient recipe groups. Age is the eighth input.
Predictions are in MPa; squared errors are in MPa squared. Laboratory identities
are unavailable. Source and exact source-row membership are in data_manifest.json.

## Models

Fit three model families at each seed 161310, 161311, 161312, with both true and
shuffled training targets: 18 fresh model fits, all retained. The paired shuffled
target permutations use NumPy default_rng seeds 161320, 161321, 161322. Reuse the
same permutation across families at a seed. No validation-based selection,
restarts, early stopping or hyperparameter searches. Training uses raw inputs
and raw targets. Tree routing converts inputs to float32 as sklearn does.

- DecisionTreeRegressor: max_depth=5, min_samples_leaf=8, criterion=squared_error,
  splitter=best, max_features=None, random_state=model seed.
- RandomForestRegressor: n_estimators=32, max_depth=5, min_samples_leaf=8,
  max_features=1.0, bootstrap=True, criterion=squared_error, n_jobs=1,
  random_state=model seed.
- GradientBoostingRegressor: n_estimators=48, learning_rate=0.05, max_depth=3,
  min_samples_leaf=8, loss=squared_error, criterion=friedman_mse, subsample=1.0,
  random_state=model seed. Default constant mean initializer.

Other settings retain the installed sklearn defaults, recorded with estimator
get_params and package versions. Main model is the first clean forest seed;
the first clean decision tree supplies the tree picture. All three clean
families and seeds plus all shuffled fits remain in performance and attribution
results. Native tree fits: 6 decision trees, 6 forests, 6 boosted ensembles,
486 component trees in total. No source changes are frozen as model weights.

## Explanation games and evaluation

Compute path-dependent TreeSHAP and empirical interventional TreeSHAP on every
explanation and audit row for all 18 models. Path absence follows weighted
training node covers (including bootstrap multiplicities). Interventional
absence uses one intact training donor row for all missing coordinates.
The primary background has 64 rows, sorted by SHA256 of
arena16.13/background/161360/{source_row}. Alternative backgrounds for the primary
forest use nested 16/128 rows and independent 64-row sets at seeds161361/161362.
No implicit TreeExplainer auto mode or background subsampling is permitted.

Exact subset oracles evaluate all 256 coalitions, on every explanation and audit
row, for the first clean seed of each model family and both games. Verify full
local attribution vectors against those oracles, not merely their sums. Compare
all 18 models and all 272 rows to installed SHAP TreeExplainer, explicitly raw
regression output, tree_path_dependent or interventional mode. Preserve every
maximum absolute error. Expect float32 internal contribution rounding in the
compiled interventional reference; use atol=2e-5, rtol=2e-6 there, and atol=1e-8,
rtol=1e-8 for the float64 exhaustive oracle. Compare actual predictions and bases.

Compute Saabas for every model/row. Save explanation-split mean-absolute feature
rankings before evaluating audit deletion. For the three first-seed clean
families, compare path SHAP, interventional SHAP and Saabas rankings by audit
squared error under the exact interventional deletion game. Retain all 100
random orders (seed161380). These curves are model/game diagnostics, not causal
ingredient interventions. Full performance includes the training-target mean.

Store all fitted tree arrays, exact games, local attributions, predictions,
background IDs, selected orders, all audit rows and controls. Fresh reproduction
must refit all 18 models from source data with no saved trees supplied, then
compare every stored numerical result and exact row identity. Measured wall
clock timings and environment paths are descriptive and compared separately.

## Mechanism and computational controls

Use exhaustive tiny path-weight enumeration, a repeated-feature threshold tree,
a constant tree and dummy features. A two-feature correlated binary population
with P(00)=P(11)=0.45 and P(01)=P(10)=0.05 distinguishes the true conditional,
interventional and cover-weighted games. Two equivalent additive trees with
opposite root-feature ordering reveal the path game's structure dependence.
Both predict f(x)=x0+2*x1 on binary inputs; query=(1,1). Show Saabas efficiency
without assuming Shapley parity.

Time both exact enumeration and dynamic programming on a fixed depth-three
balanced synthetic tree with 3/6/9/12/15 supplied features (unused extras are
explicit dummies), and on balanced depth2..8 trees using one feature per level.
Use eight fixed binary queries, seeded leaf values161390, three repeated timings;
report medians and the individual timings. Explain that an exact enumerator
that first removes dummies avoids the first axis's waste. No fitted real models
are selected by these timing results.

## Editable plays and execution

The tiny-tree play changes covers and query. The real play defaults to audit
index0, the first clean forest, primary64-row background and interventional
mode; it recomputes predictions and local contributions. Audit reuse in this
play is exploratory. All algorithms and the complete fitting/evaluation code
are learner-visible. Run on CPU, one sklearn worker, at most four BLAS threads.
The canonical own-answer execution must generate every figure from fresh model
fits. The paper connection is exact tree algorithms on a small public regression
study, not a claim to reproduce the paper's clinical benchmark or causal effects.
