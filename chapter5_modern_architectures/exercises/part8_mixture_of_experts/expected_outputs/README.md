# Expected outputs for [5.8]

This is a candidate falsification lab. The former positive-specialization
release contract fails and remains preserved. These CPU measurements do not
substitute for an accepted CUDA report.

## How this fixture was produced

`run_cpu_experiment(steps=180, seeds=(7,19,31))` initializes three independent
models, each cloned into four aux/z conditions plus an aligned equal-width dense
control. Every condition uses the same ordered training batches within a seed.
`reference_metrics.json` records these actual runs, not historical surrogate-gate
results. The shorter seed-2 smoke run tests trainability only.

## Trusted implementation

Hand-calculated softmax gates, left-to-right slot ledgers, dispatch/scatter sums,
and brute-force expert calls anchor mechanics. Switch top-1 uses its selected
probability, not a unit-valued straight-through gate. Admission does not sort
future affinities. Gradient, fixed-mask suffix, batch-independence, permutation,
and cache oracles protect these distinctions.

## Random seed

Full CPU seeds are 7, 19, and 31. Preferred experts are selected on dev routes,
then all experts are ablated and forced on final data. The sixteen shuffled
probe fits are controls within a model, not sixteen independent trained models.
The shorter smoke uses seed 2; exploratory play defaults to seed 43.

## Allowed tolerances

Exact ties, slot indices, masks, and counts use `rtol=0, atol=0`. Numerical
float32 comparisons retain their per-test tolerances. The original positive
release rule keeps its expert-ablation inequality and maximum coherent
permutation error of 3e-6. No threshold was relaxed to accept the failed claim.

## Visible results and interpretation

Fully regularized accuracy is approximately 0.9942 with hard-load entropy
0.9994, but overflow remains 0.1815. The unregularized model reaches 0.9968
accuracy while routing has zero entropy and 0.625 overflow. The context-only
probe stays near chance (0.3403 versus shuffled 0.3333).

Preferred-expert ablation drop is zero, so its claimed advantage over other
ablations is rejected. Nonpreferred forcing damages accuracy by about 0.2888,
but this is a different intervention. Longer-stream accuracy near 0.657 remains
visible. Sparse per-sequence padding executes 1,620 FFN slots versus 1,440 dense
slots; neither a matching FLOP budget nor optimized sparse-kernel speed is claimed.

The stored route/intervention heatmaps represent seed 7; aggregate effects and
seed dots include all three models. Expert identities are not aligned across seeds.

## When to regenerate

After changing gate conventions, capacity semantics, training, split, metric,
intervention, or seed list, run the exact tests, complete the learner from its
answers, execute every cell, and inspect the figure. Preserve failed hypotheses.
Any new positive claim needs a declared experiment and controls before its result
is interpreted; do not simply tune this final set until the old claim appears.
