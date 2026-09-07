# [16.7] Expected Outputs

These fixtures describe two independently checkable CPU results: the
four-example exact cold-open and the eight-row, 40-step logistic training lab.
Neither notebook loads these files.

`reference_metrics.json` records both contracts, including the training lab's
exact values, one-run score, influence/LOO comparisons, budget curve, matched
deletions, and shuffled-label relocation. `smoke_test.json` records the learner
contract.

The exact values use float64. Exact oracle and efficiency checks use an absolute
tolerance of `1e-12`. In the training lab, the seeded 256-order estimate has
maximum absolute error `0.023644`; the mean maximum error over 16 seeds decreases
from `0.272753` at four orderings to `0.043709` at 256 orderings, and every
tested budget preserves the harmful-example rank. The compact function-level
contract also retains its 512-order estimate and learning-rate stress grid.

Regenerate these values only by running the visible solution implementation and
focused tests. The optional CUDA report is separate supporting evidence and is
not a source for these fixtures.

## How this fixture was produced

The four-example cold open enumerates every coalition. The eight-example lab
runs `training_run_coalition_values` for all 256 subsets using the same 40-step
logistic update rule. `training_run_data_shapley_diagnostics` compares exact
Shapley with the independently accumulated one-run estimator and controls.

## Trusted implementation

Complete subset retraining and factorial coalition weighting form the exact
reference. The one-run gradient estimator is the approximation being tested;
it does not define its own ground truth. The immediate tests pin the eight
values and efficiency, including the harmful point at index six with value
approximately -0.618051. Relocating its bad label must relocate negative credit.

## Random seed

The 256-order estimate uses seed 0; the budget sweep uses sixteen declared
seeds. Random ranking controls use seed 13. Label relocation uses the explicit
permutation `(6, 1, 2, 3, 4, 5, 0, 7)`, keeping every feature row fixed.

## Allowed tolerances

The four-example float64 exact oracles use `rtol=0, atol=1e-12`. Eight-example
retraining reference values and budget means use `rtol=0, atol=1e-10`, with
an efficiency residual below `1e-12`. Sampling error is a measured result, not
floating-point tolerance; it must remain visible in the budget curve.

## When to regenerate

Regenerate after changes to training steps, initialization, learning rate,
coalition utility, valuation formula, evaluation set, seed, or label relocation.
Run exact retraining and the learner figures before refreshing the report.
Preserve the visible approximation failure at larger learning rates.
