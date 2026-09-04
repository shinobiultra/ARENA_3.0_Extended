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
