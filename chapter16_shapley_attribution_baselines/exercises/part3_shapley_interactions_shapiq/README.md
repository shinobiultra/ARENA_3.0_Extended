# [16.3] Shapley Interactions with shapiq Verification Assets

Generated support files for the roadmap verification contract and the real CUDA
Shapley-interaction preflight.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  CUDA-trained four-feature neural coalition game.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded GPU path in `solutions.py` trains the finite neural game, computes
pairwise Shapley interactions from real model ablation tables, validates
`shapiq` parity on that trained-model table, bounds off-target interactions,
and rejects a shuffled-label trained-model control.

Notebooks:

- `16.3_Shapley_Interactions_with_shapiq_exercises.ipynb` contains the learner
  stubs and visible-test calls.
- `16.3_Shapley_Interactions_with_shapiq_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
