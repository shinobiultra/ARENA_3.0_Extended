# [16.1] Exact Shapley on Ground-Truth Games Verification Assets

Generated support files for the roadmap verification contract and the real CUDA
neural-game preflight.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  CUDA-trained four-feature neural coalition game.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.
- `16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb` is the learner notebook.
- `16.1_Exact_Shapley_on_Ground_Truth_Games_solutions.ipynb` runs the visible tests
  and asserts the committed CUDA verification report fields.

The graded GPU path in `solutions.py` trains a small MLP on the complete binary
feature table, computes exact Shapley values from real model ablations, checks
them against the analytic data-generating game, and rejects a shuffled-label
trained-model control.
