# [16.1] Exact Shapley on Ground-Truth Games Verification Assets

ARENA-style learner path and verification assets for the exact-Shapley
ground-truth-games section.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  CUDA-trained four-feature neural coalition game.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.
- `../../instructions/assets/exact_shapley_validation_loop.svg` diagrams the
  finite-game validation loop.
- `../../instructions/assets/exact_shapley_signature_result.svg` summarizes the
  CUDA signature result.
- `16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb` is the learner notebook.
- `16.1_Exact_Shapley_on_Ground_Truth_Games_solutions.ipynb` runs the visible tests
  and asserts the committed CUDA verification report fields.

The graded GPU path in `solutions.py` trains a small MLP on the complete binary
feature table, computes exact Shapley values from real model ablations, checks
them against the analytic data-generating game, and rejects a shuffled-label
trained-model control.
