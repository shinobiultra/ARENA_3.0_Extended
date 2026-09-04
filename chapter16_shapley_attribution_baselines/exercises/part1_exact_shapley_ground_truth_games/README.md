# [16.1] Exact Shapley on Ground-Truth Games Verification Assets

ARENA-style learner path and verification assets for the exact-Shapley
ground-truth-games section. The lesson's evidence is generated in the notebook
from a complete four-player Harsanyi-dividend game; the verification report is
supporting release evidence, not the teaching result.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  CUDA-trained four-feature neural coalition game.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.
- `../../instructions/assets/exact_shapley_ground_truth_signature.png` is the
  notebook-generated signature result: exact Shapley versus the dividend
  oracle and leave-one-out, an interaction-strength sweep, and visible marginal
  contexts with an additive control.
- `16.1_Exact_Shapley_on_Ground_Truth_Games_exercises.ipynb` is the learner notebook.
- `16.1_Exact_Shapley_on_Ground_Truth_Games_solutions.ipynb` runs every visible
  test and regenerates the signature result in a fresh CPU kernel.
The learner path implements complete coalition tables, dividend games, a
closed-form Shapley estimator, an independent analytic oracle, permutation
parity, efficiency/dummy/symmetry checks, leave-one-out, and a parameterized
interaction sweep. The existing graded GPU path in `solutions.py` remains
available to release verification, but it is outside this CPU-only lesson.
