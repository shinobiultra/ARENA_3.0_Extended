# [7.2] Feature Verbalizers Verification Assets

Generated support files for the roadmap verification contract.

- `7.2_Feature_Verbalizers_exercises.ipynb` is the learner notebook with local
  stubs for example selection, explanation prediction, counterexample revision,
  intervention direction, and brevity checks.
- `7.2_Feature_Verbalizers_solutions.ipynb` executes the visible tests against
  `solutions.py` and checks the committed CUDA report highlights.
- `artifacts.lock.yml` pins the current artifact contract.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded report path loads the pinned `NeelNanda/GELU_1L512W_C4_Code`
checkpoint through TransformerLens on CUDA, constructs a real residual direction
from safe final-token activations, scores safe prompt examples by projection,
tests a concise keyword explanation on held-out and contrastive examples, and
verifies that adding the direction increases the predicted logit difference.
