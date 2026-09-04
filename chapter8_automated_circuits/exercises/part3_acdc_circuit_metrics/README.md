# [8.3] ACDC and Circuit Metrics Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current artifact contract.
- `8.3_ACDC_and_Circuit_Metrics_exercises.ipynb` is the local learner notebook
  with stubs and visible tests.
- `8.3_ACDC_and_Circuit_Metrics_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded report path loads the pinned `NeelNanda/GELU_1L512W_C4_Code`
checkpoint through TransformerLens on CUDA, prunes a residual-position circuit
from exact patch scores, and verifies faithfulness, minimality, completeness, a
same-size wrong-position baseline, and held-out prompt-template localization.
