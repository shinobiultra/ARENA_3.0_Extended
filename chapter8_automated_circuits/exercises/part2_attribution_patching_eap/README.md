# [8.2] Attribution Patching and EAP Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current artifact contract.
- `8.2_Attribution_Patching_and_EAP_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `8.2_Attribution_Patching_and_EAP_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded report path loads the pinned `NeelNanda/GELU_1L512W_C4_Code`
checkpoint through TransformerLens on CUDA, compares exact residual-stream
patching against corrupt-run attribution patching, integrated-gradient patching,
and an EAP-style edge matrix, and requires final-position agreement with
non-final gradient controls.
