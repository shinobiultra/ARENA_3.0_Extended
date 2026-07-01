# [8.1] Activation Patching Refresher Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current smoke-test artifact contract.
- `8.1_Activation_Patching_Refresher_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `8.1_Activation_Patching_Refresher_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded report path loads the pinned `NeelNanda/GELU_1L512W_C4_Code`
checkpoint through TransformerLens on CUDA, caches `blocks.0.hook_resid_post`,
patches every sequence position from the clean run into the corrupt run, and
requires final-position recovery to beat non-final wrong-position controls.
