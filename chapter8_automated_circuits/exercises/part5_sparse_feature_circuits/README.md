# [8.5] Sparse Feature Circuits Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current notebook-contract artifacts.
- `8.5_Sparse_Feature_Circuits_exercises.ipynb` is the local learner notebook
  with stubs and visible tests.
- `8.5_Sparse_Feature_Circuits_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA/official-artifact report
  checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the historical `run_smoke_test` contract hook.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The current GPU report checks the toy ladder, Pythia-70M-deduped residual
effects, released SAE state-dict and feature-attribution controls, a
100-example official-code sparse feature graph, and held-out official
faithfulness on 40 `simple_test` examples.
