# [16.7] Data Shapley in One Training Run Verification Assets

Generated support files for the roadmap verification contract and the real CUDA
one-step Data Shapley preflight.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  CUDA one-step linear training run.
- `16.7_Data_Shapley_in_One_Training_Run_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `16.7_Data_Shapley_in_One_Training_Run_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded GPU path in `solutions.py` enumerates all data coalitions on CUDA,
runs an actual one-step optimizer update, computes exact/Monte Carlo/in-run
Data Shapley evidence, and verifies the harmful-example deletion control.
