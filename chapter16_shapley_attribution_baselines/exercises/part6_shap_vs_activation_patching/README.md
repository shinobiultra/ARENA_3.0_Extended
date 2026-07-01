# [16.6] SHAP vs Activation Patching Verification Assets

Generated support files for the roadmap verification contract and the real CUDA
SHAP-vs-patching model-organism preflight.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus
  additive and interaction CUDA training runs.
- `16.6_SHAP_vs_Activation_Patching_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `16.6_SHAP_vs_Activation_Patching_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded GPU path in `solutions.py` trains an additive linear model and a
nonlinear neural coalition game, then compares exact Shapley values with
full-minus-ablated patching effects from real model outputs.
