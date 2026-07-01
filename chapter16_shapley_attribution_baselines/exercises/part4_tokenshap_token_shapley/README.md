# [16.4] TokenSHAP and TokenShapley Verification Assets

Generated support files for the roadmap verification contract and the real CUDA
TokenSHAP preflight.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  CUDA-trained four-position token scorer.
- `16.4_TokenSHAP_and_TokenShapley_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `16.4_TokenSHAP_and_TokenShapley_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded GPU path in `solutions.py` trains an embedding MLP on the complete
masked-token coalition table, computes exact and sampled TokenSHAP from real
model outputs, verifies efficiency and ranking, and rejects a shuffled-label
trained-model control.
