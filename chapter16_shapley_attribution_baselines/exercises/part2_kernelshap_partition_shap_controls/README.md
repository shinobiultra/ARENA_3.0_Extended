# [16.2] KernelSHAP and PartitionSHAP Controls Verification Assets

Generated support files for the roadmap verification contract and the real CUDA
SHAP-control preflight.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  CUDA-trained four-feature neural coalition game.
- `16.2_KernelSHAP_and_PartitionSHAP_Controls_exercises.ipynb` contains the
  learner stubs and visible tests.
- `16.2_KernelSHAP_and_PartitionSHAP_Controls_solutions.ipynb` runs the local
  reference solution, visible tests, and committed CUDA report assertions.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded GPU path in `solutions.py` trains the finite neural game, computes
KernelSHAP and PartitionSHAP from real model ablation tables, checks exact
Owen-value parity, verifies a cross-group irrelevant-player control, and rejects
a shuffled-label trained-model control.
