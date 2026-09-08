# [5.2] Gemma Scope and Feature Steering Verification Assets

Generated support files for the roadmap verification contract.

- `5.2_Gemma_Scope_and_Feature_Steering_exercises.ipynb` is the
  learner-facing notebook with local stubs for SAE metrics, direct logit
  attribution, feature validation, top activations, ablation, and steering
  controls.
- `5.2_Gemma_Scope_and_Feature_Steering_solutions.ipynb` validates the
  section-local reference implementation against the visible tests.
- `artifacts.lock.yml` pins the current notebook contract plus the real Gemma Scope
  2 1B-IT layer-13 residual JumpReLU SAE artifact preflight and authenticated
  Gemma 3 activation validation.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The current GPU report verifies the pinned SAE config, safetensor shapes,
finiteness, CUDA load, JumpReLU encode/decode, and real Gemma 3 residual
activation scoring against random-feature and label-shuffle controls.
