# [6.4] Crosscoders and Model Diffing Verification Assets

Generated support files for the roadmap verification contract.

- `6.4_Crosscoders_and_Model_Diffing_exercises.ipynb` is the learner notebook
  with local stubs for shared/model-specific reconstruction, feature
  specificity, behavior-delta prediction, paired score deltas, and ablation
  controls.
- `6.4_Crosscoders_and_Model_Diffing_solutions.ipynb` executes the visible tests
  against `solutions.py` and checks the committed CUDA report highlights.
- `artifacts.lock.yml` pins the current smoke-test artifact contract and the
  paired TransformerLens `gelu-1l` / `solu-1l` CUDA preflight.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The real-model graded path is `run_gpu_test`: it loads pinned GELU-1L and
SoLU-1L checkpoints, checks exact shared-plus-delta reconstruction of paired
residual activations, validates an SVD model-diff direction on generated
technical/everyday prompt labels, and compares top-direction ablation against an
orthogonal random-direction control.
