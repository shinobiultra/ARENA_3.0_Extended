# [1.6] Local Frontier ML Infrastructure Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current runtime artifact contract.
- `1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The graded report path requires the uv environment to expose Python 3.14,
PyTorch `2.12.1+cu132`, torchvision `0.27.1+cu132`, CUDA 13.2, BF16 support,
and a deterministic BF16 CUDA matmul. The Gemma-sized memory estimate is
reported separately from measured peak VRAM.
