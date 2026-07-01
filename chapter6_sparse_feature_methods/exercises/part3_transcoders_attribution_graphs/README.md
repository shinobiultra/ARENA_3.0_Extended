# [6.3] Transcoders and Attribution Graphs Verification Assets

Generated support files for the roadmap verification contract.

- `6.3_Transcoders_and_Attribution_Graphs_exercises.ipynb` is the learner
  notebook with local stubs for transcoder replacement, KL/logit-diff checks,
  feature contributions, graph construction, and graph controls.
- `6.3_Transcoders_and_Attribution_Graphs_solutions.ipynb` executes the visible
  tests against `solutions.py` and checks the committed CUDA report highlights.
- `artifacts.lock.yml` pins the current smoke-test artifact contract and the
  pinned TransformerLens `gelu-1l` CUDA preflight.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The real-model graded path is `run_gpu_test`: it loads `gelu-1l` at the locked
revision, checks exact MLP-feature replacement parity, trains a tiny ReLU
transcoder on real MLP activations, and validates a top-feature attribution
graph against low-effect feature controls.
