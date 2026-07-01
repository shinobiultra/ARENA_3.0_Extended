# [7.3] Mini Activation Oracles Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the smoke-test contract plus the real `gelu-1l`
  residual-direction mini-oracle preflight.
- `7.3_Mini_Activation_Oracles_exercises.ipynb` is the learner notebook with
  local stubs for activation-question batches, baseline comparisons, OOD split
  reports, random-activation controls, and activation-patching checks.
- `7.3_Mini_Activation_Oracles_solutions.ipynb` executes the visible tests
  against `solutions.py` and checks the committed CUDA report highlights.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots,
  expected OOD accuracies, random-abstention behavior, and patching answer
  changes for the pinned CUDA run.

The real-model graded run loads the pinned TransformerLens `gelu-1l` checkpoint
on CUDA, uses safe generated prompts, and records exact model/tokenizer
revisions, hook name, split counts, answer classes, controls, and measured VRAM.
It does not claim a LoRA-trained or API-backed Activation Oracle benchmark.
