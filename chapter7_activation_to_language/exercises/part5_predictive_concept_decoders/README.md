# [7.5] Predictive Concept Decoders Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the smoke-test contract plus the real `gelu-1l`
  sparse-concept PCD preflight.
- `7.5_Predictive_Concept_Decoders_exercises.ipynb` is the local learner
  notebook with stubs and visible tests.
- `7.5_Predictive_Concept_Decoders_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records sparse-concept density,
  PCD-vs-baseline, OOD, audit, stability, and top-vs-low-margin active removal
  metrics
  for the pinned CUDA run.

The real-model graded run loads the pinned TransformerLens `gelu-1l` checkpoint
on CUDA, builds signed sparse residual concepts, gates them by behavioral
question, and records exact model/tokenizer revisions, hook name, split counts,
concept names, baseline scope, controls, and measured VRAM. It does not claim a
broad PCD benchmark or a full Activation Oracle comparison.
