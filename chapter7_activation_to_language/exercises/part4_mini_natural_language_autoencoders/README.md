# [7.4] Mini Natural Language Autoencoders Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the smoke-test contract plus the real `gelu-1l`
  residual trainable text-bottleneck mini-NLA preflight.
- `7.4_Mini_Natural_Language_Autoencoders_exercises.ipynb` is the local
  learner notebook with stubs and visible tests.
- `7.4_Mini_Natural_Language_Autoencoders_solutions.ipynb` runs the reference
  implementation, visible tests, and committed CUDA report checks.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records reconstruction, baseline,
  latent-preservation, OOD, brevity, shuffled-control, and counterfactual
  metrics for the pinned CUDA run.

The real-model graded run loads the pinned TransformerLens `gelu-1l` checkpoint
on CUDA, trains a small residual-to-phrase-id encoder and phrase-id-to-residual
decoder table, transmits only compact natural-language phrase ids, and records
exact model/tokenizer revisions, hook name, split counts, text-bottleneck
format, controls, and measured VRAM. It does not claim Anthropic-scale NLA
training or faithful natural-language explanation of arbitrary activations.
