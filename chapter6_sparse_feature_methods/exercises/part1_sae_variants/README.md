# [6.1] SAE Variants Verification Assets

Generated support files for the roadmap verification contract.

- `6.1_SAE_Variants_exercises.ipynb` is the learner-facing notebook with
  local stubs for ReLU/L1, TopK, Gated, and JumpReLU encoders, sparse metrics,
  planted-feature generation, dictionary recovery, feature AUC, and
  decoder-vector steering controls.
- `6.1_SAE_Variants_solutions.ipynb` validates the section-local reference
  implementation against the visible tests and checks the committed CUDA report
  highlights.
- `artifacts.lock.yml` pins the toy contract plus the pinned Pythia-70M hidden-state TopK SAE CUDA preflight.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The real-model graded path is `run_gpu_test`: it loads
`EleutherAI/pythia-70m-deduped` at the locked revision, trains a tiny TopK SAE
on safe generated hidden-state activations, and checks held-out reconstruction,
density, feature-AUC, permuted-decoder, and decoder-steering controls. This is a
local SAE mechanics preflight, not a full GPT-2/Gemma SAE benchmark or semantic
feature-interpretation claim.
