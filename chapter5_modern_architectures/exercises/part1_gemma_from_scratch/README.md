# [5.1] Gemma from Scratch Verification Assets

Generated support files for the roadmap verification contract and the Hugging
Face Gemma reference-architecture parity preflight.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  deterministic tiny `transformers.GemmaForCausalLM` parity check.
- `5.1_Gemma_from_Scratch_exercises.ipynb` and
  `5.1_Gemma_from_Scratch_solutions.ipynb` provide an ARENA-style learner
  notebook pair for the full tiny Gemma ladder: RMSNorm, RoPE, grouped-query
  masks, SwiGLU, attention, decoder layer, full causal LM, cache parity, and
  Hugging Face tiny-reference parity. The notebooks define this path directly
  instead of importing the checked-in solution decoder.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the learner-facing full-decoder
  smoke-test contract.
- `expected_outputs/reference_metrics.json` records concrete tolerance and
  control thresholds for the CUDA/Hugging Face reference parity path.

The graded GPU path in `solutions.py` compares the local Gemma implementation
against Hugging Face's reference architecture on CUDA with matched random
weights, verifies logit/top-k/cache parity, and records measured VRAM. It does
not claim gated pretrained Gemma-weight validation.
