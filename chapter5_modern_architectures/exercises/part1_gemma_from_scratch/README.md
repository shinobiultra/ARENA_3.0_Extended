# [5.1] Gemma from Scratch Verification Assets

Learner-facing notebooks and support files for building and validating a
Gemma-style decoder from scratch.

- `artifacts.lock.yml` pins the current smoke-test artifact contract plus a
  deterministic tiny `transformers.GemmaForCausalLM` parity check.
- `5.1_Gemma_from_Scratch_exercises.ipynb` and
  `5.1_Gemma_from_Scratch_solutions.ipynb` provide the progressive learner
  build: RMSNorm, RoPE, grouped-query masks, SwiGLU, attention, decoder layer,
  a complete causal LM, activation tracing, cache parity, and Hugging Face
  tiny-reference parity. The taught implementation remains inline.
- The signature result compares every residual-stream stage and final logits
  against an independent exact reference. Removing embedding scaling, removing
  RoPE, and using the wrong GQA head order are visible architectural controls.
- The Try It Yourself cell accepts new token-ID sequences, and the bonus
  anomaly hunt localizes the first stage at which each control diverges.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the learner-facing full-decoder
  smoke-test contract.
- `expected_outputs/reference_metrics.json` records concrete tolerance and
  control thresholds for the CUDA/Hugging Face reference parity path.

The separate graded GPU path in `solutions.py` compares the local Gemma implementation
against Hugging Face's reference architecture on CUDA with matched random
weights, verifies logit/top-k/cache parity, and records measured VRAM. It does
not claim gated pretrained Gemma-weight validation.
