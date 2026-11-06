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

## Canonical migration and convention correction — 2026-10-16

The canonical authoring source is
`infrastructure/chapters/chapter5_modern_architectures/master_5_1.py`.
The generated lesson preserves all 16 historical code cells and all 18 saved
outputs with their original counts, plus every original teaching passage and
exercise. The old module remains byte-identical in `legacy_solutions.py`;
the complete pre-migration source is archived with hashes and provenance in
`docs/mastery/evidence/gemma_master_migration/`. The old verification report
retains its original bytes and historical source identity; it is not a fresh
execution receipt for the generated lesson.

The historical phrase “independent exact reference” above requires a
qualification: that local reference substantially duplicates the learner
decoder. The historical Hugging Face diagnostic also directly copies weights
between different activation and RoPE conventions. Its saved numerical output
is a tolerance diagnostic, not an exact architecture-equivalence result.

An additive CPU exercise makes students implement the GELU(tanh) gate and a
per-head Q/K row permutation from split-half to interleaved rotary coordinates.
A single separately recorded tiny random-weight comparison produces maximum
absolute logit error `1.862645149230957e-9` and MSE
`3.9970586966848386e-21`, below the predeclared `5e-6` / `1e-12` gates.
Unmatched, activation-only and basis-only variants visibly fail those gates;
all four nevertheless have perfect ordered top-five agreement. The new plot
therefore also shows why top-k agreement alone is weak evidence for parity.

There are now 17 student stubs and 18 forecast/reflection pairs, with immediate
checks, separate historical/fresh output labels, the original token-ID play,
and a new cache-position mask play. The fresh evidence covers only the named
CPU primitive subset and tiny convention comparison. It does not establish
whole-notebook freshness, pretrained Gemma 3/SmolLM2 coverage, CUDA/cache
parity, or completion of the learner walkthrough.
