# 12.3 verification ladder provenance

## How this fixture was produced

The full own-answer learner ran all 34 code cells, twelve fresh training fits, 36 final evaluations, 285 causal conditions and seven live figures. Its manifest binds the exact section sources and paired notebooks. `completed_word_study.json.gz` holds the actual generated tokens and causal results. The neighboring README gives all dataset sizes, training settings, outcomes and conditional denominators. The report records a separate fresh four-mode, seed-0 CUDA run; it is not assumed numerically identical to CPU training.

## Trusted implementation

Independent explicit-window and attention-value/gradient oracles, shifted-loss and masking checks, independently derived AdamW updates, ragged EOS tests and native clean/donor forward comparisons check the learner's own classes and functions. Engineered RGB features provide a separate exact warm-up; they are not described as a learned vision encoder. All 43 section tests pass. The raw-evidence validator recomputes scores from generated words, retains all seeds and control draws, and checks checkpoint selection and paired denominators.

## Random seed

Training retains seeds 0/1/2 in each of joint, text-only, image-only and shuffled modes. Image train/development/familiar/compositional recipes use seeds 100/200/300/400. The separate CUDA check uses seed 0 for all four modes. Its training results are reported independently, including failures. The full protocol and exact source hashes are in the published manifest and raw study.

## Allowed tolerances

Token IDs, strings, image-pair backgrounds and exact stage replacement identities use rtol=0, atol=0. Float64 mathematical oracles use explicit rtol=1e-12 to 1e-10 and atol=1e-12 to 1e-10 according to the operation; the two-step float32 optimizer comparison uses rtol=1e-5, atol=1e-6. Each immediate test contains its actual bound. Training on different devices may choose a different development checkpoint; the independent CUDA training is evaluated as a new run, not forced to match CPU weights or token outcomes.

## When to regenerate

For source or protocol changes, rebuild and run the full own-answer learner with `execute_learner.py --scope full --publish` into a new output directory, run all tests, inspect all seven figures and run the separate 12.3 CUDA report. Retain every trained seed and failed held-out combination. Zero compositional-caption accuracy remains visible and must not be retuned away using final data.
