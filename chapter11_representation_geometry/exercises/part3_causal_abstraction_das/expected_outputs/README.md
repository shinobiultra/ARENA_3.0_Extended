# Recorded causal-alignment experiments

The notebook implements and executes the methods; these archives support it.
Each archive's manifest records its persistent source directory, archive hash,
every included member hash, and hashes of large tensors kept outside Git.
Every included member was read back and verified.

## How this fixture was produced

`execute_learner.py` fills each notebook stub from its own solution dropdown,
executes the complete lesson in a fresh kernel, and saves every figure and
actual native tensor. `das_reproduction.py` reloads pinned GPT-2 independently,
recaptures all 96 final prompts and reproduces every one of the 43 conditions.
It compares all 16,512 full-vocabulary outputs and their winning tokens,
including the explicit counterfactual reference. Large tensors remain in the
persistent directories named by the manifests.

## Trusted implementation

Structural truth comes from explicit binary equations and a fixed known
rotation, rather than from the fitted alignment. Independent projector finite
differences check the QR gradient. Native full-forward hooks check the cached
continuation, including its gradient, padding and frozen weights. A fresh
pretrained reload independently checks every retained final condition.

## Random seed

The lesson exposes all seeds. Native alignment seeds are 0, 1 and 2; shuffled
targets use 400, 401 and 402; all 32 random subspaces use 700 through 731. Site
and rank development use seed 0. `das_ioi_data.py` defines deterministic prompt
splits, and `das_protocol.py` freezes the native settings before final evaluation.
The earlier development screens and the first final observation are retained.

## Allowed tolerances

The exact tests state their tolerances alongside each oracle. The independent
float64 QR derivative uses `atol=2e-8, rtol=2e-6`. Native full-forward versus
cached float32 continuation allows a maximum absolute error of 0.0003 across
the complete vocabulary, with no changed winning tokens in fresh reproduction.
The course report additionally requires fresh maximum error at most 0.00001;
the recorded comparisons have exactly zero error. These bounds do not set a
minimum scientific effect size or remove failed interventions.

## When to regenerate

Changed scientific methods, model weights, data, training protocol or executable
learner cells require new full own-answer execution, independent native
reproduction and figure review. The publisher compares all current own-answer
code ASTs and source hashes. Prose-only corrections require a new rendered
review; they cannot silently replace experimental outputs. Use a new persistent
output directory for each run and retain previous results, including failures.

## Recorded runs

- `organism_development_v1`, `own_foundations_parse_failure_v1`,
  `own_foundations_v2`, `own_foundations_v3`: the exact-model foundation,
  including the initial Python-string parse failure and both successful own
  executions. These do not claim pretrained-model evidence.
- `native_path_checks_v1`: actual pretrained native no-op/donor, padding,
  gradient, independent finite-difference, hook cleanup and weight checks.
- `native_site_screen_v1`: all six development sites, rank one, 120 updates.
- `native_rank_screen_v1`: all four development ranks, 240 updates.
- `native_controls_v1`: all three selected fits, three shuffled-target fits,
  all 32 random subspaces and observational/whole-donor baselines.
- `full_own_answer_v1`: complete 61-cell / 31-code-cell own-answer execution,
  11 figures, all 15 inputs unchanged, 292.081 seconds. Final results include
  native mistakes and all controls; no favorable-example filtering.
- `fresh_native_v1`: independent model reload and 43-condition reproduction,
  16,512 compared outputs, 17,232 native forward rows, zero full-vocabulary
  logit error and zero top-token mismatches, 41.671 seconds.

The native numerical path test briefly uses float64 for independent finite
differences; ordinary training and reported interventions use CPU float32.
The exact model runs in float64. Native tensors contain complete next-token
vocabulary logits, not selected-name-only scores. A conditional native-correct
subset is a diagnostic alongside the full unconditional denominator.

`full_own_answer_v2` archives the refined 63-cell / 32-code-cell learner,
12 figures, all 15 inputs unchanged and 283.183 seconds. `fresh_native_v2`
repeats all 43 conditions in 40.759 seconds with zero logit error, also checking
the primary IIA directly from actual winning tokens. Current prose-only math
and exercise-format refinements match all 32 executed code ASTs; browser review
records the final source and figure hashes.

`full_own_answer_v3` is the final 63-cell / 32-code-cell execution after adding
two educational test failure messages and naming the native token panel.
It passes in 306.270 seconds with all 15 inputs unchanged. All twelve figures
are byte-identical to inspected v2. `fresh_native_v4` independently reproduces
all 16,512 outputs in 47.099 seconds, with zero full-vocabulary differences,
zero changed winners and unchanged model weights. The current excerpt render
checks native equations, the expected-output/solution dropdowns and the final
signature-result heading. The normal reporting entry point performs an
additional fresh CPU comparison; its exact directory and hashes are in the
current `verification_report.json`.

The first normal report attempt failed in the generic runner's CUDA cleanup
before section execution because the GPU was occupied. The final report runs
with CUDA hidden, as this section's declared device is CPU. This does not relax
any model, gradient, native-reproduction or actual-output requirement.
`normal_report_v3` preserves the initial resource-failure log and the first
accepted normal CPU report; `fresh_native_normal_report_v3` retains that
report's actual native comparison. Later release metadata does not replace
these historical runs.

Whole-position donor accuracy matches the average DAS accuracy here; this is a
compact intervention result, not evidence of superior accuracy or unique role
identification. The prescribed development play also contains a failed learned
intervention; it is retained rather than replaced with a favorable example.
