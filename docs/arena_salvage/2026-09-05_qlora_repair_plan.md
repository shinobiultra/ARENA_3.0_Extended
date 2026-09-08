# QLoRA learner and numerical verification repair

Inspected the exercise and solution notebooks, instruction page, builder,
reference implementation, tests, and handoff. Original ARENA's transformer
module-by-module implementation and exact weight/output checks remain the style
comparison. No original chapter is modified.

Current learner failure: seven malformed answer dropdowns, a reference-class-only
update extractor that rejects notebook-defined adapters, weak forward/training
tests, and an identity/fingerprint policy that accepts thin reference wrappers.
The page is also a summary rather than the complete learner progression, and the
CUDA comparison has not been executed in the repaired environment.

Target claim: measure whether actual nested-NF4 QLoRA matches BF16 LoRA on the
counterbalanced marker task, including task counterfactuals, memory components,
protected-language behavior, and three fixed seeds. Do not assume acceptance or
weaken the existing thresholds. This optional QLoRA lab does not replace the
roadmap's full-finetuning / LoRA / DoRA / subspace study.

Signature result: learner-generated quantization and storage figures, then real
three-seed behavior, resident-state memory, and per-layer update spectra with
base-only, randomized-label, same-norm, marker-swap/removal, and name-only controls.

Student implementations: retain NF4, nested scales, QLoRA factors/forward,
counterbalanced data, training, quant-state audits and spectra in the notebook.
Fix dropdown construction and AST parity; include any newly required helper
visibly. Test exact forward values, gradients, one matched optimizer step, frozen
base preservation, notebook-defined adapter extraction, and zero-update spectra.

Provenance: fingerprint all taught class methods, reject observable direct and
closure-based reference delegation, and explicitly state the check's limits.
Hashes and object identities cannot certify arbitrary Python behavior; executed
learner code and independent numerical oracles remain the evidence.

Play cells: preserve editable block-size/outlier and marker controls; add live
real-model inspection if the completed experiment supports it.

Files: section sources/tests/builder, both notebooks, full instruction export,
reference artifacts, scope metadata and handoff. Tests: failing regressions first,
completed learner in a fresh kernel, actual CUDA comparison without fallback,
visual inspection, reference freshness and preservation checks.


## Executed outcome

Seven exercise groups and all 21 code cells complete from their own dropdowns.
The full instruction export and three play cells expose quantization, actual
three-seed behavior/memory/spectra, and a separate precision/buffer diagnostic.
The exact current bitsandbytes 0.50.2 dynamic map fixes six baseline failures;
41 tests now cover analytic forward/gradients, two Adam steps with replay,
notebook-defined adapters, zero rank, and source/evidence freshness.

QLoRA r=8 and BF16 LoRA reach 1.000 on both fixed evaluation splits. Resident
state is 38.67 versus 39.18 MiB (about 1.3% saving); masks alone occupy 32 MiB.
The unchanged BF16 parity gate fails at 0.03125 versus 0.01. FP32 agrees to
9.54e-7 in a separate fixture. The candidate remains unreleased, and the broader
core LoRA/DoRA/subspace requirement remains outstanding. Thresholds were not
weakened. The uniform-int8 scale derivation and reused replay prompts are
explicit limitations. Figures were inspected, and original ARENA is unchanged.
