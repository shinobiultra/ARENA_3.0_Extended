# I-JEPA learner review

Current learner failure: the interrupted implementation repair added image-space
key erasure, explicit shared semantic templates, a compound target ablation,
and a three-seed contract. Tests, notebook stubs, generated answers, result
tables, and lock metadata still use the old interface. Both notebooks, the page,
implementation, and original ARENA 0.2 CNN/ResNet exercises were read first.

Target section claim: in an exact predictive-key world, a small asymmetric
latent predictor uses the visible key to predict the masked object. Evaluation
uses different pixels but the same twelve semantic templates. This cannot
establish generalization to unseen object-position combinations or real images.

Signature result to create: executable learner-generated context interventions,
loss and spread curves, and per-seed results, showing both key replacement and
key erasure alongside random-feature probe failure.

Student-implemented functions: scene generation, patchification, masks, gathering,
encoder and predictor, loss, EMA, gradient boundary, collapse diagnostics,
probes, interventions, training, and experiment orchestration. Imported result
dataclasses and plotting are plumbing only.

Controls/baselines: exact patch/EMA/gradient oracles, image-space key erasure,
cyclic key replacement with non-key pixels fixed, zero context, matched random
architecture, constant features, shuffled labels, and the compound removal of
both EMA and target stop-gradient. Do not attribute its effect to EMA alone.

Play cells: choose the replacement class and seed, inspect prediction and loss,
and compare an erased key with all-zero latent context.

Files to modify: 14.2 tests, builder and generated surfaces, implementation only
for necessary contract corrections, fixture provenance, lock, report, and status
notes. Preserve prototype and TOY_ONLY status while real-image work remains.

Tests to run: focused semantic tests; answer-only completion and source parity;
clean-kernel execution including CUDA and play; visual inspection; all three
predeclared seeds; report, learner-surface, and preservation audits.
