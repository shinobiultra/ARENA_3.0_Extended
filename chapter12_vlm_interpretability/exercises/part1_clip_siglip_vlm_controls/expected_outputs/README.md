# [12.1] Expected outputs and measured evidence

`pixel_encoder_metrics.json` is written by the actual completed learner. It
contains all 18 training histories (two objectives, three seeds, three training
conditions), parameter changes and all 42 evaluation rows. The train/development/
evaluation renderer seeds are 100/200/300. There are 576 training and 288
held-out image instances, with all twelve semantic classes shared across splits.
No held-out image enters the optimizer; zero byte overlap does not itself prove
novel geometry. The 1,200-step schedule was chosen using development seed 200.

The reference CUDA run gives mean fresh-image classification 0.9711 for CLIP
and 0.9907 for SigLIP. Across seeds, frozen towers average 0.4537/0.4190 and
shuffled pairing 0.0787/0.0833. Text-to-image retrieval is 1.0 for both trained
objectives; each caption needs only one correct gallery image. All seed values,
not just means, are displayed. Do not expect bitwise identical optimizer paths
across devices. The numerical exercise fixtures use exact independent formulas
and documented float tolerances.

`typographic_metrics.json` records the learner's pretrained CLIP experiment on
48 generated scenes at seed 400. Its model revision and exact font SHA256 are
included. Conditions cross five word/card controls with background and object
placement, plus a clean baseline. Metrics distinguish accuracy, attack-class
choice, targeted flips among baseline-correct images, and margin changes
relative to clean and matching blank-card baselines. Blank-background cards
are pixel-identical to clean scenes. Scrambled letters are an imperfect
control when placed over the object; their effect is retained.

`real_checkpoint_metrics.json` records the actual learner's CLIP/SigLIP patch
rows, reconstructed-versus-official logit errors, and Qwen answers. The
reconstruction tolerances are 1e-4 for float32 CLIP and 0.05 for float16 SigLIP;
this is separate from the preserved full-patch parity threshold. The real
mechanistic path remains a two-scene preflight.

`learner_execution.json` records completion from the learner's own dropdowns,
executed code-cell count and source hashes. `execute_learner.py` reruns this
check in a fresh kernel and saves actual solution outputs and figures.

The older `smoke_test.json` and `reference_metrics.json` describe the preserved
synthetic and real-model support contracts. `verification_report.json` is
recomputed by `scripts/run_extension_verification_reports.py --section 12.1`;
it includes independent reference runs of the new training and typography
experiments. Its input manifest covers the readable lesson, source, numerical
tests, paired notebooks, measured figures and learner metrics. No prior report
was cosmetically rebound to cover these scientific changes.

## How this fixture was produced

The completed learner writes its own numerical histories, predictions and
figures; the independent release harness recomputes the reference experiments.
The legacy smoke fixtures originate in the course asset generator.

## Trusted implementation

The learner implements the methods locally. `pixel_tests.py` checks independent
formulas and gradients; `pixel_clip.py` and `typography.py` provide readable
reference bodies for the independently rerun evidence.

## Random seed

Renderer seeds are 100/200/300 for training/development/evaluation and 400 for
word overlays. Matched model seeds are 0, 1 and 2. See the data scope above.

## Allowed tolerances

Typical float32 fixtures use `rtol=1e-5`, `atol=1e-6`; the CNN/attention oracles
use float64 with stricter tolerances specified beside their assertions. Counts,
IDs, masks and byte-overlap checks are exact. The two explicit pretrained-logit
tolerances are documented above and do not weaken the original patch gate.

## When to regenerate

Rebuild and execute the learner after changing a taught computation. Recompute
the reference report after scientific or evidence-input changes. Preserve a
negative result; do not change a threshold just to make a run pass.
