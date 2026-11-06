# Sources refreshed on October 21, 2026 (Asia/Tokyo)

The [official homepage](https://deep-learning-alchemy.github.io/) was checked again
with HTTP 200. The canonical shelf now contains all 46 homepage links: every one
of the prior 38 links plus six optimization readings and lecture 6 PDF/PPTX links.
The homepage SHA256 is
`ca0f2cbcfa766319dd3e45bc144a53aba2f885c8fdb0b7f8a87c1013bfeb4c50`.
Only the homepage was fetched for this refresh; individual linked files were not
rechecked or archived. The earlier assignment checks below retain their original
dates and hashes.

Assignment 4 appears on the schedule without linked code or PDF. Its source
status remains pending. All seven companion statuses are unchanged; the optimizer
walkthrough remains deferred. The app labels quiz answers as an explicit exposure
choice while keeping every original link accessible. Source material stays
canonical and upstream. See [the source shelf](original_materials.md) and
[refresh evidence](evidence/source_shelf_20261021/).

## Historical October 15 source inventory


- [Deep Learning Alchemy official homepage](https://deep-learning-alchemy.github.io/):
  Stanford CS312, Fall 2026. The schedule links assignment code/PDF for the first
  three topics. Architecture, generalization and the unspecified exotic unit have
  no published assignment link at this snapshot. Their topics remain tentative.
- [PR #2](https://github.com/shinobiultra/ARENA_3.0_Extended/pull/2),
  `c8dec641421ba2fd833a693890e66cc82371a475`: the user-requested additive design
  and implementation plan. Its three documents were copied unchanged before
  implementing the CPU pilot. The local master-file migration is newer than
  the PR's planning parent and is preserved.
- Original ARENA: repository baseline
  `f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0`. Existing files and links remain
  canonical; this companion does not replace the original exercises.

The seven short topic labels in `mastery_hub/catalog.json` identify navigation
destinations. Companion code and explanations are original work. No Stanford
handouts, quiz questions, answer keys, images, code or assignments are vendored.
No upstream code checkout is used, so no invented code revision is attributed.
The app stays usable offline; following an external source link requires network.

`source_status` describes the checked source, independently of
`companion_status`. The optimizer pilot remains below `learner_ready` until an
actual learner completes the required walkthrough. This catalog is not a
replacement for the unfinished 86-section extension syllabus.

## Lessons checked against published assignment originals

[Assignment 1](https://deep-learning-alchemy.github.io/assets/assignments/a1_basics.pdf)
connects robust experiment design, hyperparameter interactions, budget matching,
scaling forecasts and training diagnosis. Its synthesis/report-and-prediction pattern
informs our controls, configuration records and changed-condition questions. Our
companion does not claim to reproduce its language-model run programme. Its prose
and table describe the depth/width ladder differently; a future empirical protocol
must resolve the pinned code instead of silently choosing one description.

[Assignment 2](https://deep-learning-alchemy.github.io/assets/assignments/a2_hyperparameter_scaling.pdf)
connects learning-rate/weight-decay coupling, noisy-quadratic reasoning, batch-size
transfer and parameterization diagnostics. These motivate the narrow analytic pilot
and transfer questions, not an equivalence between its heavy-ball implementation
and the source's complete optimizer experiments. The original includes example
answers: preserve them upstream, and treat reading an equivalent answer as exposure.

[Assignment 3](https://deep-learning-alchemy.github.io/assets/assignments/a3_optimization2.pdf)
studies mode connectivity, river/valley dynamics, stability and Hessian structure
using controlled comparisons and tuning. Its lessons inform later landscape and
local-approximation companions; those empirical companions are not implemented by
the current quadratic pilot. The web PDF parser failed twice, but direct HTTP returned
200 with a valid PDF, and read-only Poppler extraction verified the problem headings.

All three original PDFs returned HTTP 200 with PDF signatures. Their byte counts
and SHA256s, plus the exact three assignment release commits, are preserved in
[source_artifact_checks.json](source_artifact_checks.json). Documents were read in
memory rather than archived or rewritten locally. No source training launches ran.
