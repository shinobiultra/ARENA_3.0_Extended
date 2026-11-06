# Course-page identity and runtime recognition

## Plan recorded before edits

Current learner failure: the surface audit rejects reviewed lesson headings whose punctuation differs from the registry, and explicit runtime prose such as “runs on a CPU” or “a minute on a GPU.”

Target claim: recognize those precise forms without accepting an unrelated method title, a bare hardware mention, or missing runtime guidance. This is static course-surface recognition, not scientific or browser acceptance.

Signature result, student functions, controls and play cells: unchanged. No lesson files will be edited. Validator controls cover wrong section numbers, wrong methods, unreviewed title expansions, bare GPU mentions, durations without hardware, and missing runtime text.

Files to modify: `scripts/audit_course_surface.py`; new focused `tests/test_course_page_recognition.py`; this note and scoped `docs/mastery/evidence/2026-10-16-page-surface-recognition/`.

Tests to run: new focused fixtures, existing course-surface fixture suite, and a fresh full audit with all remaining blockers retained.

Before editing, exact validator/test bytes and the current page failure inventory were captured in `before/` and `before.json`. The initial inventory has 20 page failures. Sections 8.2 and 8.5 have no CPU/GPU/CUDA occurrence and remain actual missing-runtime blockers. The other 18 may be resolved only through the reviewed grammar and title identities.

## Implemented boundary

The title must now be a complete numbered Markdown H1, rather than a substring of prose or a longer wrong title. Normalization preserves title words and ignores only whitespace, `&` versus `and`, and the Oxford comma before `and`. Two historical headings have exact reviewed aliases, each keyed by both section number and configured title:

- 8.3, `AtP* from Scratch` → `AtP*: Fixing Attribution Patching`.
- 15.2, `LoRA vs Full Fine-Tuning` → `LoRA vs Full Finetuning on a Real Transformer`.

The existing explicit runtime interfaces remain supported. New prose recognition accepts execution on `a CPU` / `one CPU`, and a bounded numeral or small number phrase followed by
seconds/minutes/hours, optional `per run`, and `on a/one/single CPU/GPU`. It therefore recognizes the published “a minute on a GPU,” “10-40 seconds per run on a GPU,” and “7
minutes on one GPU.” A bare GPU mention, tensor placement, elapsed time without hardware, wrong method heading or unknown title suffix still fails. Existing `CUDA`, `GPU Contract`,
`run_gpu_test` and `REQUIRES_GPU` recognition is unchanged; this lane does not strengthen those older interface checks into runtime measurements.

No other validator implementation changed. The existing test file remains byte-identical to its captured preimage. All lesson notebooks, source masters, pages, outputs, reports, media, signature results, student implementations, controls and plays remain untouched by this lane.

## Evidence and verification

- Exact pre-edit validator/test source and all 20 initially rejected pages are retained. The compressed page snapshot is a JSON text archive; each page hash is checked by a frozen-inventory regression.
- Before repair: the new fixtures exposed 17 failures and passed 19 checks in 0.07 seconds.
- Final focused suite: **37 passed in 0.15 seconds**.
- Existing and new tests combined, before the final additional archive-inventory fixture: **79 passed, 1 failed in 28.92 seconds**. The sole failure is `test_current_course_surface_passes`, with 345 real remaining shared-tree blockers. Full transcript retained; no assertion weakened or removed.
- Against the frozen baseline, page recognition removes exactly 18 blockers and adds none. Sections 8.2 and 8.5 continue to fail because their captured source lacks runtime guidance. The full audit falls from 363 to 345 blockers at this worker checkpoint; all non-page blockers remain exact.
- Scoped `git diff --check` passes. No model, training, download, notebook execution, GPU, cluster, staging, commit or push occurred.

The parent separately owns any actual guidance additions and later shared-tree rebuilding. `before.json`, `after.json` and `manifest.json` are named worker checkpoints, not
continually updated claims about concurrent changes. Independent parent review passed all 37 focused checks in 0.14 seconds with no findings; `root_independent_review.json` retains
the exact source hashes and review limits. This static validator result does not establish scientific validity, live rendering or learner acceptance.
