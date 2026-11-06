# Mamba-2 / SSD canonical migration — 2026-10-16

## Plan before editing

**Current learner failure:** the complete 5.7 lesson lives under `part7_mamba2_ssd`, while the adopted syllabus expects 5.4 under `part4_mamba2_ssd`. It uses a separate
lesson/builder pair rather than the canonical master converter. Saved training, checkpoint and CUDA results need an explicit historical boundary; forecasts and reflections are
absent. Some student method groups lack immediate independent feedback.

**Target section claim:** how can recurrent, quadratic and chunked SSD compute the same causal operator, and when does recurrent state carry the delayed WRITE bit beyond a local convolution's receptive field?

**Signature result:** preserve the exact existing four-panel parity, delayed-recall/control, educational CPU timing and decay-storage figure, every saved output and all
checkpoint/CUDA results. Keep the negative CPU timing and weaker doubled-length generalization visible. Do not launch training, checkpoint inference, downloads or CUDA. Add only a
tiny no-training analytic play of the existing methods.

**Student methods:** retain all original student functions and every class method, including constructors, block/cache internals, training/orchestration, scaling/gradient routines,
stacked LM and controls, full released mixer/LM and strict tensor mapping/parity. All original supplied definitions and strict CUDA hooks remain visible. Copyable generated answers
must complete every method without relying on hidden implementations.

**Controls/baselines:** retain literal recurrence, official Listing 1, future-token perturbation, nondivisible padding and nonzero initial state, gradients, state carry ablation,
local convolution, shuffled labels, doubled length, full/cache parity, parameter-matched Mamba-1/attention and IEEE/TF32 precision distinction. Do not change seeds, 100/180-step
recipes, precision, tolerance or pinned model revision.

**Play and prediction:** retain the original chunk/state play. Add forecast/reflection pairs before each method/result group, recording confidence, reasoning and reference exposure. A small CPU-only play changes decay, carry and chunk size using the taught SSD functions; it is exact operator practice, not fresh model evidence. Propose the guided case only in scoped evidence for root to integrate.

**Files:** create `infrastructure/chapters/chapter5_modern_architectures/master_5_4.py` and generated notebook; canonical `part4_mamba2_ssd` notebooks/module/tests/utils/media;
`04_[5.4]_Mamba-2_and_State-Space_Duality.{md,py}`; `tests/mastery/test_mamba2_master_migration.py`; this migration note and scoped evidence. Preserve every old physical file and
original ARENA byte. Shared configuration/checkpoint registry are root-owned.

**Tests:** capture and seal exact original files before edits; verify all 21 historical code cells' ASTs/nonblank source lines, counts, outputs and widget state, allowing named
setup-import/identity and signature-output-path routing only. Validate all old assets and inline PNG bytes. Independently complete every student stub from its generated answer. Run
bounded CPU tensor, cache, tiny optimizer and orchestration fixtures; exclude full recall studies, official imports, checkpoint loads and CUDA. Execute only isolated new CPU
play/test cells if receipted; otherwise leave new cells explicitly unexecuted. Build without `--execute`. Independent review is required; fresh full execution and human learner
acceptance remain distinct gates.

## Authority and decision record

Root authorized this existing-lesson migration and small CPU checks. No new method/model/data study, download or GPU execution is authorized. Historical report `accepted` fields
retain their original scope and cannot certify this canonical migration or learner acceptance. The pinned checkpoint identity is preserved, not refreshed. No unresolved method
choice is needed to begin; any newly discovered scientific delta will be reported before adoption.

## Implemented increment and evidence

The new canonical master builds 102 notebook cells, 27 code cells, 11 exercise
groups and approximately 12.5k words of reference-rich prose. All 21 originally
assigned functions/classes have complete student stubs and their own generated
answers; every class constructor/helper/cache method survives. All 49 original
top-level function/class ASTs are identical. The 16 forecast/reflection pairs
include the signature, original play, released comparison and CUDA arithmetic.
New independent checks cover recall orchestration, exact scaling bytes, metric
assembly, synthetic released mapping, gate order, a hand-worked two-channel
released mixer, small LM orientation and fake full-parity result assembly.

`docs/mastery/evidence/mamba2_master_migration/REPORT.md` is the current evidence
summary. The sealed pre-edit archive, `before.json` and `output_parity.json` prove
physical/source/output preservation. All 15 historical report-input hashes matched
at capture. All 21 old code-cell outputs/counts survive, with two named route
adaptations only; six added cells now carry genuine fresh CPU subset outputs. The old six widget views lack
saved backing state; this absence remains explicit, without invented state. Both
canonical media files are the same historical PNG bytes (asset and inline copy).

Visual review found the timing plot's actual ordering differs from an old
hypothetical interpretation: chunked beats recurrent Python throughout the saved
panel, while the full-matrix path wins shortest lengths. A precise adjacent note
clarifies that without replacing the old prose, figure or method. The preserved
limitations include weak doubled-length recall (0.753 vs 1.000), synthetic task
scope, finite precision, two-prompt checkpoint coverage and no official throughput
claim. CUDA results remain historical, not refreshed parity evidence.

The bounded section selector passes 24 tests (three studies/official imports
excluded); the strict migration suite passes 33 tests after subset promotion. Builds used no `--execute`.
These checks cover exact archives, original source/prose/outputs, all own-answer
completions, tiny independent CPU oracles, incorrect-answer rejection and inert
imports. Small optimizer fixtures are method checks, not a new recall study.
Early failed builds/fixture checks are retained in evidence logs. A subsequent
ordinary nonexecuting build must preserve the same outputs and execution boundary.

The exact six new test/play cells completed in a fresh one-thread CPU kernel in
2.410160565 seconds, using only the 31 minimum visible definitions and 12 check
bodies. The prior optimizer fixture was excluded. Guards blocked actual model/data
imports, CUDA, optimizer/backward calls. Authentic outputs/counts and the analytic
PNG are published with exact origin receipts; every old full-cell object remains
unchanged. The notebook is explicitly mixed historical/fresh-subset, not freshly
executed as a whole. Changed decay/carry/chunk conditions remain unit-fixture checks.
`proposed_case.json` is ready for root-owned guided integration; no shared config
or checkpoint registry changed. Original physical `part7_mamba2_ssd` and original
ARENA remain untouched. Independent review and a human learner walkthrough remain
open; full fresh canonical execution would require separately authorized training,
pinned checkpoint/official imports and the strict CUDA cell. No study was launched.


Root subsequently authorized this exact bounded CPU subset, after independent
pre-run review, to populate the six added cells without weakening the execution
coverage gate. The 11-file pre-subset archive is sealed. Raw candidate, partials,
per-cell timings/origins and publication receipt are retained. The runtime was
Python 3.14.2 / Torch 2.12.1+cu132, CPU thread/inter-op thread one, CUDA invisible
and uninitialized; kernel 1926912 exited. The full 21 historical cell objects,
including source, IDs, metadata, counts and outputs, match the snapshot exactly.
No historical scientific runtime statement, training, official Mamba import or
checkpoint load was executed in the subset. The fresh plot is a three-token
operator exercise only, not updated language-model or CUDA parity evidence.


Independent final review passed with no P1/P2 findings: 33 focused tests in 4.65
seconds, all 21 full historical cell objects and 27 source strings unchanged,
six genuine raw output/count tuples, seven publication hashes and PNG identity.
The proposal's exact target answers were removed from pre-commit fields after
root's integration feedback; explicit inputs and exposed guided-practice status
remain, and the reviewer checked this correction. See `independent_review.json`.
This engineering review does not close human learner acceptance or produce a
fresh full-model/CUDA execution. No staging, commit, push or original ARENA edit.
