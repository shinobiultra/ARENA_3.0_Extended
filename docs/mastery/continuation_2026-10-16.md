# ARENA continuation — October 16, 2026

**Stopped for today at the user's request.** The [October 16 closeout](closeout_2026-10-16.md)
is the authoritative end-of-day state. Progress statements below describe earlier
checkpoints; they do not authorize more work, experiments or publication today.

The learner asked to continue finishing ARENA. The original-materials decision is
unchanged: canonical upstream sources with provenance, original ARENA preserved.
The human optimizer walkthrough remains deferred. Independent reviews support
engineering decisions; they do not close that acceptance gate.

## Integrated teaching

The current workbook contains 33 guided cases, up from yesterday's 19. Fourteen full
existing lessons now have canonical masters and generated learner/reference
notebooks: DAS (11.3), concept erasure (11.4), CCS (11.5), denoising-time patching
(13.2), LoRA versus full fine-tuning (15.2), QLoRA (15.3), exact Shapley (16.1),
KernelSHAP/PartitionSHAP (16.2), SAGE (16.4), checkpoint archaeology (17.1),
LoRA/DoRA (15.1), influence/TracIn (16.10), MoE (5.2), and Mamba-2/SSD (5.4).
Their migration notes under `migrations/` document preserved source, outputs,
student methods, independent sub-function tests, controls, play cells and limits.
Renumbering follows method identity, with historical source paths retained.

Visible results include causal-transport and erasure plots, CCS controls,
denoising image trajectories, adapter spectra, quantization-error plots, Shapley
credit/conservation panels, sampling-error curves, SAGE credit bars, and checkpoint
mechanism curves. Guided cases freeze the scientific denominator, solver, seeds,
selection and evidence role. Public historical results remain seen practice.

The new blind SAGE notebook joins the circuit and Data Shapley variants. It teaches
the restricted predictor, population risk game, exact attribution and marginal
contributions before a committed primary prediction and a separately committed
covariance transfer. All 12 controls passed in each engineering case; primary and
transfer together took 2.543 seconds. Exact target values, figures and completed
answers remain in private temporary receipts. The shipped 30-cell notebook is
blank. [Validation evidence](evidence/blind_sage_20261016/) records the forecast,
hashes and control counts without publishing target answers. Engineering success
does not establish unseen learner mastery.

## Repairs and evidence boundaries

- The five earlier legacy collection errors remain fixed. Today's full-suite
  diagnostic found another collection side effect: importing the two Data Shapley
  test modules changed global dtype/thread state. The tests now restore caller
  state around imports and individual tests. Existing numerical assertions and
  tolerances are unchanged.
- Historical readability proofs now resolve exactly two retired 8.3 publications
  from a sealed, hash-pinned archive in the test layer. Present-file drift and
  unknown paths fail closed. The production helper and original eleven-entry
  manifest are byte-identical; no historical proof transfers to current 8.4.
- Those two repairs passed 58 independent focused tests. The earlier full root CPU
  diagnostic is retained as **1,856 passed / 25 failed**, including concurrent
  source changes. It is not a final-tree result. [Raw diagnostic](evidence/2026-10-16-root-cpu-diagnostic/).
- KernelSHAP's partial KKT system was genuinely rank deficient. Explicit CPU
  float64 `gelsd` now gives the stable minimum-norm solution, with the result
  returned to the caller's device. Full-table direct solves remain unchanged.
  Independent rank-deficient oracles passed, and the corrected finite-game
  notebook ran all 13 cells on CPU. Both completed runs and the earlier local-socket
  startup failure are retained separately. No transfer game was executed.
- QLoRA's imported release wrapper now requires an explicit learner bundle and
  rejects reference substitution. Its notebook's scientific code and historical
  outputs are preserved. The historical BF16 error `.03125 > .01` remains a failed
  gate; plausible precision explanations are not a resolved cause.
- Output preservation now carries exact widget state as well as code outputs and
  counts across prose/navigation rebuilds. Historical state was recovered for
  13.2 and 15.2 from verified archives. Missing widget state in other original
  notebooks is disclosed; it is not fabricated.
- LoRA/DoRA's 13 added checks and play cells executed in a bounded CPU subset:
  4.415 seconds and 440 tiny optimizer steps. Actual outputs and counts accompany
  per-cell origins; all 17 historical code cells retain their original outputs.
  This is explicitly a mixed-origin notebook, not a fresh complete execution.
  Independent review verified the exact subset, historical preservation and
  immutable proposal identity. The subsequent root integration check passed
  115 tests, including both newly integrated lessons and the 31-case workbook.
  [Integration log](evidence/2026-10-16-integration/repaired-31case-focused.txt).
- Seven roadmap evidence rows now refer to their actual current method surfaces.
  All 83 existing `UNIMPLEMENTED` statuses remain. Old edge counts, models and
  acceptance claims were not transferred to different lessons by numeric ID.

## Pilot verification in progress

The corrected 7.1 lens metrics and 8.4 offset handling have independent semantic
tests. A frozen local GPU protocol uses only existing cached assets and sequential
execution, with source hashes, actual CUDA checks, memory bounds and per-cell
failure preservation. The fresh 7.1 run passed all 40 cells, including 300 tuned-lens
training steps. Its six scientific PNGs match the previous outputs exactly.
The first 8.4 admission stopped before loading a model because another GPU process
was active; that receipt remains. The process ended naturally and the unchanged
8.4 protocol then passed all 41 cells in 760.88 seconds. Both notebooks now contain
actual fresh outputs and passed independent publication review: 26 focused tests
and 15 selected structural checks. The 16 actual figures retain controls and
negative results. [Full execution record](evidence/pilot_fresh_execution_20261016/REPORT.md).
No cluster job or model download is involved.

The 12.1 and 9.1 prose repairs distinguish confounded pretrained evidence from
controlled experiments and correctly explain centered versus uncentered sign
invariance. All original code-cell objects and saved outputs were hash-preserved.

## Remaining work

The syllabus has 86 sections. Master coverage is not whole-course completion.
The [method-level inventory](migrations/method_surface_inventory_2026-10-16.md)
separates migration candidates from 25 missing or insufficient dedicated learner
lessons in its frozen snapshot. Larger briefs may exceed what an existing lesson
demonstrates. Further existing architecture lessons are being migrated.
Influence/TracIn now has an explicitly approximate historical-figure route because
its exact deletion arrays are unavailable; its budget-16 transfer was already
exposed.
Whole-tree content, provenance, hygiene and readability validators still expose
real debt; none is waived to produce a green release.

Mini VLM's unique-layer Qwen sweep and literal image-only controls passed both
independent pre-run reviews. The first frozen GPU attempt stopped before training
after 44.59 seconds: the historical notebook's two-argument PCA plotting call was
incompatible with the current three-argument helper. Ten cells succeeded and the
eleventh failed. The complete failure and partial outputs are retained; none was
published as a completed lesson. A compatibility repair is under review.

The tiny QLoRA arithmetic diagnostic has no scientific result: both admissions
stopped before its worker ran. The first could not access the driver in the
sandbox; the second saw utilization above its strict idle threshold. Its two
attempts remain separate from the historical BF16 failure.

The human walkthrough and later Alchemy empirical protocol remain open.
Claude review remains unavailable after the previously recorded OAuth failure;
independent available agents performed the reviews. No course acceptance, commit,
push, merge, recurring schedule or new paid service call is claimed.
