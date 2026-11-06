# Prediction learning throughout the ARENA extension

User requirement, 2026-10-15: preserve original materials as originals, extend their
lessons through the ARENA extension, and let the extension exercises support
prediction-based learning. Original ARENA chapters 0–4 remain byte-preserved.
The user chose **canonical originals plus provenance** and will try the learner
walkthrough later. Independent reviews continue meanwhile.

## Product contract

A learner reads the original, states a causal model, commits a specific prediction
with confidence and reasoning, changes a controlled experiment, observes the
result and controls, explains the mismatch, then predicts a changed scientific
condition. Keep the scientific method student-visible. Shared code handles
identity, records and display, never hides the taught method.

The complete [original-materials shelf](../original_materials.md) points at all
38 links from the checked official homepage. The [machine-readable index](../source_links.json)
records its date and SHA256. Sources remain upstream, intact. Publication,
companion implementation, human acceptance and scientific evidence have separate
statuses. The seven tentative Alchemy topics are thematic routes, not permission
to invent assignments. In particular, current ARENA experiments do not reproduce
the upstream switch to DNA from assignment 6.

## Two honest learning paths

1. **Guided prediction practice**, implemented as an additive workbook and links
   from current extension lessons. Each case has a concrete forecast, frozen-input
   checklist, confidence/reasoning, observed controls, discrepancy explanation and
   changed-condition transfer. Existing standard notebooks remain reference-rich.
   Prior exposure is recorded as seen or unknown; the workbook awards no unseen
   score or mastery badge. Its separate portable worksheet schema must be named
   honestly, not passed to the quadratic-only schema-1 reader.
2. **Blind local challenge variants**, still proposed. Generate an additive
   prediction notebook/page that omits target results and answer-bearing prose
   until commitment. Expected-output dropdowns and prefilled images are exposures;
   collapsing them does not hide their answers. Opening the reference lesson or
   an equivalent result makes that case practice. Local self-report remains part
   of the exposure boundary; this is not exam security.

The workbook is a useful learning artifact while blind variants are engineered.
It does not certify a completed challenge platform across all 86 destinations.

## Concrete curriculum integration

| Source theme | Existing extension connections | Boundary |
| --- | --- | --- |
| Basics/scaling | 0.6, 1.6, 10.1, 17.1/17.4 | Budgets, nulls, selection and reproducibility. Most destinations remain planned rewrites. |
| Optimization invariants | 6.1/6.4, 7.1/7.3/7.4, 12.1/12.3, 15.1–15.3, 16.9 | Fixed settings and equal-budget retuning are different questions. |
| Loss landscapes | 6.1, 7.4, 15.2/15.4, 16.9, 17.1/17.3 | A 2-D quadratic is an analytic entry point, not transformer evidence. |
| Architecture stability | 5.1/5.3–5.7, 12.3, 14.2/14.3/14.7, 17.4 | Depth, recurrence, normalization, length and memory. |
| Capacity | 5.2, 6.1–6.5, 7.3–7.5, 12.1/12.3, 15.1–15.5, 18.1/18.2 | Account for parameters, tokens and compute. |
| Data efficiency/generalization | 0.6, 5.5, 7.2–7.5, 9.1, 10.1, 12.1/12.3, 16.9–16.11, 17.1 | Thematic analogues; no claim to reproduce the upstream DNA organism. |
| Exotic/unspecified | No invented destination | Keep pending upstream. |

Current source-backed questions are authored individually for all 17 existing
masters. The other syllabus entries remain unmapped/unimplemented until their
lessons exist and meet the learner standard. Circuits, geometry and other
interpretability methods use the prediction loop directly; they need not be
forced under an unrelated Alchemy topic.

## Blind-variant engineering design (proposed)

- A versioned checkpoint manifest names source/code/config identity, prediction
  type, controls, result and interpretation anchors, source links and transfer.
- Separate learner exposure (seen/unknown/learner-declared unexposed), scientific
  data role (TRAIN/monitored evaluation/historical evaluation/untouched TEST/unknown),
  evidence class (analytic toy/recorded real/live real), and execution kind
  (replay/fresh inference/training). Unknown provenance stays unknown.
- Keep the existing quadratic schema-1 reader compatible. General experiment
  receipts need their own versioned envelope and notebook adapter; no model load
  or optimizer runs inside an `observe`/reveal helper.
- Build variants through extension-only preprocessing and sidecar anchors. The
  original converter rejects arbitrary custom cell tags; leave that converter
  and original chapter files intact. Fail on stale anchors, hashes or unclassified
  answer spans. Preserve immediate sub-function tests and nonleaking examples.
- Binary forecasts may use Brier. Numeric/ranking tasks require their own declared
  error measure; causal explanations remain ungraded without an explicit rubric.
- A live changed-case run starts only when the learner invokes the existing
  method with a displayed configuration and budget. No new training is launched
  by notebook generation, widget changes, index loading or recorded-result reveal.

## Acceptance and open work

Guided-workbook checks: exactly the current 17 master IDs; existing reference paths;
valid deterministic notebook; concrete per-case inputs/controls/transfer; immutable
commitment and stale-config rejection; truthful observation provenance; explicit
local export and replay distinguished from fresh measurement; no original edits.

Blind variants additionally require source-span and semantic leak review, matched
notebook/app record round-trip, actual changed-case controls, reviewed scoring and
learner walkthrough. Saved historical outputs do not establish a new independent
TEST result. For example, 12.1's `test_images` are evaluated repeatedly during
training and are **monitored evaluation**, whatever their historical variable name.

Full completion still requires the remaining master-file rewrites and migration
of legacy test/report APIs. Do not relabel missing lessons as implemented or
promote an agent review to human acceptance.
