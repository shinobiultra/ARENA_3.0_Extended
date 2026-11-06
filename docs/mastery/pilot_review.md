# CPU pilot and extension integration review

The October 19 continuation now supplies 42 guided cases and four blind notebooks.
Mini VLM actual outputs and the complete TinyStories migration have independent preservation reviews.
See [the dated record](continuation_2026-10-19.md); historical test counts below retain their original scope.
See [today's review and execution record](continuation_2026-10-16.md) for the new
migrations, corrected numerical methods and fresh runs. The detailed October 15
checkpoint below remains historical; its 622 passing tests do not certify today's tree.

## October 15 checkpoint

The CPU optimizer pilot is implemented and independently reviewed. Human learner
walkthrough remains **pending at the user's explicit choice**. The wider ARENA
extension is **not complete**: 86 syllabus destinations, 19 master-backed lessons,
14 marked written, and 55 canonical destinations without solution notebooks at
this audit. No existing course registry is promoted here.

## What learners can use

- Local app: `uv run --no-sync --offline streamlit run mastery_hub/app.py --browser.gatherUsageStats=false`.
  It shows actual float64 heavy-ball trajectories, loss/update plots, baseline overlays,
  synchronized step selection and tables. Challenge answers are committed with
  confidence/reasoning before reveal. Setting changes invalidate pending answers.
- The optimizer notebook keeps Hessian/update/simulation/stability methods visible
  and student-implementable, with immediate independent tests, help and solutions,
  a falsification exercise, controls and changed-regime prediction.
- Private records support resume, strict JSON export/import and seeded replay.
  Notebook records now use the same schema as the app, including all scientific
  inputs and provenance. No default export or model work occurs on Run All.
- Original resources stay canonical: [the complete source shelf](original_materials.md)
  and [checked link index](source_links.json). The user selected canonical originals
  plus provenance; no local copies of Stanford assignments or answer keys are added.
- Prediction learning across the extension follows the [integration design](designs/2026-10-15-integrated-prediction-learning.md).
  The [guided workbook](../../mastery_hub/units/extension_predictions/exercises.ipynb)
  covers all 19 current master-backed lessons with method-specific forecasts,
  frozen inputs, confidence/reasoning, observed controls, reflection and a changed-case
  forecast. All 38 generated exercise/reference notebooks link to their matching checkpoint.
  Its separate portable schema restores learner-reported worksheets, without
  executing experiments or importing them into the quadratic-only app.
- The full 11.1 calendar-geometry prototype is now a canonical master, preserving
  eleven exercises, twenty student functions, thirteen figures, all scientific
  code and negative findings. Seven forecast/reflection pairs and four play activities
  accompany its explicitly historical Mistral outputs; no fresh Mistral run is claimed.
- The full 11.2 representation-alignment lesson now has a canonical master, preserving
  six exercise groups, nine figures and three play activities. Eight forecast/reflection
  pairs emphasize that correlation, causal transport and correct-name recovery differ.
  Historical GPT-2 outputs retain checked source/output parity; no GPT-2 rerun is claimed.
- The additive blind 8.8 notebook freezes methods and exact inputs before commitment,
  then shows paired AUC plots, edge-effect heatmaps and controls from a local compiled
  transformer. Strict private receipts, separate reflection/transfer archives and
  replay preserve the learning loop. The blank notebook contains no target results.

- The blind 16.9 Data Shapley notebook compares two coalition-training normalizations
  after a committed forecast. Its visible methods have independent practice fixtures;
  paired credit bars and marginal-by-size plots accompany eleven controls, stored-result
  play and reflection/transfer. Failed measurements remain unscored portable records.

## Independent review findings and repairs

Sol and Astra reviewers checked component contracts and whole-loop behavior.
Their reviews found and prompted fixes for:

1. Persisted committed and revealed predictions could be overwritten through API
   transitions. Immutable commitments/history are now enforced across saves.
2. Portable envelopes accepted incomplete attempts. They now require revealed
   state; pending work belongs in the session format.
3. Importing an exposed result could leave a matching pending prediction unseen.
   Import, reload and equivalent dynamics now preserve practice classification.
4. Source navigation hid prerequisites and original foundation links. Both are now
   visible, with direct notebook downloads.
5. The notebook used an incompatible ad hoc prediction dictionary. The explicit
   schema-1 bridge, full JSON output and import/export/replay parity are implemented
   and independently rechecked without generating model evidence.
6. Existing 9.1/12.3 summaries still overstated null controls, harmless behavior
   preservation and five-task accuracy. Prose now acknowledges above-chance controls,
   measured capability costs and the 84.4% horizontal-task weakness.
7. 8.8 asked students for `logit_aucs` but used hidden `logit_benches` afterward.
   Its contract and exercise stub now explicitly require both.
8. Rebuilding the guided workbook could discard saved records and completed cells.
   Generation now preserves current-source execution byte-for-byte and refuses
   to overwrite learner edits, unreadable artifacts or stale executed notebooks.
9. The worksheet exporter could write a record larger than its reader accepted.
   Writers and readers now share the UTF-8 limit, including the export newline.
10. The worksheet asked learners to edit their method after freezing it. Preparation
    and sub-function tests now precede commitment; the following step runs the
    frozen experiment. Changes require a new commitment.

A separate bounded re-review closed all four lesson findings and all three workbook
findings. Temporary-file probes independently verified saved-work preservation and
writer/reader limits. These are engineering reviews, not learner acceptance.

Claude Code 2.1.299 was available. Both explicitly requested model IDs,
`claude-sonnet-5-5` and `claude-opus-5-5`, failed before inference because OAuth
was expired and refresh failed. No Claude review or verified model availability
is claimed. [Availability receipt](evidence/claude-review-availability.json).

## Verification boundaries

The [final integrated CPU checks](evidence/second-continuation-final-checks.txt)
pass **622 tests in 24.08 seconds**, with fourteen upstream Torch/Python 3.14
warnings. [Repository collection](evidence/second-continuation-collection.txt)
finds **1,522 tests without errors**. All four notebook generators' read-only checks,
37 scoped Python compilations, merged configuration, scoped whitespace and original
ARENA preservation pass. [All 38 generated notebooks](evidence/second-continuation-notebook-navigation.json)
have valid schemas, matching workbook links and no saved error outputs. The
[final source manifest](evidence/second_continuation_source_manifest.json) binds this
checkpoint; the earlier 620-test log remains preserved before the last circuit fix.

[Continued focused check output](evidence/continued-focused-checks.txt) records the
earlier combined CPU suite: **520 passed in 19.85 seconds**. Its three notebook generators' read-only
checks, Python compilation, merged config and original preservation checks pass.
The [continued source manifest](evidence/continued_source_manifest.json) identifies
this checkpoint. The [earlier manifest](evidence/verified_source_manifest.json) and
[295-test log](evidence/focused-checks.txt) remain historical records. The optimizer
solutions notebook was genuinely executed on CPU: **22 code cells, 4 PNG figures**.
Completed student answers were also executed in a separate fresh kernel.
The generator's `--check` is read-only and detects source drift.

The guided-workbook tests execute all 57 filled commitment/observation/reflection
cells with clearly synthetic fixture inputs and validate 57 JSON snapshots. They
do not execute its 19 scientific experiments. Transfers remain `forecast_only`
until separately run. Existing historical outputs stay guided practice.

[Earlier before/after hashes](evidence/reference_execution_preservation.json) verify all
17 regenerated reference notebooks retain their **778 code cells, execution counts
and saved outputs**. That earlier rebuild added navigation without scientific execution.

Original-ARENA byte/config/requirements preservation and merged config checks pass.
The path allowlist was extended narrowly for named extension master directories,
mastery files and extension build plumbing; chapter 0–4 master paths remain protected.

The five legacy collection failures are resolved with 114 tests across their
affected files. Subsequent capstone and Shapley repairs were freshly executed on
CPU; their original evidence is archived. Forty additional synthetic sparse fixture
errors were also resolved. A broader related diagnostic still fails at the first
missing legacy lock; a separate inventory records the full missing-file and stale-report
debt. See [the migration record](legacy_test_migration.md); no full
repository release or GPU success is claimed.

Browser author-smoke covered actual commit → reload → reveal, practice counts,
controls, baseline overlay, a divergent step stopping at the cap, and data tables.
[Desktop capture](evidence/browser-desktop.jpg). This is author UI evidence, not
human learning acceptance or new scientific measurement.

The earlier preview displayed the guided workbook and circuit downloads with an
18-case count; [historical navigation capture](evidence/browser-continued.png).
That circuit helper passed a repeated, explicitly seen-case CPU regression,
including strict result validation, reflection/export round-trip and replay with
model calls disabled. Its original blind engineering forecast remains preserved.
Results are stored separately from the blank learner notebook to avoid prefilling.

The current preview exposes nineteen guided cases and both blind downloads;
[current navigation capture](evidence/browser-second-continuation.png). Independent
integration review checked every workbook backlink and repaired README claims to
identify the 11.1/11.2 model studies as historical.

The final Shapley helper passed one bounded CPU target run after a recorded forecast
and independent pre-run reviews: eleven controls passed, including duplicate symmetry,
label relocation, zero learning rate and efficiency. Its exact frozen helper and blank
notebook are archived in the [validation manifest](evidence/blind_shapley_validation/manifest.json).
The final circuit helper passed a separately authorized **seen-case** CPU regression
in 5.90 seconds, matching earlier measurements exactly. An earlier v2 configuration-freeze
failure is retained; it occurred before commitment or inference. Typed finite NumPy
scalar handling, a real-configuration test with forwards forbidden, and independent
review preceded the successful attempt. The
[final circuit receipt manifest](evidence/blind_circuit_validation/known-case-v2-repaired-helper-regression-manifest.json)
binds the current helper, v2 identity, private export/reflection and model-free replay.
Target outcomes stay outside both blank learner notebooks.

## Evidence and limitations

![Fresh CPU analytic-toy signature](evidence/optimizer-signature.png)

The signature compares GD/heavy-ball, noise, an unstable step, zero learning rate
and the fixed optimum. Zero losses are omitted honestly on log axes; no curve is
shifted into apparent convergence. Finite completion is not asymptotic convergence.

Existing refusal, Mini VLM and circuit-benchmark prose was rebuilt using saved
outputs, preserving scientific code and historical execution. That is not a fresh
model rerun. Remaining scientific limits include unpinned model revisions, limited
seeds, missing image-only Mini VLM evaluation, duplicate Qwen layer 27, and separately
sampled GPT-2 evaluation without an enforced disjoint-identity audit.

Original source materials and original ARENA remain intact. No dependency pins,
model weights, GPU/cluster jobs, paid services, calendar events, commits or pushes
were added by this increment. Private preview progress stays under `/tmp`, outside Git.
Two exact `.gitignore` exceptions keep the additive PR #2 planning documents
eligible for version control; the existing `*master*.md` rule otherwise hid them.
