# Global gate triage and report diagnostics, 2026-10-16

This inventory explains the ten global gate failures within the preserved
root CPU diagnostic: **1,856 passed, 25 failed**. It does not replace that log
or claim that all findings remain current while other owners edit the tree.
The other 15 failures belonged to dtype isolation, fresh pilot execution,
directory allowlists, and historical readability proofs, with separate owners.

The read-only snapshot is
`../evidence/2026-10-16-report-diagnostics/global-triage-before.json`, captured at
`2026-10-16T03:47:50.826708+00:00`. Eight validator/config hashes were unchanged
across its capture; lesson files were not frozen. No models, downloads, cluster
jobs, or scientific report regeneration were used. Original ARENA was untouched.

## Counts and denominators

| Gate | Captured blockers | Interpretation |
| --- | ---: | --- |
| Course surface | 372 | 52 absent configured directories, 51 absent pages, 247 absent support files, 20 shape recognizer rejections, 2 chapter setup files |
| Extension verification assets | 498 paths | 364 paths under the 52 absent directories; 134 paths under 20 existing mastered directories |
| Hard exercise ladder | 11 rows | First missing provenance path per row; later row checks have not yet run |
| Report evidence | 143 absent paths + 54 freshness failures | 88 declarations: 72 missing pairs, 8 stale pairs, 8 metadata-contract passes |
| Artifact hygiene | 683 | 671 untracked required payloads, 6 missing archive contracts, 5 oversized files, 1 raw data artifact |
| Readability | 1,178 lines | 1,103 Markdown lines and 75 Python lines; zero notebook image blockers |

The five failing tests in `tests/test_extension_verification_assets.py` are
overlapping manifestations of missing assets, schema/provenance, ladder
coverage, and callable experiment hooks. They are not five independent sets
of absent lessons. The other five global failures each cover one audit above.
The report denominator includes documentation evidence records; it must not be
treated as 88 distinct configured lessons.

The earlier method inventory counted 55 absent canonical destinations; three
destinations (16.1, 16.2, 16.4) appeared before this snapshot. The earlier
classification remains frozen in `method_surface_inventory_2026-10-16.md` and
`../evidence/2026-10-16-method-surface-inventory.json`: 29 method-related
migration candidates, 3 insufficient foundations, 1 partial calendar mapping,
and 22 destinations without a dedicated source. Thus an absent destination
does not itself prove a missing scientific lesson, and an old notebook's
existence does not establish method coverage or acceptance.

## Content and provenance work still required

The 498 missing verification assets include README/provenance text, locks,
report schemas, reports, and smoke/reference fixtures. Nineteen existing
mastered destinations each lacked all seven required paths: 7.1–7.5,
8.1–8.5, 8.8, 9.1, 10.1, 12.1, 12.3, 16.2, 16.4, 16.6, and 16.9.
Section 15.3 lacked only its report. This combines missing current evidence
with migration from the historical report API to mastered lessons and
execution records. Repair requires an explicit, reviewed provenance route
for each method. Copying historical accepted reports into changed lessons,
or creating generic success files, would erase real failures.

The five test entry points are:

- `test_every_extension_notebook_has_verification_assets`;
- `test_artifact_locks_and_report_schemas_have_required_fields`;
- `test_expected_output_fixture_readmes_record_provenance_and_tolerances`;
- `test_hard_exercise_ladder_registry_tracks_all_hard_sections`;
- `test_extension_full_experiment_hooks_are_callable_not_placeholders`.

The last four originally stopped at absent 5.2 lock or solution paths. Once
those paths exist, later failures can become visible; the first exception is
not a complete inventory of those contracts.

The hard-ladder failures are absent provenance sources for 7.1–7.5, 8.1, 8.2,
10.1, 12.3, historical 8.3 ACDC, and historical 16.7 In-Run Data Shapley.
Most expect `expected_outputs/README.md`; 12.3 expects
`expected_outputs/verification_ladder.md`. The two retired method identities
need scientifically correct migration to current 8.4 and 16.9, retaining the
old rows and proof scope. `row_blockers` returns after the first absent path,
so eleven is a lower bound. Actual provenance must identify the trusted
implementation, inputs, seeds, tolerances, and regeneration policy. A CPU
smoke run cannot replace claimed CUDA reproduction evidence.

The 54 report freshness failures concern eight existing report pairs:
11.1–11.4 (9 each), 11.5 (3), 13.2 (9), 15.2 (3), and 17.1 (3).
They bind historical results to notebook, Python, or page bytes that changed
during migration. Preserve exact reports and prior sources. Either establish
a scoped compatibility proof or obtain appropriately authorized fresh
execution; updating hashes alone would falsely imply fresh evidence.
The other eight metadata-contract passes are not scientific acceptance.

## Compatibility findings and minimal safe repairs

At capture, all 20 shape-rejected pages already had introductions, objectives,
exercises, and visible tests. Seventeen failed the narrow runtime-language
recognizer; five failed exact title matching, with two overlapping. For
example, 16.9 says “Everything runs on a CPU in about a minute,” while the
old regex required “runs on CPU.” The exact title differences were:

| Section | Configured title | Published title |
| --- | --- | --- |
| 7.1 | Logit Lens, Tuned Lens and Patchscopes | Logit Lens, Tuned Lens & Patchscopes |
| 8.3 | AtP* from Scratch | AtP*: Fixing Attribution Patching |
| 11.1 | PCA, SVD and Geometry Controls | PCA, SVD, and Geometry Controls |
| 11.2 | CCA, SVCCA and Representation Alignment | CCA, SVCCA, and Representation Alignment |
| 15.2 | LoRA vs Full Fine-Tuning | LoRA vs Full Finetuning on a Real Transformer |

The safe repair is narrowly tested runtime recognition and exact reviewed title
aliases or coordinated canonical-title reconciliation. Arbitrary CPU/GPU
mentions and matching a section number alone must remain insufficient. Root
was already implementing this separate repair when this document was written;
these are capture-time findings, not a claim about the latest source.
Chapter 18 also lacked `instructions/.streamlit/config.toml` and
`instructions/requirements.txt`; normal chapter setup repairs should leave
its missing lesson gates intact.

The report aggregator defect was repaired in this lane. It previously raised
on the first unreadable or malformed file, hiding later diagnostics. It now
records missing, invalid, unreadable, non-mapping, or malformed-contract inputs
as blockers and continues. All 14 other production function ASTs, including
contract and freshness logic, and all 22 pre-existing test/helper ASTs were
preserved. No accepted state, report value, threshold, or registry was edited.

Thirteen new regressions failed against the preserved old aggregator. After
repair, the focused file had **31 passed and 1 retained global failure**:
195 blockers comprising 141 absent paths and the same 54 freshness failures.
The two-path reduction from the earlier 143-path snapshot reflects concurrent
file creation. Independent root review found no actionable issues and passed
the 31 focused tests. Exact commands, hashes, before bytes, raw logs, and review
scope are in `../evidence/2026-10-16-report-diagnostics/README.md`.

## Hygiene and readability work

The 671 untracked required payloads are pending review/integration, not 671
learner runtime defects. Review each owner's output before deliberate staging;
do not weaken the tracking gate or bulk-stage unknown work.

Six preserved tar archives lacked review-bundle contracts:
`das_master_migration`, `erasure_master_migration`,
`kernelshap_master_migration`, `qlora_master_migration`, and
`sage_master_migration` each have `published_before_master.tar.gz`, while
`pilot_fresh_execution_20261016` has `before_fresh.tar.gz`, all under
`docs/mastery/evidence/`. Inspect and seal these with the existing evidence
bundle contract, preserving exact archive bytes and hashes. The DAS, erasure,
and SAGE archives also account for three of the five oversize findings; a
valid existing review contract resolves those without a new exemption.

The other oversized files are the 12.3 exercise and solution notebooks,
5,919,866 and 5,944,519 bytes respectively, dominated by visible image outputs.
Preserve exact notebook bytes and their figures before considering lossless
figure externalization or a source-bound artifact contract. Stripping outputs
or reducing scientific image quality merely to satisfy size checks would lose
the learner-visible result and evidence.

The raw artifact
`chapter12_vlm_interpretability/exercises/part1_clip_siglip/cifar10_sample.npz`
is an active dependency, loaded by `utils.py` in that directory and taught in
`infrastructure/chapters/chapter12_vlm_interpretability/master_12_1.py`.
It must not be deleted as disposable output. A repair needs dataset provenance,
hash and license-aware fixture/cache handling with coordinated loader changes,
or a narrowly validated data-fixture contract. A blanket NPZ exemption is not
justified.

The 1,178 readability findings are source/prose lines, chiefly generated lesson
Markdown, rather than inline PNG false positives. Safe formatting should
preserve Python ASTs and rendered Markdown meaning, including fenced code,
HTML, and mathematics, and record the resulting source-provenance change.
Format the owned master where generated outputs derive from it. Do not exempt
all generated files or rewrap an exact frozen protocol snapshot; use the
existing exact-byte snapshot plus readable-mirror mechanism when appropriate.

## Acceptance boundary

No global gate was waived and no full suite was rerun during this lane.
The original `../evidence/2026-10-16-root-cpu-diagnostic/pytest.txt` retains
all 25 initial failures. This work changes diagnostics, not lesson exercises,
signature results, controls, baselines, or play cells. The learner walkthrough
remains deferred by the user; scientific and human acceptance remain distinct
from these static CPU checks.
