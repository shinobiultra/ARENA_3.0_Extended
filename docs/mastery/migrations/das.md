# 11.3 full DAS lesson master migration

## Scope and implementation plan

- **Current learner failure:** the complete published DAS lesson bypasses the canonical master pipeline; its visible methods, actual outputs, controls and negative findings must survive.
- **Target section claim:** implement a specified high-level intervention through a learned neural subspace, and distinguish output agreement from observational prediction, coordinate recovery and a unique causal abstraction.
- **Signature result:** preserve all twelve figures, including actual next-token distributions, changed cross-name role IIA, whole-donor comparison, all seeds and random/shuffled controls.
- **Student-implemented functions:** high-level interchange/output, orthonormal columns, subspace interchange, intervention loss/accuracy, exact and minibatch alignment fitting, role counterfactual targets, native continuation, observational ridge probe.
- **Controls/baselines:** known causal coordinate, predictive unused copy, mean direction, no-op/full donor, all training seeds, shuffled targets, all 32 random rank-eight subspaces, explicit counterfactual prompts, unchanged-role and conditional native-correct denominators.
- **Play cells:** unused-to-causal angle; native layer/position; donor-role pair change.
- **Files to modify:** new `master_11_3.py`, generated artifacts/media/page, migration archive and parity records, this note and focused mastery tests. The lead owns shared config, syllabus and workbook integration.
- **Tests to run:** exact archive/hash inventory; all 32 code-cell nonblank-line/AST/output/execution-count parity; every original explanatory line; own-dropdown completion; notebook validity; unrelated-directory import safety; independent tiny CPU mathematical/training oracles with ambient dtype restoration.

Original ARENA chapters 0–4 are untouched. All legacy source, evidence and assets remain physically present. No legacy publisher, model load/download, pretrained-model execution,
GPU, cluster job or new scientific study is authorized in this migration. Checked-in results remain explicitly historical. Canonical generation uses `build.py 11.3` without
`--execute`. No cleanup deletions, staging or commits.

## Canonical result and student-facing changes

`infrastructure/chapters/chapter11_representation_geometry/master_11_3.py` now generates the full published lesson: **117 cells, 32 code cells, 8,781 words of prose, nine exercises
and twelve figures**. Eleven visible student functions implement the method: `high_level_interchange`, `high_level_output`, `orthonormal_columns`, `subspace_interchange`,
`intervention_loss`, `interchange_accuracy`, `fit_alignment`, `role_interchange_targets`, `continue_native`, `fit_role_probe` and `fit_minibatch_alignment`. Immediate semantic
tests remain beside their stubs; completing each stub from its own generated answer reproduces the solution AST. A new post-exercise metric/gradient expected-output and help cell
also ensures Exercise 4's answer is available before the next code cell.

All original non-answer teaching lines survive, including the mathematical derivation, counterfactual examples, help and expected-output dropdowns, exact-model ambiguity,
pretrained path checks, full-data recipes, controls, play instructions, fixed token gallery, and limitations/readings. The canonical format adds the ARENA header, standard
objective blocks, numbered-part summaries and bonus boundary. No taught method is replaced by a hidden implementation import. Existing helpers retain data loading, native caches
and evidence plumbing; the continuation, interchange, fitting and metrics remain student-visible.

Twelve forecast/reflection pairs occur before the corresponding computation/result: exact counterfactuals; learned versus shuffled/random subspaces; coordinate recovery versus
specificity; angle play; native site selection; site play; rank selection; frozen final role transfer; unchanged-role specificity and donor-name probability; template/name cohorts;
actual token distributions; and donor-role pair play. Each forecast requests a numerical range/ranking, confidence, causal reason and prior exposure. Reflection requests observed
controls, a discrepancy explanation, alternative explanations and a forecast under a changed scientific condition. These are blank learner activities. Existing answers and figures
make this guided practice, with no unseen-score claim.

The three play cells remain: vary the unused-to-used feature angle; change GPT-2 layer/position and training seed; and change the donor-role pair while preserving the base names.
All twelve original PNGs were copied byte for byte to `media/`, retaining the exact-model distributions, learning/control curves, coordinate/specificity comparison, native
site/rank searches, all final controls, generalization subsets and actual token examples. They were visually checked as a contact sheet, with the full token-distribution gallery
also inspected individually. No new scientific figure or output was generated.

The generated Python module imports from an unrelated working directory without plotting, thread changes, directory creation, model import/loading or CUDA initialization. `MAIN`
guards retain the notebook code while protecting module imports; Python-only root discovery uses `__file__`. The compatibility `run_smoke_test` applies the visible generated
methods to independent tiny CPU oracles. Historical `run_gpu_test`/`run_full_experiment` entry points remain callable but were not called; despite the legacy name, the report path
requests native CPU reproduction and still depends on external data absent here.

## Archive and output provenance

[`before.json`](../evidence/das_master_migration/before.json) records the SHA-256 of
[`published_before_master.tar.gz`](../evidence/das_master_migration/published_before_master.tar.gz), all **122 archived files**, the 15 publication source inputs, and the twelve
figure hashes. Every archived member is reopened and hashed by the migration tests. The archive contains every existing non-pycache section file, the original instruction markdown
page and twelve original instruction assets. The old exercises/solutions/page hashes match their unchanged publication receipt. All 15 learner and seven fresh-native source-input
hashes match the physically retained sources. Every historical file remains byte-preserved at its old path except the four replaced generated publication files, whose exact bytes
are archived.

The retained readable `lesson.md` completed from its own answer dropdowns reconstructs all **63 original published cells exactly**, including all 32 code cells. Every generated
scientific code cell has the **same AST and same nonblank lines** as the archived published solution. Only empty-line/terminal-newline formatting differs.
[`output_parity.json`](../evidence/das_master_migration/output_parity.json) records per-cell published/generated source hashes, AST hashes, original/generated indices, historical
execution counts and output hashes. After this check, all **32 execution counts and 29 output-bearing code cells** were carried forward from the checked-in publication. A
subsequent canonical build confirms unchanged code and retains them using the builder's strict same-code rule.

**These are retained historical reference results, not a fresh canonical notebook execution.** This boundary appears in the master, generated notebooks/page and parity sidecar. The
old receipt certifies its archived publication; it was not rewritten to claim the new prompts were executed. The external learner directory
`/home/bitwise/.cache/arena_extended/das/full-own-answer-v3` and fresh-native directory `/home/bitwise/.cache/arena_extended/das/fresh-native-v4` are absent. The external
tensors/cache cannot be reopened here. The checked-in `expected_outputs/full_own_answer_v3/execution.tar.gz` remains available: its `completed_learner.ipynb` is inspectable, and
all 32 code ASTs, outputs and execution counts match the canonical historical solution notebook. The old receipts' `PASS` values and unavailable external tensor hashes remain
historical assertions; the migration does not fabricate missing data or rerun a publisher to reconstruct it.

## Scientific scope retained

The exact model has two binary causal variables and an unused predictive copy; its finite 1,024-pair evaluation covers all binary settings with fresh nuisances. Perfect output
agreement need not recover a unique designated coordinate. Full donor swaps can damage the base side decision; identity pairs can hide failed changed-variable interventions. The
exact-model fitting keeps all seeds 0/1/2, 300 updates at learning rate .03, shuffled targets and 32 random rank-one subspaces.

The frozen pretrained study uses GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32, eager attention, 72 training prompts and 48 development prompts. It
searches block inputs 3/6/9 at repeated-subject/final-token positions with rank 1 and 120 updates, then ranks 1/2/4/8 with 240 updates. The selected recipe remains block 9 input,
final-token (`end`) position, rank 8, 240 updates, minibatch 16 and learning rate .01. The final set has 96 prompts and 384 interventions; all three seeds, all three matching
shuffled fits and all 32 random rank-eight subspaces remain. The observational ridge penalty is .1. No performance-driven version, precision, training, cohort, selection or metric
change was introduced.

The primary metric is whole-vocabulary argmax agreement with the **base-name answer under the source's repeated-name role**, restricted to all 96 changed cross-name interventions
and including native failures. It is not donor-answer copying or an argmax restricted to a pair of names. The historical DAS scores remain 71/96, 73/96 and 72/96; whole donor
reaches 72/96, so the lesson retains its negative finding that this run does not establish an advantage over copying the full vector. The conditional
base-and-counterfactual-native-correct denominator is 56 and answers a different question. It cannot replace the primary score. The explicit counterfactual prompt is an empirical
reference, not an upper bound.

The retained fresh-native receipt reports **16,512 intervention comparisons across 43 conditions** and **17,232 total native forward rows** (16,128 tail rows, 96 baseline prompts
and 1,008 native-hook rows), plus 384 cached explicit-counterfactual comparisons. Those counts describe a prior run. Neither it nor a new pretrained path/gradient check ran during
this migration. The small dependent template/name cohorts and seed spread remain descriptive; no new independent TEST or replication is claimed.

## Guided case proposal for lead integration

[`guided_case_proposal.json`](../evidence/das_master_migration/guided_case_proposal.json) freezes a concrete numeric primary question and a changed-condition question without
prefilled learner answers. The primary forecast is seed-0 unconditional changed cross-name IIA (denominator 96). The changed condition sets source and base repeated-name roles
equal while keeping names different: seed-0 unchanged-role cross-name IIA (denominator 96). This tests specificity/preservation under a changed intervention distribution with the
same learned subspace; it is historical guided practice, with overlapping prompts and no claim of independent transfer evaluation. Its exact reference JSON paths are
`final_metrics.DAS/seed0.cross_changed.iia` and `final_metrics.DAS/seed0.cross_unchanged.iia` in `expected_outputs/native_summary.json`. The lead owns manifest/workbook integration
and validation of commitment, stale-input rejection, observation provenance and portable records. No checkpoint is silently accepted by this lane.

## Preservation exceptions

All legacy evidence and tooling remain deliberately present. The exact structural exception inventory is
[`preserved_legacy_inventory.json`](../evidence/das_master_migration/preserved_legacy_inventory.json); its relative filenames and hashes are derived from the before snapshot. This
supersedes the authoring guide's generic cleanup instruction for this migration. Original ARENA chapters 0–4 remain untouched. No untracked/ignored research files, legacy archives,
failed runs or source originals were removed.

## Validation and remaining gates

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline python infrastructure/extension/build.py 11.3
UV_CACHE_DIR=/tmp/arena-mastery-uv MPLCONFIGDIR=/tmp/arena-das-migration/mpl \
  uv run --no-sync --offline pytest -q tests/mastery/test_das_master_migration.py
```

Canonical generation passes without `--execute` and reports unchanged code/output carry-forward. The focused suite passes **21 tests in 4.52 seconds**: archive/source/asset hashes,
exact lesson reconstruction, 32 code AST/nonblank-line/output/count pairs, eleven own-answer stubs, original teaching-line preservation, notebook schemas, forecast/control/play
coverage, unrelated-directory import safety, eight existing independent exact/gradient/fitting oracles, two independent training-method oracles, the model-free prompt/role
contract, and frozen guided-case artifact/config/reference integrity without filled learner answers. Tests save/restore ambient PyTorch dtype and thread count around numeric
training checks.

Fresh canonical full-notebook execution, reopening the missing external tensors, new pretrained validation, scientific replication, human learner acceptance and course-readiness
promotion remain separate open gates. No model was loaded/downloaded, no GPU or cluster used, no publisher run, and no commit/staging/push performed. The canonical entry point is
`infrastructure/extension/build.py 11.3`; old publishers remain historical tooling.

Proposed chapter description: **Learn a distributed interchange intervention, then separate perfect probes, causal output agreement and coordinate recovery using an exact model and frozen GPT-2 role-transfer controls.**

The earlier twenty-test numeric suite also passed in 3.30 seconds with ambient PyTorch default dtype pre-set to float64; an explicit post-suite assertion confirmed float64
restoration. The new master, generated Python module and focused tests compile. Scoped whitespace checks pass after trimming the converter's extra instruction-page EOF blank. No
fresh native transformer plumbing test was run during migration; source parity preserves that exercise and its historical outputs.

Independent review correction: the missing external cache does not imply the completed notebook is unavailable. A dedicated checked-in archive parity test now verifies the available historical execution directly. The prior independently reviewed twenty-one-test suite passed; the expanded suite below includes that additional archive check.

The expanded DAS suite including direct checked-in execution-archive parity passed all **22 tests in 4.59 seconds**.
