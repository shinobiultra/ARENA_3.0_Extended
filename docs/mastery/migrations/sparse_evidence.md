# Historical sparse evidence migration

The removed `part18_spex` / `part19_proxyspex` prototypes used a joint 16.18/16.19
native study and portable replay receipt contract. The current 16.6 lesson teaches
SPEX/ProxySPEX methods directly and has no equivalent publication API. Renaming
these tests to 16.6 would erase the guarantees they test.

The retained compatibility boundary is `arena_ext/legacy_sparse_evidence/`.
It validates historical receipts and can stage historical evidence into a new
external directory. It has no model, fitting, subprocess execution, installation,
or CLI path. The deleted learner prototypes remain deleted. Original ARENA files
are untouched by this migration.

## Provenance

The source is Git commit `dec028ec8ed8accb40bf0d0a7315bb72f0c1fcc6`.
`arena_ext/legacy_sparse_evidence/PROVENANCE.json` records exact original paths,
Git blob IDs, SHA-256 digests, retained function names, and scoped changes.
Historical validator functions retain their original bodies. Publisher changes
outside those bodies adjust the repository root for the new package and remove
unused imports/constants. Synthetic fixture constructors and receipt mutation
tests now live in repository tests instead of importing deleted lesson tests.

## Guarantee map

| Historical guarantee | Status and current owner | Evidence |
| --- | --- | --- |
| Outer concurrent supervisor must complete; inner success cannot override failures, drift, wall/RSS/output limits, or wrong execution kind | Retained: `publish_study.validate_concurrent_schedule` | `test_sparse_concurrent_publication.py`: seven outer-failure mutations |
| Outer output inventory includes its own receipt; unrecorded files and links are rejected | Retained: `receipts._finalize_receipt`, `publish_study.validate_concurrent_schedule` | Synthetic receipt finalization and unrecorded-file rejection |
| Scheduling supplement, independent approval, coordinator sources, run pair, and source inventories must be hash-bound | Retained: `publish_study.validate_concurrent_schedule` | Approval/supplement/coordinator, foreign supervisor, missing schedule and false sequential-claim mutations |
| Both historical sections receive authoritative scheduling evidence | Retained: external `publish_study.stage_candidate` | Byte-for-byte staged copies for all four roles in both sections |
| Two distinct cold runs, source inventories, physical study records, comparison receipts and portable fixtures remain bound | Retained: publisher validation/staging helpers | Existing synthetic native-pair and section fixtures exercise the staging path; no model execution implied |
| Physical receipt manifests bind name, hash and byte count; unrelated sections keep the legacy fixture contract | Migrated adapter imports: `scripts/verification_fixture_contract.py` and `scripts/audit_report_evidence_contracts.py` | Integration acceptance, each receipt-size mutation, unrelated-chapter fallback |
| Historical study identity, all 1,152 conditions, nine ordered references, 32 prompts and measured fit/query counts remain required | Retained: `native_report.receipt_blockers` and helpers | Missing/duplicated conditions, missing reference/prompt/control/stage and wrong inventory mutations |
| Cached execution, replay label leakage, incomplete own-answer execution, wrong replay methods/predictions, missing or duplicate bindings, altered report outcomes are rejected | Retained: report predicates | Fourteen cross-receipt/report mutations |
| Poor fidelity, shuffled controls beating real fits, failed references, undefined normalized metrics and disclosed non-bitwise reproduction remain valid recorded outcomes | Retained unchanged | Synthetic positive contracts for both historical sections and constant-variance case |
| Both historical replay receipt layouts remain understood | Retained unchanged | Directory-record SPEX fixture and JSON-receipt ProxySPEX fixture |
| Strict historical notebook learner-contract dispatch | Retained existing script | Rejection dispatch tests for both historical notebook stems |
| Native study execution, process supervision, training, model querying, fresh reproduction, report assembly, publication approval and installation | Unsupported by this compatibility package; removed rather than restarted | No runner, report builder, installer or CLI was ported |
| Scientific completion of stopped native experiments, publication/release acceptance, or equivalence to current 16.6 | Not established | Synthetic receipt tests provide software-contract evidence only |

`verification_entrypoint` resolves historical receipt sections to the compatibility
validator file. This is an audit locator, not a runnable experiment hook. Current
16.6 retains its own lesson implementation and exercise tests. This change adds no
student-facing result, plot, baseline, or play cell; those lesson obligations remain
with the current learner material.

## Verification

```sh
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline pytest -q \
  tests/test_sparse_concurrent_publication.py \
  tests/test_sparse_verification_integration.py --tb=short
```

Result: **59 passed**, CPU/tempfile only, with no network, model, GPU, training or
cluster jobs. This verifies the retained evidence contracts and does not constitute
fresh scientific evidence. Ruff was unavailable in the existing offline environment
(`No module named ruff`); no dependency installation was attempted.

Additional checks: **379 passed** in `tests/test_verification_fixture_contract.py`;
`compileall` and scoped `git diff --check` passed. A provenance cross-check verified
all five original source hashes/blob IDs and all 35 retained function ASTs against
the pinned commit.

The initial broader diagnostic returned **398 passed, 1 failed, 40 errors**.
The 40 fixture errors came from deleted `part18_spex.build_notebooks` imports in
`tests/test_sparse_interaction_learner_contract.py`. A bounded follow-up replaces
that test's retired-builder dependency with an explicit synthetic published tree.
The remaining course-wide audit failure reads missing
`chapter5_modern_architectures/exercises/part2_mixture_of_experts/artifacts.lock.yml`;
it is separate from the sparse fixture migration.

## Historical learner-tree fixture follow-up

`tests/test_sparse_interaction_learner_contract.py` now writes only temporary
synthetic source files and notebooks. It does not import or copy retired source,
execute notebooks, call a model, or change `scripts/sparse_interaction_learner_contract.py`.
All five existing test-function ASTs, including every parametrized mutation list,
are unchanged. A new private fixture helper replaces only the old builder/source
setup; the existing byte-bound receipt and report assembly remains in the test.

The independent fixture inventory anchors retain 22 SPEX and 13 ProxySPEX taught
function names, 31/27 solution code cells, and 12/13 figure calls. The temporary
exercise notebook contains real stubs, adjacent solution dropdowns, top-level
immediate test calls, an explicit student-function namespace, and two distinct
native-study/play replay calls whose results feed figure calls. ProxySPEX's
visible bridge is compared with its temporary source-bound adapter. Source hashes,
the synthetic scientific freeze, copied replay outcomes, source/report manifests,
embedded PNGs, and deliberately unavailable historical locators use the existing
production validator unchanged.

The preserved mutation tests cover absent answers, tests hidden in a dead branch,
early/late namespace bypasses, missing play replay, hidden replay methods,
solution/signature drift, source/data/receipt tampering, error/missing image outputs,
manifest omission, audit leakage, visible prompt mismatch, missing shuffled controls,
failed pointwise predictions, and false full-notebook study claims.

These are parser and evidence-binding fixtures. Their identity functions, marker
modules, generated PNGs, execution counts, and receipts are **synthetic inputs**.
The marker immediate-test modules are never imported and their calls are never
executed; no correctness or real runtime guarantee for SPEX/ProxySPEX follows.
The fixture's padding preserves historical inventory count checks without recreating
the retired teaching flow. It does not establish historical builder output parity,
real notebook execution, original native experiment completion, live cache access,
scientific outcomes, publication acceptance, or current 16.6 certification.

Follow-up focused command:

```sh
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline pytest -q \
  tests/test_sparse_interaction_learner_contract.py \
  tests/test_sparse_concurrent_publication.py \
  tests/test_sparse_verification_integration.py --tb=short
```

Result: **100 passed** (41 learner-tree cases plus the previous 59 evidence cases).

Repeating the original broader diagnostic after this follow-up returned
**438 passed, 1 failed, 0 errors** across `test_verification_fixture_contract.py`,
`test_report_evidence_contract_audit.py`, and `test_sparse_interaction_learner_contract.py`.
The sole remaining failure is the separate missing chapter 5 lock file described
above. The changed learner fixture also passed compilation and scoped whitespace
checks. No full-suite success is claimed.
