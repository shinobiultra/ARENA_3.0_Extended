# Blind SAGE prediction: implementation plan and frozen design

**Current learner failure:** the reference-rich SAGE lesson is guided practice and exposes outputs before a learner commits a numerical explanation.

**Target question:** for one fixed zero-mean Gaussian predictor, how does allocating predictive-performance improvement change when missing inputs are conditionally inferred rather than marginally replaced? The neutral learner title is **One predictor, two missing-input rules**.

**Case frozen while unmeasured:** covariance `[[1,.6,-.3],[.6,1,.14],[-.3,.14,1]]`, truth coefficients `[1,-.5,.25]`, predictor coefficients `[.75,.25,-.5]`, independent noise
variance `.25`, CPU float64. Feature names A/B/C, ascending bitmask coalitions. The signed numeric forecast is conditional SAGE for B minus marginal SAGE for B, in squared-target
units, with no [-1,1] bound. These inputs were fixed before engineering measurement, without outcome-based selection.

**Visible student methods:** restricted linear coefficients; quadratic Gaussian risk/game; exact eight-coalition Shapley; marginal-by-predecessor-size; JSON-native assembly. Immediate independent practice fixtures use different inputs.

**Signature result:** paired conditional/marginal SAGE bars plus B's marginal contributions by predecessor size. Controls: empty/full rule agreement, efficiency, identity-covariance rule equality, zero predictor, noise .25→.75 raises risk .5 while preserving credit, feature permutation, and an independent six-ordering allocation oracle.

**Transfer:** only covariance changes to `(Sigma + I)/2`. It requires a separate forecast and immutable commitment after the primary receipt. It is not a primary control or seed replication.

**Files:** only `mastery_hub/units/sage_prediction/{lesson.py,exercises.ipynb}`, `scripts/build_sage_prediction_notebook.py`, `tests/mastery/test_sage_prediction.py`, this design, and scoped evidence. Root owns registry/UI/readmes.

**Integrity:** separate `arena-blind-sage-prediction` v1 schema; strict finite JSON/duplicate-key rejection; case, ordered input bytes, student method, helper, test and runtime
binding; no method evaluation on import/freeze/commit; before/after input/runtime/source mutation checks; independently validate returned coefficients, risk, game and assembled
arrays; immutable forecast→observation/failure→reflection links; private exclusive 0600 export; model-free replay; preserve returned actual measurements on failure. Notebook
generation preserves learner source, output, records and execution metadata.

**Tests:** non-target practice oracles, malformed/tampered record rejection, stale identities, student mutation and false assembly, failure retention, notebook blankness/order/no
answer leakage, generator read-only checking and preservation, export/replay. A production target run requires independent pre-run review and root's recorded engineering
forecast/go-ahead; no target experiment runs during implementation. Aggregate post-approval CPU allowance is 120 seconds. No model loading, downloads, GPU, cluster or Git mutation.

## Implemented API and scientific boundaries

`lesson.py` implements deterministic blank notebook authoring plus `freeze_inputs`, `commit_prediction`,
`assert_current`, `run_experiment`, `read_record`, `replay`, `replay_failure`, `plot_receipt`, `reflect`,
`read_attempt`, `commit_transfer`, `export_record` and `export_attempt`. Reference numerical code is returned
by `reveal_methods` only after a commitment. Learners edit five visible function-source cells; those exact
joined function definitions are committed and executed, including assembly and marginal-by-size logic.
Defaults/decorators/annotations capable of executing at definition time are rejected. This is ordinary
local student-code execution, not an adversarial sandbox.

The primary execution contains ten conditions: conditional/marginal rules for the primary covariance,
identity covariance, zero predictor, shifted noise and feature permutation. Each computes eight coalition
coefficient/risk/game rows. There is no sampling, training or fitted-model state. The independent replay
validator checks the stored coefficients' defining linear equations, risk quadratic, risk differences,
six-ordering allocations and strata without invoking a new linear solve or student method. Replay performs
arithmetic validation of existing measurements; it starts no experiment and reads no current runtime.

Before and after each student call, the runner checks frozen source/helper/test/runtime identities and
snapshots input shape/dtype/device/bytes. Returned coefficients, risks and games are captured before
post-call checks, and assembly receives disposable clones. Incorrect finite returned allocations are
retained as **unscored** diagnostic evidence; invalid-shaped/nonfinite allocations cannot become successful
receipts. Structured failures preserve their original commitment, stage/error/time, valid returned arrays,
completed validated results and controls when available. Failure replay/export never retries or scores.

The separate covariance-only transfer requires a primary observation, a reflection containing a new numeric
forecast/confidence/reasoning, an explicit `freeze_inputs(..., variant='transfer')`, and `commit_transfer`.
Its immutable commitment links the primary reflection and cannot change that forecast. Transfer observation
uses the same metric and its own controls. Transfer is absent from primary conditions and is not executed
by reflection, freezing or commitment. No automatic unseen status, scientific TEST claim, human learner
acceptance or general feature-importance conclusion is inferred from either receipt.

Record schemas reject unknown fields, nonfinite numbers, duplicate JSON keys, booleans masquerading as numeric
counts, unsupported versions, stale identities and broken parent links. Read limits are four million bytes
for records and six million for reflections; writers validate with their reader before returning/exporting.
Largest-source/Unicode round trips are tested. Exports use exclusive creation with mode 0600. Seals detect
accidental drift but do not authenticate a hostile local editor or exposure self-report.

The notebook generator has a fixed output path and `--check` is read-only. Current artifacts remain byte-for-byte
intact, including outputs and runtime metadata. Stale generation may replace only pristine generated source.
Learner edits, outputs, execution counts, widgets and per-cell execution metadata block a stale overwrite.
Legacy `trusted` metadata is inspected in raw JSON because nbformat removes it on read; this preservation
case is tested separately without code outputs.

## Pre-run verification

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv MPLCONFIGDIR=/tmp/arena-sage-prediction/mpl \
uv run --no-sync --offline pytest -q tests/mastery/test_sage_prediction.py
UV_CACHE_DIR=/tmp/arena-mastery-uv \
uv run --no-sync --offline python scripts/build_sage_prediction_notebook.py --check
```

Result: **68 passed in 3.06 seconds**, notebook check OK. All tests are non-target: practice uses different
one-/two-dimensional numerical fixtures and arbitrary interacting games; full runner/record tests explicitly
replace the public scientific inputs with a zero predictor, identity covariance and different truth vector.
One guard asserts that the adopted covariance never reaches those numerical runner tests. Public-case tests
inspect exact input structure and hashes only. Production import, freeze, commitment and notebook generation
perform no target numerical work. The reference implementation and all notebook code cells compile.

An independent reviewer reran all **68 tests in 3.31 seconds** and the generator check, inspected scientific
formulas, mask ordering, control/transfer semantics, mutation and failure retention, and found no blocker.
The reviewed helper SHA-256 is `94abc5badfb7a7c11994093e25447d2bcfb25eab66628f02a6f3979eb3fbb8de`;
test SHA-256 is `32b5a00b64171d25e755594d0c05e2d03f7b99cf0ffc7ce5f654aa77b278325a`.
At that historical pre-run checkpoint, the adopted public target was **uncomputed and uninspected**. Root owned the recorded
engineering forecast and explicit go-ahead before any target run; the authorized aggregate post-approval
CPU allowance remains 120 seconds and needs an external bound as well as the runner's cooperative deadline.
Changes to helper/tests/runtime require a fresh identity before the eventual validation.

Only the new SAGE variant files and this design/evidence were changed. Existing original ARENA sources,
canonical SAGE reference lesson, shared registry/UI/readmes and prior learner work are unchanged by this lane.
At that pre-run checkpoint, no model/download/GPU/cluster/scientific target run, commit, staging or push had occurred. Fresh target validation,
blind learner use and human acceptance remain distinct open gates.


## Authorized engineering validation after the pre-run checkpoint

Root subsequently recorded the immutable engineering forecasts in
`docs/mastery/evidence/blind_sage_20261016/engineering_forecast.json`, bound to the exact four reviewed
helper/test/generator/blank-notebook hashes, and explicitly authorized exactly one primary and one separately
committed covariance-only transfer. The pre-run evidence remains unchanged as the historical checkpoint.

The complete engineering lifecycle passed: visible method completion and independent practice fixtures;
primary input freeze and forecast commitment; one primary measurement, replay and plot; reflection linked to
the original receipt using the pre-recorded transfer forecast; transfer freeze and separate commitment; one
transfer measurement, replay and plot; private exclusive exports and replay round trips. The run used a hard
120-second external timeout and process CPU limit, CPU float64 inputs, one numerical thread and no initialized
CUDA context. No source hash changed before or during execution and no retry or post-result selection occurred.

Total lifecycle time was **2.543 seconds wall / 2.494 seconds CPU**. Each case retained **ten conditions,
eighty coalition rows and twelve passing controls**. All numerical receipts, two signature plots and the
completed thirteen-code-cell engineering notebook live only in the new mode-0700 directory
`/tmp/arena-blind-sage-engineering-dhrYgdeB`. Exported records and other private regular files are mode 0600.
The public [`engineering_validation.json`](../evidence/blind_sage_prediction/engineering_validation.json)
contains only file hashes, timing, execution inventory and control pass status; it contains no target answer.

The public shipped notebook remains the identical blank artifact with thirty cells, thirteen code cells,
five student implementations and zero prefilled outputs. The public case was fixed before its outcome was
computed and is now **engineering validated**. This does not establish any learner's unseen exposure,
independent TEST evaluation, causal feature importance or human mastery/acceptance. Exposure remains explicit
self-report. Shared registry/UI/readme integration belongs to root. No original ARENA source, canonical SAGE
reference lesson or existing learner work was altered.

Final post-run non-target regression: **86 passed in 6.37 seconds** across the 68 blind-variant tests and 18 canonical SAGE migration tests. The blank generator `--check`, all four reviewed source hashes, all private modes and all thirteen completed engineering execution counts were verified again. No target was rerun.
