# LoRA / DoRA canonical master migration — 2026-10-16

Current integration state: seventeen original cells retain historical outputs and
counts; thirteen added check/play cells now carry actual outputs and kernel counts
from the later, explicitly authorized CPU subset. That subset made six tiny fits
and 440 optimizer steps in 4.415 seconds. It did not execute the seventeen original
runtime statements or reproduce the historical CUDA report/signature statement.
The original proposal metadata remains immutable; current adopted-case and
code/output/count/origin bindings are separate.

## Initial plan before editing — before the later CPU-subset authorization

**Current learner failure:** the complete existing lesson is published under
`part1_lora_dora_adapter_controls`, while configuration expects `part1_lora_dora`.
The method is visible, but local checks depend on a hidden prototype reference;
saved results lack an explicit forecast/reflection and historical execution boundary.

**Target section claim / concrete question:** when a small adapter changes a frozen
classifier, does it learn the planted target coordinate rather than a distractor or
protected coordinate? Accuracy and a rank-one spectrum alone cannot establish that.

**Signature result:** preserve the full six-panel trained safe-proxy comparison
(learning curves, spectrum, ID/OOD behavior, weight direction, activation drift,
merge parity), its example-level OOD table and all historical outputs. The existing
training and signature code remains visible and unchanged; no fresh training was
run at this initial migration checkpoint. The later authorized CPU subset is
recorded below.

**Student methods:** exact scaled B@A and reports; merge parity; DoRA direction /
magnitude recomposition; protected projection and paired mechanism gates; planted
batch generation; train/evaluate; random-label and same-norm controls; signature
assembly. Preserve all thirteen original student functions and supplied helpers.

**Controls/baselines:** frozen distractor classifier, full finetuning, random-label
LoRA, same-norm full random update, protected orthogonal direction, bad mechanism
with good accuracy, and merge parity. Preserve every threshold and seed. Replace
only test plumbing with local independent hand-worked oracles; retain optional
training tests without executing them during the initial migration checkpoint.
Root later authorized their exact bounded CPU subset; that later run is recorded below.

**Play:** preserve original rank/alpha/seed/method training play; append a no-training
CPU matrix play using the taught functions to change update scale, protected
rotation and DoRA magnitudes. Plot row norms and protected projection against an
unnormalized control and same-norm wrong-direction update. Forecast/reflection
pairs ask for confidence, explanation, exposure and a changed scientific condition.

**Files:** new canonical `part1_lora_dora`, `master_15_1.py` and generated master
notebook, `01_[15.1]_LoRA_and_DoRA_from_Scratch.{md,py}`, canonical media/tests,
`tests/mastery/test_lora_dora_master_migration.py`, this note and scoped evidence.
No old files, original ARENA files, 15.2/15.3, shared config or checkpoints change.

**Verification:** archive exact old notebooks/module/page/assets/receipts before
editing; compare all seventeen original solution code cells' ASTs, nonblank lines,
outputs and counts with declared setup-route adaptations only. Complete every
stub from its own generated answer. Check local independent CPU matrix and fixed
classifier oracles, malformed answers, all play combinations and inert imports.
Build without `--execute`; preserve historical source hashes as historical records,
not a new scientific or learner acceptance. Independent review remains required.

## Implemented surface and preservation

The canonical master generates the new `part1_lora_dora` notebooks/module and
`01_[15.1]_LoRA_and_DoRA_from_Scratch.{md,py}` page. The generated master notebook
is physically present under the repository's existing ignored `master_*.ipynb`
rule. The exercise notebook has 103 cells, 30 code cells, nine exercise groups,
thirteen student functions and twelve method-specific forecast/reflection pairs.
Each exercise has immediate checks, expected output, help/interpretation,
a generated answer after its checks and its retained historical worked formulation.
All original explanations, supplied model classes, helpers, training loops,
controls, anomaly hunt, limitations and readings survive.

The sealed pre-edit archive contains twenty safe regular members. SHA-256:
`c643bc1ec8054a9e18ccf15f2ec2828fc9066ec951d6bfb220348882043bbff9`.
Seventeen old lesson/page/assets remain physically byte-identical; three source
dependencies were also captured. No nested archives were present. All fourteen
historical report-input hashes matched the physical files at capture. No original
ARENA, 15.2, 15.3, config, checkpoint registry or other worker's files were changed.

All seventeen original solution code cells preserve exact normalized ASTs,
nonblank lines, execution counts and all nine saved output objects, including the
original duplicate execution count 15. Four adaptations are confined to setup:
canonical section name, local test import, local `media/` directory and lazy
historical-release import. The lazy helper only forwards explicit release calls
to the unchanged old implementation; it contains no taught method. Python-only
root discovery uses `__file__`, and runtime calls are guarded by `MAIN`. Those
guards are removed by the notebook converter. The thirteen added test/play code cells were initially unexecuted. They now retain
actual outputs and counts from the separately authorized CPU subset, with per-cell
source/session origins. The original seventeen cells retain their historical
outputs/counts. The visible mixed-provenance banner distinguishes both origins;
the canonical thirty-cell notebook is not claimed freshly executed as one kernel run.

All four original asset bytes are copied into canonical media. The inline PNG
from the old training play is also saved byte-for-byte as a separate reference.
The signature cell itself has two figure text representations, streams and HTML
DataFrames, rather than embedded PNGs; all six of those output objects survive.
The historical saved signature PNG and behavior-table PNG remain visible beneath
the signature code on the notebook/page. Visual inspection confirmed the six-panel
signature, learned-versus-control separation, spectrum, protected drift and merge
parity summary. These are preserved renderings, not a newly measured model study.

The local tests keep original interfaces while replacing six prototype reference
calls with literal hand-worked matrix/report oracles. Additional checks evaluate
an exact fixed classifier, sample the actual same-norm control from a hand-written
correction, and exercise each signature gate using explicitly synthetic rows.
The latter tests assembly logic only and make no empirical performance claim.
The original optional training tests remain available. Initially omitted from
migration verification, three added training-test cells were later run under the
explicitly authorized CPU subset protocol described below.

The prose explicitly records the existing comparison limits: full finetuning uses
learning rate 0.05 versus 0.1 for LoRA/DoRA; method seeds differ; the same-norm random
matrix is not rank-matched. No optimizer, gate, seed, threshold or old method was
changed. Reference exposure is guided practice and cannot earn unseen credit.

## Fresh CPU play and root-owned guided proposal

A separate bounded execution used only the exact generated play cell and four
minimum definitions: `lora_delta`, `dora_recompose_weight`,
`intruder_dimension_report`, `IntruderDimensionReport`. It used no model class,
optimizer, training, pretrained weights, downloads or CUDA initialization. Its
source hash, definition hashes, runtime versions, plot hash, elapsed time and all
36 changed-condition results are in `cpu_play.json`. The visible two-panel plot is
`media/lora_dora_cpu_direction_magnitude.png`. It was visually inspected; the
unnormalized control fails requested row norms and the same-norm protected update
has projection one. This plot is fresh exact-matrix evidence, separate from the
historical trained signature and report.

For root's guided-case integration, use exactly:

- `W=[[3,4,0],[0,2,0]]`, `A=[[1,0,0]]`, `B=[[1],[-1]]`, rank one,
  alpha one, magnitudes `[10,5]`, protected direction `[0,0,1]`.
- Forecast delta `[[1,0,0],[-1,0,0]]`, update norm `sqrt(2)`, LoRA row norms
  `[sqrt(32),sqrt(5)]`, DoRA row norms `[10,5]`, pre-recomposition protected
  projection zero. The unnormalized control has row norms
  `[10*sqrt(32),5*sqrt(5)]`.
- Same-norm protected control delta `[[0,0,1],[0,0,-1]]` has norm `sqrt(2)` and
  protected projection one; it fails the unchanged projection threshold 0.2.
- Changed scientific condition: rotate A to `[0,0,1]` (90 degrees), holding scale,
  B and magnitudes fixed. LoRA norms become `[sqrt(26),sqrt(5)]`; DoRA norms remain
  `[10,5]`; protected projection becomes one. Explain why the direction failure
  coexists with exact magnitude preservation. Doubling only magnitudes instead
  doubles DoRA norms and leaves this delta-projection unchanged.

This is a seen/guided exact-matrix case, not held-out adapter generalization.
The projection metric applies to the low-rank delta before DoRA recomposition,
not the full DoRA change relative to W; the lesson states that distinction.
Zero alpha produces zero delta and both projection fractions zero. Bind any
checkpoint fingerprint to W/A/B, rank, alpha, magnitudes, angle, protected vector,
threshold and method definitions. Root owns all checkpoint/config edits.

## Verification, retained failures and remaining gates

The focused suite passes **22 tests** (4.89 seconds). It covers archive/physical
preservation, every historical AST/output/count, all thirteen own-answer
completions, full prose/figure retention, twelve independent local CPU oracles,
rejection of wrong answers, inert imports and the exact CPU-play receipt.
Importing from an unrelated directory preserves RNG and gradient state, emits no
output, changes no section file and calls neither CUDA nor an optimizer.
An ordinary subsequent build preserves all historical and fresh-subset outputs
and per-cell origins via the exact same-code checks.

Commands use the existing environment with no synchronization or network:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/arena-audit-compat-uv \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
uv run --no-sync --offline python infrastructure/extension/build.py 15.1

PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/arena-audit-compat-uv \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/arena-lora-mpl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-sync --offline python -m pytest -q -p no:cacheprovider \
  tests/mastery/test_lora_dora_master_migration.py
```

The first CPU-play harness attempt omitted the dataclass decorator while extracting
its minimum definition closure and stopped before rendering. That failure is
retained; the repaired harness includes decorators, then completed the same bounded
probe in 0.694 seconds. The first focused test run found two test assumptions about
converter presentation: generated answers follow the immediate check cells, and
page media uses canonical raw URLs rather than notebook data URLs. Those failed
checks are retained; the corrected assertions verify the actual answer and media
contracts without changing scientific source or outputs.

Remaining gates: learner acceptance and any complete canonical-notebook or
historical CUDA reproduction. Root adopted the exact guided case; the authorized
fresh tiny CPU test subset is separately documented below. No full-course green or fresh
scientific acceptance is claimed. Historical report `accepted` fields remain
historical source records and do not certify the canonical migration.

The root-owned proposal is also saved as `proposed_case.json`, validated against
the worksheet's exact twenty-key schema, with no learner target-answer fields.
`proposed_case_identity.json` binds its bytes, configuration fingerprint, canonical
source/notebooks, exact CPU receipt and plot. The primary response is numeric
pre-DoRA delta-projection; the changed 90-degree condition requests a boolean
threshold forecast. Both are explicitly exposed guided practice. Float32 cosine
at 90 degrees may retain its normal rounding residue; the strict threshold result
and stated mathematical direction remain the same.

Independent review requested one additive learner-facing correction: the new CPU
forecast initially supplied its expected numeric answers. Those answers now occur
only after the play code/result in the reflection. Every historical scientific
cell and original explanation remains unchanged. The fresh CPU play code hash did
not change, so its exact run/plot receipt remains applicable. The focused suite
also verifies the forecast/answer boundary and strict proposal bindings. Independent
reviewer `/root/remaining_lesson_inventory` signed off: initial full suite 21 passes
in 4.84 seconds, final focused prose/parity checks 3 passes in 1.56 seconds and
explicit proposal check 1 pass in 0.57 seconds; all 36 play rows and four definition
hashes independently verified. No open method or preservation finding remains.

The generated Streamlit wrapper's existing template emitted a 470-character
metadata assignment. A scoped `format_page.py` post-build step now formats only
that owned metadata and asserts identical Python AST. Initial and final readability
receipts are retained: one presentation blocker, then zero across the canonical
lesson/master/page. The shared generator is untouched. To reproduce the readable
wrapper after an ordinary build, also run:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/arena-audit-compat-uv \
uv run --no-sync --offline python \
  docs/mastery/evidence/lora_dora_master_migration/format_page.py
```


## Later authorized integration repair — mixed historical and fresh subset

Root's 31-case integration check found two issues: whole-notebook proposal hashes
became stale after navigation injection, and the thirteen added check/play cells
had no actual execution counts. Both are now resolved without changing a method,
seed, control, threshold or any original output.

The original `proposed_case.json` and `proposed_case_identity.json` are byte-immutable.
Their old whole-notebook hashes remain historical metadata. The sealed
`proposal_metadata_before_integration.tar.gz` preserves those two records and the
then-current manifest. That manifest had already been refreshed after root's
navigation update; it is not claimed to represent proposal-time notebook bytes.
Those earlier canonical notebook bytes were not archived at proposal creation.
The new `current_integration_binding.json` instead binds the exact adopted semantic
case, current master/module/test bytes, both notebooks' every code source, output,
count and session origin, and both exact CPU receipts. Tests permit navigation-only
Markdown changes while rejecting changed code or fabricated output. The proposal's
scientific input/metric/control/exposure contract remains unchanged.

Root explicitly authorized one CPU float32, one-thread, <=30-second execution of
all thirteen added cells from a minimum visible 29-definition closure. Before any
training, `fresh_subset_protocol.json` froze all cell/definition/test/module/master
hashes and the exact six fits: two rank-one LoRA fits at 60 steps; LoRA, DoRA, full
finetuning and random-label LoRA at 80 steps each. Total: 440 AdamW steps, batch 256,
original seeds/data recipes, original .1/.1/.05 learning rates and every unchanged
control/gate. No original seventeen runtime statements, historical CUDA signature
call, old training-play call or report/release hook was executed.

The sandbox initially blocked Jupyter localhost socket allocation before the kernel
started. `socket_startup_failure/` retains the failed notebook/log/receipt with zero
completed cells and zero optimizer steps. Root renewed the unused scientific budget
and explicitly authorized localhost access. The identical frozen thirteen cells
then passed in **4.415 seconds**, with actual kernel reply counts **2 through 14**
(the support/definition cell is count 1). There was no scientific cell retry, hidden
threshold relaxation, CUDA initialization, download or pretrained load.

`fresh_subset_execution.ipynb` is the actual fourteen-cell support-plus-subset run;
`fresh_subset_execution.json` binds its protocol, session, runtime, outputs and
actual reply counters. Only actual results for thirteen added canonical cells were
appended. Every original historical count and all nine original output objects
remain unchanged. The fresh play has a real inline PNG output; its separately saved
`fresh_subset_play_0.png` was visually inspected. The earlier four-definition CPU
matrix-play receipt/PNG remains a separate unchanged artifact. A subsequent ordinary
build retained the mixed outputs and origin metadata.

The integration checks pass **29 tests** in 4.20 seconds: 24 focused migration checks
plus five existing 15.1 extension checks; 161 other checks were deselected. The
existing 90% execution assertion now passes with 30 actual counted cells, transparently
from 17 historical and 13 fresh origins. This is not a claim of one fresh complete
thirty-cell kernel execution or full-course/scientific acceptance. The first
integration-test attempt incorrectly equated the root-refreshed manifest with old
proposal-time notebook digests; that failure is retained and the final checks bind
each record to its actual provenance instead.

Independent reviewer `/root/remaining_lesson_inventory` verified the later
integration with four static/provenance checks (1.24 seconds), all thirteen fresh
cell sources/outputs/counts and the 29-definition closure, all seventeen historical
outputs/counts, the mixed banner and retained zero-cell startup failure. The fresh
subset PNG was visually inspected. The reviewer did not rerun training and found
no remaining issue after the documentation correction. The prior independent
review receipts remain unchanged as records of their earlier checkpoint.
