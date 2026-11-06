# 15.3 full QLoRA lesson master migration

## Scope and implementation plan

- **Current learner failure:** the full published QLoRA companion has a legacy builder and bypasses canonical authoring. Its rich method code and unreleased
  historical failure must survive.
- **Concrete question:** can a trainable low-rank adapter recover a counterbalanced marker rule on frozen four-bit pretrained weights, and what does actual
  total training-state storage save?
- **Signature result:** preserve NF4/storage and TinyStories comparison panels, outlier play and precision/buffer diagnostic, including the failed unchanged .01
  BF16 parity gate.
- **Student methods:** exact NF4/nested-scale quantization, materialized-state audits/storage accounting, STE/QLoRA wrapper, counterbalanced data/controls,
  matched two-label plus replay objective, spectra/outlier metrics and strict loader.
- **Controls/baselines:** BF16 LoRA/full BF16, rank2/rank 8, all three seeds, random labels/same-norm adapters, marker swaps/removal, name-only baseline, exact
  backend/zero-block/gradient oracles, clean versus outlier, FP32 parity and parameter/buffer decomposition.
- **Play cells:** CPU block-size/outlier; canonical/swap/removed prompt inspection; real precision/layer/input scale diagnostic.
- **Files to modify:** new master, generated artifacts/media/page, exact original module retained as legacy reference plumbing, scoped tests and migration
  evidence/note. Root owns shared config/syllabus/checkpoints/workbook.
- **Tests to run:** exact archive and historical hashes including ignored inputs; all 21 code AST/nonblank/output/count pairs after narrowly declared import
  routing; own-answer reconstruction; original prose and failed-gate preservation; notebook validity/import safety; bounded independent CPU
  quantization/data/gradient/spectrum oracles with ambient dtype restoration. No optimizer training or pretrained checks.

Archive before overwrite and retain all physical legacy sources/assets, including ignored files. Direct filesystem verification finds all eight historical
source inputs, including
the ignored `generate_cpu_fixtures.py`, and their hashes match. The historical candidate remains unreleased (`course_release_accepted: false`). Original ARENA
remains untouched. No
training, pretrained/model loads, downloads, GPU/cluster jobs, legacy publishers, commits, staging or cleanup.

## Canonical lesson and student improvements

`infrastructure/chapters/chapter15_peft_misalignment/master_15_3.py` now generates the complete published lesson: **91 cells, 21 code cells, 7,371 words of
prose, seven exercise
groups, 31 active student definitions and four figures**. All 43 original notebook function/class definitions have identical ASTs, including the actual NF4
codebook/quantizer,
uniform-int8 nested derivation, tail-aware and allocated-storage accounting, fail-closed state audit, STE gradient path, real QLoRA wrapper, complete crossed
data/controls, matched
task/replay loop, spectra, strict loader and visible CUDA wrapper. Core methods remain editable in the notebook; the pre-existing experiment orchestration
receives the student's
callbacks rather than silently substituting methods.

All original non-answer teaching lines survive except the two reserved headings `## Learning Objectives` and `## Setup`, renamed to `## Original learning goals`
and `## Original
setup boundary`. The canonical form adds standard objective blocks, three numbered parts, summaries, bonus boundary, metadata for seven original exercises, and
immediate answer
dropdowns before their separate test cells. The active `QLoRALinear` class stub survives. Completing every stub from its own generated answer reconstructs the
corresponding full
solution AST.

Eight forecast/reflection pairs cover nested storage/audit failure, crossed marker controls, outlier block coupling, quantization/storage signature, the
pretrained rule/memory
comparison, outlier play, paired prompt play, and precision/buffer play. Each forecast requests a numerical range/ranking, confidence, causal reason and prior
exposure before its
corresponding output; reflection requests observed controls, discrepancies, alternative explanations and a changed scientific condition. No learner response is
filled in. Existing
answers and outputs remain guided practice, with no unseen score.

The three original play activities remain: change NF4 block size/outlier magnitude on CPU; inspect canonical/swapped/removed marker prompts without inferring
model behavior; and
inspect CUDA BF16/FP32 precision, target layer, input scale, parameters and buffers. No new optimizer/model-training activity was added or executed. All four
historical PNG outputs
were inspected as a contact sheet. The original NF4/storage and real comparison assets remain unchanged; both other plots remain exact notebook-output images.

## Preservation and source identity

[`published_before_master.tar.gz`](../evidence/qlora_master_migration/published_before_master.tar.gz) preserves **sixteen original files**, including every
non-pycache section file
(also ignored files), both notebooks, old module/builder/executor/generator, receipt/schema, original instruction page and both signature assets.
[`before.json`](../evidence/qlora_master_migration/before.json) records archive/file hashes and all eight historical source hashes; every one matched before
editing. The initial
`rg --files` listing omitted the ignored `generate_cpu_fixtures.py`; direct filesystem inspection found it present and hash-identical. There is no
missing-source claim. Every
archived member is reopened and hashed in tests.

The original reference module also remains physically available byte for byte as `legacy_reference.py`. Canonical notebook setup makes two explicit routing
adaptations: its
`reference` import and its explicit constants/data-record import both point to that exact module copy, preventing a canonical `solutions.py` self-import. **All
21 scientific code
cells have identical ASTs and nonblank lines after only those two named setup routes.** No numerical method, model/version/precision, objective, control,
threshold, dataset, seed,
budget or scientific selection changes. Python-only `__getattr__` retains optional old plumbing/report APIs; taught methods implemented by the notebook take
precedence. The
original builder, executor, generator, tests, README, schemas, receipt and assets remain physically present and unchanged.

The generated Python module uses Python-only `__file__` root discovery and `MAIN` guards around tests, bitsandbytes import, warning/seed/print setup, provenance
assembly, release
execution, all plotting, CUDA availability/diagnostics and experiments. Importing from an unrelated directory imports neither bitsandbytes nor Transformers,
initializes no CUDA,
changes no seed/thread configuration and creates no files. Model-loading and training functions remain callable only through deliberate invocation; none was
invoked during
migration. Existing historical release entry points and callback provenance mechanisms were preserved, not certified by a new training run.

[`output_parity.json`](../evidence/qlora_master_migration/output_parity.json) records published/adapted/generated source hashes, AST hashes, old/new indices,
output hashes and
execution counts. All **21 historical execution counts, fourteen output-bearing cells and seventy outputs** are carried directly from the archived publication.
A repeat canonical
generation retains them under the builder's strict same-code check. [`figures.json`](../evidence/qlora_master_migration/figures.json) hashes all four
inline-derived `media/` PNGs;
the receipt's two original signature hashes remain valid at the untouched original asset paths. New markdown forecasts/reflections have no execution or
learner-response claim.

**The displayed evidence is retained historical reference evidence, not a fresh canonical run, current pretrained validation or scientific replication.** The
old
`reference_metrics.json` continues to certify its archived source/notebook hashes; it is not rewritten to claim freshness for regenerated files. The legacy
freshness test
consequently belongs to the old publication and archive; focused migration tests compare those recorded hashes with the exact archive. No accepted verification
report was invented.

## Retained scientific results and failure

The manual NF4 derivation uses per-block absmax and exact nearest-neighbor lookup, including a clean zero block. The uniform signed-int8 nested-scale arithmetic
is explicitly a
derivation, separate from actual bitsandbytes dynamic-map materialization. Tail padding, the two lookup tables, offsets, nested absmax and shared storage are
counted; changing a
codebook value or inserting a non-finite nested value must fail closed. The CPU mechanics tests use independent bit/kernel/mathematical oracles, not pretrained
behavior.

The historical pretrained comparison pins `roneneldan/TinyStories-1M` revision `77f1b168e219585646439073245fe87e56b3023e`. It targets layer 6 query/value
projections, retains three
measured seeds 0/1/2, 32 updates per trained method, batch 12, adapter learning rate .02 and full-BF16 learning rate .00005. The crossed
train/template-transfer/unseen-subject sets
contain 60/40/40 examples. Both labels occur for every subject/template; OOD changes names/templates while retaining the same explicit marker rule. The task
loss is two-label
conditional NLL; protected replay uses a separate full-vocabulary next-token loss. The protected prompts are used for replay and do not supply independent
capability-preservation
evidence.

QLoRA rank 8 and BF16 LoRA rank 8 each reach 1.000 on the narrow template-transfer and OOD sets in this historical run. Random-label and same-norm random
controls remain, as do
marker swaps, paired flip/both-sides correctness, removal and name-only baselines. Seed intervals use Student-t with df=2; they describe
optimizer/batching/initialization
variability on this fixed synthetic task, not dataset uncertainty. Random adapters are also low rank, so an adapter spectrum alone is not useful-learning
evidence.

Measured resident training state is about 38.67 MiB for QLoRA, 39.18 MiB for BF16 LoRA and 60.58 MiB for full BF16: roughly 1.3% savings against BF16 LoRA, not
an eightfold
total-state reduction. The resident total is non-overlapping model tensors + gradients + actual AdamW tensor storage; peak CUDA allocation is separate. Eight
2048×2048 boolean
attention buffers occupy 32 MiB and limit the benefit from quantized projections. The precision diagnostic retains BF16 max error .03125 versus the unchanged
.01 gate, with FP32
error about9.54e-7. This helps investigate the failure but does not erase it or show every BF16 discrepancy harmless.

The historical receipt still has **`course_release_accepted: false`** and `cuda.accepted: false`. Behavior and memory checks passing do not override the failed
gate. No accepted report is registered, no threshold was relaxed and no release/readiness promotion is made by this migration.

## Frozen guided case proposal

[`guided_case_proposal.json`](../evidence/qlora_master_migration/guided_case_proposal.json) proposes a precise CPU method question without a new pretrained run:
forecast mean
squared NF4 reconstruction error **excluding weight index0**, for 512 float32 weights drawn as `.2 * randn(seed 0)`, with weight 0 set to 12 and NF4 block size
128. The exact
retained reference path is `cpu.quantization_stress.outlier.128.non_outlier_mse` in `expected_outputs/reference_metrics.json`. Compare the matched clean
weights, same outlier with
block 8 and exact zero-error anchor.

Change only raw outlier 12→6 while retaining the other 511 weights, seed, length, dtype, block size and metric. This is an **unmeasured** changed scientific
condition, to run as a
tiny CPU play only after a new learner commitment; no answer is prefilled or inferred from the existing different seed 7/length 256/block 32 play. The visible
`quantize_nf4`/`dequantize_nf4` functions support it directly. `quantization_stress_test` hardcodes 12 and does not expose an outlier parameter, so use the
explicit play tensor
with those visible functions. Config/source/reference hashes and blank response-field names are recorded in the proposal. The lead owns adopted
manifest/workbook integration and
commitment/provenance/stale-input/portable-record checks.

## Validation, exceptions and remaining gates

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline python infrastructure/extension/build.py 15.3
UV_CACHE_DIR=/tmp/arena-mastery-uv CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/arena-qlora-migration/mpl \
  uv run --no-sync --offline pytest -q tests/mastery/test_qlora_master_migration.py
```

Canonical generation passes without `--execute`; repeated generation preserves historical outputs. The first 26 focused tests passed in 6.44 seconds. They cover
the exact
archive/eight source hashes, all 21 scientific code/output/count pairs after two setup routes, four exact PNGs, seven own-answer exercises / 31 active stubs,
all original teaching
lines with named heading adaptations, all 43 notebook definitions, unrelated-directory import safety, sixteen bounded independent CPU
codebook/kernel/state/gradient/data/spectrum
oracles, explicit failed-gate/receipt retention and a brute-force scalar nearest-code quantization oracle. Tiny actual bitsandbytes matrices are explicitly
materialized on CPU; no
pretrained model is loaded and no optimizer or training loop runs. The checks assert CUDA remains uninitialized. The expanded suite adds frozen guided-case
reference/config
integrity without any learner answers. Numeric fixtures preserve ambient dtype and thread count.

[`preserved_legacy_inventory.json`](../evidence/qlora_master_migration/preserved_legacy_inventory.json) enumerates retained names/hashes requiring structural
preservation:
`README.md`, `artifacts.lock.yml`, `build_notebooks.py`, `execute_learner.py`, ignored `generate_cpu_fixtures.py`, `expected_outputs/**`,
`verification_report.schema.json`, and
exact `legacy_reference.py`. Original `tests.py` remains unchanged. Generated notebooks/module/page replace only their archived publication bytes; both old
assets stay intact.
Original ARENA chapters 0–4 were not edited. No cleanup, staging, commits or pushes occurred.

Fresh full-notebook execution, current pretrained/backend compatibility, another scientific comparison, human learner acceptance and readiness promotion remain
open. The old BF16
failure is an additional scientific/release boundary. Canonical generation, parity and CPU oracles do not resolve it. The canonical entry point is
`infrastructure/extension/build.py 15.3`; legacy publishers remain preserved historical tooling.

Proposed chapter description: **Implement and audit NF4 QLoRA, then distinguish a solved counterbalanced marker rule, actual training-state savings and an
unresolved BF16 parity failure.**

The full expanded suite passed **27 tests in 5.14 seconds** with CUDA hidden for the CPU-only runner. Existing bitsandbytes/PyTorch JIT deprecation warnings
remain visible; no numerical check failed and CUDA stayed uninitialized.

A separate run with ambient PyTorch default dtype set to float64 passed all **27 tests in 4.65 seconds**; an explicit post-suite assertion verified float64
restoration. Master/module/test compilation and scoped whitespace checks passed after trimming the converter's extra instruction-page EOF blank. Independent
review and lead integration remain separate.

## Independent-review API repair and preserved widget limitation

Review found an imported-module `run_cuda_release()` NameError because its notebook-owned `learner_functions` assembly is deliberately under `MAIN`. A
regression first reproduced
that exact failure with backend delegation mocked, without CUDA or a model. A Python-only imported API now accepts an explicit `implementation` bundle (or an
explicitly supplied
notebook bundle in module globals); absence raises a clear `ValueError` before delegation. `make_learner_functions(**callbacks)` constructs/audits a
caller-supplied bundle lazily.
It neither defaults to reference callbacks nor treats generated module-owned solutions as learner evidence. Both legacy reference objects and module-owned
solutions callbacks are
rejected, while the strict identity/direct-delegation policy remains in force. That bounded policy is not semantic correctness or learner acceptance.

The original notebook wrapper and all 21 notebook code cells/outputs remain unchanged. Mocked tests compile only the original completed-notebook definitions
into an explicit
fixture namespace, verify exact callback identity and frozen 32-step/batch 12 delegation, and reject implicit/reference/module-owned substitution. No model,
CUDA or training
operation runs in these tests. The Python-only master-source addition changes its provenance hash; the case proposal's artifact/config fingerprints are
refreshed without changing
any scientific condition. Lead integration now adds guided-workbook text, so the generated build reports 7,412 prose words versus 7,371 before that shared link.

The exact archived original notebook already contains 25 widget-view progress outputs but no `metadata.widgets` state. Those outputs remain byte-preserved and
can display
loading-widget errors; missing states were not reconstructed or fabricated. This pre-existing progress-widget limitation is distinct from the four intact
scientific PNG outputs and
from the shared builder's preservation of widget states when originals actually provide them.

The read-only BF16 analysis and a proposed unexecuted, bounded post-lane preflight are in
[`bf16_parity_diagnosis.md`](../evidence/qlora_master_migration/bf16_parity_diagnosis.md). The scalar CPU spacing illustration is separately labeled in its JSON
sidecar. No scientific fix or threshold change is adopted, and the candidate remains unreleased.

Final post-review focused suite: **31 passed in 8.04 seconds**. The three API regressions preserve explicit callbacks and reject missing, legacy-reference,
module-owned and simple module-delegating bundles. The pre-existing widget-state limitation also has a preservation check. All 21 notebook code/AST/output/count
pairs remain unchanged; compilation and scoped whitespace checks pass.
