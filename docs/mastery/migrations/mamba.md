# 5.3 Mamba from Scratch canonical migration — 2026-10-16

## Plan before editing

**Current learner failure:** this complete existing lesson uses generated notebooks without a canonical master. Eleven historical solution dropdowns have malformed indentation.
Some class stubs omit constructors/helpers, and `step` is deliberately attached in the next exercise. Tests reach into a separate implementation for tiny input/config plumbing; the
final lazy release wrapper imports `solutions`, which would recurse after canonical generation overwrites that module. References lack a forecast/reflection boundary.

**Concrete question / target claim:** how do token-dependent state updates create a causal, cached Mamba-1 language model, and how does exact recurrent state retain a WRITE beyond a finite convolutional horizon? Preserve the full architecture, pinned-checkpoint parity and exact analytical memory demonstration without creating a new model claim.

**Signature result:** preserve the existing three-panel state trace, delay-versus-read-error and long-delay causal ablation with all-context attention and finite convolution
controls. Retain all four notebook PNG output objects, saved asset, old play figure, float32/float64 associativity findings, checkpoint outputs and widgets. No
model/GPU/training/pretrained/download run. A later strictly bounded CPU method subset must be proposed and reviewed first.

**Student implementations:** exact scalar discretization, chronological affine composition, sequential/parallel scan, causal depthwise convolution, token-dependent delta/B/C,
selective recurrent/parallel scans, RMSNorm, all mixer/block methods, standalone `mamba_recurrent_step` attachment, stacked tied LM/cache/greedy generation, strict key
mapping/weight load/parity. Preserve every original notebook method body and runtime statement. Supplied data/config functions needed by tests become visible plumbing. Preserve the
exact old module at `historical_release.py` and in the archive; change only lazy release routing so explicit release calls still reach the original implementation.

**Controls/baselines:** hand-computed scalar/affine states; exact conv loop/no-future perturbation; recurrent/parallel gradients and nonzero initial state; residual-zero and
full/cache parity; tied embedding identity; all 243 checkpoint tensors, 25 hidden states/logits/cache/greedy tokens; exact selective hold/write/erase, all-context attention
ceiling, K=8 finite horizon and state intervention. Keep all original parameters, seed/schedules, tolerances and revision `1e76775f628fbf1350fbe4dbb3d971ba64af25a1`. Keep analytic
memory distinct from learned behavior and historical CPU/CUDA evidence.

**Play / prediction:** retain the existing distance/kernel play and precision anomaly hunt. Add method-specific forecast/reflection pairs (confidence, causal reasoning, exposure,
controlled discrepancy and changed condition). Propose a tiny analytical state-intervention play using existing methods only. The guided proposal has exactly 20 adopted schema
keys, explicit inputs and no pre-commit target answers; root owns adoption.

**Files:** canonical `master_5_3.py`/generated master, current `part3_mamba_from_scratch` notebooks/module/tests/utils/media and exact historical-release copy;
`03_[5.3]_Mamba_from_Scratch.{md,py}`; focused migration tests, this note and scoped evidence. Because the canonical directory is the same physical directory, archive every
overwritten original exactly first. Leave original receipts/README/assets unchanged; never edit original ARENA, other sections, shared config or checkpoints.

**Verification:** sealed exact pre-edit archive with all section files/page/assets and source dependencies; record report-input currentness before overwriting. Compare each of 17
old code cells by normalized AST, nonblank source, output/count/widget identity, with declared signature-output path and lazy release import routing only. Every old prose passage
remains, repairing indentation while retaining historical solution text and adding executable generated answers. Complete every class method/step attachment from its own answer;
imports are inert. Build without `--execute`. Propose any new-cell CPU subset and fixture budget before running; no historical scientific runtime or model imports. Independent
review precedes guided adoption; full fresh model/CUDA execution and human learner acceptance remain separate.


## Stopping checkpoint after authorized CPU publication

Root authorized and the independent reviewer cleared one exact six-cell CPU
subset (30 seconds total) and one separate primitive fixture selector (30 seconds).
They succeeded once: subset 2.286342767 seconds, eight actual checks and genuine
control plot; fixture selector 29 passed in 3.11 seconds. No optimizer, training,
pretrained/model import, checkpoint loading, historical scientific cells or GPU.
The six genuine outputs/counts were atomically published with mixed provenance,
while all 17 prior complete cells and widgets remained identical.

The user then requested no new work today. No further tests/runs/features began.
Only static byte-identity/current-state receipts were finalized. All 23 code
sources, 17 historical cell objects, six actual output/count tuples and current
publication hashes were verified; both owned processes had exited. Final
post-promotion review and structural tests remain pending, as do guided proposal
wording reconciliation and root-owned adoption/navigation. Current details and
hashes are in `REPORT.md`, `stop_checkpoint.json` and `manifest.json`.


## October 19 resumed proposal repair

The strict twenty-key proposal now names the already completed genuine six-cell
CPU subset and mixed historical/fresh origins. Previous proposal/report/manifest
bytes are retained in `pre_resume_20261019/`. No canonical notebook, code cell,
output, widget, method, figure or numerical result changed. Fifteen selected
structural/proposal checks passed in 1.20 seconds (32 numerical/import selectors
deselected), with one thread, CUDA hidden and offline flags under a 30-second
external deadline. No study or primitive computation was repeated. Independent
post-promotion review is tracked separately before root adoption.
