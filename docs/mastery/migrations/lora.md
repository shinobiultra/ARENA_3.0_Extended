# 15.2 full LoRA comparison canonical migration

## Preservation plan

Current learner failure: the complete published comparison has no canonical ARENA master or durable prediction/reflection route. Its notebook-specific execution wrappers need explicit setup and figure routing.

Target section claim: on a pinned TinyStories codebook task, LoRA and full finetuning can reach similar behavior while inducing different weight spectra and activation drift. Their learning rates differ; this does not establish broad PEFT equivalence.

Signature result to preserve: the complete four-panel real-model comparison (behavior, singular spectra, dominant drift fraction and dominant-versus-random direction ablation).

Student methods: exact scaled B@A update; transparent LoRALinear; named injection and freezing; merge/unmerge parity; template split and random-label control; matched training loop; weight spectrum; activation capture/drift; causal projection ablation; same-norm random adapter.

Controls/baselines: frozen base, full finetuning, rank1/rank4 LoRA, random training labels, same-norm random low-rank updates, random direction ablation and unrelated protected next-token replay.

Play cell: rank, intervention layer and protected replay weight, changed one at a time with a forecast.

Files to modify: new master_15_2.py, canonical generated notebooks/module/page/media, migration tests, this note and exact archive/parity evidence. Preserve all legacy physical source/artifact bytes except generated solutions.py, whose old bytes must be recoverable from the archive. No deletions, model loads/training/downloads, GPU/cluster jobs or new scientific measurements.

Tests to run: archive/report-input identity, all original prose/code/figure/output parity with narrowly named setup/route corrections, self-completed student exercises, independent tiny CPU oracles, notebook schema and unrelated-directory import safety. No --execute build.

## Canonical artifacts and preservation

The canonical source is `infrastructure/chapters/chapter15_peft_misalignment/master_15_2.py`. Current configuration generates
`15.2_LoRA_vs_Full_Fine-Tuning_{exercises,solutions}.ipynb`, `solutions.py` and registered page `02_[15.2]_LoRA_vs_Full_Fine-Tuning.{md,py}`. The original
`15.2_LoRA_vs_Full_Finetuning_{exercises,solutions}.ipynb` and `02_[15.2]_LoRA_vs_Full_Finetuning.md` remain byte-identical. Original README, tests, lockfile, report, report
schema, expected outputs and historical PNG remain physically present. No original ARENA chapter 0–4 file was edited.

`published_before_master.tar.gz` archives every original section file, original instruction page, signature asset and every file named in the original report's `report_inputs`. All
those hashes were checked before writing. `before.json` records exact member and archive hashes. The unchanged report dated **2026-09-04T23:56:55+00:00** certifies its archived
publication/source; it does not certify the newly generated module's file hash. The only pre-existing file replaced is `solutions.py`, whose exact old bytes remain recoverable in
the archive.

The original full lesson has **62 cells, 31 code cells, ten exercises, twenty output entries across fourteen output-bearing cells, one figure and one print-only play cell**. All
original explanatory prose, mathematics, help, interpretations, expected results, solution bodies, controls and limitations survive. The old learning-objectives heading is
normalized to “Learning goals” because the canonical converter reserves its standard learning-objectives syntax; its nine original objective lines are unchanged. Standard headers,
objective blocks, exercise ratings and numbered summaries are additive.

All **36 original notebook function/class definitions** match the archived reference module. Canonical generated code preserves every original nonblank code line and AST after exactly two named setup/route adaptations:

1. Setup adds `import time`, which the old notebook's CUDA wrappers referenced without importing.
2. `run_reference_experiment` uses the already-defined `asset_dir / "lora_full_signature_panel.png"` as its fallback figure path. The old fallback depended on `__file__`, which is undefined in a notebook. Model, data, fitting and metric computations are unchanged.

Python-only root discovery uses the generated module's `__file__`; notebook root discovery remains unchanged. Original exercises keep `RUN_BOUNDED_CPU=False` while original
solutions keep `True`, expressed through canonical output filters. Generated Python uses `False`. `MAIN` guards protect test execution, global setup actions, data generation, model
execution, report reads, plotting and filesystem writes. The local `transformers` import stays inside `load_pinned_model`. The module imports from an unrelated directory without
model loads, CUDA queries, training, plots or filesystem writes. Its historical strict CUDA interfaces and standalone CPU smoke helper remain available.

`output_parity.json` maps **only historical code ordinals 0–30**. Each entry includes old/new source hashes, normalized AST hash, exact historical output hash, execution count and
any named adaptation. It retains all 31 old execution counts and all twenty old output entries. The new play's output is excluded from this historical map. The historical
four-panel PNG is byte-identical in `media/`, the original asset directory and embedded exercise output; it was inspected visually. These checks establish migration fidelity, not a
fresh real-model execution.

## Added runnable CPU play and its evidence boundary

The final generated lesson has **121 cells, 32 code cells, ten exercises and 6,234 prose words including guided-workbook navigation**. Twelve forecast/reflection pairs and nine
section summaries are additive. The original ten student exercises still complete from their own generated answer dropdowns. Forecasts ask for method-specific quantities,
confidence, causal reasoning and prior exposure; no learner values are prefilled.

The historical print-only play cell remains unchanged at ordinal 30. A new **“Try It Yourself — an exact CPU matrix adaptation”** is appended at ordinal 31. Its three wired knobs
are `PLAY_FIXTURE_RANK` (1–4), `PLAY_FIXTURE_LAYER` (synthetic site 0/1) and `PLAY_FIXTURE_PROTECTED_WEIGHT` (nonnegative penalty). These control the actual inputs to a visible
`lora_matrix_play` function. Seed is also explicit. Each site prescribes a different diagonal target spectrum. The rank-constrained quadratic solution shrinks the leading singular
components; visible `toy_lora_delta`, `LoRALinear`, `summarize_spectrum`, `dominant_activation_drift` and same-norm random-adapter methods produce the update, forward-path
comparison, spectra, drift energy and controls. No checkpoint or optimizer is used.

A separate visible hand-worked diagonal oracle immediately checks the result, forward-path identity, norm matching and objective inequalities. Norm matching uses tolerance
appropriate to the preserved helper's float32 norm calculation. The two-panel plot compares target/fitted/random spectra and matrix task error versus replay displacement for zero,
rank-constrained, unconstrained and same-norm random updates. Its synthetic site is not a measured transformer layer. Its matrix replay penalty is a displacement proxy, not
protected next-token NLL. The nearby forecast and reflection ask the learner to change one scientific condition and rerun the same cell.

Only this additive cell was executed on CPU after it was initially added unexecuted. Its exact run is recorded separately in `fixture_execution.json`: code ordinal 31, source hash,
all visible method dependency hashes, exact configuration, versions/dtypes, output hash, figure hash, measured metrics and explicit no-model-loader/no-optimizer fields. The
selected rank-2/site-0/penalty-.2 fixture completed in about **0.45 seconds**. The resulting second PNG was inspected; the random update has matching norm and visibly worse task
error despite a similar spectrum. A canonical rebuild preserved both the historical outputs and this separately identified fresh fixture output through its unchanged-code check.

## Scientific scope and proposed guided case

Keep the three old execution scopes distinct: the published notebook ran a **two-update CPU preflight** with batch size 4/seed 0; `reference_metrics.json` separately retains a
**sixteen-update CPU reference** with batch size 12; the historical CUDA report and four-panel signature used **96 updates, batch size 16 and seed 0**. None was rerun here. The new
analytic CPU play is a fourth, explicitly separate evidence class.

The pinned checkpoint is `roneneldan/TinyStories-1M`, revision `77f1b168e219585646439073245fe87e56b3023e`. LoRA targets layer-6 attention `q_proj` and `v_proj`; full finetuning
exposes all parameters. Training has thirty examples across three templates; the twenty evaluation examples use two disjoint templates but reuse subjects and marker tokens. LoRA
uses learning rate **0.02**, full finetuning **0.00005**, and the batching/initialization seeds differ by method. Rank-1/rank-4 LoRA, full finetuning, random-label rank-4 LoRA,
same-norm random adapter and frozen base remain explicit. Replay has default weight .2 on four protected next-token examples. These are scoped comparisons, not broad PEFT
equivalence or an equal-hyperparameter claim.

Proposed guided question: after the 96-update rank-4 run, what is the **dominant-drift ablation target-probability drop minus the seeded-random-direction drop** at layer 6,
averaged across the same twenty evaluation prompts? Freeze exact prompts/order/labels, checkpoint, adaptation sites, alpha=rank, rank, optimizer/budgets/seeds, replay weight and
ablation position. The random-direction generator is seeded 404 once; rank-4 consumes the second sequential draw after rank-1. The target probability normalizes over the two
red/blue answer tokens. The direction is estimated from uncentered tuned-minus-base drift on these same twenty prompts, so this is an in-sample descriptive intervention comparison,
not independent direction-discovery evaluation.

The historical rank-4 drop is .3329339027404785; the random-direction drop is −.00015628337860107422; their margin is **.3330901861190796 probability units**. Full finetuning,
random labels, same-norm random updates and the protected task remain comparisons. Accuracy alone misses these mechanistic differences. The original activation summary uses an
uncentered energy fraction, despite its historical “variance” field name.

Proposed real-model transfer: change protected replay weight **.2→0** while freezing task, subjects/templates, rank, sites, training budget and all other settings; forecast both
the ablation margin and protected next-token NLL change. The altered weight must be passed to the visible `train_on_codebook` call: changing the original printed
`PLAY_PROTECTED_WEIGHT` alone does not alter the hard-coded reference study. Because `run_reference_experiment` does not expose that replay-weight argument, any propagated
argument/default change needs a new exact source commitment before rerunning. This new real-model transfer remains unexecuted. The lead owns checkpoint/workbook integration;
forecast fields remain blank. The analytic play supplies an immediately runnable, separately scoped version of the task-fit/base-preservation tradeoff.

## Validation and remaining gates

Generation without whole-notebook execution:

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline python infrastructure/extension/build.py 15.2
```

Focused checks:

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv MPLCONFIGDIR=/tmp/arena-mastery-lora/mpl \
uv run --no-sync --offline pytest -q tests/mastery/test_lora_master_migration.py
```

Result: **22 passed in 5.35 seconds**. Twelve checks run existing independent tiny CPU semantic fixtures against the generated visible methods, including exact low-rank algebra,
merge parity, budget/splits, a tiny next-token training organism, spectral/drift ground truth and causal ablation. Eight migration checks cover archive/report hashes, all
historical source/output/count parity, 36-definition module parity, figure identity, own-answer completion, teaching/prediction preservation, import safety and notebook-wrapper
timing/routing with a mocked study. Two further checks cover all **24 rank/site/replay combinations** of the new play and its independent diagonal oracle, monotonic task/replay
tradeoff, invalid inputs, disabled model/training functions, exact execution receipt and separate fresh figure provenance. Repeating with ambient PyTorch dtype pre-set to float64
yielded **22 passed in 6.86 seconds**, with explicit dtype-restoration verification.

The tiny semantic training fixture is not pretrained-model finetuning. No pretrained checkpoint was loaded/downloaded/trained, no historical publisher/report was rerun, and no
GPU/cluster job was launched. Original reports were not rewritten. No fresh whole-notebook model run, new real-model transfer, scientific replication, human learner acceptance,
readiness promotion or whole-extension completion is claimed. All five 15.2 structural checks also passed; together with the focused suite, the final combined run was **27 passed
in 7.01 seconds**. Master, generated module and migration tests compile successfully. An independent reviewer reran all 22 LoRA checks successfully in 7.52 seconds and found no
scientific or additive-play blocker. A shared builder issue remains with historical widget metadata; the lead owns that repair and final integrated validation. Human learner
acceptance remains separate.

Proposed chapter description: **Build transparent LoRA adapters, compare matched task behavior with update spectra and causal drift controls, then explore an exact CPU task-preservation tradeoff.**
