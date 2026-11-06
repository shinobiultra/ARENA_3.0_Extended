# 13.2 full denoising-time causal patching master migration

## Scope and implementation plan

- **Current learner failure:** the complete published notebook bypasses the canonical authoring pipeline. There is no readable legacy builder/prose source; notebook and page are authoritative.
- **Target section claim:** recover a planted toy bottleneck and separately test time-localized latent-region color transfer in two pinned SD1.5 cases. The real result does not establish a UNet circuit.
- **Signature result:** preserve all four PNG outputs, the toy layer/time heatmap and controls, full real timestep curves, image examples and both play cells.
- **Student methods:** diffusion schedule/noising, object-world/denoiser, activation caching/replacement, reverse trajectory, regional metric, causal sweep, spatial latent replacement, explicit DDIM loop and real-case sweep.
- **Controls/baselines:** analytic endpoints; exact masks; matched-seed unpatched; equal-size channel/location/time/shuffled/untrained toy controls; pixel-copy upper bound; real wrong-time/wrong-region/scale-matched random-latent controls.
- **Play cells:** toy layer/time/channels/mask; real case/step/patch fraction/mix.
- **Files to modify:** new master, canonical generated notebooks/module/page/media, exact legacy reference-module copy required to preserve its pre-existing plumbing, migration archive/parity records, this note and focused tests. Shared routing/config/workbook are owned by the lead.
- **Tests to run:** exact archive/report-input inventory; all 16 code-cell AST/nonblank/output/count parity with the single explicit setup import-route adaptation; own-answer reconstruction; original teaching-line preservation; schema/import safety; tiny independent CPU fixture tests only, preserving ambient dtype/threads.

Original ARENA is untouched. Archive exact original section/page/assets before overwriting. Retain all legacy physical files; no cleanup, publisher, model loading/downloads, training, GPU, cluster, commits or staging. Build canonically without `--execute`. Historical outputs remain labeled as historical, and new learner forecasts/reflections remain blank.

## Canonical lesson and learner improvements

`infrastructure/chapters/chapter13_image_generation_interpretability/master_13_2.py` generates the entire published lesson as **62 cells, sixteen code cells, ten exercises and
6,925 words of prose**. Twenty-five active learner definitions include two denoiser classes. All thirty original notebook function/class definitions—including its three report
wrappers—have identical ASTs. The original prose, equations, help/common-bug/solution/expected-output dropdowns, controls, play instructions, anomaly hunts, readings and claim
boundaries survive. The sole renamed prose heading is `## Learning Objectives` → `## Original learning goals`, because the canonical converter reserves the former phrase for
standard objective blocks. Exercise metadata received YAML fences without removing its original text.

The format adds the ARENA header, standard objective blocks, two numbered parts, summaries and bonus boundary. Four method-specific forecast/reflection pairs precede the toy
bottleneck sweep, toy layer/time/channel/mask play, real SD1.5 sweep and real patch-fraction/mix play. The real forecast precedes the original answer-bearing signature prose.
Forecasts request numerical values/ranges, confidence, causal reasons and exposure; reflection requests actual controls, discrepancies, alternative explanations and
changed-condition forecasts. No learner response is prefilled. Reference outputs make this guided practice, never an unseen-scored challenge.

Both play cells remain. The toy play changes layer, timestep, channels and mask shift, displaying donor, recipient, patched image and regional metric. The real play changes case,
call index, mask fraction and donor mix; its output is an actual historical image, not a new numeric measurement. The new prose asks learners not to infer an unmeasured CLIP score
from that image. All four retained figures were inspected as a contact sheet. The real outputs visibly include patterned backgrounds and non-geometric details; the master now
explicitly says passing the color-transfer contrast does not establish prompt or shape fidelity.

## Exact preservation and evidence boundary

[`published_before_master.tar.gz`](../evidence/denoising_master_migration/published_before_master.tar.gz) captures **sixteen exact pre-migration files**, including the original
notebooks, reference module, readable page, both saved signature assets, report inputs and global verification sources named by the historical report.
[`before.json`](../evidence/denoising_master_migration/before.json) records every archive/file hash and confirms all thirteen original report-input hashes matched at the snapshot.
Those original hashes stay in the unchanged September 4 report; they certify the archived publication, not regenerated canonical bytes.

The old `solutions.py` is physically retained, byte for byte, as `legacy_reference.py`, in addition to its exact archived copy. It retains pre-existing data records, constants,
plotting, toy-training/model-loading plumbing and report implementations. The generated module and notebook contain the taught implementations. The one explicit notebook setup
adaptation changes `import solutions as reference` to `import legacy_reference as reference`, avoiding a self-import after the canonical module replaces `solutions.py`. Python-only
`__getattr__` preserves old optional plumbing/report APIs for external callers; definitions actually implemented by the notebook take precedence. The original notebook wrappers—not
their different legacy-module counterparts—remain unchanged and delegate through `reference` exactly as before. Existing trained-toy, SD1.5 and report entry points were never
invoked.

All **sixteen code cells retain identical ASTs and nonblank lines after that single named import-route adaptation**. No method, parameter, model revision, metric, training loop,
control, cohort or selection rule changes. Python-only root discovery uses `__file__`; `MAIN` guards protect import-time plotting, tests, training, model loading, CUDA access and
experiments, while the converter removes those guards from notebook code. Importing from an unrelated directory performs none of those side effects and imports neither Diffusers
nor Transformers.

[`output_parity.json`](../evidence/denoising_master_migration/output_parity.json) records old/adapted/generated code hashes, AST hashes, cell indices, output hashes and counts. All
**sixteen historical execution counts, thirteen output-bearing code cells and 26 outputs** are retained directly from the archived published notebook. A repeat canonical build
passes its strict same-code output carry-forward check. The new forecasts and reflections have no execution or learner-response claim. **This is retained historical reference
evidence, not a fresh canonical execution, training run, pretrained-model validation or scientific replication.**

The four canonical `media/` PNGs are exact base64-decoded bytes from the historical notebook's four image outputs. The original two saved instruction assets remain exact at their
old paths and in the archive. They are not byte-identical to the inline PNGs: for example, the toy inline figure is 1411×811, while its saved asset is 2540×1460. The preserved
plotting code saves at a different raster resolution. We retain both encodings and do not assert byte equality or manufacture one from the other.
[`figures.json`](../evidence/denoising_master_migration/figures.json) hashes the four inline-derived canonical media; `before.json` separately hashes both saved assets. The
original prose asset links also remain intact.

## Keep the two claims separate

The toy world contains every combination of two shapes, three colors, two rows and two columns. Its 16×16 RGB denoiser has a planted object-feature bottleneck, an exact regional
mask, eight diffusion steps and a 350-update historical training run at seed zero. Shared noise seed 2026 supplies matched clean/corrupt trajectories. Its layer/time sweep and
equal-size channel/location/time/shuffled/untrained controls test whether the method finds a mechanism deliberately constructed to exist. Pixel copying remains an exact recovery
upper bound. This result is not pretrained-model evidence.

The separate real study uses SD1.5 revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` and CLIP ViT-B/32 revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. It retains same-seed
red-into-blue-square (seed 4) and blue-into-red-circle (seed 2) cases, exact prompt/negative-prompt strings, 512×512 generation, twenty DDIM calls, guidance scale 9, deterministic
eta zero, a centered half-height/half-width mask, and one-time donor-latent replacement. Every one of the nineteen noninitial calls is scored. The best call is selected on the same
recorded curve; it is not a held-out choice.

CLIP recovery calibrates a donor-minus-recipient text/image cosine margin as `(patched - recipient) / (donor - recipient)`. It can disagree with pixel recovery/selectivity, and
need not be bounded to [0,1]. Wrong-time, displaced-region, donor-scale random-latent and unpatched controls remain beside the result. The historical best recoveries are about .654
and .505 at call indices 2 and 4; regional selectivity at those CLIP-selected calls is about .222 in each case. Strongest-control margins are about .194 and .213. The old .10
CLIP/.05 regional/.05 control-margin gates and measured 3.235 GiB peak remain historical, with no readiness promotion.

This is a scoped latent-state color-transfer result for two selected cases. It does not localize a real UNet circuit, channels or neurons; establish faithful simple-shape
generation, broad prompt transfer or human perceptual quality; or claim a universally decisive timestep. The patterned outputs and differences between early CLIP peaks and later
pixel recovery remain inspectable rather than being replaced by prettier samples. No software, CUDA, precision or performance parameter was silently upgraded during preservation.

## Frozen guided-case proposal

[`guided_case_proposal.json`](../evidence/denoising_master_migration/guided_case_proposal.json) freezes a numeric question with exact artifact/config hashes and blank
learner-response fields. Predict case-0 calibrated CLIP recovery at fixed call index **2**, then change only denoising time to fixed call index **19**. All prompts, seed, initial
noise, scheduler, mask and mixing stay fixed. The first call is explicitly the historically selected peak; the late call is the original wrong-time control. This changes a
scientific intervention coordinate rather than merely resampling a seed. Both measurements are historical and output-exposed; there is no independent TEST or unseen transfer claim.

Exact reference paths are `metrics.gpu_test.sd15.cases[0].clip_recoveries[1]` and `[18]` in the retained `verification_report.json`. The learner should compare the complete
recorded curve, regional selectivity and all controls, not replace them with a single attractive image. The lead owns checkpoint/workbook integration and its
commitment/stale-input/provenance/portable-record checks. This lane proposes a source-backed case without accepting it on the user's behalf.

## Validation and preservation exceptions

Canonical generation ran without `--execute`:

```bash
UV_CACHE_DIR=/tmp/arena-mastery-uv uv run --no-sync --offline python infrastructure/extension/build.py 13.2
UV_CACHE_DIR=/tmp/arena-mastery-uv MPLCONFIGDIR=/tmp/arena-denoising-migration/mpl \
  uv run --no-sync --offline pytest -q tests/mastery/test_denoising_master_migration.py
```

The first twenty-four focused checks passed in 4.46 seconds. They cover the archive/report inputs; exact legacy retention; sixteen code AST/nonblank/output/count pairs; four exact
output PNGs; all ten own-answer exercises including both active class stubs; all original teaching lines with the named heading adaptation; all thirty authoritative notebook
definitions/wrappers; unrelated-directory import safety; fourteen existing independent CPU tensor/contract oracles; a two-step fake-pipeline oracle for CFG batch ordering,
deterministic scheduler updates and a single selective patch; and detached time/layer hook-cache identity. No trained model or training fixture is used. The added guided-case
integrity check verifies source/config hashes and exact reference paths without filled learner answers. Numeric fixtures save and restore ambient PyTorch dtype and thread count.

All original section files remain physically present except the overwritten generated publication files, whose exact bytes are archived; the old module also survives as
`legacy_reference.py`. [`preserved_legacy_inventory.json`](../evidence/denoising_master_migration/preserved_legacy_inventory.json) gives the exact retained names/hashes requiring
the structural preservation exception: `README.md`, `artifacts.lock.yml`, `expected_outputs/**`, `verification_report.json`, `verification_report.schema.json`, and
`legacy_reference.py`. Original `tests.py` remains byte-preserved. The original instruction assets remain untouched. No cleanup, staging, commits or pushes occurred.

Fresh full-notebook execution, current pretrained dependency compatibility, new scientific replication, learner walkthrough and human acceptance remain separate open gates. The canonical entry point is `infrastructure/extension/build.py 13.2`. No whole-course completion or user acceptance is implied.

Proposed chapter description: **Build time- and region-selective denoising interventions, validate them on a planted toy bottleneck, and inspect two pinned SD1.5 color-transfer curves with explicit controls.**

The full expanded focused suite passes **25 tests in 4.17 seconds**. A separate ambient-float64 run is recorded after the fixture explicitly reproduces the historical float32
module-import default and restores the caller's dtype: the original module creates its `COLOR_VALUES` constant during import, so importing under float64 before setting a numeric
fixture's default exposes a dtype mismatch in the old world-rendering oracle. This is a test-environment boundary, not a change to the preserved scientific source. Compiled
master/module/tests and scoped whitespace checks pass after trimming the converter's extra page EOF blank.

Lead integration decision: the proposal JSON remains the historical review input. Since it exposes call 19 as a primary control, the lead instead chose an **unmeasured** changed
condition of `patch_mix=1.0` → `.5` at the same frozen call 2, with a new input commitment in the adopted workbook case. The visible `run_sd15_latent_trajectory` and
`apply_latent_patch` support that knob directly. `run_sd15_patch_case` does not expose `patch_mix`, so the changed case must use the visible trajectory and CLIP/regional scoring
rather than rerun/reselect the sweep. The existing random-latent branch ignores `patch_mix` and performs full donor-mean/std-matched replacement; it must remain labeled a fixed
full-replacement control, not a dose-matched `.5` random control. No monotonic CLIP-response claim or new changed-case measurement is asserted here. The adopted shared
manifest/workbook is owned by the lead.

The final separate float64 run passed all **25 tests in 3.84 seconds**, and an explicit assertion verified the ambient default remained float64 after the suite. `py_compile` and scoped `git diff --check` also passed.
