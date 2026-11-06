# Draft preservation preflight: canonical 6.5 from complete legacy 6.4

Status: future planning only; no adoption or implementation. Static snapshot 2026-10-19T04:08:20.536004+00:00. This lane writes only this plan. No lesson generation, taught
imports, methods, tests, training, models, datasets, caches, network or native runs occurred.

## Learner claim and scope

**Current learner failure:** complete legacy crosscoder teaching remains at the old 6.4 path while the configured 6.5 destination has no adopted master. Preserve the whole lesson
rather than reducing it to a reference report or helper imports.

**Target preserved claim:** a learned sparse crosscoder on exact planted paired activations can be examined through reconstruction, latent ownership, behavior prediction and
targeted intervention, with held-out examples and distinct negative controls. These are retained historical outputs, not freshly reproduced findings.

**Configured breadth remains open:** the current brief calls for acausal crosscoders across layers and real base/chat models, decoder norm ratios, complete shrinkage/latent
decoupling, BatchTopK, and causal validation on a real base/fine-tuned pair. The existing learner body is a tied-encoder sparse crosscoder on a synthetic planted organism. Its
optional GELU-1L/SoLU-1L residual-delta preflight is not a full real-model learned-crosscoder replication. Existing synthetic ownership/norm reasoning cannot establish all those
broader requirements. Keep the configured title but explicitly disclose these coverage gaps; add no new science to a preservation migration.

## Verified source surface

Fresh static counts: **32 cells, 15 code cells, 13 output-bearing code cells, 4 embedded PNG outputs**. All 17 original markdown cells, both notebook metadata objects, code
execution metadata/counts, and every output must survive. Student class methods and function bodies must remain genuinely student-implemented, not filled with reference answers.

The signature has four separate existing PNG panels: training/reconstruction curves, latent ownership heatmap, behavior AUC baselines, and causal intervention bars. Top held-out
examples and the play include visible tables/text; do not discard these because they are not PNGs. Two extra historical SVG assets also exist and must remain untouched.

## Exact canonical routing and protected legacy identity

| Surface | Exact mapping / preservation |
|---|---|
| Legacy source directory | `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing` remains intact. |
| Canonical destination | `chapter6_sparse_feature_methods/exercises/part5_crosscoders/` |
| Config-derived notebook basename | `6.5_Crosscoders_and_Model_Diffing` (both `_exercises.ipynb` and `_solutions.ipynb`) |
| Proposed master | `infrastructure/chapters/chapter6_sparse_feature_methods/master_6_5.py` and generated master notebook |
| Config-derived instruction route | `chapter6_sparse_feature_methods/instructions/pages/05_[6.5]_Crosscoders_and_Model_Diffing.py` and `.md` |
| Existing legacy page | `chapter6_sparse_feature_methods/instructions/pages/04_[6.4]_Crosscoders_and_Model_Diffing.md` remains intact. |
| Legacy report/protocol/schema | Keep original bytes at original paths. Any canonical copies must use `historical_verification_report.json`, `historical_artifacts.lock.yml`, `historical_verification_report.schema.json`; never use active canonical discovery names. |
| Supporting fixtures | Keep legacy `expected_outputs/*` unchanged; any copies explicitly retain legacy regeneration instructions and identities. |

Canonical 6.4 now means Transcoders, while legacy 6.4 means Crosscoders. Route and source identity must be resolved by full path and subject, never a bare numeric ID. Do not
overwrite the legacy page, canonical transcoder master, or current 9.3 work. Root owns registry/workbook/syllabus/config and adoption status.

## Preserve the complete teaching sequence

| Zero-based original cells | Inline method and immediate checks |
|---|---|
| 0–2 | Complete introduction/objectives/setup; imported typed records and ROC-AUC primitive are visible dependencies, not replacements for the taught crosscoder. |
| 3–4 | `make_train_heldout_split`, `_orthonormal_rows`, `make_planted_crosscoder_data`; independent planted-ground-truth check. |
| 5–6 | `SparseCrosscoder` class with student `__init__`, `encode_model_a`, `encode_model_b`, `forward`; `sparse_crosscoder_loss`; shape/loss check. |
| 7–8 | `evaluate_sparse_crosscoder_reconstruction`, `train_sparse_crosscoder`; held-out reconstruction check. |
| 9–10 | `latent_ownership_table`, `_columnwise_abs_correlation`, `match_learned_to_planted_features`; specificity and planted-owner tests. |
| 11–12 | `signed_auc_binary`, `model_behavior_delta_from_reconstruction`, `ablate_latents`, `shared_only_reconstruction`, `svd_delta_direction_scores`, `svd_pair_reconstruction_mse`, `behavior_baseline_table`; baseline contract. |
| 13–14 | `crosscoder_intervention_report`, `top_activating_examples`; targeted versus same-norm/orthogonal checks. |
| 15–16 | Complete `run_crosscoder_signature_result`, `run_smoke_test` and historical main study composition; module-level guards required. |
| 17–26 | Four plotted panels and top-example table; preserve all outputs and interpretation prose. |
| 27–28 | Full Try It Yourself: `play_l1`, `play_seed`, `play_latents`, `play_steps`, separate play data and training, ownership/intervention report and baseline table. |
| 29–30 | Parent GPU contract and optional wrappers bound to unchanged legacy solutions; no automatic call. |
| 31 | Anomaly hunt, optional real paired-model context and full limitations. |

Seven original test calls explicitly pass student functions/classes; preserve their arguments rather than calling reference-only defaults:

- `tests.test_make_planted_crosscoder_data_has_exact_ground_truth(make_planted_crosscoder_data)`
- `tests.test_sparse_crosscoder_forward_and_loss_shapes(SparseCrosscoder, sparse_crosscoder_loss, make_planted_crosscoder_data)`
- `tests.test_train_sparse_crosscoder_reduces_heldout_reconstruction(train_sparse_crosscoder, evaluate_sparse_crosscoder_reconstruction, make_planted_crosscoder_data)`
- `tests.test_latent_ownership_table_classifies_specificity(latent_ownership_table)`
- `tests.test_feature_matching_recovers_planted_owners(train_sparse_crosscoder, match_learned_to_planted_features, make_planted_crosscoder_data)`
- `tests.test_behavior_baselines_include_real_controls(train_sparse_crosscoder, match_learned_to_planted_features, behavior_baseline_table, make_planted_crosscoder_data)`
- `tests.test_targeted_ablation_beats_same_norm_and_orthogonal_controls(train_sparse_crosscoder, match_learned_to_planted_features, crosscoder_intervention_report, make_planted_crosscoder_data)`

## Controls, play and prediction opportunity

Keep zero reconstruction, shared-only latents, dense SVD direction/reconstruction, label-shuffled behavior, same-norm random intervention and orthogonal intervention distinct.
Reconstruction, ownership, predictive ranking and causal intervention are different claims. Preserve train/held-out roles and the planted-owner oracle rather than picking a
successful latent/seed after inspection.

An answer-free exposed guided opportunity: before opening the historical intervention panel, predict whether removing the planted model-B-matched learned latent reduces the defined
behavior delta more than both matched controls; explain why low reconstruction error or a good predictive rank alone cannot settle this. Record confidence, reasoning and prior
exposure, leaving observation and control comparisons blank. A transfer prompt can ask how a documented existing play setting might affect ownership versus reconstruction, but it
must remain an unexecuted exploratory forecast unless separately authorized. Do not compute an answer or promise that every seed/width achieves the same outcome.

If root later wants a standard registry case, read the exact implementation metric and fixed displayed cohort/configuration before defining its binary predicate; do not invent a
normalization or reuse a similarly named metric. Bind the case to permanent legacy paths and source hashes below, not a temporary packet receipt. Published source/figures make this
historical exposed guided practice, never blind assessment or untouched TEST.

## Import, conversion and protocol hazards learned from 6.4

1. Setup currently calls `asset_dir.mkdir(exist_ok=True)` at top level. Keep any directory creation behind `if MAIN` in generated Python, alongside all study/test/plot/training
   calls. Notebook setup can remain the declared learner operation; importing canonical solutions must not create files, run training, read a report or trigger data/model access.
2. Code cell 16 invokes 500 training steps on the main planted data; cell 28 invokes a separate 300-step play, and several immediate tests train. Source-only checks must never
   execute these cells or call tests merely because their names contain smoke/unit. Optional cell 30 must stay definitions-only and call `part4_crosscoders_model_diffing.solutions`
   explicitly.
3. Setup currently sets `section_dir` to legacy 6.4, but the notebook contains no `read_text` report read; optional real context delegates through the legacy module. Keep a named
   `legacy_section_dir` for any historical lookup and a separate canonical/media directory. Do not accidentally make legacy helper paths point to canonical generated helpers.
   Shared typed dataclasses and `roc_auc_binary` are imported from `arena_ext.crosscoders`/`arena_ext.features`; preserve their hashes and semantics.
4. Student `SparseCrosscoder` has nested class-method stubs, including `__init__` retaining `super().__init__()`. A top-level-function-only EXERCISE/SOLUTION generator would leak
   complete class answers. Mark methods recursively or use exact whole-class student/solution regions, then compare student AST with the original class. Preserve every immediate
   own-function test.
5. Emit the converter-required exact `## Content & Learning Objectives` structure. Preserve the legacy separate objectives prose under a distinct heading such as `## What you will
   implement`; do not drop it or trip converter metadata parsing.
6. Existing worked-answer dropdowns should become `Historical worked solution` while preserving their complete bodies; newly generated answer dropdowns remain distinct. Otherwise
   the converter can consume a subsequent original teaching cell as a solution placeholder. Require full-content containment for all 17 original markdown cells, after declared
   heading/whitespace-only transformations.
7. Retain both original notebook metadata identities separately, especially solution Python 3.14.2 versus the student notebook metadata. Copy original code-cell execution counts
   and metadata with the historical outputs. Output preservation alone is insufficient.
8. Preserve source AST for every original code cell except a precise reviewed setup routing/import guard delta; test that setup inverse exactly. Copy and compare all four PNG
   bundles as bytes, not merely figure counts. Keep JSON readable and run the existing readability audit against a strict candidate-file allowlist.
9. The legacy report has `accepted=true` and a legacy 6.4 contract. Never publish its unchanged bytes as canonical `verification_report.json` or copy old `artifacts.lock.yml` into
   active canonical discovery. Preserve historical bytes under clear historical names or link the unchanged originals. Do not fabricate a canonical accepted/failed result.
10. Retain only permitted candidate files in a temp packet if implementation is later assigned. Root applies exact hashes after independent review. A converter/readability pass is
   structural evidence, not a learner walkthrough or acceptance of expanded crosscoder breadth.

## Historical protocol and provenance

Report date `2026-09-04T13:23:21+00:00`, recorded commit `34da1071`. The lock identifies planted dataset seed 6404, training seed 2, 1024 examples, 12 dimensions and train fraction
0.75; preserve these historical protocol values rather than silently retuning. Main notebook and play are separate conditions. The lock also retains legacy `REQUIRES_GPU: true` for
its historical parent contract despite the primary synthetic learner study being CPU; explain this distinction rather than rewriting history.

Optional paired-model pins: GELU-1L `bddc0e332f0ae84279e6a6a45d91b314899e1603`; SoLU-1L `a4ce32db5e35f13e5f09333888bd2d42660f77ce`. These were read from the lock only; neither
checkpoint nor any cache was accessed. No dedicated learner-run archive/protocol file was found in the immediate source tree beyond the listed report/lock/fixtures.

Current read-only report-input comparisons:

| Input | Hash comparison |
|---|---|
| `arena_ext/crosscoders.py` | match |
| `arena_ext/features.py` | match |
| `arena_ext/verification.py` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/6.4_Crosscoders_and_Model_Diffing_exercises.ipynb` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/6.4_Crosscoders_and_Model_Diffing_solutions.ipynb` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/README.md` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/artifacts.lock.yml` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/expected_outputs/README.md` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/expected_outputs/reference_metrics.json` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/expected_outputs/smoke_test.json` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/solutions.py` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/tests.py` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/utils.py` | match |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/verification_report.schema.json` | match |
| `chapter6_sparse_feature_methods/instructions/pages/04_[6.4]_Crosscoders_and_Model_Diffing.md` | match |
| `scripts/run_extension_verification_reports.py` | match |

All matches establish identity only; no numerical result revalidation was performed. No crosscoder text hits were found in current `infrastructure/chapters/**/master*.py` during
this scoped scan, so no already-adopted crosscoder master duplication was detected. This is not proof that no related concepts exist elsewhere. Original ARENA remains protected
append-only.

## Exact file hashes

All paths below are relative to `/home/bitwise/Documents/ARENA_3.0_Extended`; append that root for physical paths.

| File | SHA-256 |
|---|---|
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/6.4_Crosscoders_and_Model_Diffing_exercises.ipynb` | `6d2cca1ba395f8dd7f9a2b6bce6104d8786bb1a3f99a093bb883f78c56c371c7` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/6.4_Crosscoders_and_Model_Diffing_solutions.ipynb` | `dee173c9126170dc5b12a33d1983d6929398bd5105e791c23900687e2a23a482` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/README.md` | `c543de9ac302a776e831567c4e25cc25dccf66894c70e03e219e1b872931e24b` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/artifacts.lock.yml` | `45e7758106cf885eccfe66cdb593d168204c72d798fc1d74ecb510c03337f531` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/expected_outputs/README.md` | `01f0ecdc267b77c998978c275e19d948c854a6be38a970efc769a18189d7b3cd` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/expected_outputs/reference_metrics.json` | `ee703de9fdc52551f6cbb5eabdd651d129ef81e79a36cd8e5ee2bd870f1ce368` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/expected_outputs/smoke_test.json` | `2657a39592878be5a561c25d7edfba42eba5ba8ecbb80bfd6fe517f60ad2e84c` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/solutions.py` | `72be92927953612ba6e52785ac5887478787ca7ff34d7a79becaa3f8296edb16` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/tests.py` | `5648f3c673f335172ed2e6545e284a15d7edfc574ad08728b97eb571b72d1df3` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/utils.py` | `c3b2b7fe8266a5fa8c12e0fd8acfd1a06652c8b458a3fd7fd692979cdab8db52` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/verification_report.json` | `2fb0f086d774200ae7106810df297bd71b866c15403ba64ac34df52a6c7ee11b` |
| `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing/verification_report.schema.json` | `ecd833883885077ad5d7a0e239d984d01462255e94266212cf195ab4d4e27b3d` |
| `arena_ext/crosscoders.py` | `7c591605434e342a4ab2ab6cbeb7d0b6281df524610783dfdd7733457fd4e809` |
| `arena_ext/features.py` | `174023a8232b6ed60bc97c2894046485c79ff1d27136ee4e57dfc837dcb6e948` |
| `chapter6_sparse_feature_methods/instructions/pages/04_[6.4]_Crosscoders_and_Model_Diffing.md` | `f98f00fbbbc9fdff816a075321e913c5654401e7c2553880a8d8f1649088b607` |
| `chapter6_sparse_feature_methods/instructions/assets/crosscoder_behavior_auc_baselines.png` | `d4bd583eb0336deb9aabbb7bbe9119657120f257cef43234c8178e569eb60ef8` |
| `chapter6_sparse_feature_methods/instructions/assets/crosscoder_direction_control_result.svg` | `2a579611f2d9b40d6ca46e8ce55b479de7d7317bcf928b865e6c03b3403e5fc6` |
| `chapter6_sparse_feature_methods/instructions/assets/crosscoder_intervention_bars.png` | `0b9de3f431182e15b4e39bc3fb07417286eea8298718021287317f7563401b9b` |
| `chapter6_sparse_feature_methods/instructions/assets/crosscoder_latent_ownership_heatmap.png` | `451c983f8df9c946c9ad10985170ae84c9deab874dc90da41318cc8e0c7c8744` |
| `chapter6_sparse_feature_methods/instructions/assets/crosscoder_model_diffing_ladder.svg` | `e5cb692c26dbe986fd3b0ac58c25e5d98812a593bb92eba49dc8b169d5a5c881` |
| `chapter6_sparse_feature_methods/instructions/assets/crosscoder_training_reconstruction_curves.png` | `1efc2b51ef68fe0957b3db31c57529a269944b1e8a9336335e84889460f66999` |
| `infrastructure/core/config_extension.yaml` | `2b56e597201cac5fa6027ec052687b06ecc33c34bba4fa3bd605af9c1fda2364` |
| `infrastructure/extension/syllabus.yaml` | `450dd97835613ac627d97431b37724592a014400f13cfb6749a8dc50b65938f1` |

## Root decisions before any implementation

- **CC65-01:** preserve the full planted learned-crosscoder curriculum with explicit real-model/across-layer/BatchTopK gaps; no new science or requirement-completion claim is
implied by this plan.
- **CC65-02:** choose a source-only migration packet and separate review, preserving the complete class exercise and all historical material. Any training-bearing check requires
separate authority; source-only static compilation and metadata/AST/output identity checks cannot execute the taught method.
- **CC65-03:** keep report/protocol helpers historical and explicit. Root owns shared routes/registry/workbook/status and any future guided case admission.

Finish boundary: changed only this plan; original ARENA and legacy materials untouched. Student-facing changes, figures, controls and play described here are future preservation
work. Tests run: none. Remaining limitations: expanded brief gaps, fresh numerical verification and human learner acceptance remain open.
