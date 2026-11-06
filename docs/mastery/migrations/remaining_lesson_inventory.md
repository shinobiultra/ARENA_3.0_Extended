# Remaining lesson inventory — 2026-10-16

Snapshot of the shared working tree before today's migration work. The complete exact-path mapping, file hashes, source receipt checks and notebook counts are
in [the JSON inventory](../evidence/2026-10-16-lesson-inventory.json). Paths in that file are relative to its explicit repository root. This audit changed only
these two audit artifacts; it did not execute models, training, notebooks, publishers or tests. Original ARENA content was untouched.

## What is actually missing

The previous checkpoint's **55** figure counts canonical destination directories without a solution notebook. It does not mean 55 lesson bodies are absent.

| State at snapshot | Count |
|---|---:|
| Syllabus sections | 86 |
| Canonical masters present | 19 |
| Existing destination notebooks without a master | 12 |
| Empty canonical destinations with legacy source notebooks | 32 |
| Empty canonical destinations without declared source notebooks | 23 |
| Unique source notebooks for the 44 source-backed, masterless sections | 51 |

The 2026-10-15 snapshot's 58 absent exact registered filenames remains a separate historical count: 11.5, 15.2 and 17.1 already have substantial notebooks whose
legacy names differ from the registry. Multiple old sections can map to one new section, and one old section can be reused. A numeric section ID alone is
unsafe: legacy 8.5 is Sparse Feature Circuits, while current 8.5 is Causal Scrubbing.

Counts establish material availability, not scientific acceptance or coverage. All 51 candidates contain substantial source and stored outputs, but only the
seven detailed candidates below received the deeper migration checks in this audit. A larger canonical brief may add methods that the legacy lesson never
taught.

## First independent migration batch

Use separate ownership for **11.3, 11.5, 11.4 and 17.1**. Capture historical sources and outputs before generation; add masters and routes without implying
fresh scientific execution. These four lanes do not require new model runs for structural migration.

| Candidate | Cells / code / output-bearing code | PNG figures | Exercises / plays | Evidence and migration boundary |
|---|---|---:|---|---|
| [11.3 DAS](../../../chapter11_representation_geometry/exercises/part3_causal_abstraction_das) | 63 / 32 / 29 | 12 | 9 / 3 | Own-answer AST and archived execution outputs match; learner 15/15 and fresh 7/7 source hashes match. Preserve existing builder/prose. |
| [11.5 CCS](../../../chapter11_representation_geometry/exercises/part5_contrast_consistent_search) | 65 / 32 / 30 | 15 | 9 / 2 | Own-answer AST and archived execution outputs match; learner 13/13 and fresh 14/14 hashes match. Preserve sentiment scope. |
| [11.4 erasure](../../../chapter11_representation_geometry/exercises/part4_concept_erasure) | 52 / 24 / 23 | 12 | 8 / 2 | Learner 11/12 hashes match: only historical `lesson.md` differs. Fresh 13/13 and executed AST/output parity match. Preserve this prose-revision boundary. |
| [17.1 checkpoint archaeology](../../../chapter17_training_dynamics/exercises/part1_checkpoint_archaeology) | 35 / 14 / 13 | 2 | 6 / 1 | All historical report input hashes match. Migrate notebook/page directly and preserve its unique preflight wrapper. Finite mod-13 evaluation does not establish grokking or OOD generalization. |

For 11.3–11.5, all published notebook/page hashes and all 39 instruction-figure hashes match learner receipts. Each
`expected_outputs/full_own_answer_v*/execution.tar.gz` archive and every manifest-listed member were checked; archived executed code ASTs and outputs match
published notebooks. All six external learner/native cache directories named by receipts are absent, and archives contain no `.pt` tensors. Therefore
visible-output preservation is available, while full historical tensor revalidation is not available from these archives. Do not invoke the old publishers: they
require the absent external artifacts/reviews.

The three geometry lessons have `lesson.md` and `build_notebooks.py`. Their current lesson-generated student sources exactly match the exercise notebooks, and
the completed-from-dropdown ASTs match solutions. Existing Markdown pages need generated Python route wrappers. Preserve DAS random-subspace/shuffled-target
controls and three plays; erasure matched-random edits, nonlinear counterexample and two plays; CCS corrupted-pairing/random-direction controls and two plays.

## Next candidates

- [15.2 LoRA versus full fine-tuning](../../../chapter15_peft_misalignment/exercises/part2_lora_vs_full_finetuning): 62 cells, 31 code, 14 output-bearing cells,
  one PNG, ten exercises and one play. All report input hashes match and all 36 notebook definitions match `solutions.py`. No named builder was found; migrate
  the notebook/page, preserving explicit learning-rate differences, spectra/drift/ablation result and controls.
- [13.2 denoising-time patching](../../../chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching): 43 cells, 16 code, 13
  output-bearing cells, four PNGs, ten exercises and two plays. All report input hashes match. Preserve notebook-specific `run_smoke_test`, `run_gpu_test` and
  `run_full_experiment`; 27/30 definitions match `solutions.py`. Keep widget metadata and the separate toy versus SD1.5 latent-state color-transfer claims.
- [Canonical 16.4 SAGE from legacy 16.12](../../../chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance): 80 cells, 33 code, 31
  output-bearing cells, 18 PNGs, nine exercises and two plays. Learner/fresh source hashes match 15/15 each; generated learner source and completed-answer AST
  match published notebooks. This needs careful renumbering, import/link updates and preservation of the legacy route. Full archive contents were not audited in
  this lane.

The 13.2, 15.2 and 17.1 reports date to 2026-09-04. They have no per-section learner-execution receipt/archive and are not covered by yesterday's general
output-preservation manifest. Capture new preservation receipts before structural migration; do not rewrite old report provenance to suggest a new execution.

## Remaining routing work

Twelve destination directories already contain notebooks but lack masters: **0.6, 1.6, 5.1, 5.3, 6.1, 11.3, 11.4, 11.5, 13.2, 15.2, 15.3 and 17.1**. The 32
empty destinations with existing legacy material are mapped below. The JSON contains every exact notebook, page, builder and receipt path; old IDs below are
only compact labels.

| Canonical section | Existing source notebook ID(s) and subject |
|---|---|
| 5.2 | 5.8 Mixture of Experts from Scratch |
| 5.4 | 5.7 Mamba-2 and SSD |
| 5.5 | 5.4 Mamba State Tracking |
| 5.8 | 5.5 Diffusion Language Models |
| 5.9 | 5.6 Multimodal Embedding and Function Calling Models |
| 6.2 | 5.2 Gemma Scope and Feature Steering; 6.2 Gemma Scope Deep Dive |
| 6.4 | 6.3 Transcoders and Attribution Graphs |
| 6.5 | 6.4 Crosscoders and Model Diffing |
| 8.6 | 8.5 Sparse Feature Circuits |
| 8.7 | 8.4 Circuit Tracing with Attribution Graphs |
| 9.3 | 9.2 Chain of Thought Faithfulness |
| 9.4 | 9.3 Emergent Misalignment Detection |
| 9.5 | 9.4 White-box Evals and Monitors |
| 9.6 | 9.5 ROME Model Editing |
| 12.4 | 12.1 CLIP SigLIP and VLM Controls |
| 12.5 | 12.5 Object Hallucination and Modality Arbitration |
| 13.1 | 13.1 Diffusion and Image Generation Controls |
| 13.3 | 13.1 Diffusion and Image Generation Controls |
| 14.1 | 14.2 I-JEPA from Scratch |
| 14.2 | 14.1 JEPA and World Model Controls; 14.3 V JEPA from Scratch |
| 14.3 | 14.1 JEPA and World Model Controls |
| 14.5 | 14.4 Maze Transformers and Latent Planning |
| 15.1 | 15.1 LoRA DoRA and Adapter Controls |
| 16.1 | 16.1 Exact Shapley on Ground Truth Games |
| 16.2 | 16.2 KernelSHAP and PartitionSHAP Controls |
| 16.3 | 16.13 TreeSHAP from First Principles; 16.14 DeepSHAP When Does Fast Attribution Match Exact SHAP; 16.15 Expected Gradients Which Expectation Does GradientSHAP Estimate |
| 16.4 | 16.12 SAGE Global Importance |
| 16.5 | 16.16 Where Does Higher Order Credit Go; 16.17 One Query Many Interactions; 16.3 Shapley Interactions with shapiq |
| 16.7 | 16.4 TokenSHAP and TokenShapley; 16.5 VLM Modality and Region SHAP |
| 16.8 | 16.6 SHAP vs Activation Patching; 16.8 Do SHAPley and Mechanistic Interpretability Agree |
| 16.10 | 16.9 Influence Functions and TracIn |
| 16.11 | 16.10 Datamodels; 16.11 Dataset Cartography |

The 23 sections without a notebook in their declared sources are **5.6, 5.7, 6.3, 7.6, 9.2, 9.7, 11.6, 12.2, 12.6, 12.7, 13.4, 13.5, 13.6, 14.4, 14.6, 14.7,
15.4, 15.5, 17.2, 17.3, 17.4, 18.1 and 18.2**. Their syllabus briefs are requirements, not completed lessons. They require substantive authorship,
student-visible methods, real controls, signature results, play cells and execution evidence. The repository rule also keeps new-topic work behind the
priority-pilot rewrite bar.

For the 32 source-backed destination gaps, the blockers are migration and coverage reconciliation: preserve every original lesson, map old to new names without
silent content loss, identify any added brief requirements, integrate canonical masters/pages, add prediction entrypoints, and retain honest historical versus
fresh evidence. Several grouped sections (especially model-specific SHAP, Shapley interactions, and datamodels/cartography) combine multiple long lessons;
reducing them to a short summary would lose the user's requested original materials. Historical receipt success alone does not accept a new combined lesson.

The learner walkthrough remains deferred. This inventory neither changes readiness statuses nor accepts any lesson scientifically or pedagogically.
