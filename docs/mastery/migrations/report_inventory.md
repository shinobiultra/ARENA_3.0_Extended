# Legacy report inventory

Snapshot after 11.2 migration: 2026-10-15T08:42:28.482515+00:00. Source: current working tree at Git `dec028ec8ed8accb40bf0d0a7315bb72f0c1fcc6`.

The declared report audit has **88 records**: **15** have both files and **73** lack at least one. There are **72 missing locks** and **73 missing reports** (145 distinct paths).

Unchanged existing-record validation found **2 stale records with 18 freshness blockers**, **13 records without contract blockers**, and **0 evaluation exceptions**.

The previous inventory is preserved byte-for-byte in [legacy_report_inventory.before-11.2-20261015T084228Z.json](../evidence/legacy_report_inventory.before-11.2-20261015T084228Z.json) (SHA-256 `06e3e29523772de7a044e67c6379215dfb9c7ca50005b5d9d5cbce07622058f5`). It recorded one stale pair and fourteen pairs without blockers; the migrated 11.2 pair now also has freshness blockers. Missing-file counts are unchanged.

The first missing lock is `chapter5_modern_architectures/exercises/part2_mixture_of_experts/artifacts.lock.yml`. The unchanged production `report_evidence_blockers()` stops there, before it can disclose later missing records or evaluate the remaining existing ones. Fixing that first path therefore does not establish a clean audit. This inventory catches failures per record only for diagnosis; production gates and tests were not weakened.

The [JSON inventory](../evidence/legacy_report_inventory.json) lists every exact missing lock/report path, all declared pairs, input hashes, and verbatim blocker/exception diagnostics. Missing records are explicitly unevaluated.

| Existing report | Diagnostic result |
| --- | --- |
| `chapter0_fundamentals/exercises/part6_fake_interpretability_results/verification_report.json` | No contract blockers |
| `chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/verification_report.json` | No contract blockers |
| `chapter5_modern_architectures/exercises/part1_gemma_from_scratch/verification_report.json` | No contract blockers |
| `chapter5_modern_architectures/exercises/part3_mamba_from_scratch/verification_report.json` | No contract blockers |
| `chapter6_sparse_feature_methods/exercises/part1_sae_variants/verification_report.json` | No contract blockers |
| `chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/verification_report.json` | 9 blockers |
| `chapter11_representation_geometry/exercises/part2_cca_svcca_alignment/verification_report.json` | 9 blockers |
| `chapter11_representation_geometry/exercises/part3_causal_abstraction_das/verification_report.json` | No contract blockers |
| `chapter11_representation_geometry/exercises/part4_concept_erasure/verification_report.json` | No contract blockers |
| `chapter11_representation_geometry/exercises/part5_contrast_consistent_search/verification_report.json` | No contract blockers |
| `chapter13_image_generation_interpretability/exercises/part2_denoising_time_causal_patching/verification_report.json` | No contract blockers |
| `chapter15_peft_misalignment/exercises/part2_lora_vs_full_finetuning/verification_report.json` | No contract blockers |
| `chapter17_training_dynamics/exercises/part1_checkpoint_archaeology/verification_report.json` | No contract blockers |
| `docs/evidence/ioi_path_patching/verification_report.json` | No contract blockers |
| `docs/evidence/othello_gpt/verification_report.json` | No contract blockers |

The retained historical 11.1 and 11.2 reports have not been rebound to migrated current artifacts. Their exact freshness mismatches are recorded in the JSON; this inventory does not infer that preserved historical measurements became invalid.

Only unchanged `verification_report_records()` declarations are covered; its `docs/evidence` discovery already excludes reports without sibling locks. No notebooks, model studies, training, downloads or report generation ran. Contract validation checks stored metadata, metrics and current input hashes; a record without blockers does not establish fresh execution, scientific validity, publication acceptance or full-course completion. These findings are a truthful handoff, not permission to delete, relabel, regenerate or accept any historical evidence.

Validator/config source hashes and existing lock/report files stayed unchanged during their respective inventory checks.
