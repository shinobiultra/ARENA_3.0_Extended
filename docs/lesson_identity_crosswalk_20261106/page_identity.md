# Page identity: section numbers with more than one instruction page

One *page* is one filename stem (a `.md` plus its `.py` twin). `canonical` is the stem that the syllabus derives and the generated `config_extension.yaml`, the merged `config.yaml` and the build all agree on. Everything else is `stale-duplicate`, or `orphan-number-not-in-syllabus` when the number is not in the syllabus at all; original ARENA chapters 0-4 are listed but out of scope. Disk = the main checkout's working tree; HEAD = its committed tree. Columns: `disk md/py`, `HEAD md/py`, `git` state, words, `live refs` (code, tests, config outside docs/evidence that name the stem), `legacy` (named by `docs/arena_style_rewrite_status.yml`), `title now` (current section whose title matches), `lineage` (current sections whose syllabus `sources` include an exercise directory this page links to).

### 0.6 — How to Know When an Interpretability Result Is Fake (Practice)

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `06_[0.6]_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice` | ✓/✓ | ✗/✗ | untracked/untracked | 7522 | infrastructure/core/config.yaml:102 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:5 [config page_file (expected for the canonical page)]<br>infrastructure/extension/page_links.py:19 [extension page registry or contract (names the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:14338 [preservation-allowlist string (file need not exist)]<br>mastery_hub/teaching_editions/0_6.json:6 [HARD: hash-pins the page bytes]<br>scripts/audit_original_arena_preservation.py:146 [preservation-allowlist string (file need not exist)]<br>scripts/practice06_execution_checkpoint.py:21 [HARD: hash-pins the page bytes]<br>tests/test_extension_page_links.py:22 [review] | ✗ | — | 0.6 | — |
| stale-duplicate | `06_[0.6]_How_to_Know_When_an_Interpretability_Result_Is_Fake` | ✓/✗ | ✓/✗ | tracked/· | 2297 | scripts/audit_extension_artifact_hygiene.py:106 [preservation-allowlist string (file need not exist)]<br>scripts/audit_original_arena_preservation.py:102 [preservation-allowlist string (file need not exist)]<br>tests/mastery/test_preservation_scope.py:77 [preservation-allowlist string (file need not exist)] | ✓ | fuzzy:0.6 (0.92) | 0.6 | named by a legacy style-registry row |

### 1.3.1 — not in the syllabus

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| original-arena (out of scope, append-only) | `11_[1.3.1]_Linear_Probes` | ✓/✓ | ✓/✓ | tracked/tracked | 20486 | infrastructure/core/config.yaml:27 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_original.yaml:38 [config page_file (expected for the canonical page)]<br>st_dependencies.py:313 [review] | ✗ | — | — | — |
| original-arena (out of scope, append-only) | `11_[1.3.1]_Probing_for_Deception` | ✓/✓ | ✓/✓ | tracked/tracked | 20288 | — | ✗ | — | — | — |

### 1.6 — Local Frontier ML Infrastructure (Practice)

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `40_[1.6]_Local_Frontier_ML_Infrastructure_Practice` | ✓/✓ | ✗/✗ | untracked/untracked | 8262 | infrastructure/core/config.yaml:105 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:8 [config page_file (expected for the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:15235 [preservation-allowlist string (file need not exist)]<br>mastery_hub/teaching_editions/1_6.json:6 [HARD: hash-pins the page bytes]<br>scripts/audit_original_arena_preservation.py:146 [preservation-allowlist string (file need not exist)]<br>scripts/practice16_execution_checkpoint.py:19 [review] | ✗ | — | 1.6 | — |
| stale-duplicate | `40_[1.6]_Local_Frontier_ML_Infrastructure` | ✓/✗ | ✓/✗ | tracked/· | 1924 | scripts/audit_extension_artifact_hygiene.py:109 [preservation-allowlist string (file need not exist)]<br>scripts/audit_original_arena_preservation.py:104 [preservation-allowlist string (file need not exist)]<br>tests/mastery/test_preservation_scope.py:80 [preservation-allowlist string (file need not exist)]<br>tests/test_original_arena_preservation_audit.py:27 [preservation-allowlist string (file need not exist)] | ✓ | fuzzy:1.6 (0.88) | 1.6 | named by a legacy style-registry row |

### 2.2.2 — not in the syllabus

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| original-arena (out of scope, append-only) | `21_[2.2.2]_Policy_Gradient` | ✓/✓ | ✓/✓ | tracked/tracked | 5406 | — | ✗ | — | — | — |
| original-arena (out of scope, append-only) | `22_[2.2.2]_VPG` | ✓/✓ | ✓/✓ | tracked/tracked | 5406 | — | ✗ | — | — | — |

### 5.2 — Mixture of Experts from Scratch

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `02_[5.2]_Mixture_of_Experts_from_Scratch` | ✓/✓ | ✗/✗ | untracked/untracked | 11607 | infrastructure/core/config.yaml:111 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:14 [config page_file (expected for the canonical page)]<br>tests/mastery/test_reading_entrypoint.py:41 [review]<br>tests/test_exercise_provenance_preservation.py:27 [preservation-allowlist string (file need not exist)] | ✗ | — | 5.2 | — |
| stale-duplicate | `02_[5.2]_Gemma_Scope_and_Feature_Steering` | ✓/✗ | ✓/✗ | tracked/· | 4600 | tests/mastery/test_reading_entrypoint.py:57 [synthetic tmp_path fixture] | ✓ | exact:6.2 | 6.2 (inferred) | same title as current 6.2 (content renumbered); named by a legacy style-registry row |

### 5.4 — Mamba-2 and State-Space Duality

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `04_[5.4]_Mamba-2_and_State-Space_Duality` | ✓/✓ | ✗/✗ | untracked/untracked | 16234 | infrastructure/core/config.yaml:117 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:20 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/5_4.json:6 [HARD: hash-pins the page bytes]<br>tests/mastery/test_mamba2_master_migration.py:167 [review] | ✗ | — | 5.4 | — |
| stale-duplicate | `04_[5.4]_Mamba_State_Tracking` | ✓/✗ | ✓/✗ | tracked/· | 2363 | — | ✓ | — | 5.5 (inferred) | no current title matches: confirm the content is superseded before deleting; named by a legacy style-registry row |

### 5.5 — State Tracking in SSMs and Transformers

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `05_[5.5]_State_Tracking_in_SSMs_and_Transformers` | ✓/✓ | ✗/✗ | untracked/untracked | 6973 | infrastructure/core/config.yaml:120 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:23 [config page_file (expected for the canonical page)]<br>infrastructure/extension/page_links.py:26 [extension page registry or contract (names the canonical page)]<br>mastery_hub/teaching_editions/5_5.json:6 [HARD: hash-pins the page bytes]<br>tests/test_extension_page_links.py:27 [review] | ✗ | — | 5.5 | — |
| stale-duplicate | `05_[5.5]_Diffusion_Language_Models` | ✓/✗ | ✓/✗ | tracked/· | 12587 | — | ✓ | fuzzy:5.8 (0.85) | 5.8 (inferred) | named by a legacy style-registry row |

### 5.6 — Linear Attention, RetNet and RWKV

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `06_[5.6]_Linear_Attention_RetNet_and_RWKV` | ✓/✓ | ✗/✗ | untracked/untracked | 5260 | infrastructure/core/config.yaml:123 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:26 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/5_6.json:6 [HARD: hash-pins the page bytes] | ✗ | — | 5.6 | — |
| stale-duplicate | `06_[5.6]_Multimodal_Embedding_and_Function_Calling_Models` | ✓/✗ | ✓/✗ | tracked/· | 1786 | — | ✓ | fuzzy:5.9 (0.69) | 5.9 (inferred) | named by a legacy style-registry row |

### 5.7 — Griffin, xLSTM, Hyena and Hybrid Models

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `07_[5.7]_Griffin_xLSTM_Hyena_and_Hybrid_Models` | ✓/✓ | ✗/✗ | untracked/untracked | 5463 | infrastructure/core/config.yaml:126 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:29 [config page_file (expected for the canonical page)] | ✗ | — | 5.7 | — |
| stale-duplicate | `07_[5.7]_Mamba-2_and_Structured_State-Space_Duality` | ✓/✗ | ✓/✗ | tracked/· | 10150 | — | ✓ | fuzzy:5.4 (0.85) | 5.4 (inferred) | named by a legacy style-registry row |

### 5.8 — Discrete Diffusion Language Models

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `08_[5.8]_Discrete_Diffusion_Language_Models` | ✓/✓ | ✗/✗ | untracked/untracked | 15926 | infrastructure/core/config.yaml:129 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:32 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/5_8.json:6 [HARD: hash-pins the page bytes] | ✗ | — | 5.8 | — |
| stale-duplicate | `08_[5.8]_Mixture_of_Experts_from_Scratch` | ✓/✗ | ✓/✗ | tracked/· | 9240 | — | ✗ | exact:5.2 | 5.2 (inferred) | same title as current 5.2 (content renumbered) |

### 6.2 — Gemma Scope and Feature Steering

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `02_[6.2]_Gemma_Scope_and_Feature_Steering` | ✓/✓ | ✗/✗ | untracked/untracked | 12213 | infrastructure/core/config.yaml:138 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:41 [config page_file (expected for the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:9746 [preservation-allowlist string (file need not exist)]<br>mastery_hub/teaching_editions/6_2.json:6 [HARD: hash-pins the page bytes] | ✗ | — | 6.2 | — |
| stale-duplicate | `02_[6.2]_Gemma_Scope_Deep_Dive` | ✓/✗ | ✓/✗ | tracked/· | 4159 | — | ✓ | fuzzy:6.2 (0.60) | 6.2 (inferred) | named by a legacy style-registry row |

### 6.3 — Feature Splitting, Absorption and Dashboards

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `03_[6.3]_Feature_Splitting_Absorption_and_Dashboards` | ✓/✓ | ✗/✗ | untracked/untracked | 4853 | infrastructure/core/config.yaml:141 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:44 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/6_3.json:6 [HARD: hash-pins the page bytes] | ✗ | — | 6.3 | — |
| stale-duplicate | `03_[6.3]_Transcoders_and_Attribution_Graphs` | ✓/✗ | ✓/✗ | tracked/· | 4465 | — | ✓ | fuzzy:8.7 (0.66) | 6.4 (inferred) | named by a legacy style-registry row |

### 6.4 — Transcoders and Skip Transcoders

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `04_[6.4]_Transcoders_and_Skip_Transcoders` | ✓/✓ | ✗/✗ | untracked/untracked | 6236 | infrastructure/core/config.yaml:144 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:47 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/6_4.json:6 [HARD: hash-pins the page bytes] | ✗ | — | 6.4 | — |
| stale-duplicate | `04_[6.4]_Crosscoders_and_Model_Diffing` | ✓/✗ | ✓/✗ | tracked/· | 4554 | — | ✓ | exact:6.5 | 6.5 (inferred) | same title as current 6.5 (content renumbered); named by a legacy style-registry row |

### 8.3 — AtP* from Scratch

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `03_[8.3]_AtP_from_Scratch` | ✓/✓ | ✗/✗ | untracked/untracked | 11836 | infrastructure/core/config.yaml:174 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:77 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/8_3.json:6 [HARD: hash-pins the page bytes]<br>mastery_hub/units/acdc_own_method/protocol.json:165 [HARD: hash-pins the page bytes]<br>tests/mastery/test_reading_entrypoint.py:42 [review] | ✗ | — | 8.3 | — |
| stale-duplicate | `03_[8.3]_ACDC_and_Circuit_Metrics` | ✗/✗ | ✓/✗ | deleted-in-worktree/· |  | — | ✓ | exact:8.4 | — | already absent on disk (staged deletion in the main index); still in HEAD; same title as current 8.4 (content renumbered); named by a legacy style-registry row |

### 8.4 — ACDC and Circuit Metrics

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `04_[8.4]_ACDC_and_Circuit_Metrics` | ✓/✓ | ✗/✗ | untracked/untracked | 13188 | infrastructure/core/config.yaml:177 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:80 [config page_file (expected for the canonical page)]<br>mastery_hub/practice_addons/mini_vlm_checked_recovery/source_manifest.json:191 [HARD: hash-pins the page bytes]<br>mastery_hub/units/acdc_own_method/protocol.json:173 [HARD: hash-pins the page bytes]<br>tests/mastery/test_reading_entrypoint.py:43 [review] | ✗ | — | 8.4 | — |
| stale-duplicate | `04_[8.4]_Circuit_Tracing_with_Attribution_Graphs` | ✓/✗ | ✓/✗ | tracked/· | 4865 | — | ✓ | exact:8.7 | 8.7 | same title as current 8.7 (content renumbered); named by a legacy style-registry row |

### 8.5 — Causal Scrubbing

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `05_[8.5]_Causal_Scrubbing` | ✓/✓ | ✗/✗ | untracked/untracked | 10488 | infrastructure/core/config.yaml:180 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:83 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/8_5.json:6 [HARD: hash-pins the page bytes] | ✗ | — | 8.5 | — |
| stale-duplicate | `05_[8.5]_Sparse_Feature_Circuits` | ✓/✗ | ✓/✗ | tracked/· | 5944 | — | ✓ | fuzzy:8.6 (0.82) | 8.6 | named by a legacy style-registry row |

### 8.6 — Sparse Feature Circuits and SHIFT

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `06_[8.6]_Sparse_Feature_Circuits_and_SHIFT` | ✓/✓ | ✗/✗ | untracked/untracked | 9376 | infrastructure/core/config.yaml:183 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:86 [config page_file (expected for the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:9 [preservation-allowlist string (file need not exist)]<br>infrastructure/extension/study_cautions.py:9 [extension page registry or contract (names the canonical page)]<br>mastery_hub/teaching_editions/8_6.json:6 [HARD: hash-pins the page bytes]<br>tests/mastery/test_study_cautions.py:16 [review]<br>tests/test_extension_source_preservation.py:27 [preservation-allowlist string (file need not exist)]<br>tests/test_sparse_feature_circuits_8_6_migration.py:18 [review] | ✗ | — | 8.6 | — |
| stale-duplicate | `06_[8.6]_AtP-Star_from_Scratch` | ✗/✗ | ✓/✗ | deleted-in-worktree/· |  | — | ✗ | fuzzy:8.3 (0.86) | — | already absent on disk (staged deletion in the main index); still in HEAD |

### 9.1 — Refusal Directions and Representation Engineering

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `01_[9.1]_Refusal_Directions_and_Representation_Engineering` | ✓/✓ | ✗/✗ | untracked/untracked | 24151 | infrastructure/core/config.yaml:192 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:95 [config page_file (expected for the canonical page)]<br>mastery_hub/units/pilot_practice/original_source_manifest.json:140 [HARD: hash-pins the page bytes]<br>mastery_hub/units/pilot_practice/projection_map.json:2227 [review]<br>mastery_hub/units/pilot_practice/protocol.json:141 [HARD: hash-pins the page bytes]<br>tests/mastery/test_reading_entrypoint.py:44 [review] | ✗ | — | 9.1 | — |
| stale-duplicate | `01_[9.1]_Refusal_Directions_and_Safe_Steering` | ✓/✗ | ✓/✗ | tracked/· | 6308 | — | ✓ | fuzzy:9.1 (0.75) | 9.1 (inferred) | named by a legacy style-registry row |

### 9.2 — Truthfulness Directions and Hallucination Probes

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `02_[9.2]_Truthfulness_Directions_and_Hallucination_Probes` | ✓/✓ | ✗/✗ | untracked/untracked | 4762 | infrastructure/core/config.yaml:195 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:98 [config page_file (expected for the canonical page)] | ✗ | — | 9.2 | — |
| stale-duplicate | `02_[9.2]_Chain_of_Thought_Faithfulness` | ✓/✗ | ✓/✗ | tracked/· | 5078 | — | ✓ | exact:9.3 | 9.3 (inferred) | same title as current 9.3 (content renumbered); named by a legacy style-registry row |

### 9.3 — Chain-of-Thought Faithfulness

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `03_[9.3]_Chain-of-Thought_Faithfulness` | ✓/✓ | ✗/✗ | untracked/untracked | 7750 | infrastructure/core/config.yaml:198 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:101 [config page_file (expected for the canonical page)]<br>infrastructure/extension/study_cautions.py:15 [extension page registry or contract (names the canonical page)]<br>mastery_hub/teaching_editions/9_3.json:6 [HARD: hash-pins the page bytes]<br>tests/mastery/test_study_cautions.py:18 [review]<br>tests/test_cot_9_3_source_preservation.py:14 [preservation-allowlist string (file need not exist)] | ✗ | — | 9.3 | — |
| stale-duplicate | `03_[9.3]_Emergent_Misalignment_Detection` | ✓/✗ | ✓/✗ | tracked/· | 5427 | — | ✓ | exact:9.4 | 9.4 | same title as current 9.4 (content renumbered); named by a legacy style-registry row |

### 9.4 — Emergent Misalignment Detection

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `04_[9.4]_Emergent_Misalignment_Detection` | ✓/✓ | ✗/✗ | untracked/untracked | 6694 | infrastructure/core/config.yaml:201 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:104 [config page_file (expected for the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:2085 [preservation-allowlist string (file need not exist)]<br>infrastructure/extension/study_cautions.py:18 [extension page registry or contract (names the canonical page)]<br>mastery_hub/teaching_editions/9_4.json:6 [HARD: hash-pins the page bytes]<br>tests/mastery/test_study_cautions.py:19 [review]<br>tests/test_emergent_misalignment_9_4_migration.py:19 [review] | ✗ | — | 9.4 | — |
| stale-duplicate | `04_[9.4]_White_box_Evals_and_Monitors` | ✓/✗ | ✓/✗ | tracked/· | 4006 | — | ✓ | — | 9.5 | no current title matches: confirm the content is superseded before deleting; named by a legacy style-registry row |

### 9.5 — White-Box Monitors and Red-Teaming with Internals

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `05_[9.5]_White-Box_Monitors_and_Red-Teaming_with_Internals` | ✓/✓ | ✗/✗ | untracked/untracked | 6051 | infrastructure/core/config.yaml:204 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:107 [config page_file (expected for the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:3542 [preservation-allowlist string (file need not exist)]<br>mastery_hub/teaching_editions/9_5.json:6 [HARD: hash-pins the page bytes]<br>tests/test_white_box_monitors_9_5_migration.py:21 [HARD: hash-pins the page bytes] | ✗ | — | 9.5 | — |
| stale-duplicate | `05_[9.5]_ROME_Model_Editing` | ✓/✗ | ✓/✗ | tracked/· | 7420 | — | ✗ | fuzzy:6.5 (0.64) | 9.6 (inferred) | — |

### 11.5 — Contrast-Consistent Search

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `05_[11.5]_Contrast-Consistent_Search` | ✓/✓ | ✗/✗ | untracked/untracked | 11928 | infrastructure/core/config.yaml:228 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:131 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/11_5.json:6 [HARD: hash-pins the page bytes] | ✗ | — | 11.5 | — |
| stale-duplicate | `05_[11.5]_Contrast_Consistent_Search` | ✓/✗ | ✓/✗ | tracked/· | 10296 | infrastructure/chapters/chapter11_representation_geometry/master_11_5.py:3299 [HARD: hash-pins the page bytes] | ✓ | exact:11.5 | — | BLOCKER-FOR-DELETION: a test or runner hash-pins this page; named by a legacy style-registry row |

### 12.1 — CLIP and SigLIP from Scratch

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `01_[12.1]_CLIP_and_SigLIP_from_Scratch` | ✓/✓ | ✗/✗ | untracked/untracked | 28033 | infrastructure/core/config.yaml:234 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:137 [config page_file (expected for the canonical page)]<br>mastery_hub/practice_addons/mini_vlm_checked_recovery/source_manifest.json:156 [HARD: hash-pins the page bytes]<br>mastery_hub/teaching_editions/12_1.json:6 [HARD: hash-pins the page bytes]<br>tests/mastery/test_reading_entrypoint.py:45 [review] | ✗ | — | 12.1 | — |
| stale-duplicate | `01_[12.1]_CLIP_SigLIP_and_VLM_Controls` | ✓/✗ | ✓/✗ | tracked/· | 9773 | — | ✓ | fuzzy:11.1 (0.63) | 12.1;12.4 (inferred) | named by a legacy style-registry row |

### 13.1 — Diffusion Models and Cross-Attention Attribution

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `01_[13.1]_Diffusion_Models_and_Cross-Attention_Attribution` | ✓/✓ | ✗/✗ | untracked/untracked | 10045 | infrastructure/core/config.yaml:255 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:158 [config page_file (expected for the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:4465 [preservation-allowlist string (file need not exist)]<br>mastery_hub/teaching_editions/13_1.json:6 [HARD: hash-pins the page bytes]<br>tests/test_diffusion_13_1_migration.py:21 [review] | ✗ | — | 13.1 | — |
| stale-duplicate | `01_[13.1]_Diffusion_and_Image_Generation_Controls` | ✓/✗ | ✓/✗ | tracked/· | 6753 | — | ✓ | — | 13.1;13.3 | no current title matches: confirm the content is superseded before deleting; named by a legacy style-registry row |

### 14.1 — I-JEPA from Scratch

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `01_[14.1]_I-JEPA_from_Scratch` | ✓/✓ | ✗/✗ | untracked/untracked | 13217 | infrastructure/core/config.yaml:273 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:176 [config page_file (expected for the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:5226 [preservation-allowlist string (file need not exist)]<br>infrastructure/extension/study_cautions.py:27 [extension page registry or contract (names the canonical page)]<br>mastery_hub/teaching_editions/14_1.json:6 [HARD: hash-pins the page bytes]<br>tests/mastery/test_study_cautions.py:22 [review]<br>tests/test_ijepa_14_1_migration.py:27 [review] | ✗ | — | 14.1 | — |
| stale-duplicate | `01_[14.1]_JEPA_and_World_Model_Controls` | ✓/✗ | ✓/✗ | tracked/· | 3826 | — | ✓ | fuzzy:14.2 (0.72) | 14.2;14.3 | named by a legacy style-registry row |

### 14.2 — V-JEPA and Video World Models

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `02_[14.2]_V-JEPA_and_Video_World_Models` | ✓/✓ | ✗/✗ | untracked/untracked | 27298 | infrastructure/core/config.yaml:276 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:179 [config page_file (expected for the canonical page)]<br>infrastructure/extension/page_links.py:50 [extension page registry or contract (names the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:10643 [preservation-allowlist string (file need not exist)]<br>mastery_hub/teaching_editions/14_2.json:6 [HARD: hash-pins the page bytes]<br>tests/test_extension_page_links.py:47 [review]<br>tests/test_vjepa_14_2_migration.py:69 [review] | ✗ | — | 14.2 | — |
| stale-duplicate | `02_[14.2]_I-JEPA_from_Scratch` | ✓/✗ | ✓/✗ | tracked/· | 7489 | — | ✓ | exact:14.1 | 14.1 (inferred) | same title as current 14.1 (content renumbered); named by a legacy style-registry row |

### 14.3 — Action-Conditioned World Models and Latent Planning

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `03_[14.3]_Action-Conditioned_World_Models_and_Latent_Planning` | ✓/✓ | ✗/✗ | untracked/untracked | 4510 | infrastructure/core/config.yaml:279 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:182 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/14_3.json:6 [HARD: hash-pins the page bytes] | ✗ | — | 14.3 | — |
| stale-duplicate | `03_[14.3]_V_JEPA_from_Scratch` | ✓/✗ | ✓/✗ | tracked/· | 8464 | — | ✗ | fuzzy:14.1 (0.95) | 14.2 | — |

### 14.4 — The JEPA Family

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `04_[14.4]_The_JEPA_Family` | ✓/✓ | ✗/✗ | untracked/untracked | 4583 | infrastructure/core/config.yaml:282 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:185 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/14_4.json:6 [HARD: hash-pins the page bytes] | ✗ | — | 14.4 | — |
| stale-duplicate | `04_[14.4]_Maze_Transformers_and_Latent_Planning` | ✓/✗ | ✓/✗ | tracked/· | 8667 | — | ✗ | fuzzy:14.5 (0.70) | 14.5 (inferred) | — |

### 15.1 — LoRA and DoRA from Scratch

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `01_[15.1]_LoRA_and_DoRA_from_Scratch` | ✓/✓ | ✗/✗ | untracked/untracked | 9697 | infrastructure/core/config.yaml:294 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:197 [config page_file (expected for the canonical page)]<br>infrastructure/extension/page_links.py:62 [extension page registry or contract (names the canonical page)]<br>mastery_hub/teaching_editions/15_1.json:6 [HARD: hash-pins the page bytes]<br>tests/mastery/test_lora_dora_master_migration.py:138 [review]<br>tests/test_extension_page_links.py:57 [review] | ✗ | — | 15.1 | — |
| stale-duplicate | `01_[15.1]_LoRA_DoRA_and_Adapter_Controls` | ✓/✗ | ✓/✗ | tracked/· | 5898 | — | ✓ | fuzzy:11.1 (0.64) | 15.1 | named by a legacy style-registry row |

### 15.2 — LoRA vs Full Fine-Tuning

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `02_[15.2]_LoRA_vs_Full_Fine-Tuning` | ✓/✓ | ✗/✗ | untracked/untracked | 8921 | infrastructure/core/config.yaml:297 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:200 [config page_file (expected for the canonical page)] | ✗ | — | 15.2 | — |
| stale-duplicate | `02_[15.2]_LoRA_vs_Full_Finetuning` | ✓/✗ | ✓/✗ | tracked/· | 6166 | — | ✓ | fuzzy:15.2 (0.98) | 15.2 | named by a legacy style-registry row |

### 16.1 — Exact Shapley on Ground-Truth Games

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `01_[16.1]_Exact_Shapley_on_Ground-Truth_Games` | ✓/✓ | ✗/✗ | untracked/untracked | 5813 | infrastructure/core/config.yaml:309 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:212 [config page_file (expected for the canonical page)]<br>mastery_hub/teaching_editions/16_1.json:6 [HARD: hash-pins the page bytes] | ✗ | — | 16.1 | — |
| stale-duplicate | `01_[16.1]_Exact_Shapley_on_Ground_Truth_Games` | ✓/✗ | ✓/✗ | tracked/· | 3100 | — | ✓ | exact:16.1 | 16.1 | named by a legacy style-registry row |

### 16.2 — KernelSHAP and PartitionSHAP

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `02_[16.2]_KernelSHAP_and_PartitionSHAP` | ✓/✓ | ✗/✗ | untracked/untracked | 6495 | infrastructure/core/config.yaml:312 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:215 [config page_file (expected for the canonical page)]<br>infrastructure/extension/page_links.py:79 [extension page registry or contract (names the canonical page)]<br>tests/test_extension_page_links.py:72 [review] | ✗ | — | 16.2 | — |
| stale-duplicate | `02_[16.2]_KernelSHAP_and_PartitionSHAP_Controls` | ✓/✗ | ✓/✗ | tracked/· | 3558 | — | ✓ | fuzzy:16.2 (0.86) | 16.2 (inferred) | named by a legacy style-registry row |

### 16.3 — TreeSHAP, DeepSHAP and GradientSHAP (Practice)

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `03_[16.3]_TreeSHAP_DeepSHAP_and_GradientSHAP_Practice` | ✓/✓ | ✗/✗ | untracked/untracked | 100175 | infrastructure/core/config.yaml:315 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:218 [config page_file (expected for the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:16188 [preservation-allowlist string (file need not exist)]<br>mastery_hub/teaching_editions/16_3.json:6 [HARD: hash-pins the page bytes]<br>tests/mastery/test_reading_entrypoint.py:46 [synthetic tmp_path fixture]<br>tests/test_model_specific_shap_16_3_migration.py:293 [review] | ✗ | — | 16.3 | — |
| stale-duplicate | `03_[16.3]_Shapley_Interactions_with_shapiq` | ✓/✗ | ✓/✗ | tracked/· | 4139 | — | ✓ | fuzzy:16.5 (0.77) | 16.5 | named by a legacy style-registry row |
| stale-duplicate | `03_[16.3]_TreeSHAP_DeepSHAP_and_GradientSHAP` | ✓/✗ | ✗/✗ | untracked/· | 99946 | — | ✗ | fuzzy:16.3 (0.88) | 16.3 | — |

### 16.4 — SAGE Global Importance

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `04_[16.4]_SAGE_Global_Importance` | ✓/✓ | ✗/✗ | untracked/untracked | 13334 | infrastructure/core/config.yaml:318 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:221 [config page_file (expected for the canonical page)]<br>infrastructure/extension/page_links.py:86 [extension page registry or contract (names the canonical page)]<br>tests/test_extension_page_links.py:77 [review] | ✗ | — | 16.4 | — |
| stale-duplicate | `04_[16.4]_TokenSHAP_and_TokenShapley` | ✓/✗ | ✓/✗ | tracked/· | 3713 | — | ✓ | fuzzy:16.2 (0.70) | 16.7 | named by a legacy style-registry row |

### 16.5 — Shapley Interactions

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `05_[16.5]_Shapley_Interactions` | ✓/✓ | ✗/✗ | untracked/untracked | 28736 | infrastructure/core/config.yaml:321 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:224 [config page_file (expected for the canonical page)]<br>infrastructure/extension/page_links.py:93 [extension page registry or contract (names the canonical page)]<br>tests/mastery/test_shapley_interactions_master_migration.py:192 [review]<br>tests/test_extension_page_links.py:82 [review] | ✗ | — | 16.5 | — |
| stale-duplicate | `05_[16.5]_VLM_Modality_and_Region_SHAP` | ✓/✗ | ✓/✗ | tracked/· | 3744 | — | ✓ | — | 16.7 | no current title matches: confirm the content is superseded before deleting; named by a legacy style-registry row |

### 16.6 — Sparse Interaction Discovery with SPEX and ProxySPEX

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `06_[16.6]_Sparse_Interaction_Discovery_with_SPEX_and_ProxySPEX` | ✓/✓ | ✓/✓ | modified/modified | 24373 | infrastructure/core/config.yaml:324 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:227 [config page_file (expected for the canonical page)] | ✗ | — | 16.6 | — |
| stale-duplicate | `06_[16.6]_SHAP_vs_Activation_Patching` | ✓/✗ | ✓/✗ | tracked/· | 4174 | — | ✓ | exact:16.8 | 16.8 (inferred) | same title as current 16.8 (content renumbered); named by a legacy style-registry row |

### 16.7 — TokenSHAP and Region SHAP for LLMs and VLMs

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `07_[16.7]_TokenSHAP_and_Region_SHAP_for_LLMs_and_VLMs` | ✓/✓ | ✗/✗ | untracked/untracked | 16869 | infrastructure/core/config.yaml:327 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:230 [config page_file (expected for the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:12200 [preservation-allowlist string (file need not exist)]<br>tests/test_token_region_16_7_migration.py:174 [HARD: hash-pins the page bytes] | ✗ | — | 16.7 | — |
| stale-duplicate | `07_[16.7]_Data_Shapley_in_One_Training_Run` | ✗/✗ | ✓/✗ | deleted-in-worktree/· |  | — | ✓ | exact:16.9 | 16.9 | already absent on disk (staged deletion in the main index); still in HEAD; same title as current 16.9 (content renumbered); named by a legacy style-registry row |

### 16.8 — SHAP vs Activation Patching

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `08_[16.8]_SHAP_vs_Activation_Patching` | ✓/✓ | ✗/✗ | untracked/untracked | 17424 | infrastructure/core/config.yaml:330 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:233 [config page_file (expected for the canonical page)]<br>infrastructure/extension/source_preservation_contracts.json:13301 [preservation-allowlist string (file need not exist)]<br>tests/test_shap_patching_16_8_migration.py:229 [review] | ✗ | — | 16.8 | — |
| stale-duplicate | `08_[16.8]_Do_SHAPley_and_Mechanistic_Interpretability_Agree` | ✓/✗ | ✓/✗ | tracked/· | 5725 | — | ✓ | — | 16.8 | no current title matches: confirm the content is superseded before deleting; named by a legacy style-registry row |

### 16.9 — Data Shapley in One Training Run

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `09_[16.9]_Data_Shapley_in_One_Training_Run` | ✓/✓ | ✓/✓ | modified/modified | 23224 | infrastructure/core/config.yaml:333 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:236 [config page_file (expected for the canonical page)] | ✗ | — | 16.9 | — |
| stale-duplicate | `09_[16.9]_Influence_Functions_and_TracIn` | ✓/✗ | ✓/✗ | tracked/· | 12152 | tests/mastery/test_influence_tracin_master_migration.py:86 [HARD: hash-pins the page bytes] | ✓ | exact:16.10 | 16.10 (inferred) | BLOCKER-FOR-DELETION: a test or runner hash-pins this page; same title as current 16.10 (content renumbered); named by a legacy style-registry row |

### 16.10 — Influence Functions and TracIn

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `10_[16.10]_Influence_Functions_and_TracIn` | ✓/✓ | ✗/✗ | untracked/untracked | 15580 | infrastructure/core/config.yaml:336 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:239 [config page_file (expected for the canonical page)]<br>infrastructure/extension/page_links.py:100 [extension page registry or contract (names the canonical page)]<br>tests/test_extension_page_links.py:87 [review] | ✗ | — | 16.10 | — |
| stale-duplicate | `10_[16.10]_Datamodels` | ✓/✗ | ✓/✗ | tracked/· | 8447 | — | ✓ | — | 16.11 (inferred) | no current title matches: confirm the content is superseded before deleting; named by a legacy style-registry row |

### 16.11 — Datamodels and Dataset Cartography

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `11_[16.11]_Datamodels_and_Dataset_Cartography` | ✓/✓ | ✗/✗ | untracked/untracked | 24944 | infrastructure/core/config.yaml:339 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:242 [config page_file (expected for the canonical page)]<br>infrastructure/extension/page_links.py:107 [extension page registry or contract (names the canonical page)]<br>mastery_hub/teaching_editions/16_11.json:6 [HARD: hash-pins the page bytes]<br>tests/mastery/test_datamodels_cartography_master_migration.py:163 [review]<br>tests/test_extension_page_links.py:92 [review] | ✗ | — | 16.11 | — |
| stale-duplicate | `11_[16.11]_Dataset_Cartography` | ✓/✗ | ✓/✗ | tracked/· | 8384 | — | ✓ | fuzzy:16.11 (0.72) | 16.11 (inferred) | named by a legacy style-registry row |

### 16.12 — not in the syllabus

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| orphan-number-not-in-syllabus | `12_[16.12]_SAGE_Global_Importance` | ✓/✗ | ✓/✗ | tracked/· | 9505 | tests/mastery/test_sage_master_migration.py:80 [HARD: hash-pins the page bytes] | ✓ | exact:16.4 | 16.4 (inferred) | BLOCKER-FOR-DELETION: a test or runner hash-pins this page; number is not in the syllabus; no canonical page for this number; named by a legacy style-registry row |

### 16.13 — not in the syllabus

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| orphan-number-not-in-syllabus | `13_[16.13]_TreeSHAP_from_First_Principles` | ✓/✗ | ✓/✗ | tracked/· | 10003 | — | ✓ | — | 16.3 (inferred) | number is not in the syllabus; no canonical page for this number; named by a legacy style-registry row |

### 16.14 — not in the syllabus

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| orphan-number-not-in-syllabus | `14_[16.14]_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP` | ✓/✗ | ✓/✗ | tracked/· | 10276 | — | ✓ | — | 16.3 (inferred) | number is not in the syllabus; no canonical page for this number; named by a legacy style-registry row |

### 16.15 — not in the syllabus

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| orphan-number-not-in-syllabus | `15_[16.15]_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate` | ✓/✗ | ✓/✗ | tracked/· | 13571 | — | ✓ | — | 16.3 (inferred) | number is not in the syllabus; no canonical page for this number; named by a legacy style-registry row |

### 16.16 — not in the syllabus

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| orphan-number-not-in-syllabus | `16_[16.16]_Where_Does_Higher_Order_Credit_Go` | ✓/✗ | ✓/✗ | tracked/· | 12029 | — | ✓ | — | 16.5 (inferred) | number is not in the syllabus; no canonical page for this number; named by a legacy style-registry row |

### 16.17 — not in the syllabus

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| orphan-number-not-in-syllabus | `17_[16.17]_One_Query_Many_Interactions` | ✓/✗ | ✓/✗ | tracked/· | 5970 | — | ✓ | fuzzy:16.5 (0.64) | 16.5 (inferred) | number is not in the syllabus; no canonical page for this number; named by a legacy style-registry row |

### 17.1 — Checkpoint Archaeology and Grokking

| Role | Page stem | disk md/py | HEAD md/py | git md/py | words | live refs | legacy | title now | lineage | flags |
|---|---|---|---|---|---:|---|---|---|---|---|
| canonical | `01_[17.1]_Checkpoint_Archaeology_and_Grokking` | ✓/✓ | ✗/✗ | untracked/untracked | 6520 | infrastructure/core/config.yaml:342 [config page_file (expected for the canonical page)]<br>infrastructure/core/config_extension.yaml:245 [config page_file (expected for the canonical page)]<br>infrastructure/extension/study_cautions.py:33 [extension page registry or contract (names the canonical page)]<br>mastery_hub/teaching_editions/17_1.json:6 [HARD: hash-pins the page bytes]<br>tests/mastery/test_study_cautions.py:24 [review] | ✗ | — | 17.1 | — |
| stale-duplicate | `01_[17.1]_Checkpoint_Archaeology_and_Mechanism_Emergence` | ✓/✗ | ✓/✗ | tracked/· | 4776 | — | ✓ | fuzzy:17.1 (0.72) | 17.1 | named by a legacy style-registry row |

