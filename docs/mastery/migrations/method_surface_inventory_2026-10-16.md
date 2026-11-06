# Method-level surface inventory — 2026-10-16

This read-only companion refines the earlier material-availability inventory.
It records inspected learner methods, exact source notebook paths and hashes,
definition cell indices, and unresolved current-brief requirements in
[the JSON evidence](../evidence/2026-10-16-method-surface-inventory.json).
No model, training, notebook, publisher, or scientific report was executed.
Original ARENA and source material were unchanged. The shared checkout is being
edited concurrently, so this is a diagnostic snapshot, not final acceptance.

## Different denominators

The root audit emitted 404 surface messages, including 55 missing configured
exercise directories. The earlier inventory counted 86 syllabus sections: 19
masters, 12 existing destinations without masters, and 55 destinations without a
solution notebook. The two sets of 55 section IDs coincide in those snapshots,
but missing directory and missing notebook are different predicates. The older
58 absent exact registered filenames included naming mismatches such as 11.5,
15.2 and 17.1; it must not be substituted for either denominator.

The earlier 55 split into 32 with declared legacy notebooks and 23 without.
Method inspection refines that split:

| Inspected category | Count | Interpretation |
| --- | ---: | --- |
| Method-related migration candidates | 29 | Existing lessons teach relevant methods; broader brief gaps and acceptance remain. |
| Declared foundation source insufficient for dedicated lesson | 3 | 12.4, 13.3 and 14.3 have useful components, but lack the new core methods. |
| Partial adjacent calendar lesson | 1 | 11.6 can reuse part of 11.1; spatial/historical-time scope is still absent. |
| No dedicated source located | 22 | Requirements need substantive authoring; adjacent mentions do not count as lessons. |

Thus 25 lack a dedicated lesson for the current methods, although three have
declared foundation notebooks. This does not contradict the earlier 32/23
material-availability split. It adds a method-scope distinction. The 29 candidates
are not 29 complete or accepted lessons. Canonical 16.4 now exists from concurrent
SAGE migration; the frozen missing-directory inventory is retained unchanged.

The remaining original messages describe missing support files/pages, page shape,
renderer recognition, and an execution contract. They are not additional unique
missing lessons. Fourteen renderer-recognition false positives were repaired in
a separate reviewed change; real content and provenance gates remain.

## Exact mappings and scope

Every source path below is a notebook-bearing directory. The JSON records full
notebook filenames, byte hashes, observed definitions and headings. Preserve all
source lessons when several map to one destination, retain historical execution
identity, and reconcile the new brief before assigning acceptance.

### 5.2 — Mixture of Experts from Scratch

Destination: `chapter5_modern_architectures/exercises/part2_mixture_of_experts`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter5_modern_architectures/exercises/part8_mixture_of_experts`

Observed: Top-k routing, capacity/dispatch/combine, balancing and router-z losses, planted
specialization, factorial controls.

Remaining scope: Current brief also asks for small released-model routing; retain the old study
scope and receipt boundary.

### 5.4 — Mamba-2 and State-Space Duality

Destination: `chapter5_modern_architectures/exercises/part4_mamba2_ssd`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter5_modern_architectures/exercises/part7_mamba2_ssd`

Observed: Recurrent, quadratic, and chunked SSD; semiseparable matrix; official reference and
gradient parity; delayed-recall task.

Remaining scope: Algorithm parity is not automatic released-checkpoint parity or fresh CUDA
acceptance.

### 5.5 — State Tracking in SSMs and Transformers

Destination: `chapter5_modern_architectures/exercises/part5_state_tracking`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter5_modern_architectures/exercises/part4_mamba_state_tracking`

Observed: Mamba bracket-depth state tracking, recurrent-state probes and causal transplants,
matched-random state edits.

Remaining scope: Expanded Transformer and Mamba-3-style comparison is not established by this
Mamba source.

### 5.6 — Linear Attention, RetNet and RWKV

Destination: `chapter5_modern_architectures/exercises/part6_linear_attention_retnet_rwkv`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 5.7 — Griffin, xLSTM, Hyena and Hybrid Models

Destination: `chapter5_modern_architectures/exercises/part7_griffin_xlstm_hyena`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 5.8 — Discrete Diffusion Language Models

Destination: `chapter5_modern_architectures/exercises/part8_diffusion_language_models`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter5_modern_architectures/exercises/part5_diffusion_language_models`

Observed: Forward noising, masked denoising loss, remasking/reverse sampler, tiny diffusion
training and trajectories.

Remaining scope: Toy/copy-grammar and real-token inputs do not establish the expanded
TinyStories and DiffusionGemma brief.

### 5.9 — Embedding, Function-Calling and Specialist Models

Destination: `chapter5_modern_architectures/exercises/part9_specialist_models`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter5_modern_architectures/exercises/part6_multimodal_embedding_function_models`

Observed: Embedding pooling/normalization, cosine retrieval, abstaining function routing, real
BGE preflight.

Remaining scope: Does not establish every EmbeddingGemma/FunctionGemma/MedGemma component in the
expanded brief.

### 6.2 — Gemma Scope and Feature Steering

Destination: `chapter6_sparse_feature_methods/exercises/part2_gemma_scope`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter5_modern_architectures/exercises/part2_gemma_scope_feature_steering`
- `chapter6_sparse_feature_methods/exercises/part2_gemma_scope_deep_dive`

Observed: JumpReLU encoding/reconstruction, held-out feature AUROC, ablation and residual
steering; related old chapter-5 steering lesson.

Remaining scope: Keep planted-organism findings separate from real-model preflight and released
Gemma dictionary claims.

### 6.3 — Feature Splitting, Absorption and Dashboards

Destination: `chapter6_sparse_feature_methods/exercises/part3_sae_pathologies_dashboards`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 6.4 — Transcoders and Skip Transcoders

Destination: `chapter6_sparse_feature_methods/exercises/part4_transcoders`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter6_sparse_feature_methods/exercises/part3_transcoders_attribution_graphs`

Observed: Transcoder encode/decode, reconstruction decomposition, signed feature edges, graph
extraction and causal validation.

Remaining scope: No skip-transcoder implementation found in this candidate; exact colored-shape
organism is its main lesson.

### 6.5 — Crosscoders and Model Diffing

Destination: `chapter6_sparse_feature_methods/exercises/part5_crosscoders`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter6_sparse_feature_methods/exercises/part4_crosscoders_model_diffing`

Observed: Sparse crosscoder training, latent ownership, planted-feature matching and causal
ablation.

Remaining scope: Real-model residual-SVD preflight is not trained real-model crosscoder
evidence.

### 7.6 — Automated Interpretability Agents

Destination: `chapter7_activation_to_language/exercises/part6_interpretability_agents`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 8.6 — Sparse Feature Circuits and SHIFT

Destination: `chapter8_automated_circuits/exercises/part6_sparse_feature_circuits`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter8_automated_circuits/exercises/part5_sparse_feature_circuits`

Observed: Exact sparse-feature node/edge patching, EAP-IG, graph thresholds and circuit metrics;
generated SHIFT-style edit.

Remaining scope: Generated SHIFT analogue and exact planted graph do not reproduce the full
released-artifact study.

### 8.7 — Circuit Tracing with Attribution Graphs

Destination: `chapter8_automated_circuits/exercises/part7_attribution_graphs`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter8_automated_circuits/exercises/part4_circuit_tracing_attribution_graphs`

Observed: Planted graph edge ablation, integrated edge attribution, thresholding, path patches
and random/reversed controls.

Remaining scope: Exact toy graph does not by itself satisfy the expanded real-model circuit-
tracing brief.

### 9.2 — Truthfulness Directions and Hallucination Probes

Destination: `chapter9_alignment_interpretability/exercises/part2_truthfulness_hallucination`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 9.3 — Chain-of-Thought Faithfulness

Destination: `chapter9_alignment_interpretability/exercises/part3_cot_faithfulness`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter9_alignment_interpretability/exercises/part2_cot_faithfulness`

Observed: CoT probe heatmaps, forward patches, donor-margin effects and controls.

Remaining scope: Preserve negative Pythia results and distinguish the toy task from broad CoT
faithfulness.

### 9.4 — Emergent Misalignment Detection

Destination: `chapter9_alignment_interpretability/exercises/part4_emergent_misalignment`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter9_alignment_interpretability/exercises/part3_emergent_misalignment_detection`

Observed: Safe-proxy drift timeline, held-out thresholded detection and early-warning controls.

Remaining scope: A safe-proxy detector is not evidence of actual emergent misalignment.

### 9.5 — White-Box Monitors and Red-Teaming with Internals

Destination: `chapter9_alignment_interpretability/exercises/part5_white_box_monitors`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter9_alignment_interpretability/exercises/part4_white_box_evals_monitors`

Observed: White-box monitor AUROC/thresholding with surface-risk, shuffled-label and random
controls.

Remaining scope: Toy monitoring does not establish a complete internal red-team evaluation.

### 9.6 — Model Editing with ROME and MEMIT

Destination: `chapter9_alignment_interpretability/exercises/part6_rome_memit`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter9_alignment_interpretability/exercises/part5_rome_model_editing`

Observed: Covariance-preconditioned rank-one updates, activation patch sweep, value optimization
and locality controls.

Remaining scope: The candidate is course ROME-like editing; MEMIT is not implemented there.

### 9.7 — Unlearning and How to Evaluate It

Destination: `chapter9_alignment_interpretability/exercises/part7_unlearning`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 11.6 — Feature Manifolds - Days, Months, Space and Time

Destination: `chapter11_representation_geometry/exercises/part6_feature_manifolds`.

Classification: `partial_adjacent_calendar_lesson`.

Existing source material:

- `chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls`

Observed: Calendar PCA/SVD, circular regression, calendar-order nulls, probes and causal edits
in 11.1.

Remaining scope: The README explicitly leaves years, cities, geography and historical-time
interventions unfinished; no dedicated complete manifolds lesson exists.

### 12.2 — CLIP Features, SAEs and Typographic Attacks

Destination: `chapter12_vlm_interpretability/exercises/part2_clip_saes_typographic`.

Classification: `no_dedicated_learner_source_located`.

Existing source material:

- `chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls`

Observed: Adjacent 12.1 has typographic attack metrics and a pretrained CLIP study.

Remaining scope: No CLIP SAE training, latent galleries, text-feature/head localization or
ablation-mitigation lesson found; typographic metrics alone do not cover the new brief.

### 12.4 — Visual Token Flow in Real VLMs

Destination: `chapter12_vlm_interpretability/exercises/part4_visual_token_flow`.

Classification: `declared_foundation_insufficient_for_dedicated_lesson`.

Existing source material:

- `chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls`

Observed: CLIP/SigLIP embedding-row/full-vision-token patching and simple Qwen answer
generation.

Remaining scope: Source does not implement the uniform autoregressive-VLM adapter, layerwise
image-token knockout, full-vs-last decoder-token patching, visual-token logit lens or VLM
Patchscopes required by the new brief.

### 12.5 — Object Hallucination and Modality Arbitration

Destination: `chapter12_vlm_interpretability/exercises/part5_object_hallucination`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter12_vlm_interpretability/exercises/part2_object_hallucination_arbitration`

Observed: Rendered object regions, matched-region interventions, modality baselines and visual-
token counterfactual patches.

Remaining scope: Current 12.5 page already exists under the older part2 directory; bounded
rendered geometry is not generic VLM grounding.

### 12.6 — Multimodal SAEs, Transcoders and Crosscoders

Destination: `chapter12_vlm_interpretability/exercises/part6_multimodal_sparse_features`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 12.7 — Dense Vision Features - DINO and SAM

Destination: `chapter12_vlm_interpretability/exercises/part7_dense_vision_features`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 13.1 — Diffusion Models and Cross-Attention Attribution

Destination: `chapter13_image_generation_interpretability/exercises/part1_diffusion_cross_attention`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter13_image_generation_interpretability/exercises/part1_diffusion_image_controls`

Observed: Cross-attention capture/aggregation, DAAM regions, token ablation and same-seed SD1.5
prompt interventions.

Remaining scope: Preserve the exact toy and supporting SD1.5 scope; no fresh model run was
performed here.

### 13.3 — Concept Directions and Diffusion LoRA

Destination: `chapter13_image_generation_interpretability/exercises/part3_concept_directions_lora`.

Classification: `declared_foundation_insufficient_for_dedicated_lesson`.

Existing source material:

- `chapter13_image_generation_interpretability/exercises/part1_diffusion_image_controls`

Observed: latent_direction_effect_report checks supplied baseline/steered/random score tensors.

Remaining scope: A score checker does not learn a real UNet h-space direction, train a diffusion
LoRA or implement concept-slider/SVD steering.

### 13.4 — Sparse Autoencoders for Diffusion Models

Destination: `chapter13_image_generation_interpretability/exercises/part4_diffusion_saes`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 13.5 — Diffusion Transformers and ConceptAttention

Destination: `chapter13_image_generation_interpretability/exercises/part5_dit_concept_attention`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 13.6 — Autoregressive Image Tokens

Destination: `chapter13_image_generation_interpretability/exercises/part6_autoregressive_image_tokens`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 14.1 — I-JEPA from Scratch

Destination: `chapter14_jepa_world_models/exercises/part1_ijepa_from_scratch`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter14_jepa_world_models/exercises/part2_ijepa_from_scratch`

Observed: Patch/block masking, target gathering, normalized latent-prediction loss, EMA and
collapse/probe controls.

Remaining scope: Tiny controlled-scene I-JEPA is not full paper reproduction.

### 14.2 — V-JEPA and Video World Models

Destination: `chapter14_jepa_world_models/exercises/part2_vjepa`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls`
- `chapter14_jepa_world_models/exercises/part3_vjepa_from_scratch`

Observed: Tubelet positions/masks, EMA and L1 prediction, small V-JEPA training, probes and
retrieval.

Remaining scope: Persistence shortcut failure and frozen-V-JEPA preflight remain separate from
learner-trained model claims.

### 14.3 — Action-Conditioned World Models and Latent Planning

Destination: `chapter14_jepa_world_models/exercises/part3_action_conditioned_planning`.

Classification: `declared_foundation_insufficient_for_dedicated_lesson`.

Existing source material:

- `chapter14_jepa_world_models/exercises/part1_jepa_world_model_controls`

Observed: Frozen V-JEPA shape-video transition/probe preflight and latent_rollout_report
comparing supplied losses.

Remaining scope: Source limitations explicitly exclude planning and V-JEPA 2-AC action tokens;
no CEM/MPC action optimizer is taught.

### 14.4 — The JEPA Family

Destination: `chapter14_jepa_world_models/exercises/part4_jepa_family`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 14.5 — Maze Transformers and World Models

Destination: `chapter14_jepa_world_models/exercises/part5_maze_transformers`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter14_jepa_world_models/exercises/part4_maze_transformer_world_model`

Observed: BFS/distance maps, maze tokens, policy training/rollouts, state probes and shuffled-
maze/detour controls.

Remaining scope: Policy/probe evidence does not automatically demonstrate the expanded latent-
planning scope.

### 14.6 — Sudoku and Constraint World Models

Destination: `chapter14_jepa_world_models/exercises/part6_sudoku_transformers`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 14.7 — Recurrent World Models - Dreamer, TD-MPC2 and IRIS

Destination: `chapter14_jepa_world_models/exercises/part7_recurrent_world_models`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 15.1 — LoRA and DoRA from Scratch

Destination: `chapter15_peft_misalignment/exercises/part1_lora_dora`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter15_peft_misalignment/exercises/part1_lora_dora_adapter_controls`

Observed: LoRA delta/merge parity, DoRA recomposition, singular spectra and random-direction
controls.

Remaining scope: Preserve controlled adapter-study scope and historical execution identity.

### 15.4 — LoRA Phase Transitions and Misalignment Proxies

Destination: `chapter15_peft_misalignment/exercises/part4_lora_phase_transitions`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 15.5 — VLM LoRA

Destination: `chapter15_peft_misalignment/exercises/part5_vlm_lora`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 16.1 — Exact Shapley on Ground-Truth Games

Destination: `chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley_ground_truth_games`

Observed: Exact coalitions/Shapley, dividends, efficiency, permutations and leave-one-
out/interaction controls.

Remaining scope: Finite-game truth does not confer real-model causal validity.

### 16.2 — KernelSHAP and PartitionSHAP

Destination: `chapter16_shapley_attribution_baselines/exercises/part2_kernelshap_partitionshap`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter16_shapley_attribution_baselines/exercises/part2_kernelshap_partition_shap_controls`

Observed: Kernel weighting, coalition sampling, weighted regression, grouped partition games and
exact oracles.

Remaining scope: Check expanded hierarchical PartitionSHAP requirements before transferring
completeness.

### 16.3 — TreeSHAP, DeepSHAP and GradientSHAP

Destination: `chapter16_shapley_attribution_baselines/exercises/part3_model_specific_shap`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter16_shapley_attribution_baselines/exercises/part13_treeshap`
- `chapter16_shapley_attribution_baselines/exercises/part14_deepshap`
- `chapter16_shapley_attribution_baselines/exercises/part15_gradientshap`

Observed: Tree path recurrence; DeepSHAP paired forwards/rescale multipliers; Expected Gradients
schedules and quadrature.

Remaining scope: Three long source lessons have distinct games and limitations; preserve all,
rather than collapsing them to summaries.

### 16.4 — SAGE Global Importance

Destination: `chapter16_shapley_attribution_baselines/exercises/part4_sage`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance`

Observed: Marginal predictions, loss-defined games, exact/global Shapley, permutation SAGE and
path summaries.

Remaining scope: Canonical part4_sage now exists from concurrent migration; original missing-dir
count remains historical.

### 16.5 — Shapley Interactions

Destination: `chapter16_shapley_attribution_baselines/exercises/part5_shapley_interactions`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter16_shapley_attribution_baselines/exercises/part16_shapley_taylor_faith_shap`
- `chapter16_shapley_attribution_baselines/exercises/part17_shapiq_estimators`
- `chapter16_shapley_attribution_baselines/exercises/part3_shapley_interactions_shapiq`

Observed: Exact pair interactions and shapiq parity, Shapley-Taylor/Faith-Shap, higher-order
estimators and sampling error.

Remaining scope: Preserve all three source lessons and each allocation definition.

### 16.7 — TokenSHAP and Region SHAP for LLMs and VLMs

Destination: `chapter16_shapley_attribution_baselines/exercises/part7_token_region_shap`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter16_shapley_attribution_baselines/exercises/part4_tokenshap_token_shapley`
- `chapter16_shapley_attribution_baselines/exercises/part5_vlm_modality_region_shap`

Observed: Exact/permutation token coalition games, token splitting/grouping, rendered
modality/region games and pinned CLIP scoring.

Remaining scope: Token and region games are separate material; source availability is not
merged-lesson acceptance.

### 16.8 — SHAP vs Activation Patching

Destination: `chapter16_shapley_attribution_baselines/exercises/part8_shap_vs_patching`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter16_shapley_attribution_baselines/exercises/part6_shap_vs_activation_patching`
- `chapter16_shapley_attribution_baselines/exercises/part8_shapley_mechinterp_agreement`

Observed: Exact hidden coalitions/Shapley, activation intervention agreement and shuffled/random
controls; additional comparison lesson.

Remaining scope: Keep attribution-game assumptions distinct from causal activation-patching
claims.

### 16.10 — Influence Functions and TracIn

Destination: `chapter16_shapley_attribution_baselines/exercises/part10_influence_tracin`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter16_shapley_attribution_baselines/exercises/part9_influence_and_tracin`

Observed: Logistic Hessians/HVP/CG, upweight influence, SGD checkpoint TracIn and class-matched
deletion tests.

Remaining scope: Carry current scientific definitions and limitations through renumbering; do
not reuse Data Shapley identity by number.

### 16.11 — Datamodels and Dataset Cartography

Destination: `chapter16_shapley_attribution_baselines/exercises/part11_datamodels_cartography`.

Classification: `method_related_migration_candidate`.

Existing source material:

- `chapter16_shapley_attribution_baselines/exercises/part10_datamodels`
- `chapter16_shapley_attribution_baselines/exercises/part11_dataset_cartography`

Observed: Subset-response datamodels with lasso/KKT and held-out targets; ID-aligned epoch
statistics, map-region selection and retraining.

Remaining scope: Datamodels and cartography are separate long lessons, not a single
interchangeable metric.

### 17.2 — Induction Head Emergence

Destination: `chapter17_training_dynamics/exercises/part2_induction_head_emergence`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 17.3 — Fine-Tuning Diffs and Capability Emergence

Destination: `chapter17_training_dynamics/exercises/part3_finetuning_diffs_emergence`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 17.4 — Training Dynamics Across Architectures

Destination: `chapter17_training_dynamics/exercises/part4_cross_architecture_dynamics`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 18.1 — Concept Bottlenecks, Backpacks and Interchange Intervention Training

Destination: `chapter18_intrinsic_interpretability/exercises/part1_concept_bottlenecks_backpacks_iit`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

### 18.2 — Sparse and Modular Training, KANs and Sparse Trees

Destination: `chapter18_intrinsic_interpretability/exercises/part2_sparse_modular_kans`.

Classification: `no_dedicated_learner_source_located`.

Observed: No dedicated maintained learner notebook found for the configured methods.

Remaining scope: The current syllabus brief remains a requirement, not completed teaching.

## Action boundary

This inventory changes no syllabus, configuration, status, scientific method,
threshold, or report. Reconcile mappings without silently replacing original
materials, fabricating fresh execution, accepting adjacent components as a full
lesson, or dropping controls. Grouped SHAP and data-attribution sections must
retain the original long lessons. The priority-pilot and human learner-walkthrough
gates remain open under their existing owners.
