# Study guide

ARENA Extended is one route of 18 stages. It starts at original ARENA's first section, 0.0, runs through
original ARENA chapters 0–4, and brings in each extension chapter as soon as you have what it builds on. Each
stage has a few **core** lessons; the rest are **optional**, and you can take them in any order.

## Start here

- **New to ARENA?** Start at original ARENA [0.0 Prerequisites](https://learn.arena.education/chapter0_fundamentals/00_prereqs/),
  stage 1 of the route, and follow the stages in order.
- **Done ARENA chapters 0–1?** Start at 0.6, *How to Know When an Interpretability Result Is Fake*, which is stage 4.
  Original ARENA's chapters 2–4 come later on the route, before the extension chapters that build on them.

Original ARENA lessons open on [learn.arena.education](https://learn.arena.education/). Read the extension
lessons in the study reader. Run this from the repository root:

```bash
uv run --no-sync --offline streamlit run mastery_hub/read.py --server.address=127.0.0.1 --server.port=18505 --server.headless=true --browser.gatherUsageStats=false
```

Then open [the reader](http://127.0.0.1:18505/). Its sidebar shows the stage you're on, and the end of each
lesson has a **Next on the route →** link. The reader's address is your bookmark: it names the lesson and
part you're reading.

## How the route works

- **Core and optional.** Follow the core lessons in order. Take optional lessons whenever they interest you, in
  any order, once you've done what they need. A lesson's **Prerequisites** line names the lessons to do first.
- **Original ARENA on-ramps.** Some original ARENA sections sit just before the extension chapter that builds on
  them: superposition and SAEs (1.5.4, 1.3.3) before chapter 6, IOI (1.4.1) before chapter 8, linear probes
  (1.3.1) before chapter 11.
- **Chapter 10 comes twice.** Parts 1️⃣–4️⃣ of 10.1 come after the interpretability core, at stage 10. Part 5️⃣,
  your own capstone, is the route's finale.
- **Alchemy side trips.** Each [Deep Learning Alchemy](https://deep-learning-alchemy.github.io/) companion is
  an optional side trip at the stage where its unit fits. The optimizer lab goes with unit 2.
- **Pacing.** A day is about six hours of a lesson's "You should spend up to N minutes" lines, rounded to half
  days, with at least one day per lesson. Shorter lessons and afternoons, such as 10.2, count as half a day.
  Nothing is timed or recorded.

## The 18 stages

<!-- The stage table below is generated from the syllabus route: uv run python infrastructure/extension/route.py --write -->

| Stage | | Core | Optional | Core days | All days |
|---|---|---|---|---|---|
| 1 | Start | [0.0](https://learn.arena.education/chapter0_fundamentals/00_prereqs/) | — | 1 | 1 |
| 2 | Fundamentals | [0.1](https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/), [0.2](https://learn.arena.education/chapter0_fundamentals/02_cnns/), [0.3](https://learn.arena.education/chapter0_fundamentals/03_optimization/), [0.4](https://learn.arena.education/chapter0_fundamentals/04_backprop/) | [0.5.1](https://learn.arena.education/chapter0_fundamentals/51_vaes/), [0.5.2](https://learn.arena.education/chapter0_fundamentals/52_gans/), [0.7], u01 (side trip), u02 (side trip), u03 (side trip) | 4.5 | 7.5 |
| 3 | Transformers | [1.1](https://learn.arena.education/chapter1_transformer_interp/01_transformers/), [1.2](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) | [1.6](../chapter1_transformer_interp/instructions/pages/40_%5B1.6%5D_Local_Frontier_ML_Infrastructure_Practice.md), u04 (side trip) | 2 | 3 |
| 4 | Is it real? | [0.6](../chapter0_fundamentals/instructions/pages/06_%5B0.6%5D_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice.md) | — | 1 | 1 |
| 5 | Modern architectures | [5.1](../chapter5_modern_architectures/instructions/pages/01_%5B5.1%5D_Gemma_from_Scratch.md), [5.3](../chapter5_modern_architectures/instructions/pages/03_%5B5.3%5D_Mamba_from_Scratch.md) | [5.2](../chapter5_modern_architectures/instructions/pages/02_%5B5.2%5D_Mixture_of_Experts_from_Scratch.md), [5.4](../chapter5_modern_architectures/instructions/pages/04_%5B5.4%5D_Mamba-2_and_State-Space_Duality.md), [5.5](../chapter5_modern_architectures/instructions/pages/05_%5B5.5%5D_State_Tracking_in_SSMs_and_Transformers.md), [5.6](../chapter5_modern_architectures/instructions/pages/06_%5B5.6%5D_Linear_Attention_RetNet_and_RWKV.md), [5.7](../chapter5_modern_architectures/instructions/pages/07_%5B5.7%5D_Griffin_xLSTM_Hyena_and_Hybrid_Models.md), [5.8](../chapter5_modern_architectures/instructions/pages/08_%5B5.8%5D_Discrete_Diffusion_Language_Models.md), [5.9](../chapter5_modern_architectures/instructions/pages/09_%5B5.9%5D_Embedding_Function-Calling_and_Specialist_Models.md), u05 (side trip) | 2 | 9.5 |
| 6 | Features | [1.5.4](https://learn.arena.education/chapter1_transformer_interp/34_superposition/), [1.3.3](https://learn.arena.education/chapter1_transformer_interp/13_saes/), [6.1](../chapter6_sparse_feature_methods/instructions/pages/01_%5B6.1%5D_SAE_Variants.md), [6.2](../chapter6_sparse_feature_methods/instructions/pages/02_%5B6.2%5D_Gemma_Scope_and_Feature_Steering.md) | [6.3](../chapter6_sparse_feature_methods/instructions/pages/03_%5B6.3%5D_Feature_Splitting_Absorption_and_Dashboards.md), [6.4](../chapter6_sparse_feature_methods/instructions/pages/04_%5B6.4%5D_Transcoders_and_Attribution_Graphs.md), [6.5](../chapter6_sparse_feature_methods/instructions/pages/05_%5B6.5%5D_Crosscoders_and_Model_Diffing.md) | 5 | 8 |
| 7 | Circuits | [1.4.1](https://learn.arena.education/chapter1_transformer_interp/21_ioi/), [8.1](../chapter8_automated_circuits/instructions/pages/01_%5B8.1%5D_Activation_Patching_Refresher.md), [8.2](../chapter8_automated_circuits/instructions/pages/02_%5B8.2%5D_Attribution_Patching_and_EAP.md) | [1.4.2](https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/), [8.3](../chapter8_automated_circuits/instructions/pages/03_%5B8.3%5D_AtP_from_Scratch.md), [8.4](../chapter8_automated_circuits/instructions/pages/04_%5B8.4%5D_ACDC_and_Circuit_Metrics.md), [8.5](../chapter8_automated_circuits/instructions/pages/05_%5B8.5%5D_Causal_Scrubbing.md), [8.6](../chapter8_automated_circuits/instructions/pages/06_%5B8.6%5D_Sparse_Feature_Circuits_and_SHIFT.md), [8.7](../chapter8_automated_circuits/instructions/pages/07_%5B8.7%5D_Circuit_Tracing_with_Attribution_Graphs.md), [8.8](../chapter8_automated_circuits/instructions/pages/08_%5B8.8%5D_Benchmarking_Circuit_Discovery_on_Compiled_Transformers.md) | 3 | 10.5 |
| 8 | Reading activations | [7.1](../chapter7_activation_to_language/instructions/pages/01_%5B7.1%5D_Logit_Lens_Tuned_Lens_and_Patchscopes.md) | [1.3.4](https://learn.arena.education/chapter1_transformer_interp/14_activation_oracles/), [7.2](../chapter7_activation_to_language/instructions/pages/02_%5B7.2%5D_Feature_Verbalizers.md), [7.3](../chapter7_activation_to_language/instructions/pages/03_%5B7.3%5D_Mini_Activation_Oracles.md), [7.4](../chapter7_activation_to_language/instructions/pages/04_%5B7.4%5D_Mini_Natural_Language_Autoencoders.md), [7.5](../chapter7_activation_to_language/instructions/pages/05_%5B7.5%5D_Predictive_Concept_Decoders.md), [7.6](../chapter7_activation_to_language/instructions/pages/06_%5B7.6%5D_Automated_Interpretability_Agents.md) | 1 | 7 |
| 9 | Geometry | [1.3.1](https://learn.arena.education/chapter1_transformer_interp/11_probing/), [11.1](../chapter11_representation_geometry/instructions/pages/01_%5B11.1%5D_PCA_SVD_and_Geometry_Controls.md), [11.2, parts 1️⃣–2️⃣](../chapter11_representation_geometry/instructions/pages/02_%5B11.2%5D_CCA_SVCCA_and_Representation_Alignment.md) | [11.2, from part 3️⃣](../chapter11_representation_geometry/instructions/pages/02_%5B11.2%5D_CCA_SVCCA_and_Representation_Alignment.md), [11.3](../chapter11_representation_geometry/instructions/pages/03_%5B11.3%5D_Causal_Abstraction_and_Distributed_Alignment_Search.md), [11.4](../chapter11_representation_geometry/instructions/pages/04_%5B11.4%5D_Concept_Erasure.md), [11.5](../chapter11_representation_geometry/instructions/pages/05_%5B11.5%5D_Contrast-Consistent_Search.md), [11.6](../chapter11_representation_geometry/instructions/pages/06_%5B11.6%5D_Feature_Manifolds_Days_Months_Space_and_Time.md) | 3 | 7.5 |
| 10 | Research Practice I | [10.1, parts 1️⃣–4️⃣](../chapter10_capstone_research_sprint/instructions/pages/01_%5B10.1%5D_Capstone_Research_Sprint.md), [10.2] | [10.3] | 1.5 | 2 |
| 11 | RL | [2.1](https://learn.arena.education/chapter2_rl/01_intro_rl/), [2.2.1](https://learn.arena.education/chapter2_rl/21_dqn/), [2.2.2](https://learn.arena.education/chapter2_rl/22_vpg/), [2.3](https://learn.arena.education/chapter2_rl/03_ppo/) | [2.3.2](https://learn.arena.education/chapter2_rl/032_goal_misgen/), [2.4.1](https://learn.arena.education/chapter2_rl/04_rlhf/), [2.4.2](https://learn.arena.education/chapter2_rl/042_rlvr/), [2.4.3](https://learn.arena.education/chapter2_rl/043_rlaif/), [2.5](https://learn.arena.education/chapter2_rl/05_mcts_alphazero/) | 4 | 9 |
| 12 | Evals | [3.1](https://learn.arena.education/chapter3_llm_evals/01_intro_evals/), [3.2](https://learn.arena.education/chapter3_llm_evals/02_dataset_gen/), [3.3](https://learn.arena.education/chapter3_llm_evals/03_running_evals/) | [3.4](https://learn.arena.education/chapter3_llm_evals/04_llm_agents/), [3.5](https://learn.arena.education/chapter3_llm_evals/05_ai_control/) | 3 | 5 |
| 13 | Alignment | [4.1](https://learn.arena.education/chapter4_alignment_science/1_emergent_misalignment/), [1.3.2](https://learn.arena.education/chapter1_transformer_interp/12_function_vectors/), [4.4](https://learn.arena.education/chapter4_alignment_science/4_persona_vectors/), [9.1](../chapter9_alignment_interpretability/instructions/pages/01_%5B9.1%5D_Refusal_Directions_and_Representation_Engineering.md), [15.1](../chapter15_peft_misalignment/instructions/pages/01_%5B15.1%5D_LoRA_and_DoRA_from_Scratch.md) | [4.2](https://learn.arena.education/chapter4_alignment_science/2_science_misalignment/), [4.3](https://learn.arena.education/chapter4_alignment_science/3_reasoning_models/), [4.5](https://learn.arena.education/chapter4_alignment_science/5_investigator_agents/), [9.2](../chapter9_alignment_interpretability/instructions/pages/02_%5B9.2%5D_Truthfulness_Directions_and_Hallucination_Probes.md), [9.3](../chapter9_alignment_interpretability/instructions/pages/03_%5B9.3%5D_Chain-of-Thought_Faithfulness.md), [9.4](../chapter9_alignment_interpretability/instructions/pages/04_%5B9.4%5D_Emergent_Misalignment_Detection.md), [9.5](../chapter9_alignment_interpretability/instructions/pages/05_%5B9.5%5D_White-Box_Monitors_and_Red-Teaming_with_Internals.md), [9.6](../chapter9_alignment_interpretability/instructions/pages/06_%5B9.6%5D_Rank-One_Model_Editing_ROME.md), [9.7](../chapter9_alignment_interpretability/instructions/pages/07_%5B9.7%5D_Unlearning_and_How_to_Evaluate_It.md), [15.2](../chapter15_peft_misalignment/instructions/pages/02_%5B15.2%5D_LoRA_vs_Full_Fine-Tuning.md), [15.3](../chapter15_peft_misalignment/instructions/pages/03_%5B15.3%5D_QLoRA_from_Scratch.md), [15.4](../chapter15_peft_misalignment/instructions/pages/04_%5B15.4%5D_LoRA_Phase_Transitions_and_Misalignment_Proxies.md) | 5.5 | 17.5 |
| 14 | Multimodal | [12.1](../chapter12_vlm_interpretability/instructions/pages/01_%5B12.1%5D_CLIP_and_SigLIP_from_Scratch.md), [12.3](../chapter12_vlm_interpretability/instructions/pages/03_%5B12.3%5D_Mini_VLM_from_Scratch.md), [13.1](../chapter13_image_generation_interpretability/instructions/pages/01_%5B13.1%5D_Diffusion_Models_and_Cross-Attention_Attribution.md) | [12.2](../chapter12_vlm_interpretability/instructions/pages/02_%5B12.2%5D_Typographic_Attacks_and_SAE_Features_in_a_Toy_Two-Tower_Model.md), [12.4](../chapter12_vlm_interpretability/instructions/pages/04_%5B12.4%5D_Visual_Token_Flow_in_Real_VLMs.md), [12.5](../chapter12_vlm_interpretability/instructions/pages/05_%5B12.5%5D_Object_Hallucination_and_Modality_Arbitration.md), [12.6](../chapter12_vlm_interpretability/instructions/pages/06_%5B12.6%5D_Multimodal_SAEs_Transcoders_and_Crosscoders.md), [12.7](../chapter12_vlm_interpretability/instructions/pages/07_%5B12.7%5D_Dense_Vision_Features_DINO_and_SAM.md), [15.5](../chapter15_peft_misalignment/instructions/pages/05_%5B15.5%5D_VLM_LoRA.md), [13.2](../chapter13_image_generation_interpretability/instructions/pages/02_%5B13.2%5D_Denoising-Time_Causal_Patching.md), [13.3](../chapter13_image_generation_interpretability/instructions/pages/03_%5B13.3%5D_Concept_Directions_and_Diffusion_LoRA.md), [13.4](../chapter13_image_generation_interpretability/instructions/pages/04_%5B13.4%5D_Sparse_Autoencoders_for_Diffusion_Models.md), [13.5](../chapter13_image_generation_interpretability/instructions/pages/05_%5B13.5%5D_Diffusion_Transformers_and_ConceptAttention.md), [13.6](../chapter13_image_generation_interpretability/instructions/pages/06_%5B13.6%5D_Autoregressive_Image_Tokens.md) | 3 | 14 |
| 15 | World models and convergence | [14.1](../chapter14_jepa_world_models/instructions/pages/01_%5B14.1%5D_I-JEPA_from_Scratch.md), [19.1], [19.2] | [1.5.3](https://learn.arena.education/chapter1_transformer_interp/33_othellogpt/), [14.2](../chapter14_jepa_world_models/instructions/pages/02_%5B14.2%5D_V-JEPA_and_Video_World_Models.md), [14.3](../chapter14_jepa_world_models/instructions/pages/03_%5B14.3%5D_Action-Conditioned_World_Models_and_Latent_Planning.md), [14.4](../chapter14_jepa_world_models/instructions/pages/04_%5B14.4%5D_The_JEPA_Family.md), [14.5](../chapter14_jepa_world_models/instructions/pages/05_%5B14.5%5D_Maze_Transformers_and_World_Models.md), [14.6](../chapter14_jepa_world_models/instructions/pages/06_%5B14.6%5D_Sudoku_and_Constraint_World_Models.md), [14.7](../chapter14_jepa_world_models/instructions/pages/07_%5B14.7%5D_Recurrent_World_Models_Dreamer_TD-MPC2_and_IRIS.md), [19.3] | 2.5 | 11 |
| 16 | Attribution | [16.1](../chapter16_shapley_attribution_baselines/instructions/pages/01_%5B16.1%5D_Exact_Shapley_on_Ground-Truth_Games.md), [16.2](../chapter16_shapley_attribution_baselines/instructions/pages/02_%5B16.2%5D_KernelSHAP_and_PartitionSHAP.md) | [16.3](../chapter16_shapley_attribution_baselines/instructions/pages/03_%5B16.3%5D_TreeSHAP_DeepSHAP_and_GradientSHAP_Practice.md), [16.4](../chapter16_shapley_attribution_baselines/instructions/pages/04_%5B16.4%5D_SAGE_Global_Importance.md), [16.5](../chapter16_shapley_attribution_baselines/instructions/pages/05_%5B16.5%5D_Shapley_Interactions.md), [16.6](../chapter16_shapley_attribution_baselines/instructions/pages/06_%5B16.6%5D_Sparse_Interaction_Discovery_with_SPEX_and_ProxySPEX.md), [16.7](../chapter16_shapley_attribution_baselines/instructions/pages/07_%5B16.7%5D_TokenSHAP_and_Region_SHAP_for_LLMs_and_VLMs.md), [16.8](../chapter16_shapley_attribution_baselines/instructions/pages/08_%5B16.8%5D_SHAP_vs_Activation_Patching.md), [16.9](../chapter16_shapley_attribution_baselines/instructions/pages/09_%5B16.9%5D_Data_Shapley_in_One_Training_Run.md), [16.10](../chapter16_shapley_attribution_baselines/instructions/pages/10_%5B16.10%5D_Influence_Functions_and_TracIn.md), [16.11](../chapter16_shapley_attribution_baselines/instructions/pages/11_%5B16.11%5D_Datamodels_and_Dataset_Cartography.md), u06 (side trip) | 2 | 12.5 |
| 17 | Dynamics and intrinsic | [1.5.2](https://learn.arena.education/chapter1_transformer_interp/32_grokking/), [17.1](../chapter17_training_dynamics/instructions/pages/01_%5B17.1%5D_Checkpoint_Archaeology_on_Modular_Addition.md), [18.1](../chapter18_intrinsic_interpretability/instructions/pages/01_%5B18.1%5D_Concept_Bottlenecks_Backpacks_and_Interchange_Intervention_Training.md) | [1.5.1](https://learn.arena.education/chapter1_transformer_interp/31_brackets/), [17.2](../chapter17_training_dynamics/instructions/pages/02_%5B17.2%5D_Induction_Heads_in_Hand-Built_Circuits.md), [17.3](../chapter17_training_dynamics/instructions/pages/03_%5B17.3%5D_Fine-Tuning_Diffs_and_Capability_Emergence.md), [17.4](../chapter17_training_dynamics/instructions/pages/04_%5B17.4%5D_Training_Dynamics_Across_Architectures.md), [18.2](../chapter18_intrinsic_interpretability/instructions/pages/02_%5B18.2%5D_Sparse_and_Modular_Training_KANs_and_Sparse_Trees.md) | 3 | 8 |
| 18 | Finale | [10.1, from part 5️⃣](../chapter10_capstone_research_sprint/instructions/pages/01_%5B10.1%5D_Capstone_Research_Sprint.md) | u07 (side trip) | 3 | 3 |

The core route is about 50 days (about 10 weeks full-time). Everything, optional lessons included, is about 137 days (about 27 weeks). Alchemy side trips are not counted.

## The route, stage by stage

### 1. Start

Core (1 day):

- [[0.0] Prerequisites](https://learn.arena.education/chapter0_fundamentals/00_prereqs/) (original ARENA)

### 2. Fundamentals

Core (4.5 days):

- [[0.1] Ray Tracing](https://learn.arena.education/chapter0_fundamentals/01_ray_tracing/) (original ARENA)
- [[0.2] CNNs & ResNets](https://learn.arena.education/chapter0_fundamentals/02_cnns/) (original ARENA)
- [[0.3] Optimization](https://learn.arena.education/chapter0_fundamentals/03_optimization/) (original ARENA)
- [[0.4] Backpropagation](https://learn.arena.education/chapter0_fundamentals/04_backprop/) (original ARENA)

Optional, in any order:

- [[0.5.1] VAEs](https://learn.arena.education/chapter0_fundamentals/51_vaes/) (original ARENA)
- [[0.5.2] GANs](https://learn.arena.education/chapter0_fundamentals/52_gans/) (original ARENA)
- [0.7] Foundations Toolkit

Side trips:

- Alchemy u01 · Basics, an optional side trip
- Alchemy u02 · Hyperparameter scaling, an optional side trip: with the optimizer lab (`uv run streamlit run mastery_hub/app.py --server.address=127.0.0.1 --server.port=18506`)
- Alchemy u03 · Optimization, an optional side trip

### 3. Transformers

Core (2 days):

- [[1.1] Transformers from Scratch](https://learn.arena.education/chapter1_transformer_interp/01_transformers/) (original ARENA)
- [[1.2] Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) (original ARENA)

Optional, in any order:

- [[1.6] Local Frontier ML Infrastructure (Practice)](../chapter1_transformer_interp/instructions/pages/40_%5B1.6%5D_Local_Frontier_ML_Infrastructure_Practice.md)

Side trips:

- Alchemy u04 · Architecture, an optional side trip

### 4. Is it real?

Core (1 day):

- [[0.6] How to Know When an Interpretability Result Is Fake (Practice)](../chapter0_fundamentals/instructions/pages/06_%5B0.6%5D_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice.md)

### 5. Modern architectures

Core (2 days):

- [[5.1] Gemma from Scratch](../chapter5_modern_architectures/instructions/pages/01_%5B5.1%5D_Gemma_from_Scratch.md)
- [[5.3] Mamba from Scratch](../chapter5_modern_architectures/instructions/pages/03_%5B5.3%5D_Mamba_from_Scratch.md)

Optional, in any order:

- [[5.2] Mixture of Experts from Scratch](../chapter5_modern_architectures/instructions/pages/02_%5B5.2%5D_Mixture_of_Experts_from_Scratch.md)
- [[5.4] Mamba-2 and State-Space Duality](../chapter5_modern_architectures/instructions/pages/04_%5B5.4%5D_Mamba-2_and_State-Space_Duality.md)
- [[5.5] State Tracking in SSMs and Transformers](../chapter5_modern_architectures/instructions/pages/05_%5B5.5%5D_State_Tracking_in_SSMs_and_Transformers.md)
- [[5.6] Linear Attention, RetNet and RWKV](../chapter5_modern_architectures/instructions/pages/06_%5B5.6%5D_Linear_Attention_RetNet_and_RWKV.md)
- [[5.7] Griffin, xLSTM, Hyena and Hybrid Models](../chapter5_modern_architectures/instructions/pages/07_%5B5.7%5D_Griffin_xLSTM_Hyena_and_Hybrid_Models.md) (after [5.6] Linear Attention, RetNet and RWKV)
- [[5.8] Discrete Diffusion Language Models](../chapter5_modern_architectures/instructions/pages/08_%5B5.8%5D_Discrete_Diffusion_Language_Models.md)
- [[5.9] Embedding, Function-Calling and Specialist Models](../chapter5_modern_architectures/instructions/pages/09_%5B5.9%5D_Embedding_Function-Calling_and_Specialist_Models.md)

Side trips:

- Alchemy u05 · Capacity, an optional side trip

### 6. Features

Core (5 days):

- [[1.5.4] Superposition & SAEs](https://learn.arena.education/chapter1_transformer_interp/34_superposition/) (original ARENA)
- [[1.3.3] Interpretability with SAEs](https://learn.arena.education/chapter1_transformer_interp/13_saes/) (original ARENA)
- [[6.1] SAE Variants](../chapter6_sparse_feature_methods/instructions/pages/01_%5B6.1%5D_SAE_Variants.md)
- [[6.2] Gemma Scope and Feature Steering](../chapter6_sparse_feature_methods/instructions/pages/02_%5B6.2%5D_Gemma_Scope_and_Feature_Steering.md)

Optional, in any order:

- [[6.3] Feature Splitting, Absorption and Dashboards](../chapter6_sparse_feature_methods/instructions/pages/03_%5B6.3%5D_Feature_Splitting_Absorption_and_Dashboards.md)
- [[6.4] Transcoders and Attribution Graphs](../chapter6_sparse_feature_methods/instructions/pages/04_%5B6.4%5D_Transcoders_and_Attribution_Graphs.md)
- [[6.5] Crosscoders and Model Diffing](../chapter6_sparse_feature_methods/instructions/pages/05_%5B6.5%5D_Crosscoders_and_Model_Diffing.md)

### 7. Circuits

Core (3 days):

- [[1.4.1] Indirect Object Identification](https://learn.arena.education/chapter1_transformer_interp/21_ioi/) (original ARENA)
- [[8.1] Activation Patching Refresher](../chapter8_automated_circuits/instructions/pages/01_%5B8.1%5D_Activation_Patching_Refresher.md)
- [[8.2] Attribution Patching and EAP](../chapter8_automated_circuits/instructions/pages/02_%5B8.2%5D_Attribution_Patching_and_EAP.md)

Optional, in any order:

- [[1.4.2] SAE Circuits](https://learn.arena.education/chapter1_transformer_interp/22_sae_circuits/) (original ARENA)
- [[8.3] AtP* from Scratch](../chapter8_automated_circuits/instructions/pages/03_%5B8.3%5D_AtP_from_Scratch.md)
- [[8.4] ACDC and Circuit Metrics](../chapter8_automated_circuits/instructions/pages/04_%5B8.4%5D_ACDC_and_Circuit_Metrics.md)
- [[8.5] Causal Scrubbing](../chapter8_automated_circuits/instructions/pages/05_%5B8.5%5D_Causal_Scrubbing.md) (after [8.4] ACDC and Circuit Metrics)
- [[8.6] Sparse Feature Circuits and SHIFT](../chapter8_automated_circuits/instructions/pages/06_%5B8.6%5D_Sparse_Feature_Circuits_and_SHIFT.md)
- [[8.7] Circuit Tracing with Attribution Graphs](../chapter8_automated_circuits/instructions/pages/07_%5B8.7%5D_Circuit_Tracing_with_Attribution_Graphs.md) (after [6.4] Transcoders and Attribution Graphs)
- [[8.8] Benchmarking Circuit Discovery on Compiled Transformers](../chapter8_automated_circuits/instructions/pages/08_%5B8.8%5D_Benchmarking_Circuit_Discovery_on_Compiled_Transformers.md) (after [8.3] AtP* from Scratch and [8.4] ACDC and Circuit Metrics)

### 8. Reading activations

Core (1 day):

- [[7.1] Logit Lens, Tuned Lens and Patchscopes](../chapter7_activation_to_language/instructions/pages/01_%5B7.1%5D_Logit_Lens_Tuned_Lens_and_Patchscopes.md)

Optional, in any order:

- [[1.3.4] Activation Oracles](https://learn.arena.education/chapter1_transformer_interp/14_activation_oracles/) (original ARENA)
- [[7.2] Feature Verbalizers](../chapter7_activation_to_language/instructions/pages/02_%5B7.2%5D_Feature_Verbalizers.md)
- [[7.3] Mini Activation Oracles](../chapter7_activation_to_language/instructions/pages/03_%5B7.3%5D_Mini_Activation_Oracles.md)
- [[7.4] Mini Natural Language Autoencoders](../chapter7_activation_to_language/instructions/pages/04_%5B7.4%5D_Mini_Natural_Language_Autoencoders.md) (after [7.3] Mini Activation Oracles)
- [[7.5] Predictive Concept Decoders](../chapter7_activation_to_language/instructions/pages/05_%5B7.5%5D_Predictive_Concept_Decoders.md) (after [7.3] Mini Activation Oracles)
- [[7.6] Automated Interpretability Agents](../chapter7_activation_to_language/instructions/pages/06_%5B7.6%5D_Automated_Interpretability_Agents.md) (after [7.2] Feature Verbalizers and [6.3] Feature Splitting, Absorption and Dashboards)

### 9. Geometry

Core (3 days):

- [[1.3.1] Linear Probes](https://learn.arena.education/chapter1_transformer_interp/11_probing/) (original ARENA)
- [[11.1] PCA, SVD and Geometry Controls](../chapter11_representation_geometry/instructions/pages/01_%5B11.1%5D_PCA_SVD_and_Geometry_Controls.md)
- [[11.2] CCA, SVCCA and Representation Alignment, parts 1️⃣–2️⃣](../chapter11_representation_geometry/instructions/pages/02_%5B11.2%5D_CCA_SVCCA_and_Representation_Alignment.md)

Optional, in any order:

- [[11.2] CCA, SVCCA and Representation Alignment, from part 3️⃣](../chapter11_representation_geometry/instructions/pages/02_%5B11.2%5D_CCA_SVCCA_and_Representation_Alignment.md)
- [[11.3] Causal Abstraction and Distributed Alignment Search](../chapter11_representation_geometry/instructions/pages/03_%5B11.3%5D_Causal_Abstraction_and_Distributed_Alignment_Search.md)
- [[11.4] Concept Erasure](../chapter11_representation_geometry/instructions/pages/04_%5B11.4%5D_Concept_Erasure.md)
- [[11.5] Contrast-Consistent Search](../chapter11_representation_geometry/instructions/pages/05_%5B11.5%5D_Contrast-Consistent_Search.md)
- [[11.6] Feature Manifolds - Days, Months, Space and Time](../chapter11_representation_geometry/instructions/pages/06_%5B11.6%5D_Feature_Manifolds_Days_Months_Space_and_Time.md)

### 10. Research Practice I

Core (1.5 days):

- [[10.1] Capstone Research Sprint, parts 1️⃣–4️⃣](../chapter10_capstone_research_sprint/instructions/pages/01_%5B10.1%5D_Capstone_Research_Sprint.md)
- [10.2] Replicate and Critique a Fresh Claim

Optional, in any order:

- [10.3] Micro-Lab ML01

### 11. RL

Core (4 days):

- [[2.1] Intro to RL](https://learn.arena.education/chapter2_rl/01_intro_rl/) (original ARENA)
- [[2.2.1] DQN](https://learn.arena.education/chapter2_rl/21_dqn/) (original ARENA)
- [[2.2.2] VPG](https://learn.arena.education/chapter2_rl/22_vpg/) (original ARENA)
- [[2.3] PPO](https://learn.arena.education/chapter2_rl/03_ppo/) (original ARENA)

Optional, in any order:

- [[2.3.2] Goal Misgeneralisation](https://learn.arena.education/chapter2_rl/032_goal_misgen/) (original ARENA)
- [[2.4.1] RLHF](https://learn.arena.education/chapter2_rl/04_rlhf/) (original ARENA)
- [[2.4.2] RLVR](https://learn.arena.education/chapter2_rl/042_rlvr/) (original ARENA)
- [[2.4.3] RLAIF](https://learn.arena.education/chapter2_rl/043_rlaif/) (original ARENA)
- [[2.5] MCTS & AlphaZero](https://learn.arena.education/chapter2_rl/05_mcts_alphazero/) (original ARENA)

### 12. Evals

Core (3 days):

- [[3.1] Intro to Evals](https://learn.arena.education/chapter3_llm_evals/01_intro_evals/) (original ARENA)
- [[3.2] Dataset Generation](https://learn.arena.education/chapter3_llm_evals/02_dataset_gen/) (original ARENA)
- [[3.3] Running Evals with Inspect](https://learn.arena.education/chapter3_llm_evals/03_running_evals/) (original ARENA)

Optional, in any order:

- [[3.4] LLM Agents](https://learn.arena.education/chapter3_llm_evals/04_llm_agents/) (original ARENA)
- [[3.5] AI Control](https://learn.arena.education/chapter3_llm_evals/05_ai_control/) (original ARENA)

### 13. Alignment

Core (5.5 days):

- [[4.1] Emergent Misalignment](https://learn.arena.education/chapter4_alignment_science/1_emergent_misalignment/) (original ARENA)
- [[1.3.2] Function Vectors & Model Steering](https://learn.arena.education/chapter1_transformer_interp/12_function_vectors/) (original ARENA)
- [[4.4] LLM Psychology & Persona Vectors](https://learn.arena.education/chapter4_alignment_science/4_persona_vectors/) (original ARENA)
- [[9.1] Refusal Directions and Representation Engineering](../chapter9_alignment_interpretability/instructions/pages/01_%5B9.1%5D_Refusal_Directions_and_Representation_Engineering.md)
- [[15.1] LoRA and DoRA from Scratch](../chapter15_peft_misalignment/instructions/pages/01_%5B15.1%5D_LoRA_and_DoRA_from_Scratch.md)

Optional, in any order:

- [[4.2] Science of Misalignment](https://learn.arena.education/chapter4_alignment_science/2_science_misalignment/) (original ARENA)
- [[4.3] Interpreting Reasoning Models](https://learn.arena.education/chapter4_alignment_science/3_reasoning_models/) (original ARENA)
- [[4.5] Investigator Agents](https://learn.arena.education/chapter4_alignment_science/5_investigator_agents/) (original ARENA)
- [[9.2] Truthfulness Directions and Hallucination Probes](../chapter9_alignment_interpretability/instructions/pages/02_%5B9.2%5D_Truthfulness_Directions_and_Hallucination_Probes.md)
- [[9.3] Chain-of-Thought Faithfulness](../chapter9_alignment_interpretability/instructions/pages/03_%5B9.3%5D_Chain-of-Thought_Faithfulness.md)
- [[9.4] Emergent Misalignment Detection](../chapter9_alignment_interpretability/instructions/pages/04_%5B9.4%5D_Emergent_Misalignment_Detection.md)
- [[9.5] White-Box Monitors and Red-Teaming with Internals](../chapter9_alignment_interpretability/instructions/pages/05_%5B9.5%5D_White-Box_Monitors_and_Red-Teaming_with_Internals.md)
- [[9.6] Rank-One Model Editing (ROME)](../chapter9_alignment_interpretability/instructions/pages/06_%5B9.6%5D_Rank-One_Model_Editing_ROME.md)
- [[9.7] Unlearning and How to Evaluate It](../chapter9_alignment_interpretability/instructions/pages/07_%5B9.7%5D_Unlearning_and_How_to_Evaluate_It.md) (after [9.2] Truthfulness Directions and Hallucination Probes)
- [[15.2] LoRA vs Full Fine-Tuning](../chapter15_peft_misalignment/instructions/pages/02_%5B15.2%5D_LoRA_vs_Full_Fine-Tuning.md)
- [[15.3] QLoRA from Scratch](../chapter15_peft_misalignment/instructions/pages/03_%5B15.3%5D_QLoRA_from_Scratch.md)
- [[15.4] LoRA Phase Transitions and Misalignment Proxies](../chapter15_peft_misalignment/instructions/pages/04_%5B15.4%5D_LoRA_Phase_Transitions_and_Misalignment_Proxies.md) (after [15.2] LoRA vs Full Fine-Tuning)

### 14. Multimodal

Core (3 days):

- [[12.1] CLIP and SigLIP from Scratch](../chapter12_vlm_interpretability/instructions/pages/01_%5B12.1%5D_CLIP_and_SigLIP_from_Scratch.md)
- [[12.3] Mini VLM from Scratch](../chapter12_vlm_interpretability/instructions/pages/03_%5B12.3%5D_Mini_VLM_from_Scratch.md)
- [[13.1] Diffusion Models and Cross-Attention Attribution](../chapter13_image_generation_interpretability/instructions/pages/01_%5B13.1%5D_Diffusion_Models_and_Cross-Attention_Attribution.md)

Optional, in any order:

- [[12.2] Typographic Attacks and SAE Features in a Toy Two-Tower Model](../chapter12_vlm_interpretability/instructions/pages/02_%5B12.2%5D_Typographic_Attacks_and_SAE_Features_in_a_Toy_Two-Tower_Model.md)
- [[12.4] Visual Token Flow in Real VLMs](../chapter12_vlm_interpretability/instructions/pages/04_%5B12.4%5D_Visual_Token_Flow_in_Real_VLMs.md)
- [[12.5] Object Hallucination and Modality Arbitration](../chapter12_vlm_interpretability/instructions/pages/05_%5B12.5%5D_Object_Hallucination_and_Modality_Arbitration.md)
- [[12.6] Multimodal SAEs, Transcoders and Crosscoders](../chapter12_vlm_interpretability/instructions/pages/06_%5B12.6%5D_Multimodal_SAEs_Transcoders_and_Crosscoders.md) (after [6.4] Transcoders and Attribution Graphs and [6.5] Crosscoders and Model Diffing)
- [[12.7] Dense Vision Features - DINO and SAM](../chapter12_vlm_interpretability/instructions/pages/07_%5B12.7%5D_Dense_Vision_Features_DINO_and_SAM.md)
- [[15.5] VLM LoRA](../chapter15_peft_misalignment/instructions/pages/05_%5B15.5%5D_VLM_LoRA.md)
- [[13.2] Denoising-Time Causal Patching](../chapter13_image_generation_interpretability/instructions/pages/02_%5B13.2%5D_Denoising-Time_Causal_Patching.md)
- [[13.3] Concept Directions and Diffusion LoRA](../chapter13_image_generation_interpretability/instructions/pages/03_%5B13.3%5D_Concept_Directions_and_Diffusion_LoRA.md) (after [13.2] Denoising-Time Causal Patching)
- [[13.4] Sparse Autoencoders for Diffusion Models](../chapter13_image_generation_interpretability/instructions/pages/04_%5B13.4%5D_Sparse_Autoencoders_for_Diffusion_Models.md) (after [13.2] Denoising-Time Causal Patching)
- [[13.5] Diffusion Transformers and ConceptAttention](../chapter13_image_generation_interpretability/instructions/pages/05_%5B13.5%5D_Diffusion_Transformers_and_ConceptAttention.md)
- [[13.6] Autoregressive Image Tokens](../chapter13_image_generation_interpretability/instructions/pages/06_%5B13.6%5D_Autoregressive_Image_Tokens.md)

### 15. World models and convergence

Core (2.5 days):

- [[14.1] I-JEPA from Scratch](../chapter14_jepa_world_models/instructions/pages/01_%5B14.1%5D_I-JEPA_from_Scratch.md)
- [19.1] Measuring Convergence: Kernels, CKA and Mutual Neighbours
- [19.2] Do Real Models Converge? Replicate and Critique the PRH

Optional, in any order:

- [[1.5.3] OthelloGPT](https://learn.arena.education/chapter1_transformer_interp/33_othellogpt/) (original ARENA)
- [[14.2] V-JEPA and Video World Models](../chapter14_jepa_world_models/instructions/pages/02_%5B14.2%5D_V-JEPA_and_Video_World_Models.md)
- [[14.3] Action-Conditioned World Models and Latent Planning](../chapter14_jepa_world_models/instructions/pages/03_%5B14.3%5D_Action-Conditioned_World_Models_and_Latent_Planning.md) (after [14.2] V-JEPA and Video World Models)
- [[14.4] The JEPA Family](../chapter14_jepa_world_models/instructions/pages/04_%5B14.4%5D_The_JEPA_Family.md)
- [[14.5] Maze Transformers and World Models](../chapter14_jepa_world_models/instructions/pages/05_%5B14.5%5D_Maze_Transformers_and_World_Models.md) (after [1.5.3] OthelloGPT)
- [[14.6] Sudoku and Constraint World Models](../chapter14_jepa_world_models/instructions/pages/06_%5B14.6%5D_Sudoku_and_Constraint_World_Models.md)
- [[14.7] Recurrent World Models - Dreamer, TD-MPC2 and IRIS](../chapter14_jepa_world_models/instructions/pages/07_%5B14.7%5D_Recurrent_World_Models_Dreamer_TD-MPC2_and_IRIS.md) (after [14.3] Action-Conditioned World Models and Latent Planning)
- [19.3] Proving and Improving the Platonic Representation Hypothesis with JEPAs

### 16. Attribution

Core (2 days):

- [[16.1] Exact Shapley on Ground-Truth Games](../chapter16_shapley_attribution_baselines/instructions/pages/01_%5B16.1%5D_Exact_Shapley_on_Ground-Truth_Games.md)
- [[16.2] KernelSHAP and PartitionSHAP](../chapter16_shapley_attribution_baselines/instructions/pages/02_%5B16.2%5D_KernelSHAP_and_PartitionSHAP.md)

Optional, in any order:

- [[16.3] TreeSHAP, DeepSHAP and GradientSHAP (Practice)](../chapter16_shapley_attribution_baselines/instructions/pages/03_%5B16.3%5D_TreeSHAP_DeepSHAP_and_GradientSHAP_Practice.md)
- [[16.4] SAGE Global Importance](../chapter16_shapley_attribution_baselines/instructions/pages/04_%5B16.4%5D_SAGE_Global_Importance.md)
- [[16.5] Shapley Interactions](../chapter16_shapley_attribution_baselines/instructions/pages/05_%5B16.5%5D_Shapley_Interactions.md)
- [[16.6] Sparse Interaction Discovery with SPEX and ProxySPEX](../chapter16_shapley_attribution_baselines/instructions/pages/06_%5B16.6%5D_Sparse_Interaction_Discovery_with_SPEX_and_ProxySPEX.md) (after [16.5] Shapley Interactions)
- [[16.7] TokenSHAP and Region SHAP for LLMs and VLMs](../chapter16_shapley_attribution_baselines/instructions/pages/07_%5B16.7%5D_TokenSHAP_and_Region_SHAP_for_LLMs_and_VLMs.md)
- [[16.8] SHAP vs Activation Patching](../chapter16_shapley_attribution_baselines/instructions/pages/08_%5B16.8%5D_SHAP_vs_Activation_Patching.md)
- [[16.9] Data Shapley in One Training Run](../chapter16_shapley_attribution_baselines/instructions/pages/09_%5B16.9%5D_Data_Shapley_in_One_Training_Run.md)
- [[16.10] Influence Functions and TracIn](../chapter16_shapley_attribution_baselines/instructions/pages/10_%5B16.10%5D_Influence_Functions_and_TracIn.md)
- [[16.11] Datamodels and Dataset Cartography](../chapter16_shapley_attribution_baselines/instructions/pages/11_%5B16.11%5D_Datamodels_and_Dataset_Cartography.md)

Side trips:

- Alchemy u06 · Generalization, an optional side trip

### 17. Dynamics and intrinsic

Core (3 days):

- [[1.5.2] Grokking & Modular Arithmetic](https://learn.arena.education/chapter1_transformer_interp/32_grokking/) (original ARENA)
- [[17.1] Checkpoint Archaeology on Modular Addition](../chapter17_training_dynamics/instructions/pages/01_%5B17.1%5D_Checkpoint_Archaeology_on_Modular_Addition.md)
- [[18.1] Concept Bottlenecks, Backpacks and Interchange Intervention Training](../chapter18_intrinsic_interpretability/instructions/pages/01_%5B18.1%5D_Concept_Bottlenecks_Backpacks_and_Interchange_Intervention_Training.md)

Optional, in any order:

- [[1.5.1] Balanced Bracket Classifier](https://learn.arena.education/chapter1_transformer_interp/31_brackets/) (original ARENA)
- [[17.2] Induction Heads in Hand-Built Circuits](../chapter17_training_dynamics/instructions/pages/02_%5B17.2%5D_Induction_Heads_in_Hand-Built_Circuits.md)
- [[17.3] Fine-Tuning Diffs and Capability Emergence](../chapter17_training_dynamics/instructions/pages/03_%5B17.3%5D_Fine-Tuning_Diffs_and_Capability_Emergence.md)
- [[17.4] Training Dynamics Across Architectures](../chapter17_training_dynamics/instructions/pages/04_%5B17.4%5D_Training_Dynamics_Across_Architectures.md)
- [[18.2] Sparse and Modular Training, KANs and Sparse Trees](../chapter18_intrinsic_interpretability/instructions/pages/02_%5B18.2%5D_Sparse_and_Modular_Training_KANs_and_Sparse_Trees.md)

### 18. Finale

Core (3 days):

- [[10.1] Capstone Research Sprint, from part 5️⃣](../chapter10_capstone_research_sprint/instructions/pages/01_%5B10.1%5D_Capstone_Research_Sprint.md)

Side trips:

- Alchemy u07 · Not yet released, an optional side trip

<!-- End of the generated stage table. -->

## How to work through a lesson

Each extension lesson works like an original ARENA section:

1. **Read the opening and the objectives.** The opening shows you what the lesson is about; the objectives list
   what each numbered part builds.
2. **Work in the exercises notebook,** from the lesson's own folder so that its setup and files resolve. The
   solutions notebook has worked answers and every cell's output, so you can see what a cell should produce.
3. **Write each function yourself, then run its test.** Exercises carry a difficulty and importance rating and a
   time estimate. Read the estimates relative to your own pace, open a hint when you're stuck, and look at the
   solution rather than lose an afternoon. Skip an exercise that isn't worth it to you.
4. **Look at the result next to its controls.** Ask what the figure would look like if the method were wrong.
5. **Bonus sections are optional.** They point to harder exercises and further reading.
6. **Restart and run everything** when you've finished a part, to check that your work holds together.
