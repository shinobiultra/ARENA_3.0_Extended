### ARENA [slack channel](https://join.slack.com/t/arena-uk/shared_invite/zt-3d3sro2sn-lACCkkoA3Sjm8J0pvWSaGQ)

* Please report any errors/concerns with the material in #errata.

# Install Instructions

1) Close the repo
```
git clone https://github.com/callummcdougall/ARENA_3.0.git
```
2) Run the install script
```
ARENA_3.0/install.sh
```

This GitHub repo hosts the exercises and Streamlit pages for the ARENA program. (Note that the name is kept as "ARENA 3.0" for backwards compatibility, but we've stopped creating new repositories for different iterations of the program, meaning this is now the latest version of the repo and won't get replaced by a new one in the future.)

You can find a summary of each of the chapters below. For more detailed information (including the different ways you can access the exercises), click on the links in the chapter headings.

Additionally, see [this Notion page](https://arena-resources.notion.site/) for a guide to the virtual study materials available.

Scroll to the end to see our instructions for submitting PRs.

# [Chapter 0: Fundamentals](https://learn.arena.education/chapter0_fundamentals/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/headers/header-ch0.png" width="400">

The material on this page covers the first five days of the curriculum. It can be seen as a grounding in all the fundamentals necessary to complete the more advanced sections of this course (such as RL, transformers, mechanistic interpretability, training at scale, and generative models).

Some highlights from this chapter include:

* Building your own 1D and 2D convolution functions
* Building and loading weights into a Residual Neural Network, and finetuning it on a classification task
* Working with [weights and biases](https://wandb.ai/site) to optimise hyperparameters
* Implementing your own backpropagation mechanism
* Building your own GANs and VAEs, and using them to generate images

# [Chapter 1: Transformer Interpretability](https://learn.arena.education/chapter1_transformer_interp/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/headers/header-ch1.png" width="400">

The material on this page covers transformers (what they are, how they are trained, how they are used to generate output) as well as mechanistic interpretability (what it is, what are some of the most important results in the field so far, why it might be important for alignment) and other topics related to interpretability (function vectors & model steering).

Some highlights from this chapter include:

* Building your own transformer from scratch, and using it to sample autoregressive output
* Using the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library developed by Neel Nanda to locate induction heads in a 2-layer model
* Finding a circuit for [indirect object identification](https://arxiv.org/abs/2211.00593) in GPT-2 small
* Intepreting model trained on toy tasks, e.g. classification of bracket strings, or modular arithmetic
* Replicating Anthropic's results on [superposition](https://transformer-circuits.pub/2022/toy_model/index.html), and training sparse autoencoders to recover features from superposition
* Using [steering vectors](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector) to induce behavioural changes in GPT2-XL

Unlike the first chapter (where all the material was compulsory), all sections of this chapter are optional extensions other than the first two exercise sets. In these first two sets you will build and train transformers, and gain a basic understanding of mechanistic interpretability of transformer models which includes induction heads & use of TransformerLens. After this, you can pick any of the other six sets of exercises you want - there are no prerequisites!

If you've finished the compulsory material and are choosing between the other six sets of exercises, we weakly recommend choosing one of the first three (IOI, superposition, and function vectors). IOI should appeal to the experimentalists, superposition to the theorists / mathematicians, and function vectors to the engineers, so there's something for everyone!

Additionally, each optional set of exercises includes a lot of suggested bonus material / further exploration once you've finished, including suggested papers to read and replicate.

# [Chapter 2: Reinforcement Learning](https://learn.arena.education/chapter2_rl/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/headers/header-ch2.png" width="400">

Reinforcement learning is an important field of machine learning. It works by teaching agents to take actions in an environment to maximise their accumulated reward.

In this chapter, you will be learning about some of the fundamentals of RL, and working with OpenAI’s Gym environment to run your own experiments.

Some highlights from this chapter include:

* Building your own agent to play the multi-armed bandit problem, implementing methods from [Sutton & Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
* Implementing a Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) to play the CartPole game
* Applying RLHF to autoregressive transformers like the ones you built in the previous chapter

Additionally, the later exercise sets include a lot of suggested bonus material / further exploration once you've finished, including suggested papers to read and replicate.

# [Chapter 3: LLM Evaluations](https://learn.arena.education/chapter4_alignment_science/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/headers/header-ch3.png" width="400">

The material in this chapter covers LLM evaluations (what they are for, how to design and build one). Evals produce empirical evidence on the model's capabilities and behavioral tendencies, which allows developers and regulators to make important decisions about training or deploying the model. In this chapter, you will learn the fundamentals of two types of eval: designing a simple multiple-choice (MC) question evaluation benchmark and building an LLM agent for an agent task to evaluate model capabilities with scaffolding. 

Some highlights from this chapter include:

* Design and generate your own MCQ eval from scratch using LLMs, implementing Anthropic's [model-written eval](https://arxiv.org/abs/2212.09251) method
* Using the [Inspect](https://inspect.ai-safety-institute.org.uk/) library written by the UK AI Safety Institute (AISI) to run evaluation experiments
* Building a LLM agent that plays the Wikipedia Racing game
* Implementing ReAct and inflexion as elicitation methods for LLM agents 

The exercises are written in collaboration with [Apollo Research](https://www.apolloresearch.ai/), and designed to give you the foundational skills for doing safety evaluation research on language models. 

# [Chapter 4: Alignment Science](https://arena-chapter4-alignment-science.streamlit.app/)

Coming soon!

---

# ARENA Frontier Lab extension

This fork is being rewritten into a local-first frontier-model extension while
keeping the original ARENA structure intact. The current PR contains a large
prototype implementation surface plus twenty-two polished ARENA-style sections:
[5.1 Gemma from Scratch](chapter5_modern_architectures/instructions/pages/01_%5B5.1%5D_Gemma_from_Scratch.md),
[5.2 Gemma Scope and Feature Steering](chapter5_modern_architectures/instructions/pages/02_%5B5.2%5D_Gemma_Scope_and_Feature_Steering.md),
[5.3 Mamba from Scratch](chapter5_modern_architectures/instructions/pages/03_%5B5.3%5D_Mamba_from_Scratch.md),
[5.4 Mamba State Tracking](chapter5_modern_architectures/instructions/pages/04_%5B5.4%5D_Mamba_State_Tracking.md),
[5.5 Toy Discrete Diffusion Language Models and Local DiffusionGemma Proof](chapter5_modern_architectures/instructions/pages/05_%5B5.5%5D_Diffusion_Language_Models.md),
[5.6 Embedding Retrieval and Function-Calling Controls](chapter5_modern_architectures/instructions/pages/06_%5B5.6%5D_Multimodal_Embedding_and_Function_Calling_Models.md),
[6.1 SAE Variants](chapter6_sparse_feature_methods/instructions/pages/01_%5B6.1%5D_SAE_Variants.md),
[6.2 Gemma Scope Deep Dive](chapter6_sparse_feature_methods/instructions/pages/02_%5B6.2%5D_Gemma_Scope_Deep_Dive.md),
[6.3 Transcoders and Attribution Graphs](chapter6_sparse_feature_methods/instructions/pages/03_%5B6.3%5D_Transcoders_and_Attribution_Graphs.md),
[6.4 Crosscoders and Model Diffing](chapter6_sparse_feature_methods/instructions/pages/04_%5B6.4%5D_Crosscoders_and_Model_Diffing.md),
[7.1 Logit Lens, Tuned Lens, and Patchscopes](chapter7_activation_to_language/instructions/pages/01_%5B7.1%5D_Logit_Lens_Tuned_Lens_and_Patchscopes.md),
[7.2 Feature Verbalizers](chapter7_activation_to_language/instructions/pages/02_%5B7.2%5D_Feature_Verbalizers.md),
[7.3 Mini Activation Oracles](chapter7_activation_to_language/instructions/pages/03_%5B7.3%5D_Mini_Activation_Oracles.md),
[7.4 Mini Natural Language Autoencoders](chapter7_activation_to_language/instructions/pages/04_%5B7.4%5D_Mini_Natural_Language_Autoencoders.md),
[7.5 Predictive Concept Decoders](chapter7_activation_to_language/instructions/pages/05_%5B7.5%5D_Predictive_Concept_Decoders.md),
[8.1 Activation Patching Refresher](chapter8_automated_circuits/instructions/pages/01_%5B8.1%5D_Activation_Patching_Refresher.md),
[8.2 Attribution Patching and EAP](chapter8_automated_circuits/instructions/pages/02_%5B8.2%5D_Attribution_Patching_and_EAP.md),
[8.3 ACDC and Circuit Metrics](chapter8_automated_circuits/instructions/pages/03_%5B8.3%5D_ACDC_and_Circuit_Metrics.md),
[8.4 Circuit Tracing with Attribution Graphs](chapter8_automated_circuits/instructions/pages/04_%5B8.4%5D_Circuit_Tracing_with_Attribution_Graphs.md),
[8.5 Sparse Feature Circuits](chapter8_automated_circuits/instructions/pages/05_%5B8.5%5D_Sparse_Feature_Circuits.md),
[9.1 Refusal Directions and Safe Steering](chapter9_alignment_interpretability/instructions/pages/01_%5B9.1%5D_Refusal_Directions_and_Safe_Steering.md),
and [9.2 Chain-of-Thought Faithfulness](chapter9_alignment_interpretability/instructions/pages/02_%5B9.2%5D_Chain_of_Thought_Faithfulness.md).
The remaining non-course-ready extension pages are tracked as prototype scaffolds until they are
rewritten with original-ARENA pacing, diagrams, help/interpretation dropdowns,
visible signature results, and notebook surfaces. See
[ARENA style rewrite status](docs/arena_style_rewrite_status.yml).

The first added fundamentals section is [0.6 How to Know When an Interpretability Result Is Fake](chapter0_fundamentals/instructions/pages/06_%5B0.6%5D_How_to_Know_When_an_Interpretability_Result_Is_Fake.md), a GT-0 skepticism lab for leakage, cherry-picking, probe overfitting, and random-direction controls. [1.6 Local Frontier ML Infrastructure](chapter1_transformer_interp/instructions/pages/40_%5B1.6%5D_Local_Frontier_ML_Infrastructure.md) introduces the reusable verification harness for later Gemma, Mamba, diffusion, SAE/transcoder, JEPA, world-model, attribution, and alignment-interpretability notebooks.

Original ARENA is pinned at `f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0` and checked by [Original ARENA preservation contract](docs/original_preservation_contract.md). The upstream-style dependencies are preserved in `requirements-original.txt`; the extension default is the uv-locked Python 3.14 + PyTorch CUDA 13.2 stack in `pyproject.toml` / `uv.lock`, mirrored by `requirements.txt`; the old Chapter 2 JAX/Brax/EnvPool stack is isolated in `requirements-legacy-rl.txt`.

Current local verification was run in the managed `uv` environment with Python
3.14.6, PyTorch `2.12.1+cu132`, CUDA 13.2, `torchvision 0.27.1+cu132`,
`transformers 5.12.1`, `bitsandbytes 0.49.2` using its current CUDA 13.0 binary
override because the wheel does not ship `libbitsandbytes_cuda132.so`,
`mamba-ssm 2.3.2.post1`, and an RTX 5090 Laptop GPU. The full extension report refresh wrote 41 extension
verification reports and accepted all 41; the repository contains 43 accepted
CUDA-backed verification reports including the original-extension evidence
reports under `docs/evidence/`.

The modern architecture track begins with [5.1 Gemma from Scratch](chapter5_modern_architectures/instructions/pages/01_%5B5.1%5D_Gemma_from_Scratch.md), which implements a tiny Gemma-style decoder and verifies RMSNorm, RoPE, grouped-query attention, sliding-window masking, KV-cache parity, and CUDA logit/cache parity against Hugging Face's `transformers.GemmaForCausalLM` reference architecture with matched deterministic tiny weights. It continues with [5.2 Gemma Scope and Feature Steering](chapter5_modern_architectures/instructions/pages/02_%5B5.2%5D_Gemma_Scope_and_Feature_Steering.md), which adds sparse-feature metrics, held-out validation, steering controls, ablation controls, a pinned Gemma Scope 2 1B-IT residual SAE artifact CUDA preflight, and authenticated Gemma 3 activation feature validation with random-feature and label-shuffle controls. The next section, [5.3 Mamba from Scratch](chapter5_modern_architectures/instructions/pages/03_%5B5.3%5D_Mamba_from_Scratch.md), implements selective scans, causal convolution, tiny Mamba blocks, recurrent-state/cache parity checks, and a pinned Mamba-130M-HF logits/generation CUDA preflight with required fast kernels. [5.4 Mamba State Tracking](chapter5_modern_architectures/instructions/pages/04_%5B5.4%5D_Mamba_State_Tracking.md) adds parity/bracket-depth model organisms, linear probes, held-out position checks, causal state interventions, a trained tiny Mamba bracket-depth GPU preflight with random-label controls, a trained tiny causal Transformer baseline comparison on the same generated task, learned Mamba hidden-state interventions with matched random-direction controls, and pinned Mamba-130M-HF hidden-state extraction. [5.5 Toy Discrete Diffusion Language Models and Local DiffusionGemma Proof](chapter5_modern_architectures/instructions/pages/05_%5B5.5%5D_Diffusion_Language_Models.md) adds toy discrete noising, masked denoising loss, confidence remasking, oracle sampling, entropy/commitment diagnostics, activation-trajectory checks, a CUDA-trained tiny conditional diffusion LM preflight with held-out masked reconstruction, sampler exact-match, shuffled-label controls, and a scoped pinned DiffusionGemma NVFP4 local generation proof that does not claim released-checkpoint denoising-time interpretability. [5.6 Embedding Retrieval and Function-Calling Controls](chapter5_modern_architectures/instructions/pages/06_%5B5.6%5D_Multimodal_Embedding_and_Function_Calling_Models.md) adds masked mean pooling, paired retrieval, centroid probes, tool masking, function-call parsing, no-call metrics, schema-token attribution controls, a pinned public BGE embedding retrieval comparison, direct authenticated EmbeddingGemma retrieval, a pinned public FunctionGemma Mobile Actions eval on held-out real dataset rows, visible argument-fidelity failures, and direct authenticated base FunctionGemma CUDA loading. It is deliberately scoped to text retrieval and function-calling controls; broad VLM/multimodal interpretability is left to later VLM sections.

The sparse feature methods track begins with [6.1 SAE Variants](chapter6_sparse_feature_methods/instructions/pages/01_%5B6.1%5D_SAE_Variants.md), which compares ReLU/L1, TopK, Gated, and JumpReLU SAE encoders with reconstruction metrics, planted-dictionary recovery checks, feature AUC validation, nondegenerate-density checks, decoder-vector steering controls, and a pinned Pythia-70M hidden-state TopK SAE CUDA preflight with held-out reconstruction, permuted-decoder, density, feature-AUC, and decoder-projection steering controls. [6.2 Gemma Scope Deep Dive](chapter6_sparse_feature_methods/instructions/pages/02_%5B6.2%5D_Gemma_Scope_Deep_Dive.md) adds released-artifact metadata checks, tagged feature selection, held-out AUC baselines, base-vs-instruction feature deltas, ablation controls, steering safety checks, direct logit attribution, a pinned Gemma Scope 2 1B-IT layer-13 JumpReLU SAE CUDA preflight with config, shape, finiteness, and encode/decode checks, and authenticated Gemma 3 layer-13 activation validation on one benign technical-vs-narrative split with random-feature and label-shuffle controls. [6.3 Transcoders and Attribution Graphs](chapter6_sparse_feature_methods/instructions/pages/03_%5B6.3%5D_Transcoders_and_Attribution_Graphs.md) adds toy transcoder replacement checks, feature-level logit contributions, sparse input-feature-logit graphs, graph reproducibility, top-feature versus low-effect graph-damage controls, and a pinned TransformerLens `gelu-1l` CUDA preflight with exact MLP-feature replacement parity and trained tiny ReLU transcoder held-out reconstruction. [6.4 Crosscoders and Model Diffing](chapter6_sparse_feature_methods/instructions/pages/04_%5B6.4%5D_Crosscoders_and_Model_Diffing.md) adds paired-model reconstruction checks, shared/model-specific feature classification, behavior-delta prediction, paired score deltas, top-direction removal controls, and a pinned `gelu-1l` vs `solu-1l` CUDA model-diffing preflight with exact shared-plus-delta reconstruction, SVD delta-direction separation, and orthogonal random-direction controls.

The activation-to-language track begins with [7.1 Logit Lens, Tuned Lens, and Patchscopes](chapter7_activation_to_language/instructions/pages/01_%5B7.1%5D_Logit_Lens_Tuned_Lens_and_Patchscopes.md), which adds logit lens, tuned lens, attention lens, Patchscope templates, held-out decoding accuracy comparisons, counterfactual activation checks, random-activation confidence controls, and a pinned TransformerLens `gelu-1l` CUDA preflight that trains a ridge affine lens on cached activations and compares activation-conditioned clean logits against corrupt text-only baselines. [7.2 Feature Verbalizers](chapter7_activation_to_language/instructions/pages/02_%5B7.2%5D_Feature_Verbalizers.md) adds top/bottom/random/contrastive example gathering, explanation-derived held-out predictions, counterexample discovery, explanation revision, intervention-direction checks, brevity controls, and a pinned `gelu-1l` residual-direction verbalizer preflight with held-out and intervention validation. [7.3 Mini Activation Oracles](chapter7_activation_to_language/instructions/pages/03_%5B7.3%5D_Mini_Activation_Oracles.md) adds activation-question datasets, oracle-vs-baseline comparisons, OOD split reports, random-activation graceful-failure checks, activation-patching answer-change controls, and a pinned `gelu-1l` residual-direction mini-oracle CUDA preflight with text-only/probe baselines, OOD prompts, random abstention, and clean-to-corrupt answer flips. [7.4 Mini Natural Language Autoencoders](chapter7_activation_to_language/instructions/pages/04_%5B7.4%5D_Mini_Natural_Language_Autoencoders.md) adds activation-to-text-to-activation reconstruction checks, text-only baselines, target logit-diff preservation, latent-probe preservation, generated-text brevity, counterfactual explanation-change controls, and a pinned `gelu-1l` residual text-bottleneck mini-NLA CUDA preflight with compact parsed explanations, reconstruction baselines, OOD prompts, shuffled controls, and counterfactual explanation changes. [7.5 Predictive Concept Decoders](chapter7_activation_to_language/instructions/pages/05_%5B7.5%5D_Predictive_Concept_Decoders.md) adds sparse concept encoders, explicit concept-question interaction features, trained question-conditioned decoders, PCD-vs-baseline comparisons, concept sparsity and seed-stability checks, question-shuffle controls, top-concept removal, low-margin active-control removal, concept naming audits, and a pinned `gelu-1l` sparse-concept PCD CUDA preflight with signed residual concepts and explicit claim boundaries.

The automated-circuits track begins with polished [8.1 Activation Patching Refresher](chapter8_automated_circuits/instructions/pages/01_%5B8.1%5D_Activation_Patching_Refresher.md), which adds reusable clean/corrupt logit-diff metrics, activation-slice patching, recovered-fraction reports, per-component patching sweeps, target-component localization, wrong-position mean/max controls, paired ARENA-style notebooks, and a pinned TransformerLens `gelu-1l` CUDA residual-stream patching preflight with `[0, 0, 0, 0, 0, 1]` position recovery. Polished [8.2 Attribution Patching and EAP](chapter8_automated_circuits/instructions/pages/02_%5B8.2%5D_Attribution_Patching_and_EAP.md) adds first-order attribution patching, integrated-gradient patch scores, EAP-style edge scores, exact-vs-approx correlation, top-k overlap, runtime speedup reports, documented false-negative cases, paired ARENA-style notebooks, and a pinned TransformerLens `gelu-1l` CUDA exact-vs-attribution preflight where exact patching, attribution patching, IG, and EAP all localize the final residual position. Polished [8.3 ACDC and Circuit Metrics](chapter8_automated_circuits/instructions/pages/03_%5B8.3%5D_ACDC_and_Circuit_Metrics.md) adds ACDC-style edge pruning, faithfulness, minimality, completeness, same-size random circuit baselines, held-out prompt-template checks, paired ARENA-style notebooks, and a pinned TransformerLens `gelu-1l` CUDA position-circuit preflight whose one-edge circuit preserves the clean-corrupt logit-diff on three held-out templates. Polished [8.4 Circuit Tracing with Attribution Graphs](chapter8_automated_circuits/instructions/pages/04_%5B8.4%5D_Circuit_Tracing_with_Attribution_Graphs.md) adds EAP-style edge scoring, local attribution graph construction, top directed attribution paths, target-metric explanation, path perturbation tests, alternative graph baselines, counterfactual summary checks, paired ARENA-style notebooks, and a pinned TransformerLens `gelu-1l` CUDA graph preflight whose top edge is `position_5 -> position_5`. Polished [8.5 Sparse Feature Circuits](chapter8_automated_circuits/instructions/pages/05_%5B8.5%5D_Sparse_Feature_Circuits.md) adds exact sparse-feature node and edge patching, EAP-vs-EAP-IG comparison, thresholded graph and same-size random controls, generated SHIFT-style editing, paired ARENA-style notebooks, and a scoped CUDA evidence bundle with Pythia-70M-deduped residual-feature checks, released SAE state-dict and one-layer attribution controls, a 100-example official-code sparse feature graph, and held-out official faithfulness on 40 `simple_test` examples. Use `scripts/prepare_sparse_feature_circuits_artifacts.py --download --extract` to prepare the released artifacts in the ignored `external/feature-circuits/` cache.

The alignment-interpretability track begins with polished [9.1 Refusal Directions and Safe Steering](chapter9_alignment_interpretability/instructions/pages/01_%5B9.1%5D_Refusal_Directions_and_Safe_Steering.md), which adds safe prompt-pair handling, activation-cache framing, mean-difference refusal directions, projection scores, held-out separation, addition/projection-out steering reports, capability bounds, random-direction and label-shuffle controls, candidate-method comparisons, a pinned Pythia-70M-deduped hidden-state category preflight, a pinned Qwen2.5-0.5B-Instruct no-generation logit intervention preflight, and a scoped GT-2 public `josephmayo/refusal-compliance-pairs` aggregate replication path with layer/position/PCA controls and no raw prompt or completion text saved. Polished [9.2 Chain-of-Thought Faithfulness](chapter9_alignment_interpretability/instructions/pages/02_%5B9.2%5D_Chain_of_Thought_Faithfulness.md) adds hidden-answer probes before final tokens, narrow hidden-vector LM-head readout-patch checks, CoT text-only baseline comparisons, feature-level unfaithfulness detectors, no-CoT/faithful/biased/post-hoc condition reports, paired ARENA-style notebooks, and a pinned Pythia-70M-deduped hidden-answer preflight with visible-text, label-shuffle, and hidden-state readout controls. [9.3 Emergent Misalignment Detection](chapter9_alignment_interpretability/instructions/pages/03_%5B9.3%5D_Emergent_Misalignment_Detection.md) adds benign proxy drift categories, held-out white-box drift detection, crosscoder feature alignment with behavior deltas, mitigation checks with capability-loss bounds, early white-box-vs-black-box detection timing, and a pinned Pythia-70M-deduped benign proxy-drift hidden-state preflight with label-shuffle, random-direction, behavior-proxy, and projection-mitigation controls. [9.4 White-box Evals and Monitors](chapter9_alignment_interpretability/instructions/pages/04_%5B9.4%5D_White_box_Evals_and_Monitors.md) adds monitor dashboard rows, AUROC calibration, white-box catches of black-box missed failures, false-positive documentation, held-out feature-explanation validation, and a pinned Pythia-70M-deduped white-box monitor preflight with next-token black-box proxy, label-shuffle, and random-direction controls.

The capstone track contains [10.1 Capstone Research Sprint](chapter10_capstone_research_sprint/instructions/pages/01_%5B10.1%5D_Capstone_Research_Sprint.md), which adds the paper-style project readiness contract and a live CUDA mini activation-oracle sprint: a question-conditioned oracle is trained on generated latent-state activations, compared against text-only and linear-probe baselines, and checked with held-out templates, ablation, counterfactual patching, random-patch, random-activation, and label-shuffle controls.

The representation-geometry track begins with [11.1 PCA, SVD, and Geometry Controls](chapter11_representation_geometry/instructions/pages/01_%5B11.1%5D_PCA_SVD_and_Geometry_Controls.md), which adds centered PCA/SVD projections, held-out label prediction from geometry, white-noise controls, seed-stability checks, causal direction effects over random controls, template-centering checks, and a pinned Pythia-70M-deduped calendar hidden-state geometry preflight over weekday and month prompt splits with five-seed, three-setting UMAP sweeps, trustworthiness, neighborhood preservation, random-label controls, and random-token controls.

The VLM interpretability track begins with [12.1 CLIP, SigLIP, and VLM Controls](chapter12_vlm_interpretability/instructions/pages/01_%5B12.1%5D_CLIP_SigLIP_and_VLM_Controls.md), which adds CLIP-style contrastive retrieval, SigLIP-style pairwise loss, visual-token attribution locality, synthetic colored-shape scene metadata, image-grounding baselines over text-only priors, object-region patch controls, hidden visual-token activation patching with object/background/same-size random-token/full-sequence controls, object hallucination probes, visual-vs-text modality arbitration, pinned real CLIP plus SigLIP rendered-shape retrieval and hidden-token patching, and a pinned Qwen2.5-VL 3B rendered-shape generation check.

The image-generation interpretability track begins with [13.1 Diffusion and Image-Generation Controls](chapter13_image_generation_interpretability/instructions/pages/01_%5B13.1%5D_Diffusion_and_Image_Generation_Controls.md), which adds diffusion attention region maps, denoising-circuit ablation specificity, latent-direction effects over random controls, prompt-token-to-region causal drops, a supplemental pinned SD-Turbo safe-shape generation preflight, and a required pinned Stable Diffusion 1.5 safe-shape path with DAAM-style cross-attention localization, target-token ablation over random/control-token ablations, CLIP alignment, image-quality preservation, and white-noise rejection.

The JEPA and world-model track begins with [14.1 JEPA and World-Model Controls](chapter14_jepa_world_models/instructions/pages/01_%5B14.1%5D_JEPA_and_World_Model_Controls.md), which adds JEPA target-embedding prediction, held-out world-state probes, action-conditioned transition consistency, object permanence under occlusion over absent-object controls, and a pinned V-JEPA 2 ViT-L generated-video preflight with frozen-latent masked prediction, state probes, rollout heads, and causal token patching against random-token controls.

The PEFT and misalignment track begins with [15.1 LoRA, DoRA, and Adapter Controls](chapter15_peft_misalignment/instructions/pages/01_%5B15.1%5D_LoRA_DoRA_and_Adapter_Controls.md), which adds exact LoRA delta checks, DoRA row-magnitude recomposition, protected-direction projection controls, accuracy-vs-mechanism adapter acceptance checks, and a generated safe proxy PEFT GPU preflight with rank-1 LoRA merge parity, target-direction alignment, random-label controls, same-norm random-adapter controls, and a matched LoRA-vs-DoRA-vs-full-finetune comparison on the same generated task.

The Shapley attribution-baseline track begins with [16.1 Exact Shapley on Ground-Truth Games](chapter16_shapley_attribution_baselines/instructions/pages/01_%5B16.1%5D_Exact_Shapley_on_Ground_Truth_Games.md), which adds complete coalition-value tables, exact Shapley values, permutation-parity checks, efficiency checks, interaction-heavy leave-one-out failure cases, and a real CUDA-trained neural coalition-game preflight whose ablation Shapley values are checked against analytic ground truth plus a shuffled-label control. [16.2 KernelSHAP and PartitionSHAP Controls](chapter16_shapley_attribution_baselines/instructions/pages/02_%5B16.2%5D_KernelSHAP_and_PartitionSHAP_Controls.md) adds full-table KernelSHAP weighted-regression parity, grouped PartitionSHAP/Owen-value checks, and a real CUDA neural-game SHAP-control preflight with singleton, aligned-group, mismatched-group, and shuffled-label controls. [16.3 Shapley Interactions with shapiq](chapter16_shapley_attribution_baselines/instructions/pages/03_%5B16.3%5D_Shapley_Interactions_with_shapiq.md) adds exact pairwise interaction indices, target-pair negative controls, complete-table SII parity against `shapiq`, and a real CUDA neural-game interaction preflight that recovers planted positive and negative interactions from model ablations. [16.4 TokenSHAP and TokenShapley](chapter16_shapley_attribution_baselines/instructions/pages/04_%5B16.4%5D_TokenSHAP_and_TokenShapley.md) adds exact masked-token coalition tables, token-level efficiency checks, sampled permutation TokenSHAP parity, and a real CUDA-trained token-scorer preflight with analytic, ranking, efficiency, and shuffled-label controls. [16.5 VLM Modality and Region SHAP](chapter16_shapley_attribution_baselines/instructions/pages/05_%5B16.5%5D_VLM_Modality_and_Region_SHAP.md) adds image/text modality synergy checks, structured object/OCR/background region attributions, background negative controls, and a pinned CLIP rendered-shape SHAP preflight on real logits. [16.6 SHAP vs Activation Patching](chapter16_shapley_attribution_baselines/instructions/pages/06_%5B16.6%5D_SHAP_vs_Activation_Patching.md) adds additive agreement checks, interaction-heavy overcount cases, and a real CUDA model-organism comparison of Shapley values with full-minus-ablated patching effects. [16.7 Data Shapley in One Training Run](chapter16_shapley_attribution_baselines/instructions/pages/07_%5B16.7%5D_Data_Shapley_in_One_Training_Run.md) adds exact, Monte Carlo, and in-run first-order Data Shapley checks plus a real CUDA one-step training preflight with autograd per-example scores and harmful-example deletion control. [16.8 Do SHAPley and Mechanistic Interpretability Agree?](chapter16_shapley_attribution_baselines/instructions/pages/08_%5B16.8%5D_Do_SHAPley_and_Mechanistic_Interpretability_Agree.md) adds rank-correlation, top-k overlap, deletion-consequence, XOR interaction checks, and a real CUDA agreement preflight comparing model Shapley scores with known mechanistic contributions and shuffled-label controls.

The training-dynamics track begins with [17.1 Checkpoint Archaeology and Mechanism Emergence](chapter17_training_dynamics/instructions/pages/01_%5B17.1%5D_Checkpoint_Archaeology_and_Mechanism_Emergence.md), which adds stable emergence thresholds, first-crossing vs stable-crossing reports, phase-transition detection, random-control checks, toy AR/JEPA/diffusion/Mamba developmental timing comparisons, and a real CUDA modular-addition checkpoint preflight that saves/reloads checkpoints, detects stable emergence, and rejects a random-label checkpoint control.

Durable setup references:

* [Local GPU setup](docs/local_gpu_setup.md)
* [CI gates](docs/ci_gates.md)
* [Original ARENA preservation contract](docs/original_preservation_contract.md)
* [Reproducibility contract](docs/reproducibility_contract.md)
* [Verification quality policy](docs/verification_quality_policy.md)
* [Hard exercise verification ladders](docs/hard_exercise_verification_ladders.md)
* [Hard exercise ladder registry](docs/hard_exercise_ladder_registry.md)
* [GPU verification policy](docs/gpu_verification_policy.md)
* [Method registry](docs/method_registry.md)
* [Research project template](research_projects/00_project_template/README.md)
* [Generated data contract](docs/generated_data_contract.md)

---

# Submitting PRs

If you want to submit a PR to the repo (e.g. fixing a bug or typo), this would be much appreciated! The easiest way to do this for us is by editing the **master Python file** (not the notebook) in `infrastructure/master_files`, since these are the files that generate all other pages (both Colabs, Streamlit pages, solutions files). For example, if you want to edit material 2.2 (Q-Learning and DQN), you should edit just `infrastructure/master_files/master_2_2.py`. After PRs are merged, we then run code which updates all the other files based on this one (so you don't have to worry about any of those other files, only the master Python file!).

If you find the PR confusing (because you're not sure exactly what to edit in these master files), then please either send a message in the `#errata` Slack channel, or just make a PR on non-master files (e.g. the `solutions.py` file or the markdown files) and we'll be able to merge it & replicate it on the master files.
