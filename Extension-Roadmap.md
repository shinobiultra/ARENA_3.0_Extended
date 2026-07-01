# ARENA Frontier Lab Extension Roadmap

This roadmap is the target for extending ARENA 3.0, not a claim that every
prototype page in this branch is already polished course material. The original
ARENA chapters remain the canonical base. Extension material must earn
`course_ready` status in [docs/arena_style_rewrite_status.yml](docs/arena_style_rewrite_status.yml)
before it should be treated as part of the finished student-facing course.

The current polished sections are **[5.1] Gemma from Scratch**, **[5.2]
Gemma Scope and Feature Steering**, **[5.3] Mamba from Scratch**, **[5.4]
Mamba State Tracking**, **[5.5] Toy Discrete Diffusion Language Models and
Local DiffusionGemma Proof**, **[5.6] Embedding Retrieval and
Function-Calling Controls**, **[6.1] SAE Variants**, **[6.2] Gemma
Scope Deep Dive**, **[6.3] Transcoders and Attribution Graphs**, and
**[6.4] Crosscoders and Model Diffing**, and **[7.1] Logit Lens,
Tuned Lens, and Patchscopes**, **[7.2] Feature Verbalizers**, and
**[7.3] Mini Activation Oracles**, and **[7.4] Mini Natural Language Autoencoders**.
They now have ARENA-style learner pages, local
diagrams, help/interpretation dropdowns, expected-output blocks, paired
exercise/solution notebooks, CUDA-backed signature result tables, and explicit
claim boundaries.
The other extension sections are prototype scaffolds: useful
implementation and verification evidence, but still requiring the
original-ARENA rewrite pass described in `guidance_2-0.md`.

The extension goal is a local-first “ARENA Frontier Lab” rather than a pile of
topic demos. It should keep the existing ARENA transformer/mech-interp spine
intact, then add a sequence of runnable, testable notebooks that teach modern
architectures, pretrained-weight loading, local inference, local finetuning,
SAEs/transcoders/crosscoders, activation-to-language methods, automated
circuits, and alignment interpretability.

The key design rule: **every notebook must end with a verification block**. Not “I implemented Gemma/Mamba and it seems to run,” but “my implementation matches Hugging Face logits within tolerance,” “parallel and recurrent Mamba scans are numerically equivalent,” “a quantized DiffusionGemma checkpoint runs under the VRAM budget,” “my SAE recovers loss and has expected sparsity,” “my verbalizer beats text-only and probe baselines on held-out counterfactuals,” etc.

The existing ARENA material is already a strong base: it includes GPT-2-from-scratch, induction heads, TransformerLens hooks, IOI, direct logit attribution, activation/path patching, toy superposition, SAEs, NNsight, function vectors, and steering exercises.  The extension should therefore not duplicate Chapter 1; it should turn Chapter 1 into a launchpad for modern architectures and 2026 interpretability methods.

---

## 0. Core philosophy for the fork

I’d organize the fork around three tracks:

**Track A — Architecture literacy.**
Implement and verify modern model families, not just vanilla GPT-2: Gemma 3/4-style decoder blocks, Mamba/Mamba-2/Mamba-3-style SSMs, diffusion language models, multimodal Gemma-style models, and smaller specialist models like FunctionGemma or EmbeddingGemma.

**Track B — Mechanistic interpretability frontier.**
Move from classic DLA/activation patching to SAEs, JumpReLU/Gated/TopK variants, transcoders, crosscoders, attribution graphs, activation oracles, natural-language autoencoders, predictive concept decoders, and faithful verbalizers.

**Track C — Safety/alignment applications.**
Keep black-box LLM evals mostly as-is, but add white-box interpretability: refusal directions, feature-level monitoring, CoT faithfulness, emergent-misalignment detection, model-diffing, and safety-relevant circuit tracing.

This fits the current state of the field: recent open-problems work emphasizes better decomposition, stronger validation, circuit/feature methods, monitoring/control/prediction applications, and benchmarks rather than just plausible explanations.  It also matches the warning that model components like neurons, attention heads, and layers are often too coarse or polysemantic, motivating sparse dictionary learning and other decomposition methods.

---

## 1. Hardware constraint: 24GB RTX 5090 Laptop GPU

I’d make the course **explicitly local-tiered**.

### Green tier: always local

These should run comfortably with 24GB VRAM:

| Model / task                    | Use                                                |
| ------------------------------- | -------------------------------------------------- |
| GPT-2 small / medium            | TransformerLens, classic circuits, SAEs            |
| Pythia 70M–410M                 | probes, CoT faithfulness, toy safety tasks         |
| TinyStories / tiny transformers | training from scratch                              |
| Mamba 130M / 370M               | architecture implementation and SSM probes         |
| Mamba-2 130M / 370M             | SSM duality and speed comparisons                  |
| Gemma 3 1B / 4B                 | Gemma architecture, local steering, Gemma Scope    |
| FunctionGemma 270M              | function-calling / local agent behavior            |
| EmbeddingGemma 308M             | embedding-model interpretability                   |
| Toy diffusion LM                | discrete diffusion training and denoising circuits |

### Yellow tier: local with quantization, batch size 1, short contexts

| Model / task             | Use                                                                                                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gemma 3 12B int4 / 8-bit | local chat model analysis                                                                                                                                                          |
| Gemma 3 27B int4         | optional; Google says the int4 QAT 27B model fits on a 24GB RTX 3090-class GPU, so a 24GB 5090 should be plausible if the software stack cooperates. ([Google Developers Blog][1]) |
| Mamba 1.4B / 2.8B        | generation benchmark, limited activation capture                                                                                                                                   |
| Mamba-2 2.7B             | architecture comparison                                                                                                                                                            |
| DiffusionGemma quantized | Google describes DiffusionGemma as a 26B/4B-active MoE diffusion model with quantized variants fitting within 18GB VRAM. ([Google AI for Developers][2])                           |

### Red tier: do not train; inspect only if released tooling makes it easy

| Task                                             | Reason                                        |
| ------------------------------------------------ | --------------------------------------------- |
| Full-scale Activation Oracle training            | Too expensive                                 |
| Full-scale Natural Language Autoencoder training | Too expensive                                 |
| Full Gemma Scope SAE training on every layer     | Already released; retrain small replicas only |
| Full emergent-misalignment finetunes             | Safety and compute concerns                   |
| Full 27B BF16 inference                          | Not local on 24GB                             |

The fork should enforce this with a `memory_budget.py` utility: before a notebook runs, it estimates parameter memory, activation memory, KV cache, optimizer state, and whether quantization/offload is required.

---

## 2. Repository structure

I’d restructure ARENA like this:

```text
ARENA_Frontier_Lab/
  chapters/
    ch0_prereqs_delta/
    ch1_original_arena_transformers/
    ch1_5_reproducibility_harness/
    ch2_modern_architectures/
      2_1_gemma_from_scratch/
      2_2_gemma_weight_loading_and_gemma_scope/
      2_3_mamba_from_scratch/
      2_4_mamba2_mamba3_state_tracking/
      2_5_diffusion_language_models/
      2_6_multimodal_and_embedding_models/
    ch3_sparse_feature_methods/
      3_1_sae_variants/
      3_2_gemma_scope_feature_analysis/
      3_3_transcoders_and_attribution_graphs/
      3_4_crosscoders_and_model_diffing/
    ch4_activation_to_language/
      4_1_logit_lens_tuned_lens_patchscopes/
      4_2_feature_verbalizers/
      4_3_mini_activation_oracles/
      4_4_mini_natural_language_autoencoders/
      4_5_predictive_concept_decoders/
    ch5_automated_circuits/
      5_1_activation_patching_refresher/
      5_2_attribution_patching_and_eap/
      5_3_acdc_and_circuit_metrics/
      5_4_circuit_tracing_with_transcoders/
    ch6_alignment_interpretability/
      6_1_refusal_directions_and_steering/
      6_2_cot_faithfulness/
      6_3_emergent_misalignment_detection/
      6_4_white_box_evals_and_monitors/
    ch7_capstones/
  arena_ext/
    model_adapters/
    activation_store/
    evals/
    interp_metrics/
    quantization/
    tests/
    plotting/
  scripts/
    verify_hf_equivalence.py
    verify_vram_budget.py
    verify_generation.py
    download_models.py
  docs/
    literature_review.md
    local_gpu_setup.md
    reproducibility_contract.md
```

Every notebook has:

```text
Learning goal
Reading map
Implementation exercises
Pretrained-weight loading
Verification tests
Failure modes
Stretch research directions
```

---

## 3. Chapter 1.5 — Reproducibility and local GPU harness

This comes before new architectures.

### Exercises

Implement:

1. **Environment checker**

   * CUDA version, PyTorch version, GPU name, BF16 support, flash-attention availability.
   * Warn if notebook would exceed 24GB.

2. **Model loading matrix**

   * Hugging Face gated-token check.
   * Quantized model support.
   * Local cache paths.

3. **HF parity checker**

   * Given a custom implementation and an HF model, compute:

     * max absolute logit difference,
     * mean squared error,
     * KL divergence,
     * top-k agreement,
     * deterministic generation equality.
   * The uploaded ARENA-derived material already uses MSE, max-diff, cosine similarity, and KL divergence for TransformerLens-vs-HF verification, so formalizing this as a reusable harness is a natural extension.

4. **Activation-store abstraction**

   * Stream activations to disk with `safetensors` / `zarr`.
   * Avoid holding all activations in VRAM.
   * Support GPT-style, Gemma-style, and Mamba-style hooks.

5. **Notebook test contract**

   * Every notebook exposes:

     * `run_smoke_test(cpu=True)`,
     * `run_gpu_test(max_vram_gb=24)`,
     * `run_full_experiment()`.

### Verification

A notebook passes only if:

```text
pytest passes
HF logits match tolerance
generation matches at fixed seed
peak VRAM is printed
artifacts are saved with metadata
```

This is essential because a course like this can otherwise become a pile of beautiful but brittle notebooks.

---

## 4. Chapter 2 — Modern architectures from scratch

### 2.1 Gemma from scratch

Gemma is the natural “modern transformer” replacement for GPT-2 in the course. Gemma 3 is especially useful because it spans small local models and larger quantized models; Google describes Gemma 3 as a multimodal family from 1B to 27B with 128K context and local/global attention patterns, and markets it as a strong single-accelerator model family. ([arXiv][3])

#### Exercises

Implement:

1. RMSNorm.
2. RoPE.
3. Grouped-query attention.
4. Local/global attention pattern.
5. SwiGLU / gated MLP.
6. KV cache.
7. Tokenizer and chat template handling.
8. Weight loading from HF.
9. Quantized inference path.

#### Verification

Pass criteria:

```text
max_abs_diff(custom_logits, hf_logits) < architecture-specific tolerance
KL(custom || HF) below threshold
top-10 tokens match on fixed prompts
generation matches for greedy decoding
KV-cache and no-cache generation are identical
peak VRAM < configured budget
```

#### Stretch

Implement **Gemma 4 compatibility** as a later update path. Google’s current Gemma model overview says Gemma 4 models are downloadable from Kaggle and Hugging Face and that QAT variants are available. ([Google AI for Developers][4]) I would keep Gemma 3 as the main teaching target and add Gemma 4 as an optional adapter so the course does not rot when model releases change.

---

### 2.2 Gemma Scope and feature interpretability

This should be a flagship module. Google’s Gemma Scope 2 release describes open interpretability tools for Gemma 3, including SAEs and transcoders trained on every layer, with intended uses in behavior evaluation, safety, and debugging. ([Google AI for Developers][5]) Earlier Gemma Scope work released SAEs for Gemma 2 2B/9B and related checkpoints. ([Google DeepMind][6])

#### Exercises

1. Load Gemma 3 1B/4B.
2. Load Gemma Scope 2 SAE / transcoder artifacts.
3. Compute feature activations on prompts.
4. Show max-activating tokens.
5. Compute direct logit attribution from features.
6. Perform safe feature steering.
7. Compare base vs instruction-tuned feature behavior.
8. Compute SAE metrics:

   * L0,
   * loss recovered,
   * feature density,
   * dead features,
   * reconstruction KL.

This matches the standard SAE evaluation frame: sparse reconstructions, L0, loss recovered, feature density, and feature interpretability via direct logit attribution / max-activating examples.

#### Verification

A feature interpretation is accepted only if:

```text
feature fires on held-out examples
feature does not fire on matched negatives
steering changes target behavior more than random-feature steering
ablation reduces target metric
text-only baseline is weaker
```

---

### 2.3 Mamba from scratch

Mamba should be the second core architecture chapter. The official Mamba repo describes Mamba as a selective state-space architecture with hardware-aware design for information-dense data, and it provides pretrained Mamba/Mamba-2 checkpoints from 130M to 2.8B plus generation/evaluation scripts. ([GitHub][7]) Mamba-2 reframes the architecture using state-space duality and reports much faster training than Mamba in the paper. ([arXiv][8])

#### Exercises

Implement:

1. Continuous-time SSM intuition.
2. Discretization.
3. Selective scan.
4. Causal convolution.
5. Mamba block.
6. Recurrent inference path.
7. Parallel training path.
8. Weight loading for `mamba-130m`.
9. Generation benchmark vs GPT-2 / Pythia.

#### Verification

The key “proof of correctness” is numerical equivalence:

```text
parallel_scan_output == recurrent_scan_output within 1e-4
custom logits match official implementation
fixed prompt greedy generation matches
speed benchmark reports tokens/sec
state cache gives identical output to full-context pass
```

#### Mechanistic exercises

1. Probe hidden state for previous-token information.
2. Test induction-like tasks.
3. Train a tiny Mamba on synthetic copying / associative recall.
4. Compare attention induction heads vs Mamba state updates.
5. Test whether probes localize “state variables” more cleanly than in transformers.

There is already work suggesting Mamba and related SSMs can be mechanistically evaluated on induction-style behavior, including findings that Mamba can solve induction-like tasks through short convolutions. ([arXiv][9])

---

### 2.4 Mamba-3 and state tracking

Mamba-3 is frontier enough that I would make it an optional advanced notebook. The Mamba-3 paper emphasizes quality, capability, inference efficiency, and state-tracking improvements; it introduces data-dependent position encoding for state tracking and contrasts this with Mamba-2 failures on some state-tracking tasks. ([arXiv][10])

#### Exercises

1. Implement a toy state-tracking task:

   * bracket depth,
   * variable assignment,
   * parity,
   * key-value update,
   * “current room in a maze” tracking.
2. Train tiny Transformer, Mamba-2, and Mamba-3-style variants.
3. Probe whether the hidden state linearly tracks the latent variable.
4. Intervene on hidden state and verify causal control.
5. Compare recurrent state vs residual-stream features.

#### Verification

```text
task accuracy > 95% on synthetic OOD split
linear probe generalizes to longer sequences
state intervention flips model prediction as predicted
random-direction intervention fails
trained tiny models have reproducible seeds
```

I would be careful not to assume large Mamba-3 pretrained checkpoints are always conveniently available. Use the official code where possible, but make the graded exercise train a tiny local model so the notebook remains robust.

---

### 2.5 Diffusion language models and DiffusionGemma

DiffusionGemma deserves its own chapter because it teaches a qualitatively different generation algorithm. Google describes DiffusionGemma as an experimental open text-diffusion model based on a 26B/4B-active MoE Gemma 4 architecture, using discrete diffusion with autoregressive encoding plus bidirectional denoising, and says quantized variants fit within 18GB VRAM. ([Google AI for Developers][2]) Google also says DiffusionGemma generates blocks in parallel and can be up to 4× faster than comparable autoregressive models. ([blog.google][11]) vLLM announced native support for DiffusionGemma in June 2026. ([vLLM Blog][12]) NVIDIA specifically describes running DiffusionGemma with Hugging Face Transformers on a GeForce RTX 5090-class setup. ([NVIDIA | Japan Blog][13])

#### Exercises

Start with a toy model before using DiffusionGemma:

1. Implement discrete noising schedule.
2. Implement masked-token denoising objective.
3. Train tiny diffusion LM on:

   * balanced parentheses,
   * toy arithmetic strings,
   * TinyStories subset,
   * templated QA.
4. Implement sampler:

   * uniform denoising,
   * confidence-based remasking,
   * temperature schedule,
   * block generation.
5. Compare AR vs diffusion generation:

   * latency,
   * edit distance to target,
   * entropy over denoising steps,
   * commitment time per token.

Then load DiffusionGemma:

1. Run local quantized inference.
2. Capture denoising-step activations.
3. Visualize entropy collapse over steps.
4. Compare early vs late token commitments.
5. Perform “diffusion patching”: replace intermediate denoising state or hidden activations between two prompts.

#### Verification

```text
toy forward noising distribution matches analytic expectation
toy denoising loss decreases
sampler reconstructs synthetic data above threshold
DiffusionGemma loads under VRAM budget
fixed seed output is deterministic
HF/vLLM outputs match approximately where deterministic controls allow
activation cache has expected shape over denoising steps
```

#### Research stretch

“**When does a diffusion LM decide the answer?**”
For a factual prompt, compare:

```text
token entropy over denoising steps
logit lens / tuned lens over steps
feature activations over steps
patching early vs late denoising activations
```

This is exactly the kind of architecture-specific interpretability ARENA does not yet teach.

---

### 2.6 Multimodal and specialist Gemma-family models

I would add optional notebooks for:

**FunctionGemma.** Google describes FunctionGemma as a Gemma 3 270M-based function-calling model intended for fast, private local agents. ([Google AI for Developers][14]) It is small enough for great local exercises: function-call logit attribution, schema-token probes, tool-choice steering, and hallucinated-tool detection.

**EmbeddingGemma.** Google describes EmbeddingGemma as a compact multilingual embedding model for on-device use. ([Google AI for Developers][15]) This is perfect for representation geometry, linear probes, contrastive examples, and “what does an embedding model encode?” exercises.

**MedGemma.** Google describes MedGemma as open medical text/image model work, but medical use requires serious validation. ([Google for Developers][16]) I’d include it only as a “domain safety and eval design” case study, not as a source of medical advice.

---

## 5. Chapter 3 — Sparse feature methods

This chapter should upgrade ARENA’s SAE content from toy models to frontier practice.

### 3.1 SAE variants

Implement four SAE families:

1. ReLU + L1 SAE.
2. TopK SAE.
3. Gated SAE.
4. JumpReLU SAE.

Gated SAEs are especially worth teaching because they decouple feature detection from feature magnitude and address shrinkage.

#### Exercises

1. Train on toy superposition.
2. Train on GPT-2 small residual stream.
3. Train on Gemma 1B / 4B subset activations.
4. Compare:

   * L0,
   * reconstruction MSE,
   * loss recovered,
   * feature density,
   * dead features,
   * interpretability score.
5. Show feature splitting / absorption pathologies.

#### Verification

```text
toy SAE recovers planted features
loss recovered improves over random dictionary
feature density histogram is not degenerate
random-feature explanations score lower
decoder-vector steering has predictable effect
```

---

### 3.2 Gemma Scope deep dive

Use released Gemma Scope artifacts instead of training huge SAEs.

#### Exercises

1. Load a Gemma Scope SAE at a chosen layer.
2. Find features related to:

   * refusal,
   * code,
   * sentiment,
   * names,
   * languages,
   * formatting,
   * chain-of-thought markers.
3. Compare base vs instruction-tuned behavior.
4. Validate features with:

   * held-out examples,
   * contrastive prompts,
   * ablation,
   * steering,
   * direct logit attribution.

#### Verification

```text
feature AUC > baseline on held-out positives/negatives
ablation reduces target behavior
steering increases target behavior without large perplexity blow-up
random matched features fail
```

---

### 3.3 Transcoders and attribution graphs

Anthropic’s circuit-tracing work uses learned features and attribution graphs to trace chains of intermediate computation; their biology-of-Claude work describes attribution graphs as a way to trace intermediate steps, generate hypotheses, and refine them with perturbations. ([Transformer Circuits][17]) Anthropic also open-sourced circuit tracing tools in 2025. ([Anthropic][18]) Transcoders are a key tool here because they approximate MLP computations with sparse features and have been trained on models from 120M to 1.4B in prior work. ([arXiv][19])

#### Exercises

1. Train a small transcoder for GPT-2 small MLP layer.
2. Replace the MLP with the transcoder.
3. Measure loss/reconstruction.
4. Build a feature-level graph:

   * input tokens,
   * upstream features,
   * transcoder features,
   * downstream logits.
5. Reproduce a simple circuit:

   * greater-than,
   * IOI fragment,
   * factual recall toy.

#### Verification

```text
transcoder replacement KL below threshold
feature graph preserves target logit diff
removing top-k graph nodes damages behavior
removing random nodes damages less
graph is sparse and reproducible across seeds
```

---

### 3.4 Crosscoders and model diffing

Crosscoders are useful for comparing models, including base vs instruction-tuned models and cross-architecture pairs. Recent crosscoder work applies model-diffing to architecture comparisons. ([arXiv][20])

#### Exercises

1. Train tiny crosscoder on:

   * Gemma base vs Gemma IT,
   * GPT-2 vs Pythia,
   * toy Transformer vs toy Mamba.
2. Identify shared vs model-specific features.
3. Use features to explain behavioral deltas:

   * refusal,
   * verbosity,
   * tool-call formatting,
   * chain-of-thought style,
   * unsafe-code refusal.
4. Validate by steering or ablating model-specific features.

#### Verification

```text
shared features reconstruct both models
model-specific features predict behavior difference
ablation of model-specific feature reduces behavior delta
random matched features fail
```

---

## 6. Chapter 4 — Activation-to-language methods

This is where the course becomes genuinely 2026-era.

### 4.1 Logit lens, tuned lens, Patchscopes

Start with lower-compute methods. The primer material describes logit lens as projecting intermediate residual states into vocabulary space; tuned lens adds learned transformations; Patchscopes generalize activation patching to decode intermediate representations into logits, probabilities, or natural-language explanations.

#### Exercises

1. Implement logit lens.
2. Implement tuned lens for GPT-2 / Pythia.
3. Implement attention lens.
4. Implement Patchscope templates:

   * “What entity is represented here?”
   * “What token will this become?”
   * “What fact is stored here?”
5. Compare against probes.

#### Verification

```text
tuned lens improves over logit lens on held-out next-token prediction
Patchscope answer accuracy beats text-only baseline
counterfactual activations change decoded answer
random activations produce low-confidence/noisy answers
```

---

### 4.2 Feature verbalizers

Modern interpretability increasingly needs natural-language explanations, but naive max-activating examples can produce interpretability illusions. The open-problems review warns that highly activating examples are correlational and can mislead; it specifically notes interpretability illusions from dataset-dependent explanations.

#### Exercises

Implement a full verbalizer loop:

1. Select a neuron / SAE feature / transcoder feature.
2. Gather:

   * top activating examples,
   * bottom examples,
   * random examples,
   * contrastive near-misses.
3. Ask a local or API LLM to generate an explanation.
4. Turn explanation into predictions:

   * should activate / should not activate.
5. Score on held-out examples.
6. Generate counterexamples.
7. Revise explanation.

#### Verification

```text
explanation predicts held-out activations above baseline
explanation survives contrastive near-misses
explanation predicts intervention effects, not just correlations
human-readable explanation is shorter than examples-only baseline
```

This should be framed as **hypothesis generation, not truth**. Existing validation guidance emphasizes predicting activations, counterfactuals, unusual failures, adversarial examples, replacement modules, and engineering utility rather than accepting plausible stories.

---

### 4.3 Mini Activation Oracles

Anthropic introduced Activation Oracles as models trained to answer arbitrary natural-language questions about LLM activations, with LatentQA-style tasks. ([Alignment Science Blog][21]) A 2026 follow-up improves activation-oracle training with on-policy rollouts, better conversational data, more layers, and an open AObench. ([arXiv][22])

Full AO training is too expensive locally, so the course should implement a **mini Activation Oracle**.

#### Subject models

Use:

```text
Pythia 70M / 160M
GPT-2 small
Gemma 1B
tiny Mamba
```

#### Questions

Train oracle to answer:

```text
What token is represented?
Is this activation from code?
Is the prompt asking a question?
Which of two names is the indirect object?
Is the model about to refuse?
Is this a truthful or false factual completion?
Which synthetic latent variable is active?
```

#### Exercises

1. Build activation dataset.
2. Create natural-language question-answer pairs.
3. Train small LoRA oracle.
4. Compare to:

   * linear probe,
   * MLP probe,
   * SAE feature classifier,
   * text-only LLM baseline.
5. Evaluate OOD:

   * new templates,
   * new names,
   * longer contexts,
   * adversarial distractors.

#### Verification

```text
oracle beats text-only baseline
oracle beats or matches probe on at least one task
oracle generalizes to held-out templates
oracle fails gracefully on random activations
oracle answer changes under activation patching
```

---

### 4.4 Mini Natural Language Autoencoders

Anthropic’s Natural Language Autoencoders are an unsupervised method for turning LLM activations into natural-language explanations and back; the project page describes NLAs as mapping residual-stream activations to natural-language text and reconstructing activations from it. ([Transformer Circuits][23]) Anthropic’s writeup frames the method as converting activations into readable text, such as revealing internal planning information. ([Anthropic][24])

Again, do not train a full-scale NLA locally. Build a miniature.

#### Exercises

1. Subject model: Pythia 160M or GPT-2 small.
2. Activation source: residual stream at selected layers.
3. Encoder: small LM or MLP-to-token bottleneck.
4. Decoder: text-to-activation reconstructor.
5. Train on activations paired with:

   * original text spans,
   * synthetic latent labels,
   * generated explanations.
6. Evaluate reconstruction.

#### Verification

```text
activation -> text -> activation reconstruction beats text-only baseline
reconstructed activation preserves target logit diff
reconstructed activation preserves probe-predicted latent variables
generated text is shorter than original prompt
counterfactual activation changes generated explanation
```

This makes the method concrete without pretending you can reproduce Anthropic-scale results on a laptop.

---

### 4.5 Predictive Concept Decoders

Predictive Concept Decoders compress activations into sparse concepts and decode answers to natural-language behavioral questions, making the decoder auditable through sparse concepts. ([arXiv][25])

#### Exercises

1. Build sparse concept encoder over activations.
2. Train question-conditioned decoder:

   * “Will the model answer Paris?”
   * “Will the model refuse?”
   * “Will the next token be a number?”
   * “Is the hidden variable even?”
3. Compare:

   * PCD,
   * probe,
   * SAE + linear classifier,
   * mini Activation Oracle.
4. Audit selected concepts.

#### Verification

```text
PCD accuracy > probe on at least one compositional task
concept sparsity below threshold
top concepts are stable across seeds
removing top concept changes prediction
random concept removal does less
```

---

## 7. Chapter 5 — Automated circuits and circuit tracing

### 5.1 Activation patching refresher

The existing ARENA activation-patching material is still excellent: clean/corrupt runs, patch activations, measure logit-diff recovery, then localize important components.  Keep it, but turn it into a reusable library.

### 5.2 Attribution patching and EAP

Exercises:

1. Implement activation patching.
2. Implement attribution patching:
   [
   \Delta \text{metric} \approx (a_\text{clean} - a_\text{corrupt}) \cdot \nabla_a \text{metric}
   ]
3. Implement edge attribution patching.
4. Compare:

   * exact patching,
   * attribution patching,
   * integrated gradients,
   * AtP* if time allows.

Verification:

```text
correlation with exact patching > threshold
top-k overlap with exact patching > threshold
runtime improvement measured
false-negative cases documented
```

### 5.3 ACDC and circuit metrics

Circuit discovery should be graded by faithfulness, minimality, and completeness; the open-problems review notes these as standard circuit metrics, while also warning that current circuit discovery has limitations and faithfulness issues.

Exercises:

1. Reproduce IOI circuit.
2. Reproduce greater-than circuit.
3. Run ACDC.
4. Run EAP-IG.
5. Compare circuit sizes and faithfulness.
6. Test with random baselines and distribution shifts.

Verification:

```text
circuit preserves >= X% logit-diff
minimality: removing circuit nodes damages behavior
completeness: adding top omitted nodes improves little
random circuit of same size performs worse
OOD prompt templates pass
```

### 5.4 Attribution graphs

Exercises:

1. Use SAEs or transcoders as nodes.
2. Build local attribution graph for:

   * factual recall,
   * IOI,
   * refusal,
   * synthetic state tracking,
   * diffusion denoising.
3. Perturb nodes and verify predicted effects.
4. Convert graph to a written causal story.

Verification:

```text
graph explains target metric
top graph path survives perturbation tests
alternative graph baselines fail
natural-language graph summary predicts counterfactuals
```

---

## 8. Chapter 6 — Alignment interpretability

I would not replace the existing LLM eval material. Instead, add white-box extensions.

### 6.1 Refusal directions and safe steering

Exercises:

1. Reproduce refusal-direction style analysis on a small instruction model.
2. Use harmless and harmful-category labels without generating harmful procedural content.
3. Find candidate refusal direction.
4. Test addition/removal on benign refusal and allowed prompts.
5. Compare:

   * mean-difference direction,
   * probe direction,
   * SAE features,
   * Gemma Scope features.

Verification:

```text
direction separates refusal vs non-refusal prompts
steering changes refusal rate on safe benchmark
general capability degradation is small
random directions fail
```

Activation steering is a natural fit because it comes directly from the linear-representation hypothesis and is one of the better-developed interpretability-inspired control methods.

### 6.2 Chain-of-thought faithfulness

Recent work continues to stress that CoT can be unfaithful, and 2025–2026 work studies CoT faithfulness with SAEs, activation patching, and instance-level detection. ([arXiv][26])

Exercises:

1. Use toy arithmetic / GSM8K-small.
2. Compare:

   * no-CoT,
   * faithful CoT,
   * biased CoT,
   * post-hoc rationalization.
3. Train probes / SAEs for intermediate reasoning variables.
4. Patch hidden reasoning variables.
5. Detect when answer was decided before the CoT justification.

Verification:

```text
probe predicts hidden answer before final token
patching hidden answer changes output
CoT text-only baseline misses some unfaithful cases
feature-level detector improves detection
```

### 6.3 Emergent misalignment detection

Emergent misalignment work shows that narrow finetuning, such as insecure-code finetuning, can induce broad misalignment. ([arXiv][27]) OpenAI reported that emergent misalignment can be detected and mitigated using internal features/persona-style signals. ([OpenAI][28])

For a course, do **not** train harmful models. Use safe proxies:

1. Benign “persona drift” finetune:

   * overly sycophantic,
   * overconfident,
   * JSON-only,
   * pirate-style,
   * refusal-overgeneralizing.
2. Detect drift using:

   * probes,
   * SAEs,
   * crosscoders,
   * mini activation oracle.
3. Evaluate whether white-box detector catches drift earlier than black-box eval.

Verification:

```text
detector predicts drift on held-out prompts
crosscoder model-specific features align with behavior delta
mitigation reduces drift without large capability loss
```

### 6.4 White-box evals and monitors

Exercises:

1. Build a small dashboard:

   * prompt,
   * model output,
   * active SAE features,
   * refusal score,
   * hallucination score,
   * CoT faithfulness score.
2. Add thresholds and calibration curves.
3. Compare monitor vs black-box classifier.

Verification:

```text
white-box monitor has calibrated AUROC
monitor catches at least one failure black-box baseline misses
false positives documented
feature explanations validated on held-out prompts
```

---

## 9. Reading and implementation map

I’d make `docs/literature_review.md` into a living map:

| Area                | Read                                             | Implement                                      |
| ------------------- | ------------------------------------------------ | ---------------------------------------------- |
| Existing ARENA base | ARENA Ch. 1; TransformerLens                     | Keep GPT-2, induction, IOI, SAEs               |
| Gemma               | Gemma 3/4 docs; Gemma Scope                      | Gemma block + HF parity + Gemma Scope features |
| Mamba               | Mamba, Mamba-2, Mamba-3                          | Selective scan, recurrent/parallel equivalence |
| Diffusion LMs       | DiffusionGemma docs                              | Toy discrete diffusion + quantized inference   |
| SAEs                | ReLU, TopK, Gated, JumpReLU                      | Train small SAEs, evaluate L0/loss recovered   |
| Transcoders         | Sparse feature circuits, circuit tracing         | MLP replacement + attribution graphs           |
| Crosscoders         | Model diffing                                    | Base vs IT feature comparison                  |
| Verbalizers         | Patchscopes, AOs, NLAs, PCDs                     | Mini AO/NLA/PCD with baselines                 |
| Alignment           | Refusal, CoT faithfulness, emergent misalignment | White-box monitors and safe proxy finetunes    |

The wider interpretability literature now explicitly calls for moving from model-component analyses to human-interpretable features and natural-language explanations while preserving faithfulness.

---

## 10. The verification framework

This should be the heart of the fork.

### Implementation correctness

Every architecture notebook must include:

```text
shape tests
dtype tests
gradient tests
HF parity
generation parity
cache parity
speed benchmark
VRAM benchmark
```

### Mathematical equivalence

For Mamba/SSMs:

```text
parallel selective scan == recurrent selective scan
chunked scan == full scan
state cache == full-context forward
```

For RoPE:

```text
rotation preserves norm
relative-position identity holds
```

For diffusion:

```text
forward noising distribution matches schedule
reverse sampler improves denoising likelihood
entropy decreases over denoising steps
```

### Interpretability validation

For every proposed explanation:

```text
held-out examples
contrastive examples
random controls
causal ablation
causal steering
counterfactual patching
OOD prompt templates
baseline comparison
```

This follows the field’s current best-practice direction: explanations should predict activations, counterfactuals, failures, interventions, and useful engineering outcomes, not merely sound plausible.

### Circuit validation

Use:

```text
faithfulness
minimality
completeness
OOD generalization
random-circuit baseline
adversarial examples
```

### SAE validation

Use:

```text
L0
loss recovered
feature density
dead features
reconstruction KL
steering effect
ablation effect
explanation score
```

### Verbalizer validation

Use:

```text
activation -> explanation -> activation prediction
explanation predicts held-out activations
explanation predicts causal effects
text-only baseline
probe baseline
random-activation control
counterfactual activation control
```

---

## 11. Proposed phased roadmap

### Phase 1 — Stabilize fork

Goal: make the repo impossible to silently break.

Deliverables:

```text
local install guide
GPU smoke tests
HF parity harness
VRAM estimator
activation-store abstraction
CI smoke tests
```

### Phase 2 — Modern transformer: Gemma

Deliverables:

```text
Gemma block from scratch
Gemma 3 1B/4B HF parity
KV-cache parity
Gemma Scope SAE notebook
safe feature steering notebook
```

### Phase 3 — SSMs: Mamba/Mamba-2/Mamba-3

Deliverables:

```text
selective scan from scratch
parallel/recurrent equivalence
Mamba 130M weight loading
Mamba-2 SSD intuition notebook
state-tracking toy benchmark
Mamba vs Transformer interp comparison
```

### Phase 4 — Diffusion language models

Deliverables:

```text
toy discrete diffusion LM
toy AR-vs-diffusion comparison
DiffusionGemma local inference notebook
denoising-step activation analysis
diffusion patching prototype
```

### Phase 5 — Sparse feature methods

Deliverables:

```text
SAE variants
Gemma Scope deep dive
transcoder replacement
feature-level attribution graphs
crosscoder model diffing
```

### Phase 6 — Activation-to-language

Deliverables:

```text
tuned lens
Patchscopes
feature verbalizer
mini Activation Oracle
mini NLA
mini PCD
```

### Phase 7 — Alignment interpretability

Deliverables:

```text
refusal direction / SAE feature analysis
CoT faithfulness probes and patches
safe emergent-misalignment proxy
white-box monitor dashboard
```

### Phase 8 — Capstone paper-style project

Deliverables:

```text
one research question
one benchmark
one baseline suite
one mechanistic claim
one causal validation suite
one reproducible writeup
```

---

## 12. Best capstone projects

I’d choose one of these as your “PhD free-time flagship.”

### Capstone A — Gemma Scope feature robustness under quantization

Question: do Gemma Scope features survive QAT/int4 quantization?

Why it is good:

* Local.
* Practical.
* Clear metrics.
* Relevant to laptop inference.

Verification:

```text
feature activation correlation BF16 vs int4
steering effect preserved
loss recovered preserved
behavioral degradation measured
```

### Capstone B — Mamba vs Transformer state tracking

Question: do Mamba hidden states expose cleaner state variables than transformer residual streams?

Why it is good:

* Architecture frontier.
* Toy training feasible.
* Strong causal tests.

Verification:

```text
state probe generalizes OOD
state intervention causally changes output
parallel/recurrent implementation verified
comparison against transformer baseline
```

### Capstone C — Mini Activation Oracle vs probes

Question: when does a question-conditioned activation oracle beat a bank of probes?

Why it is good:

* Directly 2026-relevant.
* Local miniature feasible.
* Strong baselines possible.

Verification:

```text
oracle beats text-only baseline
oracle beats probe on compositional questions
oracle generalizes to held-out prompt templates
oracle fails on random activations
```

### Capstone D — DiffusionGemma denoising-time circuits

Question: when does a diffusion LM commit to a factual answer?

Why it is good:

* Novel architecture.
* Great visualizations.
* Could become publishable if carefully scoped.

Verification:

```text
token entropy commitment time
patch early vs late denoising states
feature activation over denoising trajectory
AR baseline comparison
```

### Capstone E — Safe emergent-misalignment proxy with crosscoders

Question: can crosscoder features detect a benign proxy for emergent misalignment before black-box evals?

Why it is good:

* Alignment-relevant.
* Avoids harmful finetuning.
* Clear white-box-vs-black-box comparison.

Verification:

```text
feature predicts drift
feature ablation reduces drift
black-box classifier baseline included
held-out prompts and seeds
```

---

## 13. What I would not do

I would **not** try to:

1. Train large Gemma-scale SAEs from scratch when Gemma Scope exists.
2. Train full-scale Anthropic-style NLAs or Activation Oracles locally.
3. Make every notebook support every architecture immediately.
4. Treat max-activating examples as sufficient explanations.
5. Train genuinely dangerous misaligned models as a course exercise.
6. Use 27B models as required exercises.
7. Accept any architecture implementation without HF parity tests.

---

## 14. The “epic ARENA extension” syllabus

If I had to name the final chapters:

```text
[1.5] Local Frontier ML Infrastructure
[2.1] Gemma from Scratch
[2.2] Gemma Scope and Feature Steering
[2.3] Mamba from Scratch
[2.4] Mamba-2, Mamba-3, and State Tracking
[2.5] Diffusion Language Models and DiffusionGemma
[2.6] Multimodal, Embedding, and Function-Calling Models
[3.1] Sparse Autoencoder Variants
[3.2] Transcoders and Feature Circuits
[3.3] Crosscoders and Model Diffing
[4.1] Logit Lens, Tuned Lens, and Patchscopes
[4.2] Feature Verbalizers
[4.3] Mini Activation Oracles
[4.4] Mini Natural Language Autoencoders
[4.5] Predictive Concept Decoders
[5.1] Attribution Patching at Scale
[5.2] Automated Circuit Discovery
[5.3] Attribution Graphs and Circuit Tracing
[6.1] Refusal, Steering, and Unlearning
[6.2] Chain-of-Thought Faithfulness
[6.3] Emergent Misalignment and White-Box Monitors
[7.1] Capstone Research Sprint
```

---

## 15. Final recommendation

The highest-value path is:

1. **Build the verification harness first.**
2. **Implement Gemma 3 and Mamba from scratch with weight parity.**
3. **Use Gemma Scope for serious local feature work.**
4. **Add DiffusionGemma as a smaller but high-novelty chapter.**
5. **Implement mini versions of Activation Oracles, NLAs, and PCDs rather than full-scale replications.**
6. **Turn alignment into white-box interpretability exercises, not just black-box evals.**
7. **End with one capstone that has paper-level standards: baselines, causal validation, OOD tests, reproducible scripts, and a clear claim.**

That would make the fork more than an update to ARENA. It would become a local, rigorous, 2026-era mechanistic-interpretability lab course: implementation-heavy, falsification-heavy, and genuinely useful for starting publishable research on a single 24GB laptop GPU.

[1]: https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/?utm_source=chatgpt.com "Gemma 3 QAT Models: Bringing state-of-the-Art AI to ..."
[2]: https://ai.google.dev/gemma/docs/diffusiongemma "DiffusionGemma model overview  |  Google AI for Developers"
[3]: https://arxiv.org/abs/2503.19786?utm_source=chatgpt.com "Gemma 3 Technical Report"
[4]: https://ai.google.dev/gemma/docs/core?utm_source=chatgpt.com "Gemma 4 model overview | Google AI for Developers"
[5]: https://ai.google.dev/gemma/docs/gemma_scope "Gemma Scope  |  Google AI for Developers"
[6]: https://deepmind.google/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/?utm_source=chatgpt.com "Gemma Scope: helping the safety community shed light on ..."
[7]: https://github.com/state-spaces/mamba "GitHub - state-spaces/mamba: Mamba SSM architecture · GitHub"
[8]: https://arxiv.org/abs/2405.21060?utm_source=chatgpt.com "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
[9]: https://arxiv.org/abs/2505.15105?utm_source=chatgpt.com "Mechanistic evaluation of Transformers and state space ..."
[10]: https://arxiv.org/html/2603.15569v1 "Mamba-3: Improved Sequence Modeling using State Space Principles"
[11]: https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/?utm_source=chatgpt.com "DiffusionGemma: 4x faster text generation"
[12]: https://vllm-project.github.io/2026/06/10/diffusion-gemma.html?utm_source=chatgpt.com "DiffusionGemma: The First Diffusion LLM (dLLM) Natively ..."
[13]: https://blogs.nvidia.co.jp/blog/rtx-ai-garage-local-gemma-diffusion/?utm_source=chatgpt.com "NVIDIA、ローカル AI の実現に向けて Google DeepMind の ..."
[14]: https://ai.google.dev/gemma/docs/functiongemma?utm_source=chatgpt.com "FunctionGemma model overview - Google AI for Developers"
[15]: https://ai.google.dev/gemma/docs/embeddinggemma?utm_source=chatgpt.com "EmbeddingGemma model overview - Google AI for Developers"
[16]: https://developers.google.com/health-ai-developer-foundations/medgemma?utm_source=chatgpt.com "MedGemma | Health AI Developer Foundations"
[17]: https://transformer-circuits.pub/2025/attribution-graphs/biology.html "On the Biology of a Large Language Model"
[18]: https://www.anthropic.com/research/open-source-circuit-tracing "Open-sourcing circuit-tracing tools \ Anthropic"
[19]: https://arxiv.org/abs/2406.11944?utm_source=chatgpt.com "Transcoders Find Interpretable LLM Feature Circuits"
[20]: https://arxiv.org/pdf/2602.11729?utm_source=chatgpt.com "Cross-Architecture Model Diffing with Crosscoders"
[21]: https://alignment.anthropic.com/2025/activation-oracles/?utm_source=chatgpt.com "Activation Oracles: Training and Evaluating LLMs as General ..."
[22]: https://arxiv.org/html/2606.02609v1 "Building Better Activation Oracles"
[23]: https://transformer-circuits.pub/2026/nla/?utm_source=chatgpt.com "Natural Language Autoencoders Produce Unsupervised ..."
[24]: https://www.anthropic.com/research/natural-language-autoencoders?utm_source=chatgpt.com "Natural Language Autoencoders"
[25]: https://arxiv.org/html/2512.15712v1 "Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants"
[26]: https://arxiv.org/abs/2507.22928?utm_source=chatgpt.com "[2507.22928] How does Chain of Thought Think ..."
[27]: https://arxiv.org/abs/2502.17424?utm_source=chatgpt.com "Narrow finetuning can produce broadly misaligned LLMs"
[28]: https://openai.com/index/emergent-misalignment/?utm_source=chatgpt.com "Toward understanding and preventing misalignment ..."

--------------------------------------------------




Yes. I’d keep **original ARENA 3.0 fully intact**, not “modernize it by breaking it.” The clean design is:

```text
ARENA_3.0/
  original/
    copied-or-submodule ARENA notebooks, frozen except bugfix patches
  frontier_extensions/
    ch2_modern_architectures/
    ch3_feature_methods/
    ch4_activation_to_language/
    ch5_jepa_world_models/
    ch6_shapley_and_attribution/
    ch7_alignment_world_model_safety/
```

The original ARENA mechanics stay as the prerequisite spine: TransformerLens, hooks, activation caching, induction heads, logit attribution, path patching, IOI, toy superposition, SAEs, and NNsight. The uploaded ARENA material explicitly has the “introduce a TransformerLens feature → apply it in an exercise” pedagogy, with induction circuits as the running theme, and learning objectives around loading HookedTransformer models, caching activations, visualizing attention heads, identifying induction heads, using hooks, and doing logit attribution.  It also already frames IOI exercises as moving from exploratory analysis to more rigorous path patching and open-ended circuit work.

So the extension should be **append-only, verified, and tiered by GPU feasibility**. The original remains the “classical transformer interpretability course”; the new material becomes “frontier model families and world-model interpretability lab.”

Also: yes, **JEPAs and world models absolutely belong**. They are not a side quest. They are almost the perfect bridge between representation learning, world modelling, multimodal interpretability, and mechanistic validation.

---

# Updated extension: add a JEPA / world-model track

I’d add a new full track:

```text
[ch5] Joint-Embedding Predictive Architectures and World Models
  [5.1] I-JEPA from scratch
  [5.2] V-JEPA and V-JEPA 2 feature prediction
  [5.3] V-JEPA 2.1 dense feature interpretability
  [5.4] VL-JEPA and embedding-space language prediction
  [5.5] Audio-JEPA, A-JEPA, MC-JEPA, 3D-JEPA, time-series JEPA
  [5.6] Action-conditioned JEPA world models
  [5.7] World-model interpretability: Othello, mazes, Sudoku, RL, robotics
  [5.8] Capstone: Do latent-prediction models learn more interpretable world states?
```

The motivating arc is simple:

**Autoregressive models predict tokens. Diffusion models denoise tokens or pixels. JEPAs predict latent representations. World models predict latent future states. Mechanistic interpretability asks what those latent states actually encode, how they are updated, and whether they support planning.**

That is a beautiful PhD-level learning arc.

---

# The JEPA family to include

I would treat “all JEPAs” as **all major families plus a living registry**, because by 2026 there are many domain-specific JEPA variants. The course should include the important conceptual branches, not every thin application paper as a core exercise.

## 1. I-JEPA

**Core idea:** predict the representation of masked target blocks of an image from context blocks, without reconstructing pixels. Meta’s official I-JEPA repo describes it as predicting representations of one part of an image from representations of other parts, and the original paper frames I-JEPA as a non-generative method for semantic image representation learning. ([GitHub][1])

Exercises:

```text
Implement:
  image patchification
  context / target block sampler
  context encoder
  EMA target encoder
  predictor
  representation-space loss
  collapse diagnostics

Verify:
  loss decreases on CIFAR-10 / STL-10 subset
  learned features beat random ViT on linear probe
  no representation collapse: covariance rank / variance tests
  official checkpoint feature extraction matches expected dimensions
```

Interpretability exercises:

```text
probe patch tokens for object part, color, position, texture
ablate context patches and measure target embedding error
train SAEs on patch embeddings
visualize nearest-neighbor feature directions
test whether features are semantic or texture-biased
```

This becomes the “image JEPA from scratch” chapter.

---

## 2. V-JEPA

**Core idea:** latent prediction over video rather than pixel reconstruction. The official Meta FAIR V-JEPA repo describes V-JEPA as a self-supervised method for learning visual representations from video. ([GitHub][2])

Exercises:

```text
Implement:
  video tubelet embedding
  temporal masking
  target encoder
  context encoder
  predictor
  latent prediction loss
  frame sampling utilities

Verify:
  model runs on toy videos
  representation-space loss decreases
  temporal order matters: shuffled clips perform worse
  embeddings support video retrieval better than random baseline
```

Interpretability exercises:

```text
probe for:
  object identity
  object position
  velocity
  contact / collision
  camera motion
  action class

patch:
  context frames
  target frames
  object crops
  temporal order
```

---

## 3. V-JEPA 2 and V-JEPA 2-AC

This should be a flagship chapter. Meta describes V-JEPA 2 as a self-supervised foundation world model trained on video, intended to understand physical reality, anticipate outcomes, and plan; the paper says it combines internet-scale video with a small amount of robot trajectory data to learn understanding, prediction, and planning in the physical world. ([Meta AI][3]) The official V-JEPA 2 GitHub says V-JEPA 2-AC is a latent action-conditioned world model post-trained from V-JEPA 2 that solves manipulation tasks without environment-specific data collection or task-specific training/calibration. ([GitHub][4]) Hugging Face Transformers also has V-JEPA 2 docs and model-loading examples, including feature extraction via `AutoModel`. ([Hugging Face][5])

Exercises:

```text
Load:
  facebook/vjepa2-vitl-fpc64-256  # required local target
  larger ViT-g checkpoints        # optional stretch

Implement:
  video preprocessing
  feature extraction
  predictor-output extraction
  frame/tubelet-level probing
  video retrieval
  simple planning in latent space

Verify:
  checkpoint loads locally
  output tensor shapes match docs
  deterministic output for fixed video and seed
  linear probe beats random baseline
  nearest-neighbor retrieval returns semantically similar clips
  VRAM < 24GB for required config
```

World-model exercises:

```text
latent rollout:
  z_t, action a_t -> predicted z_{t+k}

planning:
  choose action sequence minimizing distance to goal embedding

verification:
  predicted latent goal distance correlates with true rollout distance
  shuffled-action baseline fails
  random latent baseline fails
```

For your GPU, the required exercise should use **V-JEPA 2 ViT-L**, not ViT-G. ViT-g/G can be marked optional with frame count reduction, bf16, CPU offload, or remote fallback.

---

## 4. V-JEPA 2.1

By 2026 this belongs in the course. V-JEPA 2.1 is a 2026 family of self-supervised image/video models designed to produce dense, spatially and temporally coherent visual representations, using dense predictive loss, deep self-supervision, multimodal tokenizers, and scaling. ([arXiv][6])

Why it matters for interpretability: dense features make the model much more analyzable. Instead of only getting a global video embedding, you can inspect patch-level and tubelet-level structure.

Exercises:

```text
Compare V-JEPA 2 vs V-JEPA 2.1:
  patch feature PCA
  segmentation transfer
  object tracking consistency
  temporal correspondence
  occlusion robustness

Verify:
  dense features cluster by object/part better than V-JEPA 2
  temporal feature consistency beats shuffled-frame baseline
  linear probes recover depth / segmentation / object masks above baseline
```

This chapter should be called:

```text
[5.3] Dense World Representations: V-JEPA 2.1 as an Interpretable Video Encoder
```

---

## 5. VL-JEPA

VL-JEPA is very relevant. It predicts continuous text embeddings rather than autoregressively generating text tokens; the paper reports stronger performance than a token-space VLM baseline under a controlled comparison, with 50% fewer trainable parameters, and selective decoding that reduces decoding operations by 2.85× while preserving similar performance. ([arXiv][7])

This is almost tailor-made for a comparison with LLM logit-lens / tuned-lens methods.

Exercises:

```text
Implement:
  image/video encoder
  text embedding target encoder
  predictor from visual context to text embedding
  lightweight decoder
  selective decoding policy

Compare:
  token-space VLM loss
  embedding-space JEPA loss
  CLIP-style contrastive loss
  SigLIP-style loss
  autoregressive captioner

Verify:
  predicted text embeddings retrieve correct captions
  embedding prediction beats random and CLIP-only baselines
  selective decoding saves decoder calls
  VQA / retrieval smoke tests run locally
```

Interpretability exercises:

```text
patch visual tokens and observe text-embedding movement
decode nearest text prototypes before and after patching
probe whether embeddings encode object, action, relation, count
compare text-space explanations to feature-space explanations
```

---

## 6. Audio-JEPA / A-JEPA

There are at least two audio-relevant branches worth including. A-JEPA extends JEPA-style latent prediction to audio spectrograms and reports strong audio/speech classification performance; Audio-JEPA similarly predicts masked spectrogram patch embeddings and reports comparable performance to wav2vec 2.0 and data2vec while using less training data. ([arXiv][8])

Exercises:

```text
Implement:
  mel-spectrogram patchification
  time-frequency-aware masking
  context/target encoder
  latent spectrogram prediction

Verify:
  ESC-50 / SpeechCommands linear probe > random baseline
  pitch/time-shift augmentations behave sensibly
  embeddings cluster by class
```

Interpretability:

```text
probe:
  pitch
  phoneme-ish features
  instrument identity
  speech/music/noise
  onset/rhythm

patch:
  time bands
  frequency bands
  harmonics
```

---

## 7. MC-JEPA

MC-JEPA jointly learns motion and content features, combining optical-flow and self-supervised objectives. ([arXiv][9]) This is useful because it directly gives you a clean interpretability question:

**Are motion and content disentangled?**

Exercises:

```text
train tiny MC-JEPA on Moving-MNIST / bouncing balls
probe content identity vs velocity vs position
compute subspace overlap between motion and content features
patch motion-only features and test predicted future movement
patch content-only features and test identity preservation
```

Verification:

```text
motion probe high, content probe high
cross-patching velocity changes trajectory but not object identity
cross-patching identity changes object class but not trajectory
random subspaces fail
```

This is a perfect model-organism exercise.

---

## 8. 3D-JEPA, US-JEPA, T-JEPA, MJEPA, BiJEPA, LLM-JEPA, JEPA-VLA

These should not all be full required chapters. I’d make them a **JEPA zoo** module with one common adapter API.

Relevant long-tail variants:

```text
3D-JEPA      point clouds / 3D representations
US-JEPA      ultrasound latent prediction
T-JEPA       time-series / trajectory representation
MJEPA        multimodal audio-visual JEPA
BiJEPA       bidirectional/cycle-consistent latent prediction
LLM-JEPA     language-model / JEPA hybrid
JEPA-VLA     video-predictive embeddings for vision-language-action robotics
ACT-JEPA     imitation learning + JEPA world model for action sequences
Demo-JEPA    demonstration-conditioned future latent trajectory prediction
SparseJEPA   sparse JEPA representation learning
```

A 2026 MJEPA paper introduces a multimodal audio-visual self-supervised JEPA with a unified encoder and a JEPA objective that works within and across modalities. ([arXiv][10]) JEPA-VLA argues that predictive embeddings such as V-JEPA 2 help VLA models capture task-relevant temporal dynamics and improve sample efficiency/generalization. ([arXiv][11]) ACT-JEPA combines imitation learning with a JEPA-style latent world-model objective for action and observation sequences. ([arXiv][12])

The course implementation should expose:

```python
class JEPAAdapter:
    def encode_context(...)
    def encode_target(...)
    def predict_target(...)
    def loss(...)
    def extract_features(...)
    def patch_feature(...)
    def rollout_latent(...)
```

Then every JEPA variant plugs into the same verification suite.

---

# New world-model interpretability track

This deserves a serious standalone track, not just a few notebooks.

```text
[ch6] World-Model Interpretability
  [6.1] What is a world model?
  [6.2] Othello-GPT and linear latent board states
  [6.3] Maze transformers and causal world models
  [6.4] Sudoku / constraint-algebra world models
  [6.5] DreamerV3, TD-MPC2, IRIS, and latent dynamics
  [6.6] V-JEPA 2 / 2.1 as physical world models
  [6.7] World-model circuits, probes, patching, and planning
  [6.8] Microscope AI and scientific concept discovery
```

The existing interpretability literature already motivates this. The uploaded material says microscope AI means training a neural network to model a dataset, then interpreting the model to discover potentially novel predictors; it gives examples such as extracting chess concepts from AlphaZero and discovering scientific predictors from learned models.  It also explicitly says the field needs interpretability for broader model families and modalities, not just CNNs, BERT, and GPT-style text models.

## 6.2 Othello-GPT as the first required world-model lab

This is the obvious model organism. The TransformerLens demo ports Othello-GPT weights and asks exactly the right interpretability questions: nonlinear vs linear probes, minimal edits, component shifts, logits effects, and nontrivial circuits.  Prior Othello-GPT follow-up work found a linear representation of board state and used vector arithmetic interventions.

Exercises:

```text
replicate:
  legal-move prediction
  board-state probe
  current-player / opponent-player representation
  causal intervention on square state

extend:
  train SAE on residual stream
  discover board features
  find circuits for legality of a target square
  compare linear probe, nonlinear probe, SAE, Patchscope, causal patching
```

Verification:

```text
linear probe accuracy > threshold
board edit changes legal move logits as predicted
random edit fails
patching localized layer/position has causal effect
interpreted feature predicts held-out board states
```

## 6.3 Maze world models

Recent work finds that transformers trained on maze tasks can develop structured causal world models. ([arXiv][13])

Exercises:

```text
train tiny transformer on maze trajectories
probe wall map, current location, goal location, shortest-path distance
patch current-location representation
intervene on wall representation
compare with behavioral planning output
```

Verification:

```text
probe generalizes to unseen mazes
location patch changes next action
wall patch changes route choice
causal intervention matches predicted planning change
```

## 6.4 Sudoku / constraint world models

A 2026 paper reports that transformers trained on Sudoku solving traces build structured world models organized around rows, columns, and boxes, and identifies a naked-single circuit. ([arXiv][14])

Exercises:

```text
train tiny Sudoku trace transformer
probe row/column/box candidates
find naked-single neurons/features
ablate circuit and measure puzzle-solving degradation
```

Verification:

```text
candidate-set probe high accuracy
naked-single feature predicts correct final digit
ablation reduces solving accuracy
random neuron ablation is weaker
```

## 6.5 RL world models: DreamerV3, TD-MPC2, IRIS

These should be included as **conceptual + small local labs**, not full-scale retraining.

DreamerV3 learns a model of the environment and trains behavior by imagining future scenarios; the paper reports strong performance across 150+ tasks with one configuration, including Minecraft diamond collection from scratch. ([arXiv][15]) TD-MPC2 performs local trajectory optimization in the latent space of an implicit decoder-free world model and reports strong results across 104 continuous-control tasks. ([arXiv][16]) IRIS uses a discrete autoencoder plus autoregressive transformer world model and reports strong Atari 100k sample efficiency. ([arXiv][17])

Local exercises:

```text
Dreamer-lite:
  train on CartPole / MiniGrid / Crafter tiny
  inspect RSSM state
  probe position, velocity, reward, termination

TD-MPC2-lite:
  train/load small continuous-control world model
  probe latent state for physical variables
  visualize latent planning trajectories

IRIS-lite:
  discrete VAE tokenizer on small gridworld frames
  transformer predicts future discrete tokens
  inspect attention over past states
```

Verification:

```text
latent probes recover ground-truth state variables
latent rollout error correlates with real rollout error
planning in latent space beats random action sequences
intervening on latent position changes predicted reward/action
```

---

# JEPA interpretability: the really interesting questions

This could become publishable if done carefully.

## Question 1: Are JEPA latents more “world-state-like” than contrastive or generative latents?

Compare:

```text
I-JEPA / V-JEPA
MAE / VideoMAE
DINO / DINOv2 / DINOv3-style features
CLIP / SigLIP
diffusion encoder features
autoregressive video world model
```

Metrics:

```text
linear probe for state variables
causal patchability
counterfactual consistency
temporal stability
disentanglement of object/motion/action
OOD robustness
planning usefulness
```

V-JEPA 2.1 and related 2026 analysis make this especially timely because dense latent-prediction features are being evaluated as world-model representations. ([arXiv][6])

## Question 2: Do JEPA predictors implement transition models?

For video or toy physics:

```text
z_context -> z_target
z_t, action -> z_{t+k}
```

Interpretability tests:

```text
linearize predictor Jacobian
find velocity/contact/collision directions
patch velocity direction
patch object-identity direction
measure predicted future embedding
```

## Question 3: Can we find “object permanence” features?

Dataset:

```text
moving objects with occlusion
bouncing balls
3D toy scenes
small robot manipulation videos
```

Verification:

```text
feature remains active during occlusion
patching feature changes predicted reappearance
feature is not just texture/color
feature generalizes across object colors/backgrounds
```

## Question 4: Does action-conditioning make causal variables easier to find?

Compare:

```text
V-JEPA 2
V-JEPA 2-AC
ACT-JEPA
JEPA-WM
TD-MPC2
Dreamer-lite
```

Verification:

```text
action-conditioned latent predicts controllable variables better
uncontrollable nuisance variables are suppressed
action patch changes predicted future state in the correct direction
```

---

# Add SHAPley-value SOTA as a comparison track

Yes, add it — but with the right framing.

SHAP/Shapley methods are not mechanistic interpretability in the strict sense. They are mostly **input attribution / cooperative-game credit assignment**. The uploaded review explicitly separates local attribution methods such as SHAP, LIME, Integrated Gradients, and Grad-CAM from mechanistic interpretability, which asks how models solve general classes of problems via shared computation.  That is exactly why SHAP should be in the course: it is a strong baseline and a useful foil.

I’d add:

```text
[ch7] Shapley Values, Attribution, and Mechanistic Baselines
  [7.1] Shapley values from scratch
  [7.2] KernelSHAP, PartitionSHAP, TreeSHAP, DeepSHAP
  [7.3] SAGE and global Shapley importance
  [7.4] TokenSHAP / TokenShapley for LLMs
  [7.5] Shapley interactions: Shapley-Taylor, Faith-Shap, SHAP-IQ
  [7.6] SHAP vs gradients vs patching vs causal tracing
  [7.7] Failure modes: OOD coalitions, tokenization, interaction blindness
  [7.8] When Shapley helps mech interp, and when it misleads
```

## SHAP methods to implement

Required:

```text
Exact Shapley on tiny models
KernelSHAP
Permutation SHAP
Partition SHAP / Owen values
TreeSHAP for tree models
DeepSHAP / GradientSHAP
SAGE for global feature importance
TokenSHAP for LLM prompt attribution
Shapley-Taylor interactions
SHAP-IQ / shapiq-style interaction attribution
```

SAGE is a Shapley-based global feature importance method that quantifies a model’s dependence on each feature via predictive power. ([arXiv][18]) TokenSHAP adapts Monte Carlo Shapley attribution to tokens/substrings in LLM prompts. ([arXiv][19]) TokenShapley is a 2025 token-level context attribution method that reports improvements over attribution baselines. ([arXiv][20]) `shapiq` is a 2024/2025 package for Shapley interaction values, including applications to vision transformers and language models. ([arXiv][21]) KernelSHAP-IQ extends KernelSHAP toward interaction quantification and reports state-of-the-art performance for feature interactions. ([arXiv][22])

## Critical limitations to teach

This chapter must be skeptical. The uploaded transformer interpretability primer notes that popular attribution methods such as SHAP and Integrated Gradients have been found unreliable for predicting counterfactual model behavior in realistic settings, and that perturbation methods can create out-of-distribution inputs.

So every SHAP notebook should include:

```text
faithfulness test
counterfactual test
OOD coalition test
causal intervention comparison
random baseline
strong simple baseline
```

## Best SHAP-vs-mech-interp exercises

### Exercise 1: Exact ground-truth Boolean circuits

Train tiny networks on known functions:

```text
XOR
AND/OR
majority
multiplexer
sparse parity
modular arithmetic
```

Compare:

```text
Exact Shapley
KernelSHAP
Integrated Gradients
gradient × input
activation patching
attribution patching
SAE features
```

Verification:

```text
recover known causal variables
recover known interactions
detect where single-feature attribution fails on XOR/parity
```

### Exercise 2: SHAP on Othello-GPT

Use “players” as:

```text
input moves
board squares
latent board-state directions
SAE features
attention heads
MLP neurons
```

Compare SHAP attribution to causal interventions:

```text
Does SHAP say the same square matters as patching?
Does SHAP identify legal-move causal variables?
Does it fail when token coalitions are invalid Othello histories?
```

### Exercise 3: Shapley interactions for world models

Use toy physics:

```text
object position
object velocity
wall/contact
action
goal
```

A single Shapley value may miss interactions like:

```text
action × object position
velocity × wall contact
goal × current state
```

So this is where SHAP-IQ / Shapley-Taylor / Faith-Shap become useful.

### Exercise 4: TokenSHAP vs activation patching on LLM prompts

Dataset:

```text
factual recall
IOI
sentiment
refusal
math word problems
```

Compare:

```text
important input tokens from TokenSHAP
important components from activation patching
important features from SAEs
```

Verification:

```text
removing/patching top tokens changes output
patching top internal components changes output
top-token attribution alone fails on cases requiring hidden computation
```

---

# How this changes the final syllabus

The upgraded syllabus becomes:

```text
[0] Local GPU, testing, reproducibility, and experiment hygiene

[1] Original ARENA, preserved
    PyTorch foundations
    transformers from scratch
    TransformerLens
    induction heads
    IOI
    activation/path patching
    toy superposition
    SAEs
    NNsight
    function vectors
    RL/evals as originally included

[2] Modern language architectures
    Gemma
    DiffusionGemma
    Mamba / Mamba-2 / Mamba-3
    RWKV / RetNet optional
    multimodal and embedding models

[3] Sparse feature methods
    ReLU SAE
    TopK SAE
    Gated SAE
    JumpReLU SAE
    Gemma Scope
    transcoders
    crosscoders
    attribution graphs

[4] Activation-to-language
    logit lens
    tuned lens
    Patchscopes
    verbalizers
    mini Activation Oracles
    mini NLAs
    mini PCDs

[5] JEPAs
    I-JEPA
    V-JEPA
    V-JEPA 2
    V-JEPA 2.1
    VL-JEPA
    Audio/A-JEPA
    MC-JEPA
    3D/time-series/action-conditioned JEPA zoo

[6] World-model interpretability
    Othello-GPT
    maze world models
    Sudoku/constraint world models
    Dreamer-lite
    TD-MPC2-lite
    IRIS-lite
    V-JEPA planning
    microscope AI

[7] Shapley and attribution baselines
    exact Shapley
    KernelSHAP
    PartitionSHAP
    TreeSHAP
    DeepSHAP
    SAGE
    TokenSHAP / TokenShapley
    Shapley interactions
    SHAP vs patching vs circuits

[8] Alignment and safety interpretability
    refusal directions
    CoT faithfulness
    benign emergent-misalignment proxies
    white-box monitors
    world-model safety
    robotics/VLA safety

[9] Capstones
    one rigorous paper-style project
```

---

# What should be runnable on your 24GB RTX 5090 laptop?

## Required local targets

```text
I-JEPA ViT-B/H feature extraction
tiny I-JEPA training on CIFAR/STL subset
V-JEPA or V-JEPA 2 ViT-L feature extraction
V-JEPA 2 classification/retrieval smoke tests
V-JEPA 2.1 feature extraction if weights/tooling are stable
Othello-GPT full interpretability
maze transformer training
Sudoku tiny transformer training
Moving-MNIST / bouncing-ball JEPA
small Dreamer-lite / IRIS-lite model organisms
SHAP exact / KernelSHAP / TokenSHAP on small models
SAEs on small activations
```

## Optional / stretch

```text
V-JEPA 2 ViT-g
V-JEPA 2.1 large variants
VL-JEPA 1.6B if released weights are easy to load locally
JEPA-VLA / V-JEPA2-AC robot planning demos
larger video batches
DreamerV3 / TD-MPC2 real benchmarks
```

The course should print a mandatory footer in every notebook:

```text
peak VRAM
runtime
model checkpoint
dataset subset
seed
exact command to reproduce
pass/fail verification tests
```

---

# Verification rules for JEPA/world-model notebooks

Every notebook should have a “proof of implementation” block.

## For architecture correctness

```text
shape tests
masking tests
EMA target encoder update tests
loss decreases on toy data
collapse diagnostics pass
official checkpoint loads
feature extraction matches expected dimensions
```

## For world-model claims

```text
latent probe predicts true state
latent rollout predicts future state above baseline
action-conditioned latent rollout responds correctly to action changes
counterfactual patch changes prediction as expected
random-direction patch fails
OOD split passes
```

## For interpretability claims

```text
held-out examples
contrastive examples
causal patching
ablation
steering/intervention
random controls
simple baselines
OOD generalization
```

This matches the broader mech-interp direction in the uploaded review: interpretability methods need better validation, competitive baselines, model organisms, and benchmarks, rather than plausible stories alone.

---

# Best capstones after adding JEPA/world models/SHAP

## Capstone 1: “Do V-JEPA latents encode physical state more causally than VideoMAE latents?”

Compare:

```text
V-JEPA 2
V-JEPA 2.1
VideoMAE
DINO-style image features
CLIP/SigLIP video pooling
```

Tasks:

```text
bouncing balls
occlusion
collision
robot pushing
Something-Something subset
```

Evidence:

```text
state probes
causal patching
latent rollout
SHAP interactions
OOD robustness
```

## Capstone 2: “World-model circuits in Othello, mazes, and Sudoku”

Claim:

```text
Transformers trained on structured sequential domains learn linearly accessible world models whose geometry reflects task constraints.
```

Evidence:

```text
linear probes
causal interventions
SAEs
feature circuits
cross-domain comparison
```

## Capstone 3: “SHAP vs mechanistic patching on world models”

Claim:

```text
Input-level Shapley attribution is useful for surface credit assignment but systematically misses latent-state causal variables in world-model tasks.
```

Evidence:

```text
exact ground-truth toy tasks
Othello-GPT
maze transformer
V-JEPA toy physics
Shapley interactions
causal patching
```

## Capstone 4: “Object permanence features in latent-prediction video models”

Claim:

```text
Latent-prediction video models maintain object identity through occlusion more robustly than pixel-reconstruction models.
```

Evidence:

```text
occlusion datasets
patch-level probes
temporal feature consistency
counterfactual patching
retrieval after occlusion
```

## Capstone 5: “Action-conditioned JEPA world models expose controllable latent variables”

Compare:

```text
V-JEPA 2
V-JEPA 2-AC
ACT-JEPA
TD-MPC2-lite
Dreamer-lite
```

Evidence:

```text
action patching
goal-distance prediction
latent planning
controllable-vs-uncontrollable feature separation
```

---

# Anything else I’d add?

Yes: four more modules are worth adding.

## 1. Multimodal mechanistic interpretability

The field needs this badly. Add:

```text
CLIP / SigLIP representation geometry
DINO / DINOv2 / DINOv3 dense features
SAM/SAM2-style segmentation features
VideoPrism / VideoMAE / V-JEPA comparison
vision-language attention and cross-modal binding
```

This complements VL-JEPA and V-JEPA 2.1.

## 2. Representation geometry and causal abstraction

Add:

```text
linear probes
nonlinear probes
distributed alignment search
causal abstraction
concept erasure
subspace patching illusions
feature geometry
manifold probes
```

This is the bridge between classic mechinterp and world-model interpretability.

## 3. Training dynamics and developmental interpretability

The uploaded open-problems material explicitly highlights the need to understand how mechanisms evolve during training and how small-scale circuits connect to larger capabilities.  So add:

```text
train tiny models from scratch
save checkpoints every N steps
watch probes/circuits/features emerge
compare AR vs JEPA vs diffusion vs Mamba training dynamics
```

This is unusually valuable for a PhD because it teaches you to see mechanisms as learned developmental objects, not static artifacts.

## 4. Research-process notebooks

Make the course itself teach research taste. The uploaded material includes Neel Nanda’s Explore → Understand → Distill framework and Jacob Steinhardt’s information-rate/de-risking framing.  Turn that into a practical template:

```text
research_log.ipynb
hypothesis_tracker.md
experiment_registry.csv
baselines_checklist.md
failure_modes.md
paper_skeleton.md
```

Every capstone should end with:

```text
1–3 claims
evidence table
alternative hypotheses
failed experiments
baselines
limitations
next experiments
```

That would make the fork not just a course, but a research apprenticeship system.

---

# Bottom line

Yes: preserve ARENA fully. Then add JEPA/world-model/SHAP as **new, verified, local-first chapters**.

The resulting course becomes much stronger:

```text
Original ARENA = transformer/mechinterp foundations
Modern architecture track = Gemma, Mamba, diffusion, multimodal
Sparse feature track = SAEs, transcoders, crosscoders
Activation-to-language track = verbalizers, AOs, NLAs, PCDs
JEPA track = latent predictive representation learning
World-model track = interpretable state, dynamics, planning
SHAP track = rigorous attribution baselines and failure cases
Alignment track = white-box safety, CoT, refusal, monitors
Capstones = publishable mini-projects
```

The unifying research question becomes:

**Can we build and interpret models that do not merely predict tokens, but learn structured latent models of the world — and can we verify those structures causally, locally, and reproducibly on a 24GB GPU?**

That is an excellent PhD free-time curriculum.

[1]: https://github.com/facebookresearch/ijepa?utm_source=chatgpt.com "facebookresearch/ijepa: Official codebase for I-JEPA ..."
[2]: https://github.com/facebookresearch/jepa?utm_source=chatgpt.com "facebookresearch/jepa: PyTorch code and models for V- ..."
[3]: https://ai.meta.com/research/vjepa/?utm_source=chatgpt.com "Introducing V-JEPA 2"
[4]: https://github.com/facebookresearch/vjepa2?utm_source=chatgpt.com "PyTorch code and models for VJEPA2 self-supervised ..."
[5]: https://huggingface.co/docs/transformers/en/model_doc/vjepa2?utm_source=chatgpt.com "V-JEPA 2"
[6]: https://arxiv.org/abs/2603.14482?utm_source=chatgpt.com "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning"
[7]: https://arxiv.org/abs/2512.10942?utm_source=chatgpt.com "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
[8]: https://arxiv.org/abs/2311.15830?utm_source=chatgpt.com "A-JEPA: Joint-Embedding Predictive Architecture Can Listen"
[9]: https://arxiv.org/abs/2307.12698?utm_source=chatgpt.com "MC-JEPA: A Joint-Embedding Predictive Architecture for ..."
[10]: https://arxiv.org/html/2606.25225v1?utm_source=chatgpt.com "MJEPA: A Simple and Scalable Joint-Embedding ..."
[11]: https://arxiv.org/abs/2602.11832?utm_source=chatgpt.com "JEPA-VLA: Video Predictive Embedding is Needed for VLA Models"
[12]: https://arxiv.org/html/2501.14622v4?utm_source=chatgpt.com "ACT-JEPA: Novel Joint-Embedding Predictive Architecture ..."
[13]: https://arxiv.org/html/2412.11867v1?utm_source=chatgpt.com "Transformers Use Causal World Models in Maze-Solving ..."
[14]: https://arxiv.org/abs/2605.18847?utm_source=chatgpt.com "Transformers Linearly Represent Highly Structured World Models"
[15]: https://arxiv.org/abs/2301.04104?utm_source=chatgpt.com "Mastering Diverse Domains through World Models"
[16]: https://arxiv.org/abs/2310.16828?utm_source=chatgpt.com "TD-MPC2: Scalable, Robust World Models for Continuous Control"
[17]: https://arxiv.org/abs/2209.00588?utm_source=chatgpt.com "Transformers are Sample-Efficient World Models"
[18]: https://arxiv.org/abs/2004.00668?utm_source=chatgpt.com "Understanding Global Feature Contributions With Additive ..."
[19]: https://arxiv.org/html/2407.10114v1?utm_source=chatgpt.com "Interpreting Large Language Models with Monte Carlo ..."
[20]: https://arxiv.org/html/2507.05261v2?utm_source=chatgpt.com "Token Level Context Attribution with Shapley Value"
[21]: https://arxiv.org/abs/2410.01649?utm_source=chatgpt.com "shapiq: Shapley Interactions for Machine Learning"
[22]: https://arxiv.org/abs/2405.10852?utm_source=chatgpt.com "Weighted Least-Square Optimization for Shapley Interactions"



---------------------------------------------------------------

Yes — with one amendment: **it should not promise “every SOTA paper, fully implemented forever” as a static syllabus.** That would go stale immediately. The correct version is:

**Preserve ARENA fully, then add a living frontier layer whose rule is: every important method gets either a faithful implementation, a small-scale reproduction, a pretrained-weight verification, or a deliberately labeled “read-only / too large / no weights” treatment.**

That would genuinely let you read the important papers, implement the methods that are locally feasible, verify implementations, compare across model families, and then do targeted exploratory mech-interp projects with publishable-style rigor.

The original ARENA base should remain untouched. The current ARENA_3.0 repo says it hosts the exercises and Streamlit pages and is now the continuing latest repo rather than a repo that will be replaced by future numbered versions. ([GitHub][1]) Your existing source pack already treats ARENA + TransformerLens + NNsight as the practical foundation: transformer from scratch, intro mech interp and induction heads, IOI, toy superposition, SAEs, and function-vector / steering exercises.  That is the correct core to preserve.

---

# Final corrected answer

## Does it explore all methods?

**Almost, after the amendments below.** The previous plan covered the biggest missing blocks: Gemma, Mamba, diffusion LMs, SAEs, transcoders, crosscoders, activation-to-language methods, JEPAs, world models, Shapley/SHAP baselines, and alignment interpretability. But to make it truly comprehensive, I would explicitly add these missing or underemphasized areas:

1. **Efficient / nonstandard sequence models beyond Mamba**

   * RWKV.
   * RetNet / retention.
   * RecurrentGemma / Griffin.
   * Hyena-style implicit-convolution models.
   * xLSTM.
   * Linear attention variants.
   * Hybrid attention–SSM–RNN models.

   This matters because open-problems work explicitly says mech interp has focused heavily on CNNs, BERT, and GPT-style models, while future systems may use diffusion models, ViTs, RWKV, SSMs, and multimodal architectures; it calls cross-architecture transfer an open problem.  RecurrentGemma is especially useful because it has open weights and uses Google DeepMind’s Griffin architecture, mixing gated linear recurrences with local sliding-window attention. ([GitHub][2])

2. **Training-data attribution and influence methods**

   * Influence functions.
   * TracIn-style methods.
   * Datamodels.
   * Dataset cartography.
   * Training-run simulations.
   * SAGE/global feature importance.

   These belong because interpretability of model behavior is not only “which activation caused this output?” but also “which data caused this mechanism to form?” The uploaded primer explicitly includes training-data attribution as a branch of input attribution.

3. **Model editing, unlearning, and steerable-by-design architectures**

   * ROME / MEMIT as pedagogical examples.
   * Locate-then-edit failure modes.
   * Concept erasure.
   * Representation engineering.
   * Backpack language models.
   * Refusal direction work.
   * White-box unlearning evals.

   This should be taught skeptically: the primer notes that model editing still faces catastrophic forgetting and downstream-performance-loss risks, and that effective localization does not necessarily imply effective editing.

4. **Intrinsic interpretability**

   * Concept bottleneck models.
   * Sparse / modular training.
   * Interchange intervention training.
   * Architectures designed to be interpretable by construction.
   * KANs and other interpretable-by-design models as comparison baselines.

   This is worth adding because the open-problems review explicitly separates post-hoc mech interp from interpretable-by-design approaches and lists concept bottlenecks, Backpack LMs, KANs, and sparse trees as active interpretability-by-design work.

5. **Automated interpretability agents**

   * MAIA.
   * OpenMAIA-style local variants.
   * Activation Oracles.
   * Natural Language Autoencoders.
   * Predictive Concept Decoders.
   * Feature verbalizers.
   * LLM-generated hypotheses with held-out validation.

   This is now necessary, not optional. Anthropic’s Activation Oracles treat activations as an additional input modality and train models to answer natural-language questions about activations. ([Alignment Science Blog][3]) A 2026 follow-up tries to improve AOs with on-policy rollouts, better conversational data, more layers, and an improved injection formula. ([arXiv][4]) NLAs are a 2026 method for generating natural-language explanations of LLM activations. ([Transformer Circuits][5]) PCDs train assistants that translate internal states into sparse human-readable concepts and use them to answer behavioral questions. ([Transluce][6])

6. **Interaction methods, not just additive attribution**

   * SHAP interactions.
   * SHAP-IQ.
   * shapiq.
   * SPEX / ProxySPEX.
   * Shapley-Taylor.
   * TokenSHAP / TokenShapley.
   * Interaction-vs-circuit comparisons.

   This is important because many LLM and world-model behaviors depend on interactions, not marginal token importance. `shapiq` unifies algorithms for Shapley values and any-order Shapley interactions. ([arXiv][7]) TokenSHAP adapts Monte Carlo Shapley estimation to LLM token attribution. ([ACL Anthology][8]) TokenShapley targets token-level context attribution and reports 11–23% accuracy improvements over baselines on four benchmarks. ([arXiv][9]) SPEX/ProxySPEX should be added because they target scalable interaction discovery in LLMs rather than only marginal attributions. ([BAIR][10])

7. **Interpretability UIs and dashboards**

   * Neuronpedia.
   * sae-vis.
   * circuit-tracer frontend.
   * NNsight dashboards.
   * model-internals notebooks.
   * “white-box eval monitor” interfaces.

   This matters because open-problems work explicitly identifies human-computer interaction with model internals as an important frontier.

8. **Benchmarks and model organisms**

   * Othello-GPT.
   * Modular addition.
   * IOI.
   * Greater-than.
   * Tracr/RASP compiled transformers.
   * Maze transformers.
   * Sudoku transformers.
   * Bouncing balls / Moving-MNIST.
   * Tiny JEPA worlds.
   * Toy RL worlds.

   This is non-negotiable: the field needs testbeds with known or at least inspectable mechanisms. The open-problems review emphasizes model organisms and benchmarks as ways to validate hypotheses and avoid interpretability illusions.

So the amended answer is: **yes, it becomes comprehensive if we explicitly include these categories and maintain a living paper-ingestion protocol.**

---

# Does it let you read all the papers?

It should let you read **all central papers and all currently important frontier papers**, but not every arXiv-adjacent paper in full depth. The right structure is a three-tier reading system:

| Tier                    | What you do                                                  | Examples                                                                                                                       |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **Deep read**           | Re-derive, implement, reproduce one core result              | Transformer Circuits, Toy Superposition, IOI, SAEs, activation patching, Mamba, I-JEPA, V-JEPA 2, Gemma Scope, circuit tracing |
| **Implementation skim** | Understand enough to code the method and test it             | Gated/TopK/JumpReLU SAEs, EAP, AtP*, Patchscopes, TokenSHAP, shapiq, Mamba-2/3, V-JEPA 2.1                                     |
| **Survey / watchlist**  | Know claims, assumptions, limits, whether weights/code exist | newer JEPA variants, world-action models, large robot models, proprietary frontier interpretability, niche Shapley variants    |

This matches Neel Nanda’s research-process advice in your source pack: research should move through **Explore → Understand → Distill**, with truth-seeking, prioritization, and fast feedback loops.   The fork should teach you not just to read papers, but to ask:

```text
What is the paper’s core claim?
What experiment would falsify it?
What is the smallest local reproduction?
What is the strongest baseline?
What would count as successful verification?
What breaks when I change model family?
```

That is much more valuable than passively reading 300 papers.

---

# Does it let you implement and verify SOTA methods?

**Yes, if every module has a verification contract.** I would make this the hard rule:

```text
No method enters the course unless it has:
  1. paper summary,
  2. minimal implementation,
  3. local smoke test,
  4. official-code or official-checkpoint comparison when possible,
  5. baselines,
  6. failure cases,
  7. reproducibility metadata,
  8. VRAM report.
```

For methods too large to train locally, the course still implements the **core mechanism** at toy scale and verifies **pretrained-weight runnability** where weights exist.

Examples:

| Method               | Local implementation                                             | Verification                                                     |
| -------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| Gemma decoder block  | Implement RMSNorm, RoPE, GQA, SwiGLU, KV cache                   | HF logit parity, greedy generation parity                        |
| Gemma Scope          | Load released SAEs/transcoders                                   | Feature activation, ablation, steering, loss recovered           |
| Mamba                | Implement selective scan and recurrent cache                     | Parallel/recurrent equivalence, official checkpoint parity       |
| Mamba-3              | Implement tiny state-tracking variant                            | Synthetic state-tracking accuracy, hidden-state intervention     |
| DiffusionGemma       | Implement toy discrete diffusion LM, load real model if feasible | Denoising loss, entropy over steps, local VRAM test              |
| I-JEPA               | Train tiny image JEPA                                            | Collapse tests, linear probe > random                            |
| V-JEPA 2             | Load pretrained video encoder                                    | Shape checks, retrieval/probe tests, temporal shuffling baseline |
| VL-JEPA              | Implement embedding-prediction VLM toy                           | Caption-retrieval accuracy, selective-decoding test              |
| SAEs                 | ReLU, Gated, TopK, JumpReLU                                      | L0, loss recovered, feature density, intervention tests          |
| Transcoders          | Tiny MLP replacement                                             | KL/reconstruction, feature-circuit faithfulness                  |
| Crosscoders          | Base-vs-IT or Transformer-vs-Mamba toy                           | Shared vs model-specific features predict behavior deltas        |
| Patchscopes / lenses | Implement logit lens, tuned lens, patchscope templates           | Held-out decoding accuracy, activation counterfactuals           |
| AOs / NLAs / PCDs    | Mini local versions                                              | Beat text-only/probe baselines, fail on random activations       |
| SHAP / Shapley       | Exact, Kernel, Partition, TokenSHAP, shapiq                      | Compare to ground truth and causal patching                      |

For current frontier anchors, the plan is aligned with real releases: Gemma Scope 2 includes SAEs and transcoders trained on every layer of Gemma 3. ([Google DeepMind][11]) Mamba-3 is a 2026 SSM-style sequence model focused on inference efficiency and state-tracking improvements. ([arXiv][12]) DiffusionGemma is a June 2026 discrete diffusion language model on a Gemma 4 MoE backbone, with Google’s developer guide reporting up to 700+ tokens/s on a GeForce RTX 5090-class GPU; for your laptop 5090, I would treat this as “must verify locally,” not assume identical throughput. ([Hugging Face][13])

---

# Does it cover the various model families?

After amendment, yes. The final matrix should be:

| Family                 | Required local target                                          | Why included                             |
| ---------------------- | -------------------------------------------------------------- | ---------------------------------------- |
| GPT-style transformers | GPT-2, Pythia, TinyStories                                     | ARENA foundation, TransformerLens        |
| Modern transformers    | Gemma 3 / small Llama-like models                              | Current open-weight decoder architecture |
| MoE                    | tiny MoE, Switch-style toy, DiffusionGemma backbone inspection | Routing, expert features                 |
| SSMs                   | Mamba, Mamba-2, Mamba-3-style toy                              | Non-attention sequence modeling          |
| Recurrent hybrids      | RWKV, RecurrentGemma/Griffin, xLSTM                            | Recurrent/linear-memory comparison       |
| Diffusion LMs          | toy diffusion LM, DiffusionGemma                               | Non-autoregressive text generation       |
| Vision                 | ViT, DINO-style, CLIP/SigLIP, SAM-style features               | Multimodal feature geometry              |
| Video/JEPAs            | I-JEPA, V-JEPA, V-JEPA 2, V-JEPA 2.1                           | Latent prediction and world modeling     |
| Vision-language        | VL-JEPA, CLIP/SigLIP, small VLMs                               | Cross-modal binding                      |
| Audio                  | Audio-JEPA / A-JEPA toy                                        | Time-frequency latent prediction         |
| RL/world models        | Dreamer-lite, TD-MPC2-lite, IRIS-lite                          | Latent dynamics and planning             |
| Symbolic worlds        | Othello, mazes, Sudoku                                         | Ground-truth-ish world models            |

The JEPA track is justified: V-JEPA 2 is explicitly framed by Meta as a world-model step for understanding, prediction, and planning. ([Meta AI][14]) V-JEPA 2-AC is described in the official repo as a latent action-conditioned world model post-trained from V-JEPA 2 for robot manipulation without environment-specific data collection or task-specific calibration. ([GitHub][15]) V-JEPA 2.1 is a 2026 dense-feature image/video model with spatially structured and temporally consistent representations. ([arXiv][16]) VL-JEPA predicts continuous text embeddings rather than autoregressive tokens and reports stronger controlled-comparison performance with fewer trainable parameters. ([arXiv][17]) MC-JEPA jointly learns motion and content features, making it an excellent disentanglement/world-state lab. ([arXiv][18])

---

# Does it support real exploratory mech-interp projects?

Yes — but only if the fork includes a **research track**, not just exercises.

I would add this directory:

```text
research_projects/
  00_project_template/
    question.md
    literature_map.md
    experiment_registry.csv
    method_cards/
    baselines.md
    falsification_tests.md
    results.ipynb
    paper_skeleton.md
  01_gemma_scope_quantization/
  02_mamba_state_tracking/
  03_vjepa_object_permanence/
  04_shap_vs_patching_world_models/
  05_diffusiongemma_commitment_time/
  06_crosscoder_base_vs_instruction/
  07_white_box_refusal_monitor/
```

Every project should produce:

```text
1. One precise claim.
2. One strong baseline.
3. One causal intervention.
4. One OOD test.
5. One negative-control experiment.
6. One failure-mode section.
7. One reproducible script.
8. One paper-style figure.
9. One short writeup.
```

That directly addresses the field’s current weakness: many explanations sound plausible but are under-validated. The open-problems review emphasizes that explanations need validation, benchmarks, model organisms, competitive baselines, and attention to average/worst-case behavior, not just best-case demos.  It also explicitly says mech interp should aim at concrete engineering/scientific goals: monitoring, control, prediction, training/inference improvement, and microscope AI.

---

# The final amended syllabus

Here is the corrected “yes, this is enough” version:

```text
0. Local GPU + reproducibility infrastructure
   pytest, deterministic seeds, VRAM budget, model cache, artifact logging

1. Original ARENA, preserved
   PyTorch, transformers from scratch, TransformerLens, induction heads,
   IOI, path patching, toy superposition, SAEs, NNsight, RL/evals

2. Modern model implementation
   Gemma, Llama-like models, MoE, Mamba, Mamba-2, Mamba-3,
   RWKV, RetNet, Griffin/RecurrentGemma, xLSTM, Hyena, diffusion LMs

3. Sparse feature methods
   ReLU SAE, Gated SAE, TopK SAE, BatchTopK, JumpReLU,
   Gemma Scope, transcoders, skip-transcoders, crosscoders,
   feature geometry, feature splitting/absorption

4. Circuit methods
   activation patching, path patching, attribution patching,
   EAP, EAP-IG, AtP*, ACDC, causal scrubbing, attribution graphs,
   circuit tracing, sparse feature circuits

5. Activation-to-language
   logit lens, tuned lens, attention lens, Patchscopes,
   feature verbalizers, MAIA/OpenMAIA, Activation Oracles,
   Natural Language Autoencoders, Predictive Concept Decoders

6. Concept and representation methods
   probes, causal probes, CCS, DAS, concept erasure,
   representation engineering, steering vectors, refusal directions,
   causal abstraction, subspace-patching illusions

7. JEPA and predictive representation learning
   I-JEPA, V-JEPA, V-JEPA 2, V-JEPA 2.1, VL-JEPA,
   Audio-JEPA/A-JEPA, MC-JEPA, 3D-JEPA, TC-JEPA,
   JEPA-VLA / action-conditioned JEPA as stretch

8. World-model interpretability
   Othello-GPT, maze transformers, Sudoku transformers,
   Dreamer-lite, TD-MPC2-lite, IRIS-lite, V-JEPA 2-AC,
   object permanence, latent planning, controllable variables

9. SHAPley and attribution baselines
   exact Shapley, KernelSHAP, PartitionSHAP, TreeSHAP,
   DeepSHAP, GradientSHAP, SAGE, TokenSHAP, TokenShapley,
   SHAP-IQ, shapiq, SPEX/ProxySPEX, interaction-vs-circuit tests

10. Alignment and safety interpretability
   refusal, hallucination, CoT faithfulness, white-box evals,
   benign emergent-misalignment proxies, unlearning, model editing,
   monitors, red-teaming with internals

11. Training dynamics and developmental interpretability
   induction-head emergence, grokking, checkpoint archaeology,
   finetuning diffs, data influence, capability emergence

12. Capstone research sprint
   one claim, one baseline suite, one causal validation suite,
   one paper-style writeup
```

That is the version I would consider **realistically comprehensive**.

---

# The key amendment: add a method registry

To prevent the course from becoming obsolete, add:

```text
docs/method_registry.csv
```

with columns:

```text
method_name
paper
year
category
model_family
has_code
has_weights
local_24gb_status
implementation_status
verification_status
baseline_status
notes
```

Statuses:

```text
REQUIRED_IMPLEMENT
REQUIRED_LOAD_WEIGHTS
TOY_REPRO_ONLY
READ_ONLY_TOO_EXPENSIVE
WAIT_FOR_WEIGHTS
DEPRECATED_BY_NEWER_METHOD
```

This turns “all SOTA” into an auditable process.

---

# The local feasibility rule

On your 24GB RTX 5090 laptop GPU:

**Required:** implement concepts locally, run small models, load quantized/open weights where feasible.

**Not required:** train frontier-scale SAEs, full Activation Oracles, full NLAs, full V-JEPA 2.1, large robot world models, or 27B+ BF16 models.

Instead:

```text
Toy implementation proves understanding.
Small reproduction proves method works.
Official weights prove real-model runnability.
Verification suite proves correctness.
Capstone proves research usefulness.
```

That is the right standard.

---

# Final verdict

**Yes — after these amendments, this really does what you want.**

It preserves ARENA, extends it across current frontier architectures, covers the major mech-interp method families, includes JEPAs and world-model interpretability, adds SHAP/Shapley as rigorous attribution baselines, and gives you a route to real targeted research projects.

The only thing I would refuse to promise is “every SOTA paper implemented fully at frontier scale.” The stronger and more useful promise is:

**Every important method is read, categorized, implemented at the smallest faithful scale, verified against a baseline or official implementation where possible, and either applied to a targeted mech-interp project or explicitly marked as read-only / too expensive / awaiting weights.**

That would be a genuinely epic ARENA fork: not just a course, but a local-first mechanistic-interpretability research lab.

[1]: https://github.com/callummcdougall/ARENA_3.0?utm_source=chatgpt.com "callummcdougall/ARENA_3.0"
[2]: https://github.com/google-deepmind/recurrentgemma?utm_source=chatgpt.com "google-deepmind/recurrentgemma"
[3]: https://alignment.anthropic.com/2025/activation-oracles/?utm_source=chatgpt.com "Activation Oracles: Training and Evaluating LLMs as General ..."
[4]: https://arxiv.org/html/2606.02609v2?utm_source=chatgpt.com "Building Better Activation Oracles"
[5]: https://transformer-circuits.pub/2026/nla/?utm_source=chatgpt.com "Natural Language Autoencoders Produce Unsupervised ..."
[6]: https://transluce.org/pcd?utm_source=chatgpt.com "Predictive Concept Decoders"
[7]: https://arxiv.org/abs/2410.01649?utm_source=chatgpt.com "shapiq: Shapley Interactions for Machine Learning"
[8]: https://aclanthology.org/2024.nlp4science-1.1.pdf?utm_source=chatgpt.com "TokenSHAP: Interpreting Large Language Models with ..."
[9]: https://arxiv.org/abs/2507.05261?utm_source=chatgpt.com "TokenShapley: Token Level Context Attribution with Shapley Value"
[10]: https://bair.berkeley.edu/blog/2026/03/13/spex/?utm_source=chatgpt.com "Identifying Interactions at Scale for LLMs"
[11]: https://deepmind.google/models/gemma/gemma-scope/?utm_source=chatgpt.com "Gemma Scope"
[12]: https://arxiv.org/abs/2603.15569?utm_source=chatgpt.com "Mamba-3: Improved Sequence Modeling using State Space Principles"
[13]: https://huggingface.co/google/diffusiongemma-26B-A4B-it?utm_source=chatgpt.com "google/diffusiongemma-26B-A4B-it"
[14]: https://ai.meta.com/research/vjepa/?utm_source=chatgpt.com "Introducing V-JEPA 2"
[15]: https://github.com/facebookresearch/vjepa2?utm_source=chatgpt.com "PyTorch code and models for VJEPA2 self-supervised ..."
[16]: https://arxiv.org/abs/2603.14482?utm_source=chatgpt.com "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning"
[17]: https://arxiv.org/abs/2512.10942?utm_source=chatgpt.com "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
[18]: https://arxiv.org/abs/2307.12698?utm_source=chatgpt.com "MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features"

-----------------------------------------


Yes — **IE/IG feature and circuit attribution should be mandatory**, and **Sparse Feature Circuits** should be one of the central replication notebooks.

I’d add this as a named block in the circuit chapter:

```text
[ch4] Circuit Methods and Feature Attribution
  [4.1] Direct, total, and indirect effects
  [4.2] Attribution patching from scratch
  [4.3] Integrated gradients and integrated effects
  [4.4] Edge attribution patching: EAP, EAP-IG, AtP*
  [4.5] Sparse Feature Circuits replication
  [4.6] Sparse feature editing: SHIFT-style interventions
  [4.7] Extending sparse feature circuits to VLMs
```

This fits perfectly with the course’s existing spine. Your source pack already treats attribution, causal interventions, activation patching, EAP, integrated gradients, sparse dictionary learning, SAEs, transcoders, and crosscoders as core mech-interp methods rather than side material.  It also explicitly highlights **Sparse Feature Circuits** as an important SAE-circuit paper using attribution patching plus an integrated-gradients variant, and notes its downstream editing application for reducing spurious correlations.

The specific paper should be replicated because it does exactly what we want the course to teach: it replaces coarse circuit nodes like heads/neurons with **SAE features and SAE error terms**, uses efficient causal-attribution approximations to find feature-level graphs, and then edits behavior by ablating human-judged spurious features. The paper introduces sparse feature circuits as causally implicated subnetworks of human-interpretable features, uses SAEs for attention outputs, MLP outputs, and residual streams in Pythia-70M, and releases code/data/autoencoders for replication. ([arXiv][1])

---

## Final required notebook: `[4.5] Sparse Feature Circuits`

Core question:

```text
Can we discover a causal graph over human-interpretable SAE features,
rather than over polysemantic heads, neurons, or layers?
```

### Concepts taught

```text
direct effect
indirect effect / IE
integrated gradients / IG
integrated effects
attribution patching
edge attribution patching
EAP-IG
SAE feature nodes
SAE error nodes
feature-level causal graphs
faithfulness
completeness
minimality
feature editing
```

The notebook should make the notation explicit:

```text
Exact indirect effect:
IE(node) = metric(clean with node patched from corrupt) - metric(clean)

First-order attribution patching:
IE(node) ≈ grad_metric(node_clean) · (node_corrupt - node_clean)

Integrated-gradient version:
IE_IG(node) ≈ (node_corrupt - node_clean)
              · average_t grad_metric(node_t)

where node_t interpolates between clean and corrupt activations.
```

Then extend this from **nodes** to **edges**:

```text
edge u -> v:
  measure how changing sender u affects receiver v’s contribution to metric
```

EAP-IG should be presented as the “more faithful but more expensive” bridge between brute-force edge patching and cheap first-order attribution. The public EAP-IG repo describes input- and activation-interpolation variants of edge attribution patching with integrated gradients, where the activation variant is more principled but more expensive. ([GitHub][2])

---

## Implementation sequence

The notebook should be written ARENA-style:

### Part 1 — Warmup on exact patching

Use GPT-2 small or Pythia-70M on a tiny task.

```text
Task:
  subject–verb agreement
  IOI
  greater-than
  factual recall toy
```

Implement:

```python
run_clean()
run_corrupt()
compute_metric()
patch_node_exact()
patch_edge_exact()
```

Verify:

```text
exact patching changes metric in expected direction
random node patching does less
clean/corrupt pairs differ only in target variable
```

---

### Part 2 — Attribution patching

Implement:

```python
attrib_patch_score = grad_metric_at_clean * (act_corrupt - act_clean)
```

Compare to exact patching:

```text
correlation with exact patching
top-k overlap
runtime improvement
false positives
false negatives
```

This should explicitly teach the limitation: attribution patching is a **first-order approximation**, so it can fail when the path is nonlinear, saturated, or affected by backup/self-repair.

---

### Part 3 — Integrated effects / integrated gradients

Implement:

```python
for alpha in torch.linspace(0, 1, steps):
    act_alpha = act_clean + alpha * (act_corrupt - act_clean)
    grad_alpha = grad(metric, act_alpha)
average_grad = mean(grad_alpha)
ie_ig = (act_corrupt - act_clean) @ average_grad
```

Then compare:

```text
exact patching
plain attribution patching
IG / IE attribution
EAP
EAP-IG
```

Verification:

```text
EAP-IG correlates better with exact patching than plain EAP on at least one task
EAP is faster
exact patching remains ground truth for small cases
```

---

### Part 4 — Sparse Feature Circuits replication

Use the paper’s local-replication target:

```text
Model:
  Pythia-70M

Task:
  subject–verb agreement

Nodes:
  SAE features
  SAE error terms

Graph:
  sparse feature circuit
```

The official repo includes replication scripts, subject–verb agreement data, Pythia-70M dictionaries, circuit discovery, faithfulness evaluation, and plotting utilities. ([GitHub][3])

Implement:

```text
load Pythia-70M
load or train SAEs
encode activations into sparse features
include SAE reconstruction error as graph nodes
compute node attribution
compute edge attribution
threshold nodes and edges
plot sparse causal graph
```

Verify:

```text
reproduce at least one subject–verb agreement circuit
feature circuit is more interpretable than neuron circuit
faithfulness/completeness curves match the qualitative result
feature annotations agree with top activating examples
random-feature graph performs worse
```

---

### Part 5 — SHIFT-style editing

Add the editing part, not just discovery.

Core question:

```text
Can we use a sparse feature circuit to remove a spurious feature
without destroying the intended behavior?
```

Replicate a small version of SHIFT:

```text
Dataset:
  Bias-in-Bios-style classifier, or a tiny synthetic classifier first

Target:
  preserve profession signal
  remove gender/spurious signal

Method:
  find sparse features used by classifier
  identify task-irrelevant / spurious features
  ablate or suppress them
```

Verify:

```text
spurious-feature reliance decreases
target-task accuracy mostly preserved
OOD generalization improves
random feature ablation does worse
black-box baseline included
```

The Sparse Feature Circuits paper introduces SHIFT for removing sensitivity to unintended signals and demonstrates it in a worst-case profession/gender classifier setting. ([arXiv][1])

---

## Add a follow-up notebook: `[4.6] Feature-circuit methods compared`

This should compare:

```text
exact activation patching
exact edge/path patching
attribution patching
EAP
EAP-IG
AtP*
ACDC
Sparse Feature Circuits
transcoder attribution graphs
```

Metrics:

```text
faithfulness
completeness
minimality
sparsity
runtime
interpretability
editability
robustness to random seeds
robustness to prompt distribution shift
```

The course should make the methodological tradeoff clear:

```text
Exact patching:
  most trusted, too expensive at scale

Attribution patching:
  very fast, first-order approximation

EAP:
  fast edge-level localization

EAP-IG / IE:
  slower, usually more faithful where gradients vary along path

Sparse Feature Circuits:
  interpretable feature-level graph, depends on SAE quality

Transcoder attribution graphs:
  more mechanism-like for MLP computation, but requires replacement model
```

---

## Add this to VLMs too

Because VLMs are your main topic, I’d add a VLM version later:

```text
[ch5.12] Sparse Multimodal Feature Circuits
```

Core question:

```text
Can we build sparse causal graphs from image features to answer tokens?
```

Nodes:

```text
CLIP/SigLIP SAE features
vision-encoder patch features
projector SAE features
VLM image-token residual features
VLM text-token residual features
answer-token residual features
SAE error nodes
```

Edges:

```text
image patch feature -> projector feature
projector feature -> LLM visual-token feature
visual-token feature -> answer-token feature
answer-token feature -> output logit
```

Tasks:

```text
object identity
color
OCR text
spatial relation
object hallucination
typographic attacks
modality conflict
```

Verification:

```text
feature circuit localizes object/color/OCR evidence
patching object-region features changes answer
background-region features matter less
text-only baseline fails
random feature graph fails
feature ablation reduces hallucination or typographic vulnerability
```

That would give you a direct research bridge from **Sparse Feature Circuits in LMs** to **sparse feature circuits in VLMs**.

---

## Final amendment to the syllabus

The corrected circuit section is:

```text
[4] Circuit Methods and Feature Attribution
  [4.1] Activation patching and causal intervention basics
  [4.2] Direct, total, and indirect effects
  [4.3] Attribution patching from scratch
  [4.4] Integrated gradients, integrated effects, and EAP-IG
  [4.5] ACDC and automated circuit discovery
  [4.6] Sparse Feature Circuits replication
  [4.7] SHIFT-style sparse feature editing
  [4.8] Transcoder and attribution-graph circuits
  [4.9] Circuit faithfulness, completeness, and minimality
  [4.10] Failure modes: saturation, backup, self-repair, SAE error, OOD patching
```

So yes: **IE/IG attribution is explicitly in**, and **Sparse Feature Circuits becomes a required replication target**, with from-scratch implementation, official-code comparison, faithfulness/completeness verification, and an extension path to VLM feature circuits.

[1]: https://arxiv.org/html/2403.19647v1 "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models"
[2]: https://github.com/hannamw/eap-ig?utm_source=chatgpt.com "hannamw/EAP-IG"
[3]: https://github.com/saprmarks/feature-circuits "GitHub - saprmarks/feature-circuits · GitHub"


Yes — final amendment: **the ARENA fork should keep the original style, and VLM interpretability should become one of the flagship chapters, not an appendix.**

The final version should feel like ARENA, not like a survey pasted into notebooks. ARENA’s style is part of why it works: clear learning objectives, friendly but technically serious exposition, diagrams, setup cells, implementation exercises, solution dropdowns, expected outputs, tests, bonus sections, and open-ended anomaly hunting. The existing materials explicitly describe ARENA as a practical TransformerLens-based introduction “full of snippets to copy” with exercises and solutions, covering GPT-2 from scratch, induction heads, IOI, direct logit attribution, activation patching, and related techniques.  The IOI sections also have the exact style we should preserve: learning objectives, implementation from scratch, replication, faithfulness/completeness/minimality, and bonus exploration.

So the updated fork should have a **style contract** and a **full VLM interpretability chapter**.

---

# Final style contract for the ARENA extension

Every new notebook should follow this template:

```text
# N.M Title

## Core question
A concrete, exciting question:
"Where does the VLM store object identity?"
"Does the model read the image, or just language priors?"
"Can we causally edit a visual concept?"

## Learning objectives
3–7 bullets, written in ARENA style.

## Why this matters
Short, intuitive motivation.

## The model
Small local model first.
Then one real open-weight model.

## Build it from scratch
Implement the relevant mechanism.

## Load real weights
Run the real model locally.

## Interpret it
Hooks, caches, probes, patching, attribution, SAEs, steering.

## Verify it
Unit tests, parity tests, causal tests, baselines, VRAM report.

## Solution dropdowns
Readable, explanatory solutions.

## Bonus / exploring anomalies
Open-ended research prompts.

## Capstone ideas
Possible paper-style projects.
```

The voice should stay: **concrete, curious, mildly playful, implementation-first, not overformal.** The research-process material in your source pack says exploration should prioritize information gain, fast feedback loops, visualizations, and flexible tooling, while the understanding phase should be skeptical and hypothesis-driven.  The writing material also emphasizes “inform, not persuade,” precision over obfuscation, rigorous evidence, limitations, and good figures.  That is exactly the house style.

Each notebook should have a final cell like:

```python
verify_notebook(
    model_name=model_name,
    required_vram_gb=24,
    tests=[
        "shape_tests",
        "official_model_parity",
        "activation_cache_sanity",
        "causal_intervention_effect",
        "random_control_fails",
        "ood_split_passes",
    ],
)
```

And every serious claim should be backed by:

```text
clean/corrupt contrast
causal patch or ablation
random baseline
text-only baseline where relevant
image-only baseline where relevant
OOD examples
failure cases
```

That matters especially for VLMs, because attention maps and saliency maps are seductive but can easily become interpretability theater.

---

# New flagship chapter: Vision-Language Model Interpretability

I would make this a major chapter:

```text
[ch5] Vision-Language and Generative-Vision Interpretability
  [5.0] Why VLM interpretability is different
  [5.1] CLIP and SigLIP from scratch
  [5.2] CLIP-like model interpretability
  [5.3] Building a mini multimodal LLM
  [5.4] PaliGemma, LLaVA, Qwen-VL, Molmo: real VLMs locally
  [5.5] Visual-token flow and cross-modal circuits
  [5.6] Object hallucination, grounding, and modality arbitration
  [5.7] Multimodal SAEs, transcoders, and crosscoders
  [5.8] VLM patchscopes, verbalizers, and activation-to-language
  [5.9] SHAPley and attribution baselines for VLMs
  [5.10] Diffusion and autoregressive image-generation interpretability
  [5.11] VLM capstone projects
```

This chapter would become one of the most important parts of the fork, because the open-problems review explicitly argues that interpretability needs to expand beyond GPT-style language models to broader model families and multimodal systems.

---

# 5.0 — Why VLM interpretability is different

Core question:

```text
When a VLM answers a question about an image, where exactly does the answer come from?
```

This section introduces the central VLM failure modes:

```text
visual grounding failure
object hallucination
OCR hallucination
counting failure
spatial relation failure
text priors overriding image evidence
image evidence overriding text evidence
modality collapse
typographic attacks
overreliance on language priors
```

The notebook should start with simple examples:

```text
Image: blue banana
Question: What color is the banana?
Possible failure: yellow

Image: dog with text "cat" written on it
Question: What animal is shown?
Possible failure: cat

Image: two cups, one left of the other
Question: Which cup is on the left?
Possible failure: spatial confusion
```

Then introduce the core interpretability questions:

```text
Where is object identity represented?
Where is color represented?
Where is OCR text represented?
Where are spatial relations represented?
When does the LLM integrate image tokens?
Does the answer come from vision, language priors, or both?
Can we causally change the answer by patching internal states?
```

This also introduces the central warning: **VLM interpretability cannot just reuse last-token LLM patching blindly.** A 2026 VLM paper argues that standard last-token interventions often do not affect VLMs, while full-sequence activation patching can substantially change outputs in visual/language conflict settings. ([arXiv][1])

---

# 5.1 — CLIP and SigLIP from scratch

This is the VLM equivalent of “build GPT-2 from scratch.”

## Exercises

Implement:

```text
image encoder
text encoder
projection heads
normalized embeddings
temperature / logit scale
contrastive image-text loss
zero-shot classification
image-text retrieval
typographic attack dataset
```

Start with a tiny CLIP trained on:

```text
CIFAR-10 with template captions
small image-caption subset
synthetic colored-shapes dataset
```

Then load real models:

```text
OpenCLIP ViT-B/32
OpenCLIP ViT-L/14
SigLIP small/base
```

CLIP is foundational because OpenAI’s original multimodal-neuron work found high-level multimodal neurons in CLIP, including neurons responding across images, text, and abstract concepts. ([OpenAI][2]) The existing ARENA-adjacent source pack also singles out CLIP as a key multimodal model and describes its contrastive image/text shared-latent-space objective.

## Verification

```text
contrastive loss decreases on toy data
image-text retrieval beats random
zero-shot accuracy beats random
image/text embedding norms are ~1 after normalization
text-image and image-text logits are transposes
loaded OpenCLIP outputs match reference outputs
typographic attacks reduce accuracy
```

## Interpretability intro

```text
patch-level embeddings
image-token PCA
text-token PCA
image-text logit attribution
embedding arithmetic
concept directions
typographic vulnerability analysis
```

This notebook should end with the first “aha” moment: **CLIP-like models do not merely see pixels; their late representations often organize images around language-like concepts.**

---

# 5.2 — CLIP-like model interpretability

Core question:

```text
What features does a contrastive vision-language model learn, and are they causal?
```

## Required methods

```text
CLIP neuron / channel analysis
CLIP-Dissect-style automatic descriptions
patch-level logit attribution
dogit lens / patch-to-class attribution
SAEs on CLIP ViT residual streams
SAEs on CLS token vs patch tokens
feature steering
typographic attack mitigation
```

The new vision-mech-interp ecosystem makes this realistic. Prisma is an open-source toolkit for vision and video mechanistic interpretability; its paper describes support for 75+ vision/video transformers, SAE/transcoder/crosscoder training, 80+ pretrained SAE weights, activation caching, circuit analysis, and visualization tools. ([arXiv][3]) Recent CLIP-SAE work trains SAEs on CLIP’s vision transformer and reports that SAEs expose many more steerable features than the base neuron basis, with applications to disentanglement tasks and typographic attacks. ([arXiv][4]) Another 2025 paper focuses on hierarchical SAEs for CLIP/SigLIP-style multimodal representations. ([arXiv][5])

## Exercises

```text
Train a small SAE on CLIP patch activations.
Load a pretrained vision SAE where available.
Find features for:
  object identity
  color
  texture
  OCR/text
  style
  background
  human attributes
  spatial layout
  typographic text
Compare neuron basis vs SAE basis.
Steer or suppress a feature.
Measure effect on zero-shot classification.
```

## Verification

```text
SAE reconstruction MSE below threshold
loss/retrieval degradation from SAE reconstruction measured
feature top-activating images are non-cherry-picked
feature predicts held-out positives and negatives
feature ablation changes target CLIP logits
random feature ablation does less
steering improves or worsens a controlled target metric
typographic feature suppression reduces typographic attack success
```

This section should be very careful about **interpretability illusions**. The existing source pack warns that max-activating examples can be misleading across datasets, and that CLIP’s multimodal neurons are fascinating but may not correspond cleanly to one-dimensional human concepts.

---

# 5.3 — Building a mini multimodal LLM

Core question:

```text
How does a VLM turn image features into language-model tokens?
```

Implement a tiny LLaVA/PaliGemma-style model:

```text
vision encoder -> projector / connector -> language model
```

## Exercises

Implement:

```text
patch embedding
ViT vision encoder or frozen OpenCLIP encoder
linear projector / MLP projector
special image tokens
concatenation of image tokens + text tokens
causal language-model decoding
teacher-forced captioning loss
visual question answering loss
```

Use a toy model first:

```text
colored shapes
CLEVR-style mini-scenes
synthetic OCR
synthetic spatial relations
```

Then load real VLMs.

Good required local targets:

```text
PaliGemma 3B / PaliGemma 2 3B
Qwen2.5-VL 3B or 7B, quantized if needed
LLaVA-OneVision 7B, quantized if needed
Molmo 7B-D, quantized if needed
```

PaliGemma is described in Hugging Face Transformers as a VLM combining a SigLIP vision encoder with a Gemma decoder, available in 3B, 10B, and 28B sizes. ([Hugging Face][6]) Qwen2.5-VL is available in 3B, 7B, and 72B variants and includes a ViT encoder, dynamic resolution processing, window attention, and multimodal RoPE-like mechanisms for spatial/temporal information. ([Hugging Face][7]) LLaVA-OneVision is a VLM for text generation conditioned on one or more images or videos, using a SigLIP vision encoder and Qwen2 language backbone in the Transformers implementation. ([GitHub][8]) Molmo is an open multimodal model family with released training/usage code. ([GitHub][9])

## Verification

```text
toy model overfits tiny caption dataset
image-token shapes match expected architecture
projector output dimensions match LLM embedding dimension
greedy decoding deterministic at fixed seed
real model loads locally
real model answers simple image questions
peak VRAM printed
quantized and non-quantized outputs compared where possible
```

The point is to make VLMs stop feeling magical: they are usually **vision tokens inserted into an LLM computation graph through a connector**.

---

# 5.4 — Visual-token flow and cross-modal circuits

Core question:

```text
Where does visual information enter the language computation, and how does it move?
```

This becomes the VLM version of induction heads / IOI.

## Required experiments

```text
visual token residual-stream patching
text token residual-stream patching
full-sequence patching
image-region patching
visual encoder layer patching
projector patching
LLM layer patching
attention-head patching
MLP patching
image-token deletion
image-token shuffling
image-token compression
```

Important contrast pairs:

```text
clean image, clean question
corrupted image, same question
same image, corrupted question
image contradicts text prior
text prompt contradicts image evidence
image contains object A vs object B
image contains red object vs blue object
```

The LLaVA interpretability literature is now mature enough to make this a real notebook. The ICLR 2025 LLaVA paper studies how visual tokens are processed inside the language-model component, including object-information localization, visual-token representation evolution, and integration of visual information for predictions. ([OpenReview][10]) “What Do VLMs NOTICE?” applies causal intervention-style methods to VLMs with semantic image corruption and text token replacement. ([ACL Anthology][11]) Fine-grained Cross-modal Causal Tracing analyzes visual and textual tokens, attention, FFNs, and hidden states across decoder layers for object perception and hallucination mitigation. ([AAAI Publications][12])

## Exercises

```text
Implement VLM activation cache.
Patch only visual tokens.
Patch only text tokens.
Patch entire sequence.
Patch projector outputs.
Patch visual encoder outputs.
Patch selected object-region tokens.
Compare last-token patching vs full-sequence patching.
Plot causal effect by layer × token type.
```

## Verification

```text
patching image object A -> object B changes answer
patching irrelevant background does less
full-sequence patching beats last-token patching on conflict cases
random image patching fails
text-only baseline fails on image-dependent examples
image-only baseline fails on language-dependent examples
causal effect localizes consistently across held-out examples
```

This should create a VLM version of the IOI mental model:

```text
Visual encoder detects object/color/text.
Projector converts visual features into LLM-compatible residual directions.
Early/middle LLM layers integrate image evidence.
Later layers arbitrate between visual evidence and language priors.
Final layers express the answer in vocabulary space.
```

But every part of that story should be treated as a hypothesis to test.

---

# 5.5 — Object hallucination, grounding, and modality arbitration

Core question:

```text
When a VLM hallucinates an object, is the problem perception, transfer, arbitration, or decoding?
```

This should be one of the most important subchapters for your interests.

## The four failure hypotheses

```text
H1: Perceptual failure
The vision encoder never represented the object correctly.

H2: Transfer failure
The vision encoder knew, but the projector/connector failed to pass it to the LLM.

H3: Arbitration failure
The LLM received visual evidence but language priors overrode it.

H4: Decoding failure
The model had the right internal state but generated the wrong answer.
```

The 2026 “Arbitration Failure, Not Perceptual Blindness” paper is directly aligned with this structure: it studies cases where visual evidence disagrees with language biases and uses full-sequence activation patching to establish causal relationships. ([arXiv][1]) A 2026 dual-pathway paper studies object hallucination circuits in VLMs, explicitly framing hallucination as a mechanistic question rather than only a behavioral one. ([arXiv][13]) CVPR 2025 work on “Devils in Middle Layers” reports hallucination mitigation by adjusting visual-information processing in middle layers. ([CVF Open Access][14])

## Datasets

Use small, local subsets:

```text
POPE-style object presence
HallusionBench-style visual illusion cases
MMHal-style hallucination examples
synthetic color-object conflict
synthetic OCR conflict
synthetic spatial relations
Winoground-style compositional pairs
```

## Exercises

```text
Build clean/corrupt image-question pairs.
Train probes for object presence in:
  vision encoder
  projector
  visual-token residual stream
  answer-token residual stream
Run activation patching to locate causal layers.
Run visual-token ablation and background controls.
Compare VLM output to CLIP retrieval and image classifier baselines.
Patch visual evidence into hallucination cases.
Patch hallucination-prone hidden states into clean cases.
```

## Verification

```text
probe detects object above random and text-only baseline
causal patch changes answer, not just probe score
object-region patch beats background patch
intervention reduces hallucination rate on held-out examples
intervention does not destroy normal VQA accuracy
random direction/control intervention fails
```

This would give you a direct path to targeted research.

---

# 5.6 — Multimodal SAEs, transcoders, and crosscoders

Core question:

```text
Can sparse feature methods decompose multimodal representations into meaningful visual, textual, and cross-modal features?
```

Train or load sparse dictionaries on:

```text
CLIP vision encoder patch activations
CLIP text encoder activations
SigLIP vision activations
VLM projector outputs
VLM image-token residual stream
VLM text-token residual stream
VLM answer-token residual stream
diffusion-model cross-attention activations
DiT residual streams
```

## Exercises

```text
Train ReLU, TopK, Gated, and JumpReLU SAEs on CLIP patch activations.
Train small SAE on VLM projector outputs.
Compare feature density across:
  CLS token
  patch tokens
  visual tokens after projector
  text tokens
  answer tokens
Find features for:
  object identity
  color
  texture
  text in image
  spatial relation
  style
  safety-relevant content
  hallucination-prone priors
Train crosscoders between CLIP and VLM representations.
Train crosscoders between base VLM and instruction-tuned VLM.
```

## Verification

```text
loss/reconstruction frontier measured
feature-density histograms inspected
held-out feature classifier AUC measured
feature steering has causal behavioral effect
feature ablation affects target behavior more than random controls
crosscoder shared features predict cross-model transfer
model-specific features predict behavioral differences
```

This chapter should explicitly compare **neuron basis, attention-head basis, residual directions, SAE features, and crosscoder features**. The open-problems source warns that neurons, attention heads, and layers are often too coarse or polysemantic, motivating sparse dictionary learning and stronger decomposition methods.

---

# 5.7 — VLM patchscopes, verbalizers, and activation-to-language

Core question:

```text
Can we ask a VLM what an internal visual activation represents, and then verify the answer?
```

## Exercises

Implement:

```text
image-token logit lens
visual-token tuned lens
patchscope for image activations
caption-scope: activation -> natural-language description
counterfactual activation decoding
feature verbalizer for CLIP/VLM SAE features
```

Example prompts:

```text
"This image token represents: ___"
"The object at this patch is: ___"
"The color represented here is: ___"
"The model is likely to answer: ___"
"The visual evidence supports: ___"
```

## Verification

```text
activation explanation predicts held-out image patches
decoded object matches ground truth above baseline
decoded color matches ground truth above baseline
counterfactual patch changes decoded explanation
text-only verbalizer baseline fails on image-dependent cases
random activation produces low-confidence or nonsense explanation
```

This is where VLM interpretability can become more powerful than text-only mech interp: the model’s internal states can potentially be tied to image patches, language labels, and generated explanations.

---

# 5.8 — SHAPley and attribution baselines for VLMs

Core question:

```text
When does a VLM use the image, the text, or both?
```

This should connect your earlier SHAP request directly to VLMs.

## Methods

```text
MM-SHAP
CC-SHAP for VLMs
PixelSHAP
image-region Shapley
text-token Shapley
modality-level Shapley
object-level Shapley
Shapley interaction values
SHAP vs activation patching
SHAP vs visual-token patching
```

MM-SHAP uses Shapley values to quantify how much multimodal models rely on individual modalities. ([arXiv][15]) CC-SHAP for VLMs extends consistency/faithfulness analysis to vision-language decoders. ([GitHub][16]) PixelSHAP extends Shapley-style attribution to structured visual entities for generative VLMs. ([arXiv][17])

## Exercises

```text
Compute modality Shapley:
  image present / absent
  text present / absent
  both present
  both corrupted

Compute object Shapley:
  remove object A
  remove object B
  remove background
  remove OCR text

Compare to:
  activation patching
  causal tracing
  visual-token ablation
  attention maps
  Grad-CAM / relevancy maps
```

## Verification

```text
Shapley attribution predicts counterfactual answer changes
object-level attribution beats pixel-level attribution on structured scenes
Shapley and activation patching agree on simple cases
Shapley fails on interaction-heavy cases, and the notebook explains why
modality-collapse cases are detected
```

The framing should be: **SHAP is a useful baseline and audit tool, not a substitute for mechanistic explanation.**

---

# 5.9 — Diffusion and autoregressive image-generation interpretability

This should be included, but as a supporting pillar after VLMs.

Core question:

```text
How does text become image structure inside a generative model?
```

## Diffusion models

Required local targets:

```text
Stable Diffusion 1.5
SDXL if local memory allows
small DiT / PixArt-style model
Flux-schnell or similar only if quantized/local stack works
```

## Methods

```text
cross-attention maps
DAAM-style word-to-region attribution
ConceptAttention for DiTs
SAEs on U-Net / DiT activations
DIFFLENS-style bias features
cross-attention head relevance vectors
denoising-step causal patching
prompt-token ablation
latent-region ablation
```

DAAM performs text-image attribution in Stable Diffusion by aggregating cross-attention maps. ([ACL Anthology][18]) ConceptAttention repurposes Diffusion Transformer attention-layer outputs to produce sharper concept localization maps than ordinary cross-attention. ([arXiv][19]) DIFFLENS uses sparse semantic features to analyze and mitigate bias mechanisms in diffusion models. ([CVF Open Access][20])

## Exercises

```text
Generate image from prompt.
Record cross-attention over denoising steps.
Track where each noun/adjective is localized.
Patch one prompt token's attention.
Patch one denoising timestep's activation.
Ablate a concept feature.
Compare U-Net cross-attention vs DiT ConceptAttention.
```

## Verification

```text
word-region attribution matches segmentation masks above baseline
cross-attention patch changes corresponding image region
random token patch does less
early/mid/late denoising interventions have distinct effects
feature steering changes target concept without global collapse
```

## Autoregressive image generation

Implement toy version:

```text
VQ-VAE / VQGAN tokenizer
image-token transformer
text-conditioned image-token generation
image-token logit lens
patching image-token generation path
```

The reason to include this is that autoregressive image generators use the same core idea as language models — sequential token prediction — but over image tokens; recent image-generation histories still treat AR image generation as a major branch alongside diffusion. ([arXiv][21])

---

# 5.10 — VLM model zoo and local feasibility

The required models should be chosen for **runnability and interpretability access**, not leaderboard prestige.

## Green tier: required

```text
OpenCLIP ViT-B/32 or ViT-L/14
SigLIP small/base
PaliGemma 3B
Qwen2.5-VL 3B
LLaVA-OneVision small/7B quantized if needed
Molmo 7B-D quantized if needed
Stable Diffusion 1.5
toy VQGAN + AR image transformer
toy multimodal LLM
```

## Yellow tier: optional

```text
Qwen2.5-VL 7B
PaliGemma 10B
LLaVA-OneVision 7B
SDXL
small DiT
V-JEPA 2 / V-JEPA 2.1 feature extractors
```

## Red tier: read-only / remote / optional

```text
Qwen2.5-VL 72B
PaliGemma 28B
large Molmo variants
large video VLMs
full-scale VLM SAEs across all layers
large diffusion transformer SAEs
```

Every local notebook should print:

```text
model name
parameter count where available
precision
quantization
image resolution
number of visual tokens
context length
peak VRAM
runtime
seed
```

---

# The VLM chapter should have its own research project track

The course should include:

```text
research_projects/vlm_interp/
  00_template/
  01_visual_token_flow/
  02_modality_arbitration/
  03_object_hallucination_circuits/
  04_clip_sae_typographic_attacks/
  05_vlm_crosscoder_base_vs_instruct/
  06_vlm_shap_vs_patching/
  07_diffusion_prompt_to_region_circuits/
  08_visual_world_model_features/
```

## Best VLM capstones

### Capstone A — “Where does object identity enter the LLM?”

Claim:

```text
Object identity is represented in the vision encoder, transformed by the projector, and becomes causally available to answer-token computation in a specific band of LLM layers.
```

Evidence:

```text
vision probe
projector probe
LLM residual probe
visual-token patching
full-sequence patching
held-out objects
background controls
```

### Capstone B — “VLM hallucinations are arbitration failures, not perception failures”

Claim:

```text
In a subset of hallucination cases, the model internally represents the correct visual evidence but answer generation follows language priors.
```

Evidence:

```text
visual encoder knows object/color
projector passes object/color
LLM mid-layer hidden state contains object/color
late-layer intervention flips answer
text-prior controls
blue-banana / typographic / object-presence datasets
```

This directly connects to the 2026 arbitration-failure framing. ([arXiv][1])

### Capstone C — “SAE features can mitigate typographic attacks in CLIP/VLMs”

Claim:

```text
A sparse set of OCR/text-like visual features mediates typographic attack vulnerability; suppressing them improves robustness without broadly damaging zero-shot performance.
```

Evidence:

```text
CLIP/VLM typographic attack benchmark
SAE feature identification
feature suppression
random-feature control
zero-shot accuracy preservation
OOD typography variants
```

This builds naturally on CLIP SAE steering work. ([arXiv][4])

### Capstone D — “Do CLIP, SigLIP, and VLM image-token streams learn the same visual concepts?”

Claim:

```text
Some high-level object/color/style features are shared across CLIP-like and VLM representations, but VLM-specific features emerge for instruction-following, OCR, refusal, and answer formatting.
```

Evidence:

```text
crosscoders
feature matching
shared-feature steering
model-specific feature ablation
held-out datasets
```

### Capstone E — “SHAP vs causal patching in VLMs”

Claim:

```text
Object-level SHAP is useful for auditing surface reliance but misses internal arbitration failures that causal patching can detect.
```

Evidence:

```text
MM-SHAP / PixelSHAP
visual-token patching
full-sequence activation patching
cases where both agree
cases where they disagree
ground-truth synthetic scenes
```

### Capstone F — “Prompt-to-region circuits in diffusion models”

Claim:

```text
Text-to-image models localize prompt concepts through identifiable cross-attention or DiT-attention pathways whose causal role varies by denoising timestep.
```

Evidence:

```text
DAAM / ConceptAttention maps
token-level interventions
denoising-step patching
segmentation overlap
random-token controls
```

---

# The final updated syllabus

The full fork now becomes:

```text
0. Local GPU, testing, reproducibility, and notebook style
1. Original ARENA, preserved exactly
2. Modern language architectures
3. Sparse feature methods
4. Circuit methods
5. Vision-language and generative-vision interpretability
6. JEPAs and predictive representation learning
7. World-model interpretability
8. Activation-to-language methods
9. SHAPley and attribution baselines
10. Alignment and safety interpretability
11. Training dynamics and developmental interpretability
12. Capstone research sprint
```

But because VLMs are your main interest, I would make chapter 5 one of the longest chapters.

The final priority order should be:

```text
1. Preserve ARENA style and original content.
2. Build reusable local GPU + hook/cache/verification infrastructure.
3. Implement CLIP/SigLIP from scratch.
4. Implement a mini VLM from scratch.
5. Load PaliGemma/Qwen2.5-VL/LLaVA/Molmo locally.
6. Build VLM activation patching and visual-token patching tools.
7. Add CLIP/VLM SAEs and steering.
8. Add hallucination/grounding/arbitration projects.
9. Add SHAP baselines.
10. Add diffusion/AR image-generation interpretability as the generative-vision extension.
```

---

# Final verdict

Yes: the final course should now be understood as:

```text
Original ARENA preserved
+
Frontier architecture implementation
+
VLM-first multimodal mechanistic interpretability
+
CLIP/SigLIP representation analysis
+
multimodal LLM visual-token circuit analysis
+
VLM hallucination and grounding research
+
diffusion / AR image-generation interpretability
+
JEPA and world-model interpretability
+
SHAPley baselines
+
paper-style capstone projects
```

The most important amendment is this:

**VLM interpretability becomes a full flagship chapter with its own model zoo, tools, implementation path, verification suite, and capstone track.**

That makes the fork aligned with your actual research interest while preserving the “beautiful ARENA notebook” feel: readable, runnable, exercise-driven, visual, skeptical, and constantly pushing from toy implementation to real pretrained models and causal verification.

[1]: https://arxiv.org/html/2604.09364v1?utm_source=chatgpt.com "Arbitration Failure, Not Perceptual Blindness: How Vision- ..."
[2]: https://openai.com/index/multimodal-neurons/?utm_source=chatgpt.com "Multimodal neurons in artificial neural networks"
[3]: https://arxiv.org/html/2504.19475v1?utm_source=chatgpt.com "Prisma : An Open Source Toolkit for Mechanistic ..."
[4]: https://arxiv.org/abs/2504.08729?utm_source=chatgpt.com "Steering CLIP's vision transformer with sparse autoencoders"
[5]: https://arxiv.org/html/2502.20578v1?utm_source=chatgpt.com "Interpreting CLIP with Hierarchical Sparse Autoencoders"
[6]: https://huggingface.co/docs/transformers/en/model_doc/paligemma?utm_source=chatgpt.com "PaliGemma"
[7]: https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl?utm_source=chatgpt.com "Qwen2.5-VL"
[8]: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llava_onevision.md?utm_source=chatgpt.com "LLaVA-OneVision - huggingface/transformers"
[9]: https://github.com/allenai/molmo?utm_source=chatgpt.com "Code for the Molmo Vision-Language Model"
[10]: https://openreview.net/forum?id=chanJGoa7f&utm_source=chatgpt.com "Towards Interpreting Visual Information Processing in..."
[11]: https://aclanthology.org/2025.naacl-long.571.pdf?utm_source=chatgpt.com "What Do VLMs NOTICE? A Mechanistic Interpretability ..."
[12]: https://ojs.aaai.org/index.php/AAAI/article/view/40431?utm_source=chatgpt.com "Causal Tracing of Object Representations in Large Vision ..."
[13]: https://arxiv.org/html/2605.13156v1?utm_source=chatgpt.com "Dual-Pathway Circuits of Object Hallucination in Vision- ..."
[14]: https://openaccess.thecvf.com/content/CVPR2025/papers/Jiang_Devils_in_Middle_Layers_of_Large_Vision-Language_Models_Interpreting_Detecting_CVPR_2025_paper.pdf?utm_source=chatgpt.com "Interpreting, Detecting and Mitigating Object Hallucinations via ..."
[15]: https://arxiv.org/abs/2212.08158?utm_source=chatgpt.com "[2212.08158] MM-SHAP: A Performance-agnostic Metric ..."
[16]: https://github.com/Heidelberg-NLP/CC-SHAP-VLM?utm_source=chatgpt.com "CC-SHAP for VLMs 🖼️"
[17]: https://arxiv.org/html/2503.06670v1?utm_source=chatgpt.com "Attention, Please! PixelSHAP Reveals What Vision ..."
[18]: https://aclanthology.org/2023.acl-long.310.pdf?utm_source=chatgpt.com "What the DAAM: Interpreting Stable Diffusion Using Cross ..."
[19]: https://arxiv.org/html/2502.04320v1?utm_source=chatgpt.com "Diffusion Transformers Learn Highly Interpretable Features"
[20]: https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_Dissecting_and_Mitigating_Diffusion_Bias_via_Mechanistic_Interpretability_CVPR_2025_paper.pdf?utm_source=chatgpt.com "Dissecting and Mitigating Diffusion Bias via Mechanistic ..."
[21]: https://arxiv.org/html/2603.07455v1?utm_source=chatgpt.com "Image Generation Models: A Technical History"


----------------------------------------------------

Yes — this is the final missing amendment. The course should absolutely contain **a refusal-direction replication notebook**, **representation-geometry notebooks with PCA/UMAP/t-SNE**, **image/VLM/image-generation direction geometry**, and **LoRA/DoRA interpretability for alignment and misalignment**.

And I agree with your “if the correct result is white noise, it is useless” criterion. The final extension should explicitly reject pretty plots unless they survive quantitative and causal checks.

The final design principle becomes:

> **Every visualization must cash out in prediction, intervention, or falsification.**
> PCA, UMAP, t-SNE, attention maps, saliency maps, and feature dashboards are exploration tools. They do not count as interpretability evidence unless they predict labels, transfer OOD, or support causal interventions.

That principle fits the research style in your source pack: explore broadly, prioritize information gain, then move into skeptical hypothesis testing and rigorous evidence rather than elegant narratives.  The writing notes also explicitly emphasize precise methods, rigorous supporting evidence, clear figures, and avoiding overclaiming.

---

# Final amendment A: refusal direction is mandatory

Yes, **`Refusal is Mediated by a Single Direction`** should be a required exercise. It is simple, beautiful, important, local, and pedagogically perfect.

The paper reports that across 13 open-source chat models up to 72B, a single residual-stream direction can be found such that erasing it suppresses refusal while adding it elicits refusal on otherwise benign prompts. ([arXiv][1]) Your source pack also singles it out as one of the most compelling practical uses of model-internals work: adding the direction causes refusal, ablating it prevents refusal, and the method is cheap compared with finetuning.

The notebook should be:

```text
[8.1] Refusal Is Mediated by a Direction
```

## Core exercise

Use a small local chat model and sanitized prompt pairs.

```text
Dataset:
  harmless instruction prompts
  refusal-eliciting policy-category prompts, sanitized and non-procedural

Model:
  small open-weight instruction model, preferably <= 7B
  optional larger quantized model

Representations:
  residual stream at every layer
  final user-token position
  assistant-start position
  first generated-token position
```

Implement:

```python
d_refusal[layer] =
    mean(resid_harmful[layer]) - mean(resid_harmless[layer])

d_refusal = normalize(best_layer_direction)
```

Then test:

```text
1. Addition:
   Add +α d_refusal to harmless prompts.
   Does refusal rate increase?

2. Ablation:
   Project out d_refusal from refusal-eliciting prompts.
   Does refusal rate decrease?

3. Layer sweep:
   Which layers work best?

4. Position sweep:
   Which token positions work best?

5. Direction robustness:
   Does the direction transfer across held-out prompts?

6. Model robustness:
   Does the method work across at least two small local models?
```

The notebook should not teach “how to jailbreak models” as the point. It should teach **what safety finetuning implemented internally, how brittle it is, and how to validate a representation-level claim**. Outputs should be evaluated as aggregate refusal/non-refusal labels, not as procedural harmful completions.

## Verification

A pass requires:

```text
refusal classifier / rubric has high agreement on held-out examples
d_refusal separates harmful-vs-harmless activations
adding d_refusal increases refusal on harmless prompts
projecting out d_refusal decreases refusal behavior on sanitized refusal prompts
random direction control fails
label-shuffled direction fails
OOD prompt templates still work
general capability degradation is measured
```

## Required extensions

This notebook should not stop at the paper’s headline result.

Add:

```text
PCA of harmful/harmless activation differences
SVD of all layer-wise refusal directions
fraction of refusal variance explained by PC1
cosine similarity between mean-difference direction and PC1
linear probe vs mean-difference direction
DAS / learned 1D direction vs mean-difference direction
multi-direction refusal basis
```

The core question:

```text
Is refusal really one-dimensional, or is one dimension just a convenient control knob?
```

This is important because later work argues that different refusal/non-compliance categories can correspond to geometrically distinct directions, even if many directions act like a shared refusal-control knob. ([arXiv][2]) So the course should teach the original result **and** the natural skeptical extension.

---

# Final amendment B: representation geometry becomes a full chapter

Add:

```text
[ch3.5] Representation Geometry: PCA, UMAP, t-SNE, Directions, and Manifolds
```

This chapter should sit between “linear representations / steering” and “SAEs / circuits.”

## 3.5.1 PCA is not just plotting

Core exercise:

```text
Given activation vectors for concept examples,
find whether a concept is:
  one-dimensional,
  low-dimensional,
  circular / periodic,
  clustered,
  hierarchical,
  or not cleanly represented.
```

Methods:

```text
PCA
SVD
linear probes
logistic regression
CCA / SVCCA
cosine similarity matrices
explained variance ratio
kNN classification
UMAP
t-SNE
Procrustes alignment across models/layers
```

Verification:

```text
plot has labels hidden during dimensionality reduction
nearest-neighbor accuracy beats random
linear probe beats random
PCA dimensions predict known variables
held-out examples land in expected region
random token/control concept gives white noise
permuted labels destroy structure
causal intervention along discovered direction changes behavior
```

## 3.5.2 Days, months, years, space, and time

Yes, the “days in LLMs” notebook should be included.

There are two closely related strands:

First, **Language Models Represent Space and Time** shows evidence that LLMs learn linear representations of spatial and temporal coordinates across several datasets and scales. ([arXiv][3])

Second, the “days of the week / months of the year” style geometry is a great example of **multi-dimensional feature geometry**, not just one direction. A 2025 ICLR paper explicitly analyzes the “days of the week” cluster along PCA components and argues that some features are not one-dimensionally linear. ([arXiv][4]) A 2026 feature-manifold line of work also discusses days/months forming interpretable circular structures. ([arXiv][5])

Notebook:

```text
[3.5.2] Days, Months, Space, and Time
```

Exercises:

```text
Extract residual-stream activations for:
  days of week
  months
  years
  cities
  countries
  historical figures
  artworks/news dates

Run:
  PCA
  UMAP
  t-SNE
  linear regression to known coordinates
  circular fit for days/months
  layer sweep
  prompt-template sweep
```

Verification:

```text
days form a non-random periodic structure
months form a non-random periodic structure
space/time coordinates are linearly decodable above baseline
structure persists across prompt templates
permuted labels fail
unrelated token sets do not produce the same clean geometry
interventions along discovered coordinates have predictable logit effects where possible
```

This is a cornerstone because it teaches the student that **“direction” is not always the right abstraction**. Some concepts are one-dimensional; others are planes, rings, manifolds, or local geometries.

---

# Final amendment C: “white noise detector” for every visualization

Every visualization notebook should include this cell:

```python
def reject_pretty_but_useless_plot(embeddings, labels, controls):
    """
    A visualization is only allowed to count as evidence if it passes:
      1. quantitative separability,
      2. held-out generalization,
      3. random-label failure,
      4. random-token failure,
      5. causal or predictive downstream test.
    """
```

The course should explicitly say:

```text
A good UMAP that cannot predict labels is decoration.
A beautiful t-SNE that disappears under seed changes is decoration.
A PCA plot that works only on cherry-picked examples is decoration.
A feature cluster that cannot be patched, steered, probed, or predicted is decoration.
```

Required metrics:

```text
silhouette score
kNN accuracy
linear probe accuracy
cross-validation accuracy
bootstrap stability
Procrustes alignment across seeds
trustworthiness / continuity for UMAP/t-SNE
label permutation test
random concept control
OOD concept split
```

For UMAP/t-SNE specifically:

```text
Never trust one seed.
Never trust one perplexity / n_neighbors.
Never trust one hand-picked dataset.
Never infer global geometry from t-SNE.
Always pair with PCA/probes/distances.
```

---

# Final amendment D: VLM geometry — clothes, objects, colors, styles

Yes: **UMAP of clothes in VLMs** is exactly the kind of thing that belongs in the VLM flagship chapter.

Notebook:

```text
[5.3] VLM Representation Geometry: Clothes, Objects, Colors, Styles
```

Core question:

```text
Do VLMs organize visual concepts by object category, color, texture, style, function, language label, or dataset bias?
```

Dataset:

```text
small local clothing image dataset
synthetic colored-clothes dataset
product-image subset
CLIP/SigLIP image-text pairs
optional generated images with controlled prompts
```

Concept labels:

```text
garment type:
  shirt, dress, pants, coat, shoe

color:
  red, blue, black, white, green

material:
  denim, leather, cotton, wool

style:
  formal, casual, athletic, vintage

viewpoint:
  front, side, close-up

textual label:
  caption says "red dress" vs actual image color
```

Models:

```text
CLIP / OpenCLIP
SigLIP
PaliGemma-style VLM
Qwen-VL-style VLM
small local VLM
```

Representations:

```text
CLIP image embedding
CLIP text embedding
vision encoder CLS token
vision patch tokens
VLM projector output
VLM visual-token residual stream
answer-token residual stream
```

Exercises:

```text
PCA/UMAP/t-SNE of clothes embeddings
color-coded by type, color, material, style
linear probes for each factor
concept directions for color/style/type
crosscoder between CLIP and VLM visual tokens
image-region patching for garment part
text-vs-image conflict examples
```

Verification:

```text
clothing type is decodable above baseline
color is decodable above baseline
style is harder and must be treated skeptically
synthetic controls separate color from garment type
text-label corruption reveals whether model follows image or caption prior
object-region patch beats background patch
random-image patch fails
plot structure survives multiple seeds/settings
```

This also becomes a good place to teach **spurious visual features**. For example:

```text
Does "doctor" cluster by white coat?
Does "formal" cluster by background?
Does "athletic" cluster by pose rather than clothing?
Does "dress" cluster by gender presentation rather than garment?
```

Those are real VLM interpretability questions, not just cute plots.

---

# Final amendment E: image-generation model directions and geometry

Yes, do this too.

Add:

```text
[5.10] Image-Generation Interpretability: Directions, Denoising, and Concept Geometry
```

This covers diffusion and AR image generation.

## Diffusion direction exercise

Core question:

```text
Do text-to-image models have interpretable activation or latent directions for visual concepts?
```

Relevant work exists. CVPR 2024 work studies self-discovered interpretable diffusion latent directions for responsible text-to-image generation. ([CVF Open Access][6]) Apple’s 2025 Activation Transport work explicitly frames activation steering as modality-agnostic and applies to language and diffusion models. ([Apple Machine Learning Research][7]) Recent diffusion interpretability work also studies how interpretable concepts emerge and evolve through the denoising process. ([arXiv][8])

Exercises:

```text
Generate controlled image pairs:
  red car vs blue car
  smiling face vs neutral face
  day scene vs night scene
  watercolor vs photo
  one object vs two objects

Collect:
  text encoder activations
  cross-attention activations
  U-Net / DiT block activations
  latent states across timesteps
  final image embeddings via CLIP/SigLIP

Find directions:
  mean-difference direction
  PCA direction
  linear probe direction
  sparse feature direction
  activation-transport direction
```

Verification:

```text
direction changes target concept in generated image
direction does not globally destroy image quality
CLIP/SigLIP classifier detects intended concept change
human-inspection sample grid included but not sufficient
random direction fails
wrong-timestep direction is weaker
opposite direction reverses effect
seed sweep succeeds
```

## UMAP/t-SNE for image-generation

Allowed, but only as exploration.

```text
UMAP of generated images by CLIP embedding
UMAP of denoising latents by timestep
UMAP of concept-direction endpoints
UMAP of LoRA style adapters
```

Required sanity checks:

```text
If UMAP separates by random seed or brightness rather than concept, mark it as failure.
If prompt concept is not recoverable by an independent classifier, mark it as failure.
If intervention produces white noise, mark it as failure.
If it only works on cherry-picked images, mark it as failure.
```

---

# Final amendment F: LoRA / DoRA interpretability and misalignment

Your P.S. is important. Add a full PEFT chapter:

```text
[ch9] Finetuning, LoRA, DoRA, and Adapter Interpretability
  [9.1] LoRA from scratch
  [9.2] DoRA from scratch
  [9.3] LoRA vs full finetuning: do they learn the same mechanism?
  [9.4] Adapter SVD, rank, and intruder dimensions
  [9.5] LoRA alignment and refusal directions
  [9.6] LoRA emergent-misalignment model organisms
  [9.7] VLM LoRA and visual-instruction tuning interpretability
  [9.8] Diffusion LoRAs: style, identity, concept, and safety directions
```

## 9.1 LoRA from scratch

Implement:

```python
W_eff = W_frozen + alpha / r * B @ A
```

Exercises:

```text
insert LoRA into attention Q/K/V/O
insert LoRA into MLP up/down/gate
train on toy task
merge LoRA into base weights
compare merged vs unmerged outputs
```

Verification:

```text
merged and unmerged logits match
only adapter weights get gradients
rank of update <= r
training loss decreases
base model weights unchanged
```

LoRA is worth including because it is now central to cheap model adaptation. It also creates a natural interpretability object: the low-rank update itself.

## 9.2 DoRA from scratch

DoRA should be included too. The DoRA paper decomposes pretrained weights into magnitude and direction components, using LoRA for directional updates, and reports improved learning capacity and stability over LoRA without extra inference overhead. ([arXiv][9]) The official repo describes the same magnitude/direction decomposition. ([GitHub][10])

Exercises:

```text
implement weight normalization into magnitude and direction
apply LoRA to direction update
compare LoRA vs DoRA on toy task
compare gradient norms and stability
merge adapter
```

Verification:

```text
DoRA merged output matches unmerged output
direction/magnitude decomposition numerically reconstructs W
DoRA trains at least as stably as LoRA on toy task
rank and norm of updates are logged
```

## 9.3 LoRA vs full finetuning interpretability

This is a cornerstone. The course should not assume LoRA and full finetuning are mechanistically equivalent.

Two key papers make this a real exercise:

```text
LoRA Learns Less and Forgets Less
LoRA vs Full Fine-tuning: An Illusion of Equivalence
```

The first finds that standard low-rank LoRA can underperform full finetuning on target tasks while preserving more base-model performance outside the target domain; it also reports that full finetuning learns much higher-rank perturbations than typical LoRA settings. ([arXiv][11]) The second argues that LoRA and full finetuning can achieve similar task performance while learning different spectral structures, including “intruder dimensions” in LoRA-updated weight matrices. ([arXiv][12])

Exercises:

```text
finetune tiny transformer with:
  full finetuning
  LoRA rank 1
  LoRA rank 4
  LoRA rank 16
  DoRA rank 4
  QLoRA optional

Tasks:
  sentiment style shift
  JSON formatting
  refusal-overgeneralization safe proxy
  toy arithmetic skill
  synthetic VLM color-answering task
```

Analyze:

```text
SVD of weight deltas
rank of effective updates
cosine similarity of update subspaces
singular value spectra
intruder dimensions
activation-space shift
feature activation shift
crosscoder base-vs-finetuned features
```

Verification:

```text
LoRA and full FT reach similar in-distribution performance on at least one task
OOD behavior differs
LoRA update rank is constrained
full FT update effective rank is higher
adapter directions explain some activation shift
crosscoder finds model-specific features
random LoRA of same norm fails
```

## 9.4 LoRA alignment and misalignment

Add a safe alignment notebook:

```text
[9.5] Alignment Changes as Low-Rank Updates
```

Core question:

```text
Does a LoRA adapter add a small number of behavior directions, or does it rewrite distributed mechanisms?
```

Exercises:

```text
Train benign adapters:
  politeness adapter
  refusal-overgeneralization adapter
  sycophancy-style safe proxy
  JSON-only adapter
  medical-domain caution adapter, non-advice
  toy misinformation-vs-correction adapter

Measure:
  adapter B-vector directions
  residual-stream shift
  refusal direction cosine
  PCA of adapter-induced activation changes
  crosscoder features
  SAE feature changes
```

Verification:

```text
adapter-induced behavior is measurable
activation shift is low-rank-ish or not, measured honestly
top adapter direction causally steers behavior
projection-out reduces adapter behavior
random adapter direction fails
OOD prompts tested
capability degradation measured
```

For emergent misalignment, use **safe proxies**, not harmful training. But the phenomenon belongs. Recent work reports convergent activation-space representations of emergent misalignment and uses LoRA patching/ablation experiments. ([arXiv][13]) There is also already ARENA-style alignment-science material around loading LoRA-adapted emergent-misalignment model organisms and inspecting qualitative differences. ([learn.arena.education][14])

Required safe version:

```text
Train a toy rank-1 LoRA that induces a benign but broad behavioral drift.
Example:
  overconfident answers
  excessive refusal
  sycophancy on harmless opinions
  always-rhyming answers
  JSON-only behavior
```

Then test:

```text
Does rank-1 LoRA learn one dominant activation direction?
Does the direction appear suddenly during training?
Does PCA of checkpoints show a phase transition?
Can projection-out undo the behavioral drift?
Does full finetuning produce the same direction?
Does DoRA produce the same direction?
```

This is a beautiful bridge between:

```text
LoRA interpretability
training dynamics
alignment
misalignment
representation geometry
activation steering
```

## 9.5 VLM LoRA interpretability

This belongs because many VLMs are adapted with LoRA/QLoRA.

Notebook:

```text
[9.7] VLM LoRA: What Does Visual Instruction Tuning Change?
```

Tasks:

```text
synthetic VQA
OCR answering
color answering
spatial relation answering
clothing classification
caption style adaptation
```

Analyze:

```text
LoRA updates in vision encoder
LoRA updates in projector
LoRA updates in LLM attention
LoRA updates in LLM MLP
visual-token residual shift
text-token residual shift
answer-token residual shift
```

Verification:

```text
adapter improves target VQA task
adapter shifts visual-token representations more than text-only control
patching adapter-induced direction changes answer
adapter direction transfers across held-out images
random adapter direction fails
LoRA vs DoRA vs full finetuning compared
```

## 9.6 Diffusion LoRA interpretability

Image-generation LoRAs are everywhere, so include them.

Notebook:

```text
[9.8] Diffusion LoRAs: Style, Identity, and Concept Directions
```

Exercises:

```text
load a small style LoRA
compute weight delta spectra
compare style LoRAs by SVD
generate same prompts with/without LoRA
extract CLIP image embeddings
UMAP style clusters
patch / scale LoRA strength
inspect cross-attention changes
```

Verification:

```text
LoRA strength monotonically changes independent style classifier score
image quality does not collapse
random LoRA delta fails
same concept appears across seeds
UMAP cluster corresponds to style label, not seed or brightness
```

---

# Final amendment G: the final “cornerstones” list

Here is the final list of methods that must be included for the ARENA extension to feel complete.

## 1. Original ARENA, untouched

Keep:

```text
PyTorch foundations
transformers from scratch
TransformerLens
activation caching
induction heads
IOI
activation patching
path patching
DLA
toy superposition
SAEs
NNsight
function vectors
LLM evals
alignment chapter as-is or updated append-only
```

## 2. Representation directions and geometry

Required:

```text
mean-difference directions
PCA / SVD
linear probes
DAS / causal subspaces
activation addition
concept erasure
UMAP / t-SNE
feature manifolds
space/time/days/months
Othello board geometry
VLM object/color/style geometry
diffusion latent geometry
```

## 3. Refusal, truthfulness, hallucination, misalignment

Required:

```text
refusal direction replication
truthfulness directions
hallucination probes
multi-refusal directions
misalignment directions
safe LoRA misalignment proxy
white-box monitor
```

## 4. IE/IG/feature/circuit attribution

Required:

```text
direct effect
indirect effect
integrated gradients
integrated effects
attribution patching
EAP
EAP-IG
AtP*
ACDC
Sparse Feature Circuits
SHIFT-style editing
transcoder attribution graphs
```

Your source pack explicitly notes that Sparse Feature Circuits uses attribution patching and an integrated-gradients variant, and that this is one of the promising ways to do circuit analysis in the SAE basis.  The primer also explains that EAP requires only two forward passes and one backward pass for all edge scores, while EAP-IG improves faithfulness and is used for sparse feature circuits.

## 5. Sparse decomposition

Required:

```text
ReLU SAE
Gated SAE
TopK SAE
BatchTopK SAE
JumpReLU SAE
attention-output SAEs
residual-stream SAEs
VLM SAEs
CLIP/SigLIP SAEs
SAE evaluation
feature density
loss recovered
SAE error nodes
```

SAE evaluation should include L0, loss recovered, feature-density histograms, DLA, and max-activating examples, as in the primer.

## 6. VLM interpretability

Required:

```text
CLIP/SigLIP from scratch
CLIP/SigLIP feature geometry
mini VLM from scratch
PaliGemma/Qwen-VL/LLaVA/Molmo-style local model adapters
visual-token patching
full-sequence patching
VLM object hallucination
modality arbitration
VLM SAEs
VLM crosscoders
VLM feature circuits
VLM SHAP baselines
VLM LoRA interpretability
```

## 7. Image-generation interpretability

Required:

```text
Stable Diffusion / small diffusion model
cross-attention attribution
DAAM-style prompt-to-region maps
DiT attention maps
denoising-step activation patching
diffusion latent directions
diffusion LoRA analysis
AR image-token transformer toy model
```

## 8. Modern architectures

Required:

```text
Gemma-style transformer
Gemma Scope
Mamba / Mamba-2 / Mamba-3-style state tracking
DiffusionGemma / toy diffusion LM
RecurrentGemma / Griffin
RWKV / RetNet / xLSTM optional
JEPAs and V-JEPA
world models
```

## 9. PEFT and finetuning interpretability

Required:

```text
LoRA
DoRA
QLoRA optional
adapter merging
SVD of updates
intruder dimensions
LoRA vs full finetuning
LoRA vs DoRA
adapter-induced activation directions
crosscoder base-vs-finetuned diffing
safe emergent-misalignment proxy
VLM LoRA
diffusion LoRA
```

## 10. SHAPley and attribution baselines

Required:

```text
exact Shapley
KernelSHAP
PartitionSHAP
TreeSHAP
DeepSHAP
GradientSHAP
SAGE
TokenSHAP
TokenShapley
SHAP-IQ / shapiq
VLM modality SHAP
image-region SHAP
SHAP vs causal patching
```

---

# Final amended syllabus

The final ARENA fork should be:

```text
0. Course Style, Local GPU, and Reproducibility
   ARENA-style notebook contract
   testing harness
   VRAM budget
   visualization sanity checks
   research log template

1. Original ARENA, Preserved
   transformers
   induction heads
   IOI
   patching
   superposition
   SAEs
   NNsight
   evals

2. Modern Architectures
   Gemma
   Mamba
   recurrent hybrids
   diffusion LMs
   multimodal variants

3. Representation Geometry and Directions
   PCA/SVD
   UMAP/t-SNE
   days/months/time/space
   Othello geometry
   one-dimensional vs multi-dimensional features
   white-noise controls

4. Sparse Feature Methods
   SAE variants
   Gemma Scope
   transcoders
   crosscoders
   feature geometry
   feature dashboards

5. Circuit Attribution
   activation patching
   path patching
   direct/indirect effects
   integrated gradients
   EAP/EAP-IG/AtP*
   Sparse Feature Circuits
   SHIFT editing
   attribution graphs

6. Vision-Language Interpretability
   CLIP/SigLIP from scratch
   mini VLM from scratch
   real local VLMs
   visual-token circuits
   object hallucination
   modality arbitration
   VLM SAEs/crosscoders
   VLM geometry of clothes/objects/colors
   VLM SHAP
   VLM LoRA

7. Image-Generation Interpretability
   diffusion attention maps
   denoising circuits
   latent directions
   diffusion LoRAs
   AR image-token models
   prompt-to-region causal tests

8. JEPAs and World Models
   I-JEPA
   V-JEPA
   VL-JEPA
   Othello
   mazes
   Sudoku
   RL world models
   object permanence

9. Finetuning, LoRA, DoRA, and Misalignment
   LoRA from scratch
   DoRA from scratch
   LoRA vs full FT
   intruder dimensions
   rank-1 adapter directions
   refusal and alignment adapters
   safe emergent-misalignment proxies
   VLM and diffusion adapters

10. Alignment Interpretability
   refusal direction replication
   multi-refusal geometry
   hallucination/truthfulness directions
   CoT faithfulness
   white-box monitors
   unlearning/model editing

11. SHAPley and Attribution Baselines
   Shapley values
   interaction values
   TokenSHAP
   VLM SHAP
   SHAP vs patching
   failure cases

12. Training Dynamics and Developmental Interpretability
   checkpoint archaeology
   grokking
   induction emergence
   LoRA phase transitions
   fine-tuning diffs
   capability emergence

13. Capstone Research Sprint
   one claim
   one model family
   one strong baseline
   one causal intervention
   one OOD test
   one negative control
   one paper-style writeup
```

---

# Final acceptance criteria

A notebook is not accepted unless it has:

```text
1. ARENA-style exposition
2. implementation from scratch where feasible
3. pretrained-weight loading where relevant
4. shape/unit tests
5. local 24GB VRAM report
6. at least one baseline
7. at least one negative control
8. at least one causal intervention where applicable
9. at least one OOD/generalization check
10. a "what would make this result useless?" section
```

For direction/geometry notebooks specifically:

```text
PCA alone is not enough.
UMAP alone is not enough.
t-SNE alone is not enough.
A pretty cluster is not enough.
A max-activating-example collage is not enough.
A feature name from an LLM is not enough.
```

The notebook only passes if:

```text
the geometry predicts held-out labels
the geometry survives controls
the direction has causal effect
the result is stable across seeds/prompts
the plot does not collapse into white noise under fair tests
```

---

# The last cornerstone I would add

One final thing: add a small but serious notebook called:

```text
[0.4] How to Know When an Interpretability Result Is Fake
```

It should teach:

```text
label leakage
cherry-picked examples
post-hoc storytelling
UMAP hallucinations
t-SNE seed instability
probe overfitting
subspace patching illusions
SAE feature absorption
SAE feature splitting
random-direction steering
self-repair / Hydra effects
OOD patching artifacts
text-prior confounds in VLMs
image-region confounds in VLMs
LoRA matching accuracy but not mechanism
```

This is the notebook that protects the whole course from becoming vibes.

---

# Final verdict

Yes: the final amended ARENA extension should include the refusal-direction replication, PCA/SVD/UMAP/t-SNE representation geometry, days/months/space/time manifolds, VLM clothes/object/color geometry, diffusion/image-generation directions, and LoRA/DoRA interpretability for alignment and misalignment.

The course is now not just:

```text
ARENA + newer papers
```

It is:

```text
ARENA preserved
+
a local-first frontier mechanistic-interpretability lab
+
VLM-first multimodal interpretability
+
direction/geometry/circuit/SAE/LoRA/world-model methodology
+
strong verification culture
+
paper-style research capstones
```

The most important final amendment is:

**Every cool visual or direction must earn its keep.**
It must predict, intervene, transfer, or fail honestly. Otherwise it is a pretty picture, not mechanistic interpretability.

[1]: https://arxiv.org/abs/2406.11717?utm_source=chatgpt.com "Refusal in Language Models Is Mediated by a Single Direction"
[2]: https://arxiv.org/abs/2602.02132?utm_source=chatgpt.com "There Is More to Refusal in Large Language Models than a Single Direction"
[3]: https://arxiv.org/abs/2310.02207?utm_source=chatgpt.com "Language Models Represent Space and Time"
[4]: https://arxiv.org/html/2405.14860v2?utm_source=chatgpt.com "Not All Language Model Features Are Linear"
[5]: https://arxiv.org/html/2510.01025v1?utm_source=chatgpt.com "Shape Happens: Automatic Feature Manifold Discovery in ..."
[6]: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Self-Discovering_Interpretable_Diffusion_Latent_Directions_for_Responsible_Text-to-Image_Generation_CVPR_2024_paper.pdf?utm_source=chatgpt.com "Self-Discovering Interpretable Diffusion Latent Directions for ..."
[7]: https://machinelearning.apple.com/research/transporting-activations?utm_source=chatgpt.com "Controlling Language and Diffusion Models by ..."
[8]: https://arxiv.org/html/2504.15473v1?utm_source=chatgpt.com "Emergence and Evolution of Interpretable Concepts in ..."
[9]: https://arxiv.org/abs/2402.09353?utm_source=chatgpt.com "DoRA: Weight-Decomposed Low-Rank Adaptation"
[10]: https://github.com/NVlabs/DoRA?utm_source=chatgpt.com "NVlabs/DoRA: [ICML2024 (Oral)] Official PyTorch ..."
[11]: https://arxiv.org/abs/2405.09673?utm_source=chatgpt.com "LoRA Learns Less and Forgets Less"
[12]: https://arxiv.org/abs/2410.21228?utm_source=chatgpt.com "LoRA vs Full Fine-tuning: An Illusion of Equivalence"
[13]: https://arxiv.org/html/2506.11618v2?utm_source=chatgpt.com "Convergent Linear Representations of Emergent ..."
[14]: https://learn.arena.education/chapter4_alignment_science/1_emergent_misalignment/?utm_source=chatgpt.com "4.1 Emergent Misalignment"


----------------------------------------------------------------


## Appendix: Agent Implementation and Verification Guidance

This appendix is intended to be appended after the ARENA 3.0 extension specification. It defines what it means for an exercise, notebook, artifact, dataset, and replication to be “done correctly.” It is written for implementation agents, reviewers, and future maintainers.

The course must preserve the original ARENA style: implementation-first, readable exposition, expected outputs, tests, solutions, and bonus exploration. The existing source pack treats ARENA, TransformerLens, and NNsight as the hands-on spine: building transformers from scratch, induction heads, IOI, toy superposition, SAEs, and function-vector / steering work.   New material must feel like a natural continuation of that style, not a collection of disconnected survey notebooks.

---

# 1. Core agent contract

Every new exercise must satisfy this contract:

```text
An exercise is accepted only if it has:
  1. a clear learning objective,
  2. a runnable implementation task,
  3. a verification target,
  4. at least one baseline or negative control,
  5. expected outputs,
  6. reproducibility metadata,
  7. local 24GB GPU feasibility, unless explicitly marked optional,
  8. a short “what would make this result bogus?” section.
```

Agents must not treat a good-looking visualization, attention map, UMAP, feature dashboard, or generated image as success by itself. A result must predict, intervene, transfer, or fail honestly.

The course should explicitly follow the research-process philosophy from the source pack: explore quickly, then move to skeptical hypothesis testing, then distill claims with strong evidence.

---

# 2. Ground-truth tiers

Not every exercise can have the same kind of ground truth. Agents must label each notebook with a `GT_TIER`.

## GT-0: Exact mathematical ground truth

Use when the model or task is fully specified.

Examples:

```text
Boolean circuits
modular addition
finite-state automata
RASP / Tracr-compiled models
toy attention-only transformers
toy diffusion noising process
toy Mamba selective-scan recurrence
toy JEPA dynamics
toy VLM colored-shapes scenes
toy Othello / maze / Sudoku labels
```

Acceptance:

```text
implementation exactly matches analytic result
unit tests pass on edge cases
random controls fail
known-circuit / known-variable recovery succeeds
```

This tier is ideal for teaching new methods. The open-problems source explicitly recommends model organisms and benchmarks with known ground-truth explanations, including models created by compiling simple programs into model weights.

---

## GT-1: Reference implementation parity

Use when the exercise implements a known architecture or algorithm.

Examples:

```text
Gemma block
Mamba selective scan
LoRA / DoRA merge
Diffusion sampler
CLIP contrastive loss
PaliGemma / Qwen-VL preprocessing
V-JEPA feature extraction
SAE forward pass
JumpReLU / TopK / Gated SAE
EAP-IG attribution
SHAP / Shapley interaction algorithms
```

Acceptance:

```text
custom implementation matches official implementation
tensor shapes match documented shapes
logits or embeddings match within tolerance
generation is deterministic at fixed seed where applicable
merged/unmerged adapter outputs match
parallel/recurrent algorithms match numerically
```

Agents must save:

```text
reference_output.safetensors
student_output.safetensors
parity_report.json
```

---

## GT-2: Published replication target

Use when reproducing a paper result.

Examples:

```text
IOI circuit
refusal direction
Sparse Feature Circuits
Gemma Scope feature analysis
Othello-GPT linear board state
Mamba induction/state-tracking task
CLIP typographic attacks
DAAM prompt-to-region maps
LoRA-vs-full-finetuning spectra
```

Acceptance:

```text
same qualitative phenomenon appears
same direction of effect appears
metrics fall within expected range
paper’s main figure can be approximately recreated
known limitations are documented
```

Do not require exact numerical equality unless the original repo provides deterministic expected values. Many replications are not bitwise stable.

---

## GT-3: Controlled proxy ground truth

Use for real models where true mechanisms are unknown, but labels are controlled.

Examples:

```text
VLM object/color/spatial/OCR synthetic scenes
VLM clothing geometry
image-generation concept directions
refusal/safety prompt categories
hallucination/arbitration datasets
LoRA-induced behavior drift
JEPA object-permanence datasets
```

Acceptance:

```text
labels are known from generation metadata
held-out split passes
counterfactual split passes
random-label split fails
text-only / image-only baselines are included where relevant
causal intervention changes target behavior
```

This is the main tier for VLM work.

---

## GT-4: Exploratory-only

Use when the goal is surface-area building, not a claim.

Examples:

```text
UMAP of new embedding space
feature-dashboard browsing
manual SAE feature exploration
qualitative generated-image grids
unlabeled VLM hallucination case studies
```

Acceptance:

```text
notebook runs
artifacts are saved
interesting observations are logged
no strong mechanistic claim is made
follow-up hypotheses are listed
```

GT-4 notebooks must not be graded as successful mechanistic explanations. They are exploration tools only.

---

# 3. Notebook style requirements

Each notebook must follow the ARENA rhythm:

```text
1. Motivation
2. Learning objectives
3. Setup
4. Tiny / toy implementation
5. Tests
6. Real model or published artifact
7. Interpretation / analysis
8. Verification
9. Bonus exercises
10. Research extensions
```

Use ARENA-style exercise blocks:

```text
### Exercise - implement `function_name`

> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend 10–20 minutes on this exercise.
```

Each exercise should have:

```text
- function stub
- docstring
- shape comments
- tests immediately after implementation
- expected output dropdown
- solution dropdown
- common bugs
```

This mirrors the original ARENA style, including expected-output dropdowns, difficulty/importance metadata, and implementation cells.  The IOI notebook structure is the canonical example: model setup, logit attribution, activation patching, path patching, paper replication, and bonus work.

---

# 4. Required files per notebook

Each new notebook directory must contain:

```text
chapter_X_Y/
  X_Y_title.ipynb
  X_Y_title_solutions.ipynb
  tests/
    test_X_Y.py
  assets/
    expected_outputs/
      smoke_test.json
      reference_metrics.json
      expected_plot.png        # only if plot has deterministic structure
    diagrams/
  data/
    README.md                  # never large raw data committed
  artifacts.lock.yml
  verification_report.schema.json
  README.md
```

`artifacts.lock.yml` is mandatory.

Example:

```yaml
notebook_id: "5.4_visual_token_flow"
gt_tier: "GT-3"
required_gpu_gb: 16
max_allowed_gpu_gb: 24
models:
  - id: "google/paligemma2-3b-mix-224"
    source: "huggingface"
    revision: "<commit hash>"
    precision: "bf16"
    gated: true
datasets:
  - id: "synthetic_vlm_colored_shapes_v1"
    source: "generated"
    seed: 12345
expected_metrics:
  object_probe_auc_min: 0.90
  color_probe_auc_min: 0.90
  background_patch_effect_max: 0.20
  object_patch_effect_min: 0.50
controls:
  - random_region_patch
  - text_only_baseline
  - image_only_baseline
  - label_permutation
```

Agents must pin exact revisions, seeds, and package versions. Do not depend on floating `main` branches.

---

# 5. Universal verification report

Each notebook must end by writing:

```text
verification_report.json
```

Schema:

```json
{
  "notebook_id": "...",
  "date_run": "...",
  "git_commit": "...",
  "gt_tier": "...",
  "gpu_name": "...",
  "peak_vram_gb": 0.0,
  "wall_clock_seconds": 0.0,
  "models": [],
  "datasets": [],
  "tests_passed": true,
  "metrics": {},
  "baselines": {},
  "negative_controls": {},
  "ood_tests": {},
  "known_failures": [],
  "safety_notes": [],
  "accepted": true
}
```

A notebook with no `verification_report.json` is incomplete.

---

# 6. Global acceptance checklist

A reviewer should be able to answer “yes” to all of these:

```text
Does the notebook run top-to-bottom on a fresh machine?
Does it run within the declared 24GB VRAM budget?
Does it avoid hidden API dependencies unless explicitly optional?
Does it pin models, data, and revisions?
Does it have unit tests?
Does it have at least one baseline?
Does it have at least one negative control?
Does it distinguish exploration from evidence?
Does it save reproducible artifacts?
Does it include expected outputs?
Does it explain failure modes?
```

For interpretability work, add:

```text
Does the method predict held-out activations or behavior?
Does an intervention change the model in the predicted direction?
Does a random intervention fail?
Does a label-shuffled / prompt-shuffled control fail?
Does the result survive OOD examples?
Does the notebook say what result would falsify the hypothesis?
```

The source pack emphasizes that interpretability hypotheses need validation, stress tests, benchmarks, average/worst-case evaluation, and known-ground-truth settings rather than plausible stories.

---

# 7. Visualization acceptance rules

Plots are allowed only as exploration unless backed by metrics.

## PCA / UMAP / t-SNE

Required controls:

```text
run at least 5 random seeds
run at least 3 hyperparameter settings
include PCA baseline
include random-label control
include random-token / random-image control
include held-out nearest-neighbor or probe metric
report trustworthiness or neighborhood-preservation metric
```

Required metrics:

```text
kNN accuracy
linear probe accuracy
silhouette score where appropriate
bootstrap stability
Procrustes alignment across seeds
OOD split performance
```

A UMAP is accepted only if it supports a predictive or causal claim. A pretty cluster is not evidence.

## Attention maps / saliency maps / cross-attention maps

Required controls:

```text
random prompt token
random image region
counterfactual prompt
counterfactual image
model-randomization sanity check where feasible
```

Required metrics:

```text
IoU with segmentation mask, if available
pointing-game accuracy, if available
change in target logit / score after patching
comparison to Grad-CAM / DAAM / SHAP baseline where relevant
```

## Generated image grids

Required controls:

```text
fixed seed sweep
random direction
opposite direction
independent classifier score
human-inspection notes marked as qualitative only
```

A generated-image exercise fails if the intervention produces white noise, global collapse, or only cherry-picked examples.

---

# 8. Module-specific acceptance criteria

## 8.1 Refusal direction

Required notebook:

```text
[10.1] Refusal Is Mediated by a Direction
```

Resources:

```text
paper: arxiv:2406.11717
repo: refusal-direction
datasets: sanitized refusal/compliance prompt pairs
```

The paper reports that refusal can be mediated by a one-dimensional subspace across open-source chat models, and the public repo contains code/results for the paper. ([arXiv][1])

Required implementation:

```text
mean-difference direction
layer sweep
position sweep
addition
projection-out / ablation
PCA / SVD of refusal differences
linear probe baseline
random-direction control
label-shuffled control
```

Acceptance:

```text
held-out harmful-vs-harmless activation AUC >= 0.85
adding direction increases refusal on harmless prompts
projecting direction out decreases refusal on sanitized refusal prompts
random direction effect < 25% of true direction effect
label-shuffled direction fails
PC1 variance explained is reported
multi-direction basis compared to 1D direction
capability degradation measured
```

Safety rule:

```text
Do not include procedural harmful completions in notebook outputs.
Use sanitized prompts, refusal labels, and aggregate metrics.
```

---

## 8.2 Sparse Feature Circuits

Required notebook:

```text
[5.6] Sparse Feature Circuits
```

Resources:

```text
paper: Sparse Feature Circuits
repo: saprmarks/feature-circuits
model: Pythia-70M
task: subject–verb agreement
```

The official repo includes code, data, and autoencoders for replicating the experiments, and the paper defines sparse feature circuits as causally implicated subnetworks of interpretable features. ([GitHub][2]) The source pack also highlights Sparse Feature Circuits as a central SAE-circuit method using attribution patching and an integrated-gradients variant.

Required implementation:

```text
load model
load SAEs or small local SAEs
encode activations into SAE features
include SAE error nodes
compute EAP scores
compute EAP-IG / integrated effects
threshold feature graph
measure faithfulness, completeness, minimality
compare to architecture-node circuit
```

Acceptance:

```text
feature graph is sparse
faithfulness curve beats random graph
completeness curve detects omitted important features
EAP-IG better matches exact patching than plain EAP on at least one early-layer case
SAE reconstruction metrics reported
SAE error-node contribution reported
```

SAE metrics are mandatory:

```text
L0
loss recovered
feature density histogram
dead feature fraction
reconstruction KL / loss delta
direct logit attribution
max activating examples
```

These metrics are standard in the source pack’s SAE evaluation discussion.

---

## 8.3 Integrated gradients / integrated effects / EAP-IG

Required notebook:

```text
[5.4] Integrated Effects and EAP-IG
```

Required implementation:

```text
plain gradients
gradient × activation
integrated gradients
activation patching
attribution patching
edge attribution patching
EAP-IG
```

Integrated gradients approximate an integral of gradients along a path between baseline and input; the source pack also groups IG, SHAP, and perturbation methods as core attribution baselines.

Acceptance:

```text
exact patching implemented on small case
IG approximates exact patching better than plain gradient in at least one saturated case
EAP computes all edge scores with two forward passes and one backward pass where applicable
EAP-IG runtime and faithfulness tradeoff reported
random baseline fails
```

---

## 8.4 Representation geometry

Required notebooks:

```text
[3.1] PCA, SVD, UMAP, and t-SNE Are Not Evidence
[3.2] Days, Months, Space, and Time
[3.3] Directions vs Manifolds
```

Acceptance:

```text
PCA/SVD implemented from scratch on small data
UMAP/t-SNE used only as visualization
linear probe / kNN metrics reported
random-label control fails
prompt-template OOD split passes
seed stability reported
causal direction intervention tested where meaningful
```

A result is rejected if:

```text
structure disappears under seed changes
structure appears for random labels
only UMAP shows the effect
no quantitative metric supports the plot
```

---

## 8.5 VLM interpretability

Required notebooks:

```text
[6.1] CLIP / SigLIP from Scratch
[6.2] Mini VLM from Scratch
[6.3] Visual Token Flow
[6.4] Object Hallucination and Modality Arbitration
[6.5] VLM Feature Geometry: Clothes, Objects, Colors, Styles
[6.6] Multimodal SAEs and Crosscoders
[6.7] VLM SHAP vs Causal Patching
```

Start with synthetic controlled data before real VLMs.

Minimum synthetic dataset schema:

```json
{
  "image_id": "...",
  "objects": [
    {
      "shape": "cube",
      "color": "red",
      "material": "matte",
      "x": 0.25,
      "y": 0.70,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "question": "What color is the cube?",
  "answer": "red",
  "counterfactual_answer": "blue",
  "spurious_text": null
}
```

Acceptance:

```text
vision-only baseline
text-only baseline
image+text model
object-region patch
background-region patch
visual-token patch
full-sequence patch
random-region control
counterfactual image control
held-out object/color/style split
```

For real VLMs, acceptable local targets include PaliGemma, Qwen2.5-VL, LLaVA-OneVision, and Molmo. PaliGemma combines a SigLIP vision encoder with a Gemma decoder, Qwen2.5-VL has a 3B Instruct checkpoint, LLaVA-OneVision has Qwen2-based 7B checkpoints, and Molmo-7B-D uses a Qwen2-7B language model with an OpenAI CLIP vision backbone. ([Hugging Face][3])

---

## 8.6 CLIP / SigLIP

Required implementation:

```text
contrastive image-text loss
normalized embeddings
temperature / logit scale
zero-shot classification
image-text retrieval
typographic attack test
SAE on image embeddings or patch tokens
```

Acceptance:

```text
toy CLIP trains above random retrieval
OpenCLIP / SigLIP feature extraction runs locally
retrieval recall@k reported
zero-shot accuracy reported
typographic attack reduces performance
SAE feature ablation changes target logits
random feature ablation does less
```

OpenCLIP provides CLIP checkpoints through Hugging Face, and SigLIP / SigLIP2 checkpoints are available from Google; SigLIP replaces CLIP’s softmax contrastive loss with a sigmoid loss, while SigLIP2 emphasizes improved semantic understanding, localization, and dense features. ([GitHub][4])

---

## 8.7 Image-generation interpretability

Required notebooks:

```text
[7.1] Stable Diffusion Attention Maps
[7.2] Denoising-Time Causal Patching
[7.3] Diffusion Concept Directions
[7.4] Diffusion LoRA Interpretability
[7.5] Toy AR Image-Token Transformer
```

Required models:

```text
Stable Diffusion 1.5
SDXL optional
FLUX.1-schnell optional / stretch
toy diffusion model
toy AR image-token model
```

Stable Diffusion 1.5 and SDXL are available on Hugging Face, and FLUX.1-schnell is a 12B rectified-flow transformer text-to-image model. ([Hugging Face][5]) DAAM is the required baseline for Stable Diffusion prompt-to-region attribution. ([GitHub][6])

Acceptance:

```text
fixed seed generation
cross-attention maps saved
DAAM baseline included
segmentation / mask IoU when possible
token ablation changes corresponding region
random token ablation does less
concept direction changes classifier score
image quality degradation measured
white-noise outputs marked as failure
```

---

## 8.8 LoRA / DoRA / PEFT interpretability

Required notebooks:

```text
[9.1] LoRA from Scratch
[9.2] DoRA from Scratch
[9.3] LoRA vs Full Finetuning
[9.4] Adapter Geometry and SVD
[9.5] Safe LoRA Misalignment Proxy
[9.6] VLM LoRA Interpretability
[9.7] Diffusion LoRA Interpretability
```

LoRA and PEFT are supported by Hugging Face tooling, while DoRA decomposes pretrained weights into magnitude and direction and uses LoRA for directional updates. ([Hugging Face][7])

Required implementation:

```text
LoRA forward pass
adapter merge
adapter unmerge
rank check
DoRA magnitude/direction decomposition
SVD of weight deltas
activation shift measurement
adapter-induced direction
projection-out intervention
```

Acceptance:

```text
merged and unmerged outputs match within tolerance
only adapter parameters train
rank(update) <= declared rank
DoRA reconstructs base weight decomposition
adapter improves target task
OOD behavior measured
full finetuning comparison included on toy task
random adapter delta fails
```

For misalignment, use safe proxies only:

```text
excessive refusal
overconfidence
sycophancy on harmless opinions
JSON-only behavior
rhyme-only behavior
verbosity drift
```

Do not train or publish harmful adapters.

---

## 8.9 JEPA and world models

Required notebooks:

```text
[8.1] I-JEPA from Scratch
[8.2] V-JEPA 2 Feature Extraction
[8.3] Toy Action-Conditioned JEPA
[8.4] Othello-GPT World Model
[8.5] Maze World Models
[8.6] Sudoku / Constraint World Models
[8.7] JEPA Object Permanence
```

I-JEPA predicts representations of image parts from other image parts, while V-JEPA 2 is released as a video-understanding / world-model family with HF checkpoints such as `facebook/vjepa2-vitl-fpc64-256`. ([Hugging Face][8])

Acceptance:

```text
toy latent-prediction loss decreases
collapse diagnostics pass
linear state probes beat random baseline
latent rollout predicts future state above baseline
action-shuffled baseline fails
object-permanence feature remains active through occlusion
causal latent patch changes predicted future
```

---

## 8.10 SHAP / Shapley attribution

Required notebooks:

```text
[11.1] Exact Shapley on Ground-Truth Games
[11.2] KernelSHAP and PartitionSHAP
[11.3] Shapley Interactions with shapiq
[11.4] TokenSHAP / TokenShapley
[11.5] VLM Modality and Region SHAP
[11.6] SHAP vs Activation Patching
```

Use exact ground-truth games first. `shapiq` provides algorithms and benchmarks for Shapley interaction values, TokenSHAP implements Monte Carlo token attribution, and SAGE provides global Shapley-style feature importance. ([GitHub][9])

Acceptance:

```text
exact Shapley matches brute force on small games
KernelSHAP approximates exact values within tolerance
interaction methods recover XOR/parity interactions
TokenSHAP identifies causal tokens on synthetic tasks
SHAP fails honestly on interaction-heavy cases
activation patching comparison included
```

---

# 9. Artifact registry: required starter resources

Agents should create and maintain:

```text
docs/artifact_registry.md
docs/artifact_registry.csv
docs/artifact_registry.lock.yml
```

Columns:

```text
name
type
provider
repo_or_source_id
license
gated
revision
local_status
max_vram_gb
used_in_notebooks
gt_tier
notes
```

Statuses:

```text
REQUIRED
OPTIONAL
STRETCH
READ_ONLY
GENERATED_BY_COURSE
DEPRECATED
WAIT_FOR_WEIGHTS
```

---

# 10. Core tooling registry

| Resource                                |                      Type |   Status | Use                                                 |
| --------------------------------------- | ------------------------: | -------: | --------------------------------------------------- |
| `callummcdougall/ARENA_3.0`             |               course repo | REQUIRED | preserve original course, fork append-only          |
| `TransformerLensOrg/TransformerLens`    |       mech-interp library | REQUIRED | hooks, activation cache, GPT-style model organisms  |
| `ndif-team/nnsight` / NNsight           |      intervention tooling | REQUIRED | activation editing, remote/local intervention graph |
| `SAELens`                               |               SAE tooling | REQUIRED | SAE loading, training, visualization                |
| `dictionary-learning`                   |  SAE / dictionary tooling | REQUIRED | Sparse Feature Circuits, experimental SAE variants  |
| `pyvene`                                |      intervention tooling | REQUIRED | causal interventions, trainable interventions       |
| `Tracr` / RASP                          | GT model organism tooling | REQUIRED | compiled ground-truth transformer programs          |
| `CircuitsVis`, `sae-vis`, `Neuronpedia` |             visualization | REQUIRED | ARENA-style interpretability displays               |
| `captum`, `shap`, `shapiq`, `inseq`     |               attribution | REQUIRED | IG, SHAP, Shapley interactions, token attribution   |
| `peft`, `bitsandbytes`, `accelerate`    |       PEFT / quantization | REQUIRED | LoRA, DoRA, QLoRA, local inference                  |
| `diffusers`                             |          image generation | REQUIRED | Stable Diffusion / SDXL / DiT experiments           |

The source pack already identifies TransformerLens, NNsight, Pyvene, SAELens, dictionary-learning, Neuronpedia, sae-vis, Tracr, and attribution libraries as central interpretability tooling.

---

# 11. Core language-model artifacts

| Resource                                        |           Status | Use                                           |
| ----------------------------------------------- | ---------------: | --------------------------------------------- |
| `gpt2-small` / TransformerLens GPT-2 small      |         REQUIRED | IOI, induction heads, patching, SAEs          |
| `EleutherAI/pythia-70m`                         |         REQUIRED | Sparse Feature Circuits, small real LM        |
| `EleutherAI/pythia-14m-deduped`                 |         REQUIRED | tiny LM smoke tests                           |
| `roneneldan/TinyStories-1M` / `TinyStories-3M`  |         REQUIRED | small generation, finetuning, LoRA            |
| `NeelNanda/Othello-GPT-Transformer-Lens`        |         REQUIRED | world-model interpretability                  |
| `google/gemma-3-1b-it` / `google/gemma-3-4b-it` |         REQUIRED | modern transformer, refusal, Gemma Scope      |
| `google/gemma-scope-2-*`                        |         REQUIRED | Gemma SAEs/transcoders                        |
| `state-spaces/mamba-130m`                       |         REQUIRED | Mamba implementation parity                   |
| `state-spaces/mamba2-130m`                      |         REQUIRED | Mamba-2 comparison                            |
| `google/diffusiongemma-26B-A4B-it`              | OPTIONAL/STRETCH | diffusion LM inference and denoising analysis |

Pythia is intended for interpretability/scaling research, TinyStories provides small language-model targets, and Othello-GPT is already ported for TransformerLens-style work. ([Hugging Face][10]) Gemma 3 and Gemma Scope 2 provide modern model and SAE/transcoder artifacts, while Mamba and Mamba-2 provide official small checkpoints. ([Hugging Face][11])

---

# 12. SAE and feature artifacts

| Resource                                   |                        Status | Use                                    |
| ------------------------------------------ | ----------------------------: | -------------------------------------- |
| `jbloom/GPT2-Small-SAEs`                   |                      REQUIRED | GPT-2 small residual-stream SAEs       |
| `jacobcd52/gpt2-small-sparse-autoencoders` |                      REQUIRED | alternative GPT-2 SAE set              |
| `google/gemma-scope-2-4b-it`               |                      REQUIRED | Gemma 3 4B IT SAEs/transcoders         |
| `google/gemma-scope-2-1b-it`               |                      REQUIRED | smaller Gemma Scope target             |
| `google/gemma-scope-2-270m*`               | REQUIRED if available locally | tiny Gemma Scope target                |
| Neuronpedia feature pages                  |                      REQUIRED | feature inspection / expected displays |
| `saprmarks/feature-circuits` artifacts     |                      REQUIRED | Sparse Feature Circuits replication    |

GPT-2 SAE artifacts and Gemma Scope artifacts should be used before training large SAEs from scratch. Gemma Scope 2 includes SAEs/transcoders across Gemma 3 layers, while GPT-2 SAE sets are available on Hugging Face. ([Hugging Face][12])

---

# 13. VLM / vision / multimodal artifacts

| Resource                                   |                          Status | Use                             |
| ------------------------------------------ | ------------------------------: | ------------------------------- |
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`    |                        REQUIRED | CLIP baseline                   |
| `laion/CLIP-ViT-L-14-laion2B-s32B-b82K`    |                        OPTIONAL | larger CLIP                     |
| `google/siglip-base-patch16-224`           |                        REQUIRED | SigLIP baseline                 |
| `google/siglip2-base-patch16-224`          |                        REQUIRED | dense/localized visual features |
| `google/paligemma-3b-pt-224`               |                        REQUIRED | base VLM architecture           |
| `google/paligemma2-3b-mix-224`             |                        REQUIRED | practical VQA/caption/OCR tasks |
| `Qwen/Qwen2.5-VL-3B-Instruct`              |                        REQUIRED | local VLM                       |
| `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`          | REQUIRED if quantization needed | low-VRAM VLM                    |
| `llava-hf/llava-onevision-qwen2-7b-si-hf`  |                        OPTIONAL | LLaVA-OneVision                 |
| `allenai/Molmo-7B-D-0924`                  |                        OPTIONAL | Molmo VLM                       |
| synthetic colored-shapes VLM dataset       |                        REQUIRED | GT-3 controlled VLM tests       |
| synthetic OCR / typographic attack dataset |                        REQUIRED | modality arbitration            |

PaliGemma, Qwen2.5-VL, LLaVA-OneVision, and Molmo are appropriate real-model VLM targets; CLIP/OpenCLIP and SigLIP/SigLIP2 are the required CLIP-like baselines. ([Hugging Face][13])

---

# 14. VLM and vision datasets

| Dataset                          |   Status | Use                                 |
| -------------------------------- | -------: | ----------------------------------- |
| synthetic colored shapes         | REQUIRED | object/color/count/spatial GT       |
| CLEVR                            | REQUIRED | compositional visual reasoning      |
| Super-CLEVR                      | OPTIONAL | robustness / domain shift           |
| VQAv2                            | REQUIRED | real VQA baseline                   |
| Winoground                       | REQUIRED | compositional image-text matching   |
| POPE                             | REQUIRED | object hallucination                |
| Fashion-MNIST                    | REQUIRED | tiny clothing geometry              |
| Fashionpedia                     | REQUIRED | clothing objects, parts, attributes |
| DeepFashion / multimodal fashion | OPTIONAL | clothes/caption/style geometry      |
| fashion-product-images-small     | OPTIONAL | garment type/color/style metadata   |

CLEVR is explicitly designed as a diagnostic visual-reasoning dataset with detailed annotations, Winoground tests compositional image-text understanding, POPE targets object hallucination, and Fashionpedia / Fashion-MNIST provide clothing labels suitable for VLM geometry. ([Computer Science][14])

---

# 15. Image-generation artifacts

| Resource                                      |                  Status | Use                                 |
| --------------------------------------------- | ----------------------: | ----------------------------------- |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` |                REQUIRED | diffusion interpretability          |
| `stabilityai/stable-diffusion-xl-base-1.0`    |                OPTIONAL | larger diffusion baseline           |
| `black-forest-labs/FLUX.1-schnell`            |                 STRETCH | rectified-flow transformer          |
| DAAM repo                                     |                REQUIRED | prompt-to-region attribution        |
| generated prompt-pair dataset                 |                REQUIRED | concept direction tests             |
| generated mask annotations                    | REQUIRED where possible | IoU evaluation                      |
| course-trained diffusion LoRAs                |     GENERATED_BY_COURSE | style/concept LoRA interpretability |
| toy VQ tokenizer + AR image transformer       |     GENERATED_BY_COURSE | AR image-generation GT              |

Do not rely on random public LoRAs for required exercises. Generate small course LoRAs where possible so labels, prompts, seeds, and intended effects are known.

---

# 16. JEPA and world-model artifacts

| Resource                         |           Status | Use                                    |
| -------------------------------- | ---------------: | -------------------------------------- |
| `facebook/ijepa_vith16_1k`       |         REQUIRED | I-JEPA feature extraction              |
| `facebook/vjepa2-vitl-fpc64-256` |         REQUIRED | V-JEPA 2 local target                  |
| `facebook/vjepa2-vitg-fpc64-256` | OPTIONAL/STRETCH | larger V-JEPA 2                        |
| `facebook/jepa-wms`              |         OPTIONAL | JEPA world-model checkpoint            |
| synthetic bouncing-balls         |         REQUIRED | object permanence, velocity, collision |
| synthetic gridworld videos       |         REQUIRED | action-conditioned latent dynamics     |
| Othello-GPT data                 |         REQUIRED | board-state world model                |
| generated mazes                  |         REQUIRED | planning/world-model probes            |
| generated Sudoku traces          |         REQUIRED | constraint-world model                 |

The V-JEPA 2 ViT-L checkpoint is the required local target, while larger ViT-g checkpoints are stretch. ([Hugging Face][15])

---

# 17. Refusal / alignment / safety datasets

| Resource                                        |                       Status | Use                                |
| ----------------------------------------------- | ---------------------------: | ---------------------------------- |
| `josephmayo/refusal-compliance-pairs`           |                     REQUIRED | refusal direction, sanitized pairs |
| `libr-ai/do-not-answer`                         |                     REQUIRED | refusal / abstention evals         |
| `Anthropic/hh-rlhf`                             | OPTIONAL / SAFETY-RESTRICTED | harmlessness labels only           |
| course-generated harmless/refusal proxy prompts |                     REQUIRED | safe GT-3 refusal experiments      |
| course-generated sycophancy proxy prompts       |                     REQUIRED | LoRA misalignment proxy            |
| course-generated overconfidence proxy prompts   |                     REQUIRED | LoRA/DoRA behavior drift           |

Use only labels and sanitized prompt forms when needed. Do not include procedural harmful completions in notebooks, tests, or expected outputs. HH-RLHF and Do-Not-Answer are useful but safety-restricted; agents should prefer sanitized generated prompt pairs for required exercises. ([Hugging Face][16])

---

# 18. Text datasets

| Dataset                                  |   Status | Use                                  |
| ---------------------------------------- | -------: | ------------------------------------ |
| `stas/openwebtext-10k`                   | REQUIRED | smoke tests, activation caches       |
| `roneneldan/TinyStories`                 | REQUIRED | tiny LM training                     |
| The Pile / Pythia data access            | OPTIONAL | Pythia distribution checks           |
| generated IOI prompts                    | REQUIRED | IOI/path patching                    |
| generated subject–verb agreement prompts | REQUIRED | Sparse Feature Circuits              |
| generated days/months/time/space prompts | REQUIRED | geometry                             |
| generated factual recall prompts         | REQUIRED | patching, hallucination, attribution |

OpenWebText-10k is a lightweight test subset, TinyStories supports small model training, and Pythia provides a controlled scaling/interpretability model family. ([Hugging Face][17])

---

# 19. PEFT / LoRA artifacts

Required adapters should usually be generated by the course, not downloaded.

| Artifact                 |              Status | Use                               |
| ------------------------ | ------------------: | --------------------------------- |
| rank-1 sentiment LoRA    | GENERATED_BY_COURSE | direction geometry                |
| rank-4 JSON-format LoRA  | GENERATED_BY_COURSE | adapter behavior                  |
| over-refusal LoRA proxy  | GENERATED_BY_COURSE | safe alignment drift              |
| sycophancy proxy LoRA    | GENERATED_BY_COURSE | safe misalignment proxy           |
| VLM color-answering LoRA | GENERATED_BY_COURSE | VLM LoRA interpretability         |
| diffusion style LoRA     | GENERATED_BY_COURSE | image-generation adapter geometry |
| public PEFT examples     |            OPTIONAL | API sanity only                   |

Acceptance for generated adapters:

```text
training script included
data generation script included
seed fixed
adapter weights saved
base model specified
rank specified
merge/unmerge verified
behavior shift measured
OOD behavior measured
```

---

# 20. Prompt artifact format

All prompt sets must be stored as JSONL.

Schema:

```json
{
  "id": "refusal_0001",
  "task": "refusal_direction",
  "prompt": "...",
  "label": "refusal_expected",
  "safe_to_display": true,
  "contains_sensitive_content": false,
  "target_token": null,
  "counterfactual_id": "refusal_0001_cf",
  "metadata": {
    "template": "template_03",
    "split": "train",
    "source": "generated",
    "seed": 123
  }
}
```

For safety-sensitive prompts:

```json
{
  "safe_to_display": false,
  "prompt": "[REDACTED]",
  "prompt_hash": "...",
  "label": "refusal_expected"
}
```

Agents must never put unsafe procedural content into solution cells, expected outputs, or screenshots.

---

# 21. Dataset generation rules

Generated datasets are preferred for ground truth. Every generator must save:

```text
generator script
seed
schema
train/val/test split
OOD split
counterfactual mapping
label function
example preview
```

Minimum splits:

```text
train: 60%
validation: 20%
test: 20%
OOD: separate, not sampled from same templates
```

Required controls:

```text
label permutation
template permutation
spurious-correlation split
counterfactual split
random-input split
```

For VLM/image datasets, save:

```text
image
mask
bounding boxes
scene graph
question
answer
counterfactual image id
counterfactual answer
```

---

# 22. Quality gates

Agents should implement the following automated gates.

## Gate 1: runnability

```text
pytest tests pass
notebook executes top-to-bottom
no cell requires manual editing
no hidden API key required for required path
```

## Gate 2: local feasibility

```text
peak VRAM <= declared max
runtime for smoke test <= reasonable notebook target
batch size configurable
CPU fallback for unit tests where possible
```

## Gate 3: correctness

```text
GT-0: exact expected result
GT-1: reference parity
GT-2: published replication target
GT-3: controlled proxy metric
GT-4: no strong claim
```

## Gate 4: interpretability evidence

```text
correlational evidence included
causal intervention included where applicable
baseline included
negative control included
OOD check included
```

## Gate 5: exposition quality

```text
student knows why the exercise matters
student knows what to implement
student sees expected output
student sees common bugs
student can open solution
bonus section exists
```

---

# 23. “Looks good” criteria

For every notebook, agents should include a small expected-output block.

Examples:

```text
A successful refusal-direction notebook should show:
  - layer sweep with a clear band of high separation,
  - addition increases refusal,
  - ablation decreases refusal,
  - random controls near zero,
  - PCA plot maybe helpful, but metrics decisive.

A successful VLM visual-token notebook should show:
  - object-region patch effect > background patch effect,
  - text-only baseline failing on image-dependent questions,
  - image-only baseline failing on language-dependent questions,
  - layer/token heatmap with stable causal band.

A successful diffusion notebook should show:
  - word-region map roughly aligned with object mask,
  - token ablation changing the corresponding image region,
  - random-token ablation not doing so,
  - no global collapse / white noise.

A successful LoRA notebook should show:
  - merged/unmerged logits matching,
  - low-rank update spectrum,
  - adapter-induced behavior shift,
  - OOD side-effect measurement.

A successful geometry notebook should show:
  - quantitative separability,
  - stable visualization,
  - random-label failure,
  - held-out prediction.
```

If the expected result is “white noise,” the notebook must say this is a negative result and explain why. Otherwise, white noise is failure.

---

# 24. Safety and misuse constraints

Agents must avoid:

```text
jailbreak instructions
harmful procedural examples
public release of unsafe adapters
unsafe image-generation LoRAs
private/personal data
copyrighted dataset redistribution
raw gated-model weights committed to repo
```

Refusal-direction, misalignment, red-teaming, and hallucination notebooks must operate through:

```text
sanitized prompts
labels
aggregate metrics
toy proxies
safe behavior categories
```

The course may teach the existence and mechanics of refusal directions, but not provide an operational harmful-use playbook.

---

# 25. Final completeness check for the whole extension

The extension is complete only when these milestone checks pass:

```text
Original ARENA preserved unchanged except compatibility patches.

At least one GT-0 notebook exists for:
  - circuit discovery
  - representation geometry
  - SHAP/Shapley
  - world models
  - VLMs
  - diffusion / image generation
  - LoRA/DoRA

At least one GT-1 notebook exists for:
  - Gemma
  - Mamba
  - CLIP/SigLIP
  - VLM loading
  - LoRA/DoRA
  - diffusion model loading
  - JEPA loading

At least one GT-2 notebook exists for:
  - refusal direction
  - Sparse Feature Circuits
  - Othello-GPT
  - IOI / path patching
  - Gemma Scope feature analysis

At least one GT-3 notebook exists for:
  - VLM hallucination / grounding
  - VLM clothing/object/color geometry
  - image-generation concept directions
  - safe LoRA misalignment proxy
  - JEPA object permanence

Every chapter has:
  - tests
  - expected outputs
  - verification report
  - artifact lockfile
  - baseline
  - negative control
```

If all of the above passes, the fork is not merely “ARENA plus some papers.” It is a local-first, ARENA-style, verification-heavy mechanistic interpretability lab course.

[1]: https://arxiv.org/abs/2406.11717?utm_source=chatgpt.com "Refusal in Language Models Is Mediated by a Single ..."
[2]: https://github.com/saprmarks/feature-circuits?utm_source=chatgpt.com "saprmarks/feature-circuits"
[3]: https://huggingface.co/docs/transformers/en/model_doc/paligemma?utm_source=chatgpt.com "PaliGemma"
[4]: https://github.com/mlfoundations/open_clip?utm_source=chatgpt.com "mlfoundations/open_clip: An open source implementation ..."
[5]: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5?utm_source=chatgpt.com "stable-diffusion-v1-5/stable-diffusion-v1-5"
[6]: https://github.com/castorini/daam?utm_source=chatgpt.com "castorini/daam: Diffusion attentive attribution maps for ..."
[7]: https://huggingface.co/docs/peft/package_reference/lora?utm_source=chatgpt.com "LoRA"
[8]: https://huggingface.co/facebook/ijepa_vith16_1k?utm_source=chatgpt.com "facebook/ijepa_vith16_1k"
[9]: https://github.com/mmschlk/shapiq?utm_source=chatgpt.com "shapiq: Shapley Interactions for Machine Learning ..."
[10]: https://huggingface.co/EleutherAI/pythia-70m?utm_source=chatgpt.com "EleutherAI/pythia-70m"
[11]: https://huggingface.co/google/gemma-3-4b-it?utm_source=chatgpt.com "google/gemma-3-4b-it"
[12]: https://huggingface.co/google/gemma-scope-2-4b-it?utm_source=chatgpt.com "google/gemma-scope-2-4b-it"
[13]: https://huggingface.co/google/paligemma-3b-pt-224?utm_source=chatgpt.com "google/paligemma-3b-pt-224"
[14]: https://cs.stanford.edu/people/jcjohns/clevr/?utm_source=chatgpt.com "CLEVR: A Diagnostic Dataset for Compositional Language ..."
[15]: https://huggingface.co/facebook/vjepa2-vitl-fpc64-256?utm_source=chatgpt.com "facebook/vjepa2-vitl-fpc64-256"
[16]: https://huggingface.co/datasets/Anthropic/hh-rlhf?utm_source=chatgpt.com "Anthropic/hh-rlhf · Datasets at Hugging Face"
[17]: https://huggingface.co/datasets/stas/openwebtext-10k?utm_source=chatgpt.com "stas/openwebtext-10k · Datasets at Hugging Face"


----------------------------------------------

## Final tiny add-on: In-Run Data Shapley and SHAPley-vs-mechinterp agreement tests

Add this as a required subsection to the SHAPley / attribution chapter:

```text
[11.7] Data Shapley in One Training Run
[11.8] Do SHAPley and mechanistic interpretability agree?
```

I interpret your “in-run Shapley paper” as **Data Shapley in One Training Run / In-Run Data Shapley**. It should absolutely be included. The paper introduces In-Run Data Shapley as a way to attribute training-data contribution during a single training run, rather than repeatedly retraining on many data subsets; the OpenReview page lists it as an ICLR 2025 Oral, last modified May 18, 2025. ([OpenReview][1]) The project page describes it as a scalable “contribution accountant” for a specific training run, using first- and second-order Taylor approximations plus ghost dot-product / ghost gradient-Hessian-gradient-product tricks. ([Jiachen T. Wang][2]) There is also an official-looking `GhostSuite` codebase for fast per-sample gradient information, intended as a drop-in implementation for computing gradient similarities between validation loss and individual training samples. ([GitHub][3])

This belongs in the course because the uploaded source pack already treats Shapley values, SHAP, Integrated Gradients, training-data attribution, causal interventions, circuit discovery, TransformerLens, and NNsight as adjacent tools that need to be compared rather than worshipped in isolation.

---

# Add required notebook: `[11.7] Data Shapley in One Training Run`

## Core question

```text
Which training examples made this particular trained model behave this way?
```

This is different from the usual mechinterp question:

```text
Which input tokens, activations, features, heads, MLPs, or edges caused this output?
```

That contrast is exactly why the notebook is valuable.

## ARENA-style exercise sequence

```text
Part 1 — Exact Data Shapley on a tiny problem
Part 2 — Truncated / Monte Carlo Data Shapley
Part 3 — In-Run Data Shapley from scratch
Part 4 — Ghost dot-product implementation
Part 5 — Compare to influence functions and TracIn-style attribution
Part 6 — Compare to mechanistic interpretability methods
Part 7 — Data pruning / data debugging mini-project
```

Use tiny, local targets first:

```text
logistic regression on synthetic data
tiny MLP on MNIST / Fashion-MNIST
tiny transformer on synthetic IOI / subject–verb agreement
TinyStories-1M small LM
toy VLM colored-shapes model
```

Do **not** require foundation-model pretraining. The paper’s large-scale motivation is important, but the course implementation should prove the idea on small controlled models.

## Verification

The notebook passes only if:

```text
exact Data Shapley matches brute force on n <= 8 training examples
Monte Carlo Data Shapley converges toward exact values
In-Run first-order scores correlate with exact / MC scores on a tiny training run
negative-value examples are actually harmful under deletion/retraining tests
positive-value examples improve validation loss under addition tests
random data attribution fails
label-shuffled attribution fails
runtime overhead is reported
```

Optional stretch:

```text
second-order In-Run Data Shapley
GhostSuite integration
Adam-aware In-Run Data Shapley
```

The Adam-aware extension should be marked **stretch / frontier**, because a 2026 preprint argues that SGD-based In-Run proxies can diverge under Adam and proposes an Adam-aware variant with a fixed-state approximation and linearized ghost approximation. ([arXiv][4])

---

# Add required notebook: `[11.8] Do SHAPley and mechanistic interpretability agree?`

## Core question

```text
When different attribution methods say “this was important,”
are they pointing to the same causal story?
```

The answer should often be:

```text
sometimes yes, sometimes no, and the disagreement is scientifically useful.
```

## Methods to compare

### Input-level attribution

```text
exact token Shapley
KernelSHAP
PartitionSHAP
TokenSHAP / TokenShapley
Integrated Gradients
gradient × input
leave-one-token-out ablation
contrastive token attribution
```

The primer material explicitly distinguishes gradient-based attribution, perturbation-based attribution, SHAP-style methods, contrastive attribution, and their limitations; it also warns that popular input-attribution methods can disagree or fail to predict counterfactual behavior.

### Training-data attribution

```text
exact Data Shapley
Monte Carlo Data Shapley
In-Run Data Shapley
influence functions
TracIn-style scores
gradient similarity
BM25 / nearest-neighbor retrieval baseline
```

### Mechanistic methods

```text
direct logit attribution
activation patching
path patching
attribution patching
EAP
EAP-IG
Sparse Feature Circuits
SAE feature ablation
transcoder attribution graphs
```

Circuit validation should use faithfulness, completeness, and minimality; the source material explicitly names these as core circuit-validation criteria and also warns that circuit faithfulness can be low or metric-dependent.

---

# Agreement / disagreement protocol

For each task, compute attribution rankings over different “players”:

```text
training examples
input tokens
image regions
model components
SAE features
circuit edges
LoRA directions
```

Then compare them with:

```text
Spearman rank correlation
Kendall tau
top-k overlap
deletion curves
insertion curves
causal patch effect
faithfulness AUC
completeness AUC
random baseline gap
```

The important point: **methods with different player sets must be compared through their consequences**, not just their raw scores.

So the notebook should ask:

```text
If method A says X is important, does removing X damage behavior?
If method B says Y is important, does patching Y recover behavior?
Do A and B identify the same causal bottleneck?
If not, is one method identifying upstream evidence while the other identifies internal computation?
```

---

# Required comparison tasks

## Task 1 — Ground-truth Boolean circuits

Use:

```text
AND
OR
XOR
majority
sparse parity
multiplexer
```

Compare:

```text
exact Shapley
SHAP interactions
Integrated Gradients
activation patching
edge patching
SAE features if using a trained tiny network
```

Expected result:

```text
ordinary Shapley handles additive / simple cases
interaction methods are needed for XOR/parity
mechanistic patching should recover the known causal variables
```

This is the GT-0 sanity check.

---

## Task 2 — IOI / subject–verb agreement

Use:

```text
GPT-2 small or Pythia-70M
IOI prompts
subject–verb agreement prompts
```

Compare:

```text
TokenSHAP says which input tokens matter
activation patching says which components matter
Sparse Feature Circuits says which SAE features / edges matter
In-Run Data Shapley says which training examples, in a tiny trained model, made the behavior emerge
```

Expected result:

```text
input-token methods should highlight names / subject tokens
mechanistic methods should localize heads/features/edges
training-data attribution should highlight examples with matching syntactic structure
```

Disagreement is useful:

```text
TokenSHAP may identify the surface token.
Activation patching may identify the circuit that transports the token.
Sparse Feature Circuits may identify interpretable internal features.
In-Run Data Shapley may identify training examples that taught the model this pattern.
```

---

## Task 3 — VLM object/color grounding

Use:

```text
synthetic colored-shapes VLM dataset
optional clothing dataset
CLIP/SigLIP + tiny VLM
PaliGemma/Qwen-VL-style local model as stretch
```

Compare:

```text
image-region SHAP
text-token SHAP
modality SHAP
visual-token patching
projector patching
VLM SAE feature ablation
CLIP/SigLIP feature directions
```

Expected result:

```text
image-region SHAP should identify the object region
visual-token patching should causally flip answers when object/color evidence is patched
text SHAP should dominate in text-prior or typographic-conflict cases
```

Disagreement cases become research targets:

```text
SHAP says the image region matters, but patching internal visual tokens does not.
Possible interpretation: the VLM did not actually route that evidence through the answer path.

Patching visual tokens flips the answer, but SHAP gives low image attribution.
Possible interpretation: the visual evidence is latent and only becomes causal inside the model.
```

---

## Task 4 — LoRA / DoRA behavior drift

Use:

```text
rank-1 or rank-4 LoRA on a safe proxy behavior
JSON-only behavior
over-refusal proxy
sycophancy proxy
style drift
VLM color-answering LoRA
```

Compare:

```text
In-Run Data Shapley over finetuning examples
influence / gradient-similarity scores
adapter SVD directions
activation steering directions
crosscoder base-vs-LoRA features
behavioral deletion tests
```

Expected result:

```text
data attribution identifies which examples drove the adapter behavior
adapter SVD identifies low-rank parameter directions
activation geometry identifies residual-stream behavior directions
crosscoders identify changed features
```

This is the cleanest bridge between **training data**, **parameter updates**, and **activation mechanisms**.

---

# Required “agreement matrix” artifact

Every SHAP-vs-mechinterp notebook should output:

```text
agreement_matrix.csv
```

Schema:

```csv
task,method_a,method_b,player_type_a,player_type_b,metric,value,interpretation
IOI,TokenSHAP,ActivationPatching,input_token,attention_head,topk_overlap,0.42,"surface tokens agree partly with causal heads"
IOI,EAPIG,SparseFeatureCircuits,edge,sae_feature_edge,faithfulness_auc,0.81,"feature circuit preserves behavior"
VLM,ObjectSHAP,VisualTokenPatching,image_region,visual_token,causal_agreement,0.76,"object-region evidence routes through visual tokens"
LoRA,InRunDataShapley,AdapterSVD,training_example,parameter_direction,rank_corr,0.31,"data influence and parameter direction only weakly aligned"
```

And:

```text
deletion_curves.png
insertion_curves.png
topk_overlap_heatmap.png
method_disagreement_examples.md
```

---

# Add to artifact registry

```yaml
- name: data_shapley_in_one_training_run
  type: paper
  status: REQUIRED
  source: arxiv:2406.11011 / OpenReview HD6bWcj87Y
  used_in:
    - 11.7_data_shapley_in_one_training_run
    - 11.8_shapley_vs_mechinterp_agreement
  local_status: toy_reimplementation_required
  notes: "Do exact tiny Data Shapley first; then In-Run approximation."

- name: GhostSuite
  type: code
  status: OPTIONAL
  source: Jiachen-T-Wang/GhostSuite
  used_in:
    - 11.7_data_shapley_in_one_training_run
  local_status: optional_reference
  notes: "Use after from-scratch implementation; do not hide core ideas behind library calls."

- name: adam_aware_in_run_data_shapley
  type: paper
  status: STRETCH
  source: arxiv:2602.00329
  used_in:
    - 11.7_bonus_adam_aware_in_run_shapley
  local_status: read_and_toy_reimplementation
  notes: "Relevant because most modern training uses Adam/AdamW, not plain SGD."
```

---

# Acceptance criteria

This add-on is complete only if students can answer:

```text
Did SHAPley/token attribution identify the same evidence as activation patching?
Did training-data Shapley identify examples that plausibly taught the mechanism?
Did mechanistic methods identify where the computation happened?
When methods disagreed, was the disagreement explained by different player sets,
different causal notions, or method failure?
```

A notebook passes only if:

```text
exact Shapley is verified on a tiny game
In-Run Data Shapley correlates with exact / MC Data Shapley on a tiny model
SHAP/IG/token attribution is compared against causal patching
feature/circuit methods are compared against input attribution
at least one agreement case and one disagreement case are shown
disagreements are tested, not hand-waved
```

Final addition to the specification:

```text
SHAPley methods are not only a baseline chapter.
They become a comparison lens across the whole course:
  data -> input -> activation -> feature -> circuit -> behavior.
```

That makes the course much stronger, because it forces the student to confront the central attribution question:

**When we say something was “important,” important to what — the training run, the input-output decision, the internal computation, or the causal mechanism?**

[1]: https://openreview.net/forum?id=HD6bWcj87Y&utm_source=chatgpt.com "Data Shapley in One Training Run"
[2]: https://jiachen-t-wang.github.io/data-shapley.github.io/?utm_source=chatgpt.com "Data Shapley in One Training Run"
[3]: https://github.com/Jiachen-T-Wang/GhostSuite?utm_source=chatgpt.com "GhostSuite (Official Codebase for \"Data Shapley in One ..."
[4]: https://arxiv.org/abs/2602.00329?utm_source=chatgpt.com "In-Run Data Shapley for Adam Optimizer"

---------------------------------------------------

Yes — amend this explicitly. It was implied by the verification harness, but it should be a **first-class requirement** because ARENA’s best hard exercises work by testing sub-functions before the final function. This is especially important for long implementations like Mamba scans, Gemma weight loading, VLM patching, EAP-IG, Sparse Feature Circuits, LoRA/DoRA merging, SHAPley estimators, and diffusion samplers.

Add the following appendix.

---

# Appendix: Step-by-Step Verification for Hard Exercises

Every complex exercise must be decomposed into a **test ladder**: small sub-functions, local invariants, reference outputs, then full end-to-end verification. Do not ask students to implement a 200-line algorithm and only test the final output.

This follows the ARENA style and the broader research-process guidance: de-risk hard work early, get fast feedback loops, use small examples first, compare against brute-force/reference implementations, and only then scale.

---

## 1. Test ladder requirement

For any exercise rated difficulty 3/5 or above, agents must provide:

```text
1. shape tests
2. dtype / device tests
3. small hand-computed examples
4. sub-function tests
5. brute-force reference comparison where possible
6. randomized property tests
7. integration test
8. real-model smoke test
9. full verification report
```

The notebook should be structured like this:

```text
Exercise 1: implement helper A
Test helper A

Exercise 2: implement helper B
Test helper B

Exercise 3: combine A and B
Test combined result on toy example

Exercise 4: optimize / vectorize
Compare optimized version to brute-force version

Exercise 5: run on real model
Compare to reference implementation or expected behavior
```

A hard exercise without intermediate tests is incomplete.

---

## 2. Required exercise metadata

Every nontrivial implementation block should include:

```python
# Exercise metadata
EXERCISE_ID = "5.4.eap_ig.edge_scores"
GT_TIER = "GT-1"
DIFFICULTY = 4
IMPORTANCE = 5
EXPECTED_RUNTIME = "seconds on toy, minutes on real model"
REQUIRES_GPU = False  # or True
```

And the function stub should include expected shapes:

```python
def compute_eap_ig_scores(
    clean_acts: Float[Tensor, "batch edges d_model"],
    corrupt_acts: Float[Tensor, "batch edges d_model"],
    gradients: Float[Tensor, "steps batch edges d_model"],
) -> Float[Tensor, "edges"]:
    """
    Return integrated-gradient edge attribution scores.

    Expected:
        - output shape: [edges]
        - scores are zero when clean_acts == corrupt_acts
        - scores match exact patching on the tiny linear test case
    """
```

---

## 3. Standard test cell format

Every exercise should be followed immediately by a visible test cell:

```python
tests.test_compute_eap_ig_scores_basic(compute_eap_ig_scores)
tests.test_compute_eap_ig_scores_zero_delta(compute_eap_ig_scores)
tests.test_compute_eap_ig_scores_matches_bruteforce(compute_eap_ig_scores)
```

The notebook should print an ARENA-style success message:

```text
All tests in `test_compute_eap_ig_scores_*` passed!
```

For hard tasks, do not hide all tests in CI. Students need partial feedback inside the notebook.

---

## 4. Brute-force before optimized

Whenever possible, require two versions:

```text
slow_correct_version
fast_vectorized_version
```

The slow version should be obviously correct and used as an oracle.

Examples:

```text
Mamba:
  recurrent selective scan = slow oracle
  parallel selective scan = optimized implementation

SHAPley:
  exact Shapley over all coalitions = oracle
  KernelSHAP / Monte Carlo Shapley = approximation

EAP:
  exact edge patching = oracle on tiny graph
  EAP / EAP-IG = scalable approximation

Diffusion:
  explicit noising distribution = oracle
  vectorized sampler = optimized implementation

LoRA:
  explicit W + BA multiplication = oracle
  merged adapter module = optimized implementation

VLM patching:
  full activation replacement = oracle
  cached/batched patching = optimized implementation
```

Acceptance rule:

```text
fast_vectorized_version must numerically match slow_correct_version
on toy cases before being used on real models.
```

---

## 5. Property-based tests

Agents should add randomized tests for mathematical invariants.

Examples:

### RoPE

```text
rotation preserves vector norm
relative-position identity holds
inverse rotation recovers original vector
```

### RMSNorm

```text
output RMS is approximately controlled
scaling input by positive constant gives expected invariance
```

### LoRA

```text
rank(delta_W) <= rank r
merged and unmerged forward passes match
zero LoRA weights exactly recover base model
```

### DoRA

```text
magnitude × normalized direction reconstructs weight
zero adapter update recovers base model
merged and unmerged outputs match
```

### Mamba

```text
chunked scan equals full scan
recurrent scan equals parallel scan
state cache generation equals full-context generation
```

### SHAPley

```text
efficiency: sum Shapley values equals f(all) - f(empty)
symmetry: identical players receive equal attribution
dummy: unused player gets zero attribution
linearity: Shapley(f + g) = Shapley(f) + Shapley(g)
```

### Integrated gradients

```text
zero input difference gives zero attribution
linear model IG equals gradient × input difference
sum attributions approximately equals output difference
```

### Diffusion noising

```text
q(x_t | x_0) matches analytic schedule
t=0 gives original data
t=T approaches noise / mask distribution
```

---

## 6. Golden fixtures

Each hard notebook must include tiny frozen fixtures:

```text
assets/expected_outputs/
  toy_input.pt
  expected_helper_A.pt
  expected_helper_B.pt
  expected_final.pt
  expected_metrics.json
```

Agents should generate these once from a trusted implementation and pin them.

The fixture README must say:

```text
How this fixture was produced
Which implementation produced it
Which random seed was used
Which tolerances are allowed
When it should be regenerated
```

---

## 7. Tolerance policy

Every test must specify tolerances explicitly.

Example:

```python
torch.testing.assert_close(
    actual,
    expected,
    rtol=1e-4,
    atol=1e-5,
)
```

Recommended defaults:

```text
exact integer / token tests: exact equality
float32 toy tests: rtol=1e-5, atol=1e-6
bf16 model tests: rtol=1e-2, atol=1e-2
quantized model tests: compare behavior/top-k/KL, not exact logits
stochastic generation: fixed seed or distributional metric
```

Never use vague checks like:

```python
assert looks_reasonable(output)
```

unless followed by quantitative criteria.

---

## 8. Checkpoint cells for long exercises

Long notebooks must include checkpoint cells:

```python
# Checkpoint 1: shapes are correct
# Checkpoint 2: toy numerical example passes
# Checkpoint 3: vectorized version matches brute force
# Checkpoint 4: real model smoke test passes
# Checkpoint 5: causal intervention / final verification passes
```

This is especially required for:

```text
Gemma from scratch
Mamba from scratch
Diffusion LM sampler
VLM activation patching
Sparse Feature Circuits
EAP-IG
Data Shapley / In-Run Shapley
LoRA vs full finetuning
VLM SAE training
JEPA latent rollout
```

---

## 9. Example: Sparse Feature Circuits test ladder

```text
Step 1:
  test SAE encode/decode shapes

Step 2:
  test SAE reconstruction error on frozen activations

Step 3:
  test feature activation sparsity / L0

Step 4:
  test exact node patching on tiny graph

Step 5:
  test exact edge patching on tiny graph

Step 6:
  test attribution patching approximates exact patching

Step 7:
  test EAP-IG improves over plain EAP on tiny nonlinear case

Step 8:
  test feature graph thresholding preserves metric

Step 9:
  test random feature graph performs worse

Step 10:
  run Pythia-70M subject–verb agreement smoke replication
```

The final sparse feature circuit graph is not accepted unless all earlier steps pass.

---

## 10. Example: VLM visual-token patching test ladder

```text
Step 1:
  generate synthetic image with known object/color/bbox

Step 2:
  test image preprocessing preserves expected dimensions

Step 3:
  test vision encoder output shape

Step 4:
  test projector output shape equals LLM embedding dimension

Step 5:
  test text-only baseline fails on image-dependent question

Step 6:
  test object-region patch changes answer more than background patch

Step 7:
  test visual-token patching clean -> corrupt flips target logit

Step 8:
  test random visual-token patch has smaller effect

Step 9:
  test held-out colors/objects generalize

Step 10:
  run real VLM smoke test
```

No VLM notebook should jump directly to real PaliGemma/Qwen-VL patching before passing the synthetic controlled version.

---

## 11. Example: refusal-direction test ladder

```text
Step 1:
  load sanitized harmless/refusal prompt pairs

Step 2:
  test labels and splits are balanced

Step 3:
  cache residual activations for one layer

Step 4:
  compute mean-difference direction

Step 5:
  test direction separates train activations

Step 6:
  test direction separates held-out activations

Step 7:
  run PCA/SVD and report explained variance

Step 8:
  add direction to harmless prompts

Step 9:
  project direction out of refusal prompts

Step 10:
  compare against random and label-shuffled directions
```

The direction is not accepted merely because a PCA plot separates the classes. It must steer or ablate behavior.

---

## 12. Example: LoRA/DoRA test ladder

```text
Step 1:
  implement low-rank update B @ A

Step 2:
  test rank(delta_W) <= r

Step 3:
  test zero adapter gives base output

Step 4:
  test merged and unmerged outputs match

Step 5:
  train LoRA on tiny task

Step 6:
  test target behavior improves

Step 7:
  compute SVD of adapter update

Step 8:
  test projection along adapter-induced direction changes behavior

Step 9:
  compare LoRA, DoRA, and full finetuning on same toy task

Step 10:
  test OOD side effects
```

---

## 13. Example: SHAPley-vs-mechinterp test ladder

```text
Step 1:
  implement exact Shapley on n <= 8 players

Step 2:
  test Shapley axioms

Step 3:
  implement KernelSHAP / Monte Carlo Shapley

Step 4:
  compare approximations to exact Shapley

Step 5:
  run activation patching on the same toy task

Step 6:
  compare top-k important variables

Step 7:
  run deletion/insertion curves

Step 8:
  add interaction-heavy XOR/parity case

Step 9:
  show where marginal Shapley fails and interaction Shapley succeeds

Step 10:
  compare to SAE feature / circuit attribution
```

The point is not “SHAP agrees with mechinterp.” The point is to identify when they agree, when they disagree, and why.

---

## 14. CI markers

Tests should be split by cost:

```text
@pytest.mark.unit
@pytest.mark.cpu
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.requires_model
@pytest.mark.requires_gated_model
@pytest.mark.requires_large_download
```

Required CI:

```text
unit + cpu tests run on every commit
gpu smoke tests run before merging chapter
slow/full tests run before release
```

A notebook can be merged only if:

```text
unit tests pass
CPU smoke path passes where applicable
GPU smoke path has been run at least once and report is committed
```

---

## 15. Failure messages must teach

Bad:

```python
assert actual.shape == expected.shape
```

Better:

```python
assert actual.shape == expected.shape, (
    f"Expected shape {expected.shape}, got {actual.shape}. "
    "Common bug: you may have forgotten to keep the batch dimension, "
    "or you transposed [batch, seq, d_model] into [seq, batch, d_model]."
)
```

Every common bug should have an explanatory failure message.

---

## 16. Required “debug mode”

Complex functions should support:

```python
debug=True
```

and return intermediate activations:

```python
out, debug_cache = function(..., debug=True)
```

Example:

```python
debug_cache = {
    "q": q,
    "k": k,
    "v": v,
    "attn_scores": attn_scores,
    "attn_probs": attn_probs,
    "pre_norm": pre_norm,
    "post_norm": post_norm,
}
```

For VLMs:

```python
debug_cache = {
    "image_pixels": image_pixels,
    "vision_tokens": vision_tokens,
    "projected_tokens": projected_tokens,
    "input_embeddings": input_embeddings,
    "visual_token_positions": visual_token_positions,
}
```

For EAP-IG:

```python
debug_cache = {
    "clean_acts": clean_acts,
    "corrupt_acts": corrupt_acts,
    "interpolated_acts": interpolated_acts,
    "gradients": gradients,
    "edge_scores": edge_scores,
}
```

Debug mode makes student errors much easier to localize.

---

## 17. Final amendment to acceptance criteria

Add this to the global checklist:

```text
For every hard exercise:
  - Are sub-functions tested before the full function?
  - Is there a toy case with known ground truth?
  - Is there a brute-force or reference implementation where possible?
  - Does the optimized implementation match the reference?
  - Are failure messages educational?
  - Are partial expected outputs shown?
  - Can a student identify which sub-step failed?
```

If not, the exercise is incomplete.

---

## Final wording for the specification

```text
Hard exercises must be implemented as step-by-step verification ladders,
not monolithic final-answer tests. Each major function should be decomposed
into smaller helpers with visible notebook tests, expected outputs, common
bug notes, and at least one toy oracle or reference implementation where
possible. Final model-level verification is necessary but not sufficient:
students must receive ARENA-style partial feedback that localizes mistakes
to the smallest feasible sub-function.
```

That amendment should definitely be included. It is one of the main things that will keep the extension beautiful, learnable, and robust rather than becoming a pile of impressive but fragile frontier notebooks.
