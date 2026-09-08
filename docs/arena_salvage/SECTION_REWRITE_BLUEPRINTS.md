# Section Rewrite Blueprints

These are concrete rewrite plans for the broken or high-priority sections.

## 7.1 Logit Lens, Tuned Lens, and Patchscopes

### Current problem

The current notebook teaches functions on tiny arbitrary tensors and ends with a dict of metrics. The student sees numbers but not a model, token, prompt, layer, activation, or mechanism. The real result is hidden in a committed GPU report.

### New section claim

```text
By the end of this notebook, you will have shown that residual-stream activations in a trained transformer become increasingly decodable as the model computes the next token, and that Patchscope-style activation insertion carries answer information beyond text-only prompt priors.
```

### Required learner arc

1. Load a tiny real model or train a tiny transformer on a simple generated language task.
2. Pick a concrete prompt with a known next token.
3. Run the model with activation cache.
4. Implement `logit_lens(resid, W_U)`.
5. Show layer-by-layer top decoded tokens.
6. Implement a tiny tuned lens or ridge affine lens on held-out next-token targets.
7. Show tuned lens improves but can overfit.
8. Implement Patchscope activation replacement.
9. Compare patched prompt vs text-only prompt.
10. Add random activation and counterfactual controls.
11. Let student change prompt/layer/source activation.

### Required visible outputs

- A table: prompt, layer, top decoded tokens, correct token rank.
- A line plot: correct-token logit/rank across layers.
- A bar chart: logit lens vs tuned lens held-out accuracy.
- A table: Patchscope vs text-only vs random activation on 6–20 examples.
- At least 3 concrete qualitative examples with actual tokens.

### Required implementation exercises

```text
Exercise 1: implement `logit_lens`
Exercise 2: implement `top_token_table`
Exercise 3: implement `fit_ridge_tuned_lens`
Exercise 4: implement `evaluate_lens_on_heldout`
Exercise 5: implement `replace_activation_at_hook`
Exercise 6: implement `patchscope_eval`
Exercise 7: implement `random_activation_control`
```

### Banned outcomes

- No final dict as the main output.
- No “tuned lens accuracy 1.0” on a two-example artificial tensor.
- No signature result only loaded from `verification_report.json`.
- No Patchscope claim unless text-only prompt fails or is clearly weaker.

### Minimum acceptance

```text
held-out examples >= 20 unless exact toy theorem
logit lens improves across layers or notebook explains why not
text-only baseline < patched activation path
random activation confidence low
student can change prompt/layer/source in a play cell
```

## 8.3 ACDC and Circuit Metrics

### Current problem

The notebook looks more ARENA-like, but the result is mostly a metrics preflight: final-position localization in `gelu-1l`. It repeatedly says it is not real ACDC. That makes it a weak main section.

### New section claim

```text
By the end of this notebook, you will have implemented ACDC-style pruning on a known toy circuit and validated a small real-model circuit fragment using faithfulness, minimality, completeness, and same-size random controls.
```

### Required learner arc

1. Create a hand-coded toy computational graph with a known two-edge circuit.
2. Implement exact node/edge patching on the toy graph.
3. Implement pruning by threshold.
4. Show the discovered graph matches the known graph.
5. Compute faithfulness/minimality/completeness.
6. Add same-size random graph control.
7. Move to a small transformer task: e.g. subject-verb agreement, IOI mini-fragment, or greater-than mini-fragment.
8. Use activation/path patching to produce a nontrivial candidate circuit.
9. Compare exact patching to attribution approximation if included.

### Required visible outputs

- Diagram of toy graph with ground-truth circuit highlighted.
- Heatmap or table of edge patching scores.
- Discovered circuit graph.
- Faithfulness/minimality/completeness curves.
- Same-size random circuit comparison.
- One real-model fragment plot.

### Banned outcomes

- Do not make final-position localization the signature result.
- Do not call a position-circuit preflight “ACDC.”
- Do not hide exact patching implementation.

## 12.1 VLM Interpretability / CLIP Controls

### Current problem

Too much is squeezed into one notebook. It should be split into a chapter. VLMs are the flagship topic and must be the most polished.

### Split into these notebooks

```text
12.1 CLIP and SigLIP from Scratch
12.2 CLIP/SigLIP Feature Geometry
12.3 Mini VLM from Scratch
12.4 Visual Token Flow in Real VLMs
12.5 Object Hallucination and Modality Arbitration
12.6 Multimodal SAEs and Crosscoders
```

### 12.1 claim

```text
A tiny CLIP trained on controlled colored-shape image/caption pairs learns a shared image-text embedding space, because image-text retrieval succeeds on held-out pairs while random-caption and mismatched-label controls fail.
```

### Required visible outputs

- Generated image grid with captions.
- Training loss curve.
- Retrieval heatmap with strong diagonal.
- Top-k retrieval examples.
- Random-label failure.
- Typographic attack or image/text conflict example.

### 12.3 Mini VLM claim

```text
A frozen vision encoder plus learned projector can make a tiny decoder answer simple visual questions, and visual-token patching can causally flip the answer on controlled scenes.
```

Required visible outputs:

- Controlled scene images.
- VQA accuracy curve.
- Text-only baseline failure.
- Object-region patch > background-region patch.
- Layer/position patching heatmap.

### Real VLM escalation

Only after toy VLM works, add PaliGemma/Qwen-VL/Molmo smoke paths. Real-model claims must have:

- exact model revision;
- at least 10 controlled examples;
- object-region/background-region controls;
- text-only/image-only baselines;
- patching effect metric.

## 9.1 Refusal Direction

### Required section claim

```text
A mean-difference residual-stream direction separates refusal-eliciting and harmless prompts in a small chat model, and adding/removing this direction causally changes refusal behavior on sanitized held-out prompts.
```

### Required visible outputs

- Refusal/non-refusal prompt examples, sanitized.
- Layer sweep of linear separation.
- PCA/SVD explained variance of refusal differences.
- Steering curve for adding direction to harmless prompts.
- Ablation/projection-out curve for refusal prompts.
- Random-direction and label-shuffled controls.
- Capability side-effect table.

### Required play cells

```python
prompt = "Write a polite birthday message to my friend."
alpha = 4.0
run_with_refusal_direction(prompt, alpha)
```

and

```python
layer = 18
plot_refusal_projection_distribution(layer)
```

## 5.5 Diffusion Language Models / DiffusionGemma

### What to keep

The claim boundary is good: toy discrete diffusion works; NVFP4 local generation is a runtime proof; BF16 is deferred; no denoising-step interpretability is claimed yet.

### What to improve

Add the missing ARENA learner result:

- show noising schedule visually;
- show denoising trajectory for a few examples;
- plot entropy/commitment time by position;
- show shuffled-label control side-by-side;
- let the student change prefix tokens and watch denoising.

### Required signature result

```text
A toy discrete diffusion LM reconstructs [a,b] -> [a,a,b,b] on held-out examples, while shuffled-label training fails; the denoising trajectory shows suffix positions committing over time.
```

The NVFP4 proof should be a final “real model loading proof,” not the main interpretability result.

## 11.1 Representation Geometry

### Required section claim

```text
Some concepts form stable low-dimensional geometry in activations, while others produce white noise; PCA/UMAP/t-SNE only count as evidence when paired with probes, seed stability, and controls.
```

### Required visible outputs

- days/months PCA or circular plot;
- random-label UMAP failure;
- linear probe/kNN table;
- seed stability table;
- one intervention along a discovered direction when meaningful;
- VLM clothing/object/color geometry as a later subchapter.

### Banned outcomes

- Do not present UMAP as evidence by itself.
- Do not claim geometry from 2D projection without quantitative validation.

## 15.1 LoRA / DoRA

### Required section claim

```text
LoRA and DoRA implement low-rank adapter updates whose parameter-space directions and activation-space effects can be measured, compared to full finetuning, and causally tested on safe proxy behaviors.
```

### Required visible outputs

- merge/unmerge parity table;
- rank and singular value spectrum;
- training curves for LoRA vs DoRA vs full finetuning;
- OOD side-effect table;
- activation shift PCA/SVD;
- projection-out intervention reducing adapter behavior;
- random rank-matched adapter control.

## 16.8 SHAPley vs Mech Interp Agreement

### Required section claim

```text
Input-level, data-level, activation-level, and circuit-level attribution methods sometimes agree and sometimes disagree because they assign credit to different kinds of players; the disagreement is itself informative.
```

### Required visible outputs

- exact Shapley on known Boolean games;
- interaction Shapley recovering XOR/parity;
- TokenSHAP vs activation patching agreement matrix;
- deletion/insertion curves;
- one case where they agree;
- one case where they disagree and a follow-up test explains why.
