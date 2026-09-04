# ARENA 3.0 Extended PR Review: guidance for an ARENA-style rewrite

I could not clone the GitHub repository inside the container because DNS resolution for `github.com` failed. I therefore reviewed the PR through the GitHub web/raw views, the repository file tree, representative notebooks/instruction pages/tests, and the uploaded `FOR_REVIEW` packet. That is enough to diagnose the **course-design, style, preservation, and evidence-quality problems**, but I did **not** independently re-run the full test suite locally.

The current PR is technically impressive but **not yet ARENA-like**. It looks like a huge, verification-heavy research scaffold plus many generated notebooks, rather than a polished continuation of the original ARENA teaching sequence. The right next move is not to throw it away. The right move is to treat it as a **prototype artifact branch**, then extract a smaller number of flagship sections and rewrite them in the original ARENA style.

The PR itself claims 41 accepted extension sections, a new `arena_ext` library, notebooks, Streamlit pages, verification reports, artifact locks, audit scripts, and a DiffusionGemma NVFP4 proof; it also claims 337 tests passed and several audit scripts passed. ([GitHub][1]) But the files page shows the scale problem clearly: 768 files changed, +173,611 / −142 lines, including 84 notebooks, 172 JSON files, 151 Markdown files, and 251 Python files. ([GitHub][2]) This is far too large to review as a single course PR, and it explains why the result feels “weirdly put together.”

---

## 1. Executive verdict

**Do not merge this PR as the final ARENA extension.**

Treat it as:

```text
A valuable prototype / implementation dump / verification artifact branch.
```

Not as:

```text
A polished ARENA-style course continuation.
```

The main failure is not “the agent did nothing.” It did a lot. The failure is that it optimized for **coverage and verification checkboxes** rather than the actual thing that makes ARENA great:

```text
A carefully paced notebook experience where the student builds intuition,
implements functions step by step, sees beautiful expected outputs,
understands why the result matters, and then uses the result to do real
mechanistic interpretation.
```

The current extension often has the ingredients — tests, reports, artifact locks, toy tasks, real-model preflights — but not the **pedagogical shape**.

---

## 2. What went right

### 2.1 Original-preservation intent exists

The repo does have a preservation contract. It says the extension is append-only relative to original ARENA, that original exercises/instruction pages/assets should not be edited except through explicit extension paths, and that the original base commit is pinned. ([GitHub][3]) This is good and should remain.

The root tree also still contains the original ARENA chapter directories alongside the new extension chapters, including `chapter0_fundamentals`, `chapter1_transformer_interp`, `chapter2_rl`, `chapter3_llm_evals`, and `chapter4_alignment_science`. ([GitHub][4]) That is directionally correct.

But the preservation needs to be made stricter. The original ARENA should be treated as a **frozen upstream subtree/submodule**, not something casually co-edited with extension infrastructure.

---

### 2.2 The verification culture is much better than a normal “AI-generated course”

The repo has a real verification policy. It explicitly says tests should not be fake, that shared tests must validate mathematical invariants or reusable APIs, that notebook-local tests should support step-by-step feedback, and that verification reports must distinguish accepted, blocked, and deferred claims. ([GitHub][5])

It also has a hard-exercise verification ladder document requiring subfunction tests, toy oracles, brute-force/reference comparisons, integration tests, and final reports. ([GitHub][6])

This is genuinely good. Preserve it.

The problem is that these policies are often **better than the notebooks themselves**. The policies describe what a great ARENA extension should do; the notebooks do not yet consistently deliver that experience.

---

### 2.3 DiffusionGemma is handled with unusually honest claim boundaries

The uploaded review packet is the strongest part of the work. It says the section 5.5 diffusion-language-model exercise has a valid CUDA toy-model learning ladder, deterministic controls, and strict claim boundaries. It also records a real released-checkpoint DiffusionGemma generation proof for the 24GB route using the NVIDIA NVFP4 checkpoint in an isolated vLLM runtime on the RTX 5090 Laptop GPU.

It also explicitly does **not** overclaim: the packet says BF16 direct local loading is deferred because the Google BF16 shards are 48.10 GiB while the GPU has 23.46 GiB, and it says not to claim denoising-step activation capture, diffusion-time patching, throughput, broad quality, or full interpretability parity.

That is exactly the right epistemic style. The current result proves:

```text
toy discrete diffusion learning works;
negative controls fail;
NVFP4 DiffusionGemma can generate locally through an isolated vLLM path.
```

It does **not** yet prove:

```text
DiffusionGemma interpretability;
denoising-time circuits;
diffusion-time activation patching;
quality benchmarking;
throughput benchmarking;
BF16 local feasibility.
```

Keep that honesty.

---

### 2.4 Some exercises have the right skeleton

The Gemma section is a good example of a technically reasonable skeleton. The notebook includes RMSNorm, RoPE, GQA helpers, MLP, attention, layer, model, memory-budget checks, and verification. ([GitHub][7]) The corresponding instruction page has ARENA-ish elements such as difficulty/importance, expected output, solution, common bug notes, and tests. ([GitHub][8])

That is the right direction. It is not yet beautiful, but it is salvageable.

---

### 2.5 The VLM section has the right instincts

The VLM instruction page has an excellent standard: it says the goal is not merely “the caption looks plausible,” but paired retrieval, localization, hallucination, modality controls, and real-model evidence. ([GitHub][9])

That is exactly the right principle for VLM interpretability. It should remain the cornerstone of the VLM chapter.

---

## 3. What went wrong

### 3.1 The PR is trying to be the whole future at once

The new chapter layout adds a huge set of chapters: modern architectures, sparse feature methods, activation-to-language, automated circuits, alignment interpretability, capstones, representation geometry, VLM interpretability, image-generation interpretability, JEPA/world models, PEFT/misalignment, Shapley baselines, and training dynamics. ([GitHub][4])

The README also summarizes a very large frontier-lab extension, with many accepted sections across many domains. ([GitHub][4])

That may be a good **roadmap**, but it is not a good **course PR**.

ARENA works because each notebook feels like a crafted path:

```text
concept → implementation → tests → expected output → interpretation → extension.
```

This PR often feels like:

```text
method registry → generated notebook → tests → report JSON → next method.
```

That is not the same thing.

---

### 3.2 The new notebooks often lack the original ARENA voice

The raw Gemma notebook is mostly setup, stubs, tests, and verification. It has the right technical order, but not enough conceptual exposition, diagrams, intuitive explanations, expected-output discussion, or “why this result is cool” narrative. ([GitHub][7])

The VLM notebook is more concerning: it is a giant single JSON line in raw form, combines many different ideas into one large section, and reads more like a test harness than a carefully paced notebook. ([GitHub][10])

Original ARENA notebooks are much more explanatory. They have expected-output dropdowns, help dropdowns, interpretation notes, and visible plots with detailed explanations. For example, the source pack includes an ARENA-style expected-output block followed by a “Help — I don’t understand the interpretation of these plots” dropdown explaining an induction-head attention-score decomposition.

The extension needs much more of that.

---

### 3.3 Some generated files are effectively unreadable

Several raw Python and notebook-support files appear as one or two enormous lines. For example, the Gemma solutions file is reported as a single-line raw file. ([GitHub][11]) The VLM solutions and tests are similarly compressed into tiny line counts despite containing substantial code. ([GitHub][12])

This is a serious quality issue.

ARENA is educational software. Students and reviewers must be able to read the code. A file that technically runs but is minified into one line is not acceptable.

Add a hard CI rule:

```text
No generated .py, .md, or .ipynb source representation may be minified.
No Python file may have a median line length above 120 characters.
No Python file may contain a line above 240 characters unless explicitly allowlisted.
All Python must pass ruff format or black.
All Markdown must pass mdformat or equivalent.
All notebooks must round-trip through jupytext without collapsing into unreadable JSON.
```

---

### 3.4 The style audit is too shallow

The repo has an `audit_arena_style_depth.py` script, which sounds good. But the raw script checks for textual markers like expected output, solution, common bug, difficulty, and importance. ([GitHub][13])

That is not enough. A generated file can contain those words and still not feel like ARENA.

The audit should check for:

```text
visible explanatory markdown before hard code;
at least one expected-output dropdown with actual output;
at least one help/interpretation dropdown;
hidden solutions, not inline answers;
non-minified source;
signature result artifact;
step-by-step tests after subfunctions;
no “verification report only” final result;
human-readable figure captions;
at least one real interpretability conclusion or honest negative result.
```

The current audit verifies **vocabulary**, not **pedagogy**.

---

### 3.5 Too many sections are “verification report first” rather than “student result first”

Verification reports are useful, but in ARENA the student sees the thing.

They see:

```text
an attention pattern;
an induction stripe;
a logit attribution heatmap;
a superposition geometry;
a learned SAE feature;
a patching recovery plot;
a circuit diagram.
```

The extension often seems to end in:

```text
read verification_report.json;
assert accepted == true;
print a summary table.
```

That is valuable for CI, but it is not enough for a course.

Each notebook needs a **signature result**. For example:

| Notebook                | Required visible signature result                                       |
| ----------------------- | ----------------------------------------------------------------------- |
| Gemma from scratch      | HF parity table + greedy generation match + KV-cache equivalence        |
| Mamba                   | recurrent-vs-parallel scan equality plot + state intervention           |
| Diffusion LM            | denoising trajectory + entropy/commitment plot + shuffled-label failure |
| Refusal direction       | layer sweep + addition/ablation steering curve + PCA/SVD variance       |
| Sparse Feature Circuits | feature graph + faithfulness/completeness curves                        |
| CLIP/SigLIP             | retrieval heatmap + typographic attack failure                          |
| VLM visual-token flow   | layer × token causal patching heatmap                                   |
| VLM hallucination       | perception/transfer/arbitration/decoding diagnostic table               |
| Diffusion image-gen     | token-to-region map + causal token ablation result                      |
| LoRA/DoRA               | SVD spectrum + merge parity + side-effect matrix                        |
| Shapley vs mechinterp   | agreement/disagreement matrix + deletion curves                         |

If a section cannot produce a convincing visible result, it should stay in the roadmap, not the student-facing course.

---

### 3.6 The extension is too interleaved with original infrastructure

The root config contains original chapter mappings and extension mappings together. The raw config shows original Chapter 1 mappings and new extension mappings in the same `conversion_map`.  It also appends many new extension chapters directly into the same course config.

This is risky.

The original ARENA should be the holy grail. Do not let extension infrastructure mutate the original course source of truth.

Use:

```text
infrastructure/core/config_original.yaml
infrastructure/core/config_extension.yaml
scripts/build_merged_config.py
```

Then the build process can generate a merged site, but the original source remains untouched.

---

### 3.7 The roadmap/spec text leaked into the repository as course-like material

The `Extension-Roadmap.md` raw file begins like a pasted planning response rather than a polished course design document: “Here’s how I’d do it…” and then a large conceptual plan. ([GitHub][14])

That is fine in an archive:

```text
docs/archive/original_planning_conversation.md
```

It is not fine as a student-facing roadmap unless rewritten.

The student-facing roadmap should be concise:

```text
What this extension adds.
Why these chapters exist.
What is required vs optional.
What runs locally.
What is verified.
What remains deferred.
```

---

### 3.8 VLM interpretability is overloaded into one giant section

The VLM chapter is supposed to be your main topic. It should be the most polished part of the extension.

Right now, the VLM section tries to cover CLIP/SigLIP losses, visual-token attribution, synthetic scenes, clothing geometry, nearest centroids, baseline comparisons, region patching, sequence patching, hallucination, arbitration, report generation, and real Qwen/PaliGemma-style checks in one large section. ([GitHub][10])

That should be split into a sequence of beautiful notebooks.

A VLM chapter should not feel like a checklist. It should feel like:

```text
We build a tiny CLIP.
We see retrieval work.
We break it with typographic attacks.
We build a tiny VLM.
We patch visual tokens.
We diagnose hallucination.
We move to real VLMs.
We then ask real research questions.
```

---

## 4. What “ARENA 10.0-Ultra” should actually look like

The joke name is useful because it points to the right ambition: not “more files,” but **a course that exceeds ARENA’s clarity while preserving ARENA’s spirit**.

The final extension should have fewer chapters than the PR currently creates, but each should be much better.

Recommended top-level structure:

```text
Original ARENA 3.0
  Completely preserved.

Extension Chapter 5: Modern Architectures
  5.1 Gemma from Scratch
  5.2 Mamba from Scratch
  5.3 Diffusion Language Models
  5.4 Optional: Recurrent / hybrid architectures

Extension Chapter 6: Frontier Mechanistic Tools
  6.1 Representation Directions and Geometry
  6.2 Refusal Direction Replication
  6.3 SAE Variants and Gemma Scope
  6.4 Sparse Feature Circuits
  6.5 Activation-to-Language: Patchscopes, Verbalizers, Mini-AOs

Extension Chapter 7: Vision-Language Model Interpretability
  7.1 CLIP and SigLIP from Scratch
  7.2 CLIP/SigLIP Feature Geometry
  7.3 Mini VLM from Scratch
  7.4 Visual Token Flow in Real VLMs
  7.5 Object Hallucination and Modality Arbitration
  7.6 Multimodal SAEs and Crosscoders
  7.7 VLM Research Capstones

Extension Chapter 8: Generative Vision and Image-Generation Interpretability
  8.1 Stable Diffusion Attention and DAAM
  8.2 Denoising-Time Patching
  8.3 Diffusion Concept Directions
  8.4 Diffusion LoRA Interpretability
  8.5 Toy AR Image-Token Models

Extension Chapter 9: JEPAs and World Models
  9.1 I-JEPA from Scratch
  9.2 V-JEPA / V-JEPA 2 Feature Extraction
  9.3 Othello-GPT World Models
  9.4 Maze and Sudoku World Models
  9.5 Object Permanence and Action-Conditioned Latents

Extension Chapter 10: Attribution, Shapley, and Data Influence
  10.1 Exact Shapley on Ground-Truth Games
  10.2 TokenSHAP / IG / Input Attribution
  10.3 In-Run Data Shapley
  10.4 SHAPley vs Mechanistic Patching

Extension Chapter 11: Finetuning and Adapter Interpretability
  11.1 LoRA from Scratch
  11.2 DoRA from Scratch
  11.3 LoRA vs Full Finetuning
  11.4 Safe Misalignment Proxies
  11.5 VLM and Diffusion LoRA Interpretability

Extension Chapter 12: Capstone Research Sprint
  Paper-style projects with strong evidence, baselines, causal tests, and limitations.
```

This is still ambitious, but it is much less weirdly fragmented than chapters 5–17.

---

## 5. Required notebook style

Every notebook should follow this template.

```markdown
# [7.4] Visual Token Flow in Vision-Language Models

## Learning objectives

By the end of this notebook, you should be able to:

1. Explain how image tokens enter a multimodal LLM.
2. Build clean/corrupt image-question pairs.
3. Patch visual-token activations between prompts.
4. Distinguish perception, transfer, arbitration, and decoding failures.
5. Validate a VLM interpretability claim with baselines and negative controls.

## Setup

Imports, model loading, small synthetic dataset.

## Background: what is the computation we are studying?

Short explanation with diagrams.

## Exercise 1: generate a controlled image-question pair

Student implements small function.

Tests immediately follow.

Expected output dropdown.

## Exercise 2: cache visual tokens

Student implements hook/cache function.

Tests immediately follow.

Expected shape output.

## Exercise 3: patch object-region visual tokens

Student implements patching.

Expected heatmap.

## Interpreting the result

A help dropdown explains what the result means.

## Exercise 4: random-region and background controls

Student implements controls.

## Exercise 5: run on real VLM

Optional / GPU path.

## Final verification

Causal effect table, controls, OOD split, limitations.

## Bonus

Research extensions.
```

The tone should be:

```text
clear;
concrete;
slightly playful;
mathematically serious;
not survey-like;
not benchmark-dump-like;
not generated-spec-like.
```

---

## 6. New merge strategy

Do not merge all 768 files as the final course.

Use this sequence:

### PR 1 — preservation and infrastructure only

Keep:

```text
original preservation contract
environment split
artifact registry
verification-report schema
style audit skeleton
arena_ext minimal utilities
```

Do not include 84 notebooks yet.

### PR 2 — one polished pilot notebook

Pick one of:

```text
Gemma from Scratch
CLIP/SigLIP from Scratch
Refusal Direction
Sparse Feature Circuits
```

This pilot must match original ARENA style before any other chapter is accepted.

### PR 3 — VLM flagship chapter skeleton

Add only:

```text
7.1 CLIP/SigLIP from Scratch
7.2 Mini VLM from Scratch
7.3 Visual Token Flow
```

No hallucination, LoRA, SHAP, diffusion, or crosscoders until these are beautiful.

### PR 4+ — expand one chapter at a time

Each chapter should be reviewable in isolation.

---

## 7. Rewrite guidance for the VLM chapter

The current VLM section should be split into the following sequence.

### 7.1 CLIP and SigLIP from Scratch

Core result:

```text
A toy CLIP model learns image-text retrieval on controlled colored-shape data.
```

Exercises:

```text
normalize embeddings
compute contrastive logits
implement CLIP loss
implement SigLIP loss
train on synthetic image-caption pairs
evaluate retrieval
run typographic attack
```

Expected visible results:

```text
image-text retrieval heatmap with strong diagonal
retrieval@1 / retrieval@5 table
typographic attack examples
random-label control failure
```

This should cite:

* [CLIP](https://arxiv.org/abs/2103.00020)
* [SigLIP](https://arxiv.org/abs/2303.15343)
* [Multimodal Neurons](https://openai.com/index/multimodal-neurons/)

---

### 7.2 CLIP/SigLIP feature geometry

Core result:

```text
CLIP/SigLIP representations encode clothing category, color, and style to different degrees;
some beautiful UMAPs are real, and some are white-noise decoration.
```

Exercises:

```text
extract embeddings from Fashion-MNIST / Fashionpedia / synthetic clothing
run PCA
run UMAP
run t-SNE
train linear probes
run kNN classification
run random-label controls
run seed-stability checks
```

Expected visible results:

```text
PCA / UMAP plots
probe accuracy table
random-label failure
“white noise detector” result
```

Acceptance rule:

```text
No plot counts unless it predicts held-out labels or supports a causal/behavioral test.
```

---

### 7.3 Mini VLM from Scratch

Core result:

```text
A frozen vision encoder + learned projector + tiny language model can answer simple visual questions,
and visual-token patching can causally flip its answer.
```

Exercises:

```text
build synthetic colored-shape dataset
freeze tiny CLIP/SigLIP encoder
train projector into tiny decoder
answer color/object/spatial questions
cache visual tokens
patch clean/corrupt visual tokens
```

Expected visible results:

```text
toy VQA accuracy curve
text-only baseline failure
image-only / multimodal success
object-region patch flips target logit
background patch does less
```

This is the VLM equivalent of “build GPT-2 from scratch.” It should come before real PaliGemma/Qwen-VL.

---

### 7.4 Visual-token flow in real VLMs

Core result:

```text
In a real local VLM, object/color evidence enters through visual tokens,
is transformed by the projector, and becomes causally available to answer-token computation
in a specific layer band.
```

Models:

```text
PaliGemma 3B / PaliGemma 2 3B
Qwen2.5-VL 3B
optional: LLaVA-OneVision 7B quantized
optional: Molmo 7B-D quantized
```

Exercises:

```text
load real VLM
render synthetic image-question pairs
cache vision encoder outputs
cache projector outputs
cache visual-token residual stream
patch object-region tokens
patch background-region tokens
full-sequence patching
last-token patching comparison
```

Expected visible results:

```text
layer × token-type causal-effect heatmap
object patch > background patch
full-sequence patch > last-token patch on conflict cases
random visual-token control fails
```

---

### 7.5 Object hallucination and modality arbitration

Core result:

```text
Some VLM hallucinations are not perception failures; they are transfer/arbitration/decoding failures.
```

Use the four-hypothesis frame:

```text
H1: Perception failure
H2: Transfer failure
H3: Arbitration failure
H4: Decoding failure
```

Exercises:

```text
synthetic POPE-style object-presence data
typographic conflict data
vision-encoder probe
projector probe
answer-token probe
causal patching
modality SHAP baseline
```

Expected visible results:

```text
diagnostic table assigning failures to H1/H2/H3/H4
causal patch examples
controls and OOD examples
```

---

### 7.6 Multimodal SAEs and crosscoders

Core result:

```text
Sparse features can identify visual, textual, and cross-modal concepts,
but they must be validated causally.
```

Exercises:

```text
SAE on CLIP/SigLIP activations
SAE on VLM projector outputs
SAE on VLM visual-token residual stream
feature top examples
held-out feature classifier
feature ablation
feature steering
crosscoder between CLIP and VLM representations
```

Expected visible results:

```text
feature dashboard
feature-density histogram
ablation/steering effect table
random-feature control
```

---

## 8. Rewrite guidance for Gemma

Gemma is one of the closest sections to being salvageable. It needs to become a **beautiful implementation narrative**, not just a tested implementation.

Add:

```text
diagram of Gemma block
RMSNorm intuition
RoPE geometry
GQA diagram
KV-cache diagram
memory-budget table
HF parity table
```

Exercises should be:

```text
Exercise 1: RMSNorm
Exercise 2: RoPE
Exercise 3: grouped-query attention
Exercise 4: SwiGLU MLP
Exercise 5: Gemma decoder block
Exercise 6: KV cache
Exercise 7: full model forward pass
Exercise 8: load HF weights
Exercise 9: compare logits
Exercise 10: greedy generation parity
```

Expected visible result:

```text
max_abs_diff / MSE / KL / top-k agreement table
cache-vs-no-cache equality
peak VRAM estimate
short deterministic generation
```

Do not call this complete until a student can understand **why** each architectural change exists.

---

## 9. Rewrite guidance for DiffusionGemma

The DiffusionGemma section should be renamed more conservatively:

```text
[5.3] Toy Discrete Diffusion Language Models and Local DiffusionGemma Proof
```

Do not title it as though full DiffusionGemma interpretability is done.

The uploaded review packet makes the correct claim boundary: toy discrete diffusion works, NVFP4 released-checkpoint generation is proven in isolated vLLM, but denoising-step activation capture and diffusion-time patching are still unsupported.

To become ARENA-quality, add:

```text
denoising trajectory visualization
entropy over denoising steps
commitment-time plot
side-by-side shuffled-label failure
prompt suite for NVFP4
VRAM/runtime table
optional denoising-step activation capture when tooling supports it
```

The toy results are strong — held-out masked accuracy 1.0, sampler exact match 1.0, shuffled-label control accuracy 0.0714, and tiny peak VRAM — but the real checkpoint result is still a one-prompt proof, not an interpretability result.

---

## 10. Rewrite guidance for Sparse Feature Circuits

This should be a flagship notebook, not a buried method checkbox.

Notebook sequence:

```text
[6.4] Sparse Feature Circuits
```

Exercises:

```text
1. exact patching on a tiny known graph
2. attribution patching
3. integrated gradients
4. EAP
5. EAP-IG
6. SAE encode/decode
7. SAE error nodes
8. feature graph thresholding
9. subject–verb agreement replication
10. SHIFT-style editing
```

Expected visible results:

```text
tiny exact-vs-approx attribution table
EAP vs EAP-IG comparison
feature graph diagram
faithfulness curve
completeness curve
random-feature graph failure
spurious-feature editing result
```

The source pack treats Sparse Feature Circuits as an important SAE-circuit method using attribution patching and an integrated-gradients variant, with editing applications to spurious correlations.  That should be reflected in a full pedagogical arc.

---

## 11. Rewrite guidance for refusal directions

This should be short, beautiful, and mandatory.

Notebook:

```text
[6.2] Refusal Is Mediated by a Direction
```

Exercises:

```text
1. build harmless/refusal prompt pairs
2. cache residual stream activations
3. compute mean-difference direction
4. layer sweep
5. position sweep
6. add direction to harmless prompts
7. project direction out of refusal prompts
8. random-direction control
9. label-shuffled control
10. PCA/SVD extension
```

Expected visible results:

```text
layer separation plot
steering strength curve
ablation strength curve
PC1 explained variance table
random-direction failure
capability degradation table
```

Core question:

```text
Is refusal really one-dimensional, or does one dimension merely provide a strong control knob?
```

This should be one of the clearest “wow” sections in the extension.

---

## 12. Rewrite guidance for LoRA / DoRA

Do not make LoRA/DoRA a giant disconnected chapter. Make it part of:

```text
How finetuning changes mechanisms.
```

Notebook sequence:

```text
[11.1] LoRA from Scratch
[11.2] DoRA from Scratch
[11.3] LoRA vs Full Finetuning
[11.4] Adapter-Induced Directions and Safe Misalignment Proxies
```

Required visible results:

```text
merged/unmerged parity
rank(delta_W) <= r
SVD spectrum
LoRA-vs-full-finetuning OOD behavior table
adapter-induced activation direction
projection-out behavior reduction
random adapter control
```

For VLM and diffusion LoRAs, make them **bonus sections** until the core PEFT chapter is polished.

---

## 13. Rewrite guidance for Shapley and data attribution

This should be one clean chapter:

```text
[10] Attribution, Shapley, and Data Influence
```

Do not make it a pile of method demos. The central question should be:

```text
When different attribution methods say “this was important,” do they agree?
```

Notebook sequence:

```text
10.1 Exact Shapley on Ground-Truth Games
10.2 SHAP / Integrated Gradients / Token Attribution
10.3 In-Run Data Shapley
10.4 SHAPley vs Mechanistic Patching
```

Expected visible results:

```text
exact Shapley matches brute force
Shapley axioms pass
interaction Shapley recovers XOR/parity
TokenSHAP vs activation patching agreement matrix
Data Shapley identifies influential training examples in tiny run
cases where SHAP and mechinterp disagree
```

This should explicitly distinguish:

```text
training-example importance
input-token importance
activation importance
feature importance
circuit-edge importance
behavioral causal importance
```

---

## 14. Rewrite guidance for representation geometry

Representation geometry should be beautiful, but ruthless.

Notebook:

```text
[6.1] Directions, PCA, UMAP, and Feature Geometry
```

Expected visible results:

```text
days/months PCA or circular structure
space/time coordinate decoding
refusal PCA/SVD
VLM clothing geometry
random-label failure
seed-stability table
```

Acceptance rule:

```text
A UMAP that cannot predict labels is decoration.
A t-SNE that disappears under seeds is decoration.
A PCA plot that only works on cherry-picked examples is decoration.
```

Quantitative requirements:

```text
linear probe
kNN accuracy
silhouette score where appropriate
random-label control
bootstrap stability
OOD prompt/template split
```

---

## 15. Required style and build-system changes

### 15.1 Use a master-file workflow

Original ARENA’s README tells contributors not to edit notebooks or instruction pages directly, but to edit master Python files in `infrastructure/master_files`, from which notebooks, Streamlit pages, and solutions are generated. ([GitHub][4])

The extension should follow the same pattern:

```text
infrastructure/extension_master_files/
  master_5_1_gemma.py
  master_7_1_clip_siglip.py
  master_7_4_visual_token_flow.py
```

Generated outputs:

```text
chapterX/.../*.ipynb
chapterX/.../*_solutions.ipynb
chapterX/.../instructions/pages/*.md
```

Do not hand-maintain divergent notebooks, pages, and solution files.

---

### 15.2 Add anti-minification audits

Add CI:

```bash
ruff format --check arena_ext chapters tests scripts
black --check arena_ext chapters tests scripts
mdformat --check docs chapters
python scripts/audit_no_minified_files.py
python scripts/audit_notebook_roundtrip.py
```

`audit_no_minified_files.py` should reject:

```text
.py file with line length > 240 unless allowlisted
.md file with line length > 400 unless table/URL allowlisted
.ipynb raw JSON not round-trippable through jupytext
solution file without readable functions
test file without readable function names
```

---

### 15.3 Add real ARENA-style audit

Replace the current style audit with checks for:

```text
learning objectives
conceptual intro
at least 3 exercises
visible test cell after each hard exercise
expected-output dropdown
solution dropdown
help / interpretation dropdown
signature result figure or table
common bugs section
limitations section
bonus section
artifact lockfile
verification report
```

Also check that:

```text
solutions are hidden;
solutions are not immediately inline;
expected outputs are not just JSON;
figures have captions;
plots are generated or linked as stable artifacts;
notebook has an actual narrative.
```

---

## 16. Definition of done for a polished section

A section is not done when tests pass.

A section is done when a student can say:

```text
I know what problem this method solves.
I implemented the key functions myself.
I saw each subfunction pass a test.
I saw a beautiful expected output.
I understand what the output means.
I saw a baseline fail.
I saw a negative control fail.
I saw the method work on a toy case.
I saw the method connect to a real model or artifact.
I know what claims are not supported.
I know what research question this unlocks.
```

Every final section should have:

```text
one core claim;
one signature result;
one strong baseline;
one negative control;
one OOD/generalization check;
one causal intervention if relevant;
one limitations box;
one set of further research prompts.
```

---

## 17. Minimum reading links to include in the rewritten course

Each notebook should include only the few papers needed for that notebook. A global reading index can include the broader list.

### Original ARENA foundations

* [ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0)
* [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
* [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
* [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
* [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
* [Indirect Object Identification](https://arxiv.org/abs/2211.00593)

### Frontier mechinterp

* [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647)
* [Sparse Feature Circuits repo](https://github.com/saprmarks/feature-circuits)
* [Refusal is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
* [Gemma Scope](https://ai.google.dev/gemma/docs/gemma_scope)
* [Mamba](https://github.com/state-spaces/mamba)
* [Mamba-2](https://arxiv.org/abs/2405.21060)

### VLMs and vision-language

* [CLIP](https://arxiv.org/abs/2103.00020)
* [Multimodal Neurons](https://openai.com/index/multimodal-neurons/)
* [SigLIP](https://arxiv.org/abs/2303.15343)
* [PaliGemma docs](https://huggingface.co/docs/transformers/model_doc/paligemma)
* [Qwen2.5-VL docs](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
* [LLaVA-OneVision docs](https://huggingface.co/docs/transformers/model_doc/llava_onevision)
* [Molmo](https://github.com/allenai/molmo)

### Image generation

* [DAAM](https://github.com/castorini/daam)
* [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
* [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

### LoRA / DoRA / PEFT

* [LoRA](https://arxiv.org/abs/2106.09685)
* [DoRA](https://arxiv.org/abs/2402.09353)
* [Hugging Face PEFT LoRA docs](https://huggingface.co/docs/peft/package_reference/lora)

### Shapley / data attribution

* [SHAP](https://github.com/shap/shap)
* [shapiq](https://github.com/mmschlk/shapiq)
* [Data Shapley in One Training Run](https://openreview.net/forum?id=HD6bWcj87Y)

### Research-process and writing guidance

The extension should explicitly include a short “how to use this course for research” page. The source pack emphasizes Explore → Understand → Distill, truth-seeking, prioritization, fast feedback loops, and rigorous evidence over elegant narratives.  It also emphasizes that good ML writing requires precise methods, clear figures, rigorous evidence, and limitations.

---

## 18. PR review comments ready to paste

### Blocking comment 1 — PR scope

```markdown
This PR is too large to merge as a course extension. It changes 768 files and adds 41 accepted sections. This should be split into infrastructure/preservation first, then one polished pilot notebook, then one chapter at a time. The current branch is valuable as a prototype/artifact branch, but not as a final ARENA-style course PR.
```

### Blocking comment 2 — original preservation

```markdown
The original ARENA content must remain frozen as the canonical upstream source. Extension chapters should not be interleaved directly into the original config source of truth. Please split original and extension configs and generate a merged config at build time.
```

### Blocking comment 3 — notebook style

```markdown
The new notebooks do not yet match the original ARENA style. They need richer explanatory prose, diagrams, expected-output dropdowns, help/interpretation dropdowns, hidden solutions, common-bug explanations, visible signature results, and a clearer conceptual arc. Passing tests is not sufficient for course quality.
```

### Blocking comment 4 — minified files

```markdown
Several generated Python/Markdown/notebook-support files are effectively minified into one or two enormous lines. This is not reviewable or educational. Add formatting CI and regenerate all files in readable form.
```

### Blocking comment 5 — verification reports are not enough

```markdown
A final verification_report.json is useful for CI, but not sufficient for an ARENA notebook. Each notebook needs visible intermediate tests and a signature result figure/table that the student can interpret.
```

### Blocking comment 6 — VLM chapter structure

```markdown
The VLM section is too overloaded. Split it into CLIP/SigLIP from scratch, CLIP geometry, mini VLM, real VLM visual-token flow, hallucination/arbitration, multimodal SAEs/crosscoders, and VLM capstones. This is the main research topic and should be the most polished chapter.
```

### Blocking comment 7 — claim boundaries

```markdown
The DiffusionGemma claim boundary is good and should be preserved: toy discrete diffusion and NVFP4 local generation are supported, but denoising-step activation capture, diffusion-time patching, throughput, quality benchmarking, and BF16 local inference are not yet supported.
```

---

## 19. The target feel

The final course should feel like this:

```text
I open a notebook.
I immediately understand why the topic matters.
I implement a small thing.
A test tells me if I got it right.
I implement the next thing.
A beautiful plot appears.
The text explains why the plot matters.
A baseline fails.
A causal intervention works.
A real model connection appears.
A limitations box prevents overclaiming.
A bonus section points me toward a real research project.
```

Not like this:

```text
I open a notebook.
There are many imports and stubs.
Tests pass.
A JSON report says accepted.
There is a huge list of methods.
I am not sure what I learned.
```

---

## 20. Final recommendation

Keep the agent’s work, but demote it from “course” to “prototype branch.”

Then rebuild as:

```text
ARENA original: sacred, frozen, preserved.
Extension infrastructure: clean, minimal, separate.
Pilot notebook: one section polished to original ARENA quality.
VLM chapter: rewritten as the flagship.
Other methods: added only when they have a signature result, stepwise tests, controls, and real explanatory prose.
```

The current PR is a good sign that the implementation surface is possible. It is not yet a good sign that the teaching surface is right.

The rewritten version should be smaller, clearer, more opinionated, more visual, and much more ruthless about what counts as a finished result. It should match ARENA by making students **build, see, test, interpret, and extend** — not merely run a verified method registry.

[1]: https://github.com/shinobiultra/ARENA_3.0_Extended/pull/1 "[codex] Extend ARENA 3.0 frontier lab by shinobiultra · Pull Request #1 · shinobiultra/ARENA_3.0_Extended · GitHub"
[2]: https://github.com/shinobiultra/ARENA_3.0_Extended/pull/1/files "[codex] Extend ARENA 3.0 frontier lab by shinobiultra · Pull Request #1 · shinobiultra/ARENA_3.0_Extended · GitHub"
[3]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/docs/original_preservation_contract.md "raw.githubusercontent.com"
[4]: https://github.com/shinobiultra/ARENA_3.0_Extended/tree/codex/arena-frontier-lab-extension "GitHub - shinobiultra/ARENA_3.0_Extended at codex/arena-frontier-lab-extension · GitHub"
[5]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/docs/verification_quality_policy.md "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/docs/hard_exercise_verification_ladders.md "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/chapter5_modern_architectures/exercises/part1_gemma_from_scratch/5.1_Gemma_from_Scratch_exercises.ipynb "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/chapter5_modern_architectures/instructions/pages/01_%5B5.1%5D_Gemma_from_Scratch.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/chapter12_vlm_interpretability/instructions/pages/01_%5B12.1%5D_CLIP_SigLIP_and_VLM_Controls.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/12.1_CLIP_SigLIP_and_VLM_Controls_exercises.ipynb "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/chapter5_modern_architectures/exercises/part1_gemma_from_scratch/solutions.py "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/solutions.py "raw.githubusercontent.com"
[13]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/scripts/audit_arena_style_depth.py "raw.githubusercontent.com"
[14]: https://raw.githubusercontent.com/shinobiultra/ARENA_3.0_Extended/codex/arena-frontier-lab-extension/Extension-Roadmap.md "raw.githubusercontent.com"
