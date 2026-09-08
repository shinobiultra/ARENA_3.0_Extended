# Codex CLI `/goal` Prompts for Salvaging ARENA 3.0 Extended

Use these with Codex CLI. Codex documentation says `/goal` is for long-running tasks with a verifiable stopping condition, so every goal below has a concrete stop condition.

## Goal 0: stop adding content, audit learner surface

```text
/goal You are salvaging ARENA_3.0_Extended so it genuinely feels like
original ARENA. Stop adding new topics. First audit the learner-facing
notebooks and instruction pages, especially 7.1, 8.3, 12.1, 9.1, 5.5,
and 11.1. Compare each to original ARENA chapter1 notebooks. Produce
docs/arena_salvage/current_surface_audit.md with: what the student
implements, what outputs they see, whether the signature result is
generated live or loaded from a report, whether there is a play cell,
whether controls fail convincingly, and whether the result teaches a
mechanism. Do not modify notebooks yet. Stopping condition: the audit
names the top 10 failures and proposes the first 3 rewrite targets.
```

## Goal 1: rewrite 7.1 as a real ARENA notebook

```text
/goal Rewrite chapter7_activation_to_language/exercises/part1_lenses_patchscopes
so it becomes an ARENA-style notebook, not a verification-report wrapper.
Preserve the original ARENA content elsewhere. The new 7.1 must teach
logit lens, tuned lens, and Patchscopes through a concrete model/task
where students can see actual prompts, tokens, layers, activations,
decoded top tokens, and controls. Do not hide the core implementation
in utils or solutions; students must implement logit_lens, top-token
tables, a simple tuned/ridge lens, activation replacement, patchscope
evaluation, and random/text-only controls. Include visible expected
outputs, interpretation dropdowns, solution dropdowns, and at least one
Try It Yourself cell where the student changes prompt/layer/source
activation. The signature result must be a visible table/plot generated
in the notebook, not only a dictionary or verification_report.json.
Stopping condition: 7.1 has a clear one-sentence claim, at least 6
exercises with visible subfunction tests, a layer-by-layer decoding
plot/table, a tuned-lens held-out comparison, a
Patchscope-vs-text-only-vs-random table on >=20 held-out examples or an
exact toy theorem, and all tests pass.
```

## Goal 2: rewrite 8.3 from preflight into actual ACDC/circuit metric teaching

```text
/goal Rewrite chapter8_automated_circuits/exercises/part3_acdc_circuit_metrics
so it no longer presents a final-position preflight as the main result.
Build an ARENA-style path: first a hand-coded toy computational graph
with known ground-truth circuit, then exact node/edge patching, then
ACDC-style pruning, then faithfulness/minimality/completeness/same-size
random controls, then a small real-model fragment. Students must
implement the core patching/pruning/metric functions in the notebook.
The signature result must be a visible circuit graph plus curves/tables
showing the discovered circuit preserves behavior, is minimal, is
complete, and beats same-size random circuits. Stopping condition: the
notebook recovers a known toy circuit exactly, shows at least one
nontrivial real-model circuit fragment or explicitly labels it optional,
and no longer relies on `position_5` localization as the main result.
```

## Goal 3: split and rebuild the VLM flagship chapter

```text
/goal The VLM chapter is the flagship. Split chapter12_vlm_interpretability/part1
into a proper ARENA sequence: 12.1 CLIP/SigLIP from Scratch, 12.2
CLIP/SigLIP Feature Geometry, 12.3 Mini VLM from Scratch, 12.4 Visual
Token Flow in Real VLMs, 12.5 Object Hallucination and Modality
Arbitration, 12.6 Multimodal SAEs/Crosscoders as optional stretch.
Start by implementing 12.1 and 12.3 only. 12.1 must train or load a
tiny CLIP-like model on controlled colored-shape image/caption pairs
and show a retrieval heatmap plus random-caption failure. 12.3 must
build a tiny VLM where object/color/spatial VQA works, then visual-token
patching flips answers and background patching fails. Include images,
captions, examples, tests, expected outputs, controls, and Try It
Yourself cells. Stopping condition: 12.1 and 12.3 are polished,
runnable, and visibly convincing; real VLM paths are only introduced
after the toy VLM works.
```

## Goal 4: make refusal direction a beautiful minimal notebook

```text
/goal Rewrite chapter9_alignment_interpretability/exercises/part1_refusal_directions_safe_steering
into a short, beautiful ARENA-style replication of "Refusal is Mediated
by a Single Direction" on sanitized prompts. Students must cache
activations, compute mean-difference direction, run layer/position
sweeps, add the direction to harmless prompts, project it out of refusal
prompts, and compare random and label-shuffled directions. Add PCA/SVD
explained variance and a one-vs-multi-direction comparison. Keep safety:
do not include harmful procedural completions, only aggregate refusal
labels and sanitized prompts. Stopping condition: notebook shows layer
sweep, steering curve, ablation curve, PCA/SVD table, random-control
failure, and capability side-effect table.
```

## Goal 5: improve Diffusion LM notebook without overclaiming

```text
/goal Improve chapter5_modern_architectures/exercises/part5_diffusion_language_models
so the toy diffusion result is visually and pedagogically compelling.
Keep the honest claim boundary around DiffusionGemma NVFP4: one-prompt
generation proves local loading/generation, not interpretability or
quality. Add noising schedule visualization, denoising trajectory
display, entropy/commitment-time plot, shuffled-label side-by-side
failure, and a Try It Yourself prefix cell. Do not claim denoising-step
activation patching on released DiffusionGemma unless implemented and
verified. Stopping condition: the toy section has a clear mechanism
students can see, with held-out exact-match >95% and shuffled-label near
chance; DiffusionGemma remains a separate runtime proof box.
```

## Goal 6: add learner-surface audit gates

```text
/goal Add a stricter learner-surface audit beyond keyword checks. It
should fail notebooks whose signature result is only a
verification_report.json, whose main output is only a JSON/dict, whose
core implementation is hidden in utils/solutions, or which lack a Try It
Yourself cell, visible expected output, interpretation dropdown,
baseline/control, and limitations. Add scripts/audit_learner_surface.py
and integrate it into the hosted CPU gate without breaking original
ARENA. Stopping condition: the audit fails the current bad 7.1 pattern
and passes a rewritten pilot notebook.
```

## Goal 7: master-file workflow and readability

```text
/goal Ensure all rewritten extension sections follow the original ARENA
master-file workflow rather than hand-edited divergent notebooks/pages/solutions.
Add or adapt infrastructure/extension_master_files so notebooks,
solutions, and instruction pages are generated from readable master
files. Add anti-minification checks for Python, Markdown, and notebooks.
Stopping condition: one pilot section, preferably 7.1, is generated from
a master file; generated files are readable; formatting checks pass.
```

## Goal 8: freeze expansion until pilots pass

```text
/goal Freeze all non-pilot extension sections by marking them roadmap/prototype,
not course_ready, until at least three pilot sections pass human
ARENA-style review. The pilots are 7.1, 12.1/12.3, and 8.3 or 9.1.
Update docs/arena_style_rewrite_status.yml and README to distinguish
prototype coverage from polished course-ready content. Stopping
condition: README no longer claims 41 sections are course-ready unless
they pass the stricter learner-surface audit and human-style checklist.
```

## Goal 9: end-to-end PR-ready salvage

```text
/goal Prepare a PR that salvages the extension by focusing on quality over
coverage. It should include: updated ARENA-style guidance docs, stricter
learner-surface audit, the rewritten 7.1 pilot notebook, and no
unrelated topic expansion. Preserve original ARENA exactly. Run hosted
CPU tests and the new learner-surface audit. Stopping condition: PR diff
is reviewable, 7.1 is genuinely ARENA-style, original ARENA is
unchanged, tests pass, and docs explain what remains prototype vs
course-ready.
```
