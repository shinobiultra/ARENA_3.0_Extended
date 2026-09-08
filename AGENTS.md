# AGENTS.md — ARENA 3.0 Extended Salvage Rules

Use this file at the repository root while salvaging the ARENA extension.

## Sacred rule

Original ARENA is the canonical base. Preserve it append-only. Do not edit original ARENA notebooks, instruction pages, or assets unless the task explicitly says it is an upstream compatibility patch.

## Current problem

The extension currently passes many verification gates but often fails the learner experience. A notebook that imports hidden utilities, asks students to fill stubs, then prints a JSON-like report is not ARENA-style.

## Definition of ARENA-style

A section must have:

- one concrete question;
- one convincing visible result;
- student-implemented core functions;
- immediate sub-function tests;
- expected outputs;
- help/interpretation dropdowns;
- solution dropdowns;
- controls and baselines;
- at least one “Try It Yourself” play cell;
- real-model or paper connection;
- honest limitations.

## Do not add topics

Do not add new methods, papers, chapters, or model families until the pilot sections are rewritten to this standard. The priority pilots are:

1. 7.1 Logit Lens / Tuned Lens / Patchscopes.
2. 12.1/12.3 CLIP/SigLIP and Mini VLM.
3. 8.3 ACDC and Circuit Metrics.
4. 9.1 Refusal Direction.

## Signature result bar

Every section needs a result the student can see and understand. Examples:

- layer-by-layer decoded tokens;
- attention or patching heatmap;
- retrieval matrix;
- circuit graph;
- faithfulness/minimality/completeness curves;
- steering/ablation curves;
- denoising trajectory;
- LoRA singular-value spectrum;
- SHAP vs patching agreement matrix.

A dict, scalar report, or `verification_report.json` is not enough.

## Hidden implementation rule

Shared utilities are allowed for boring plumbing. The method being taught must be implemented in the notebook. If the student cannot see and modify the core method, the notebook fails.

## Verification reports

Verification reports are supporting evidence. They must not be the main teaching artifact.

## Controls

Every plot or headline metric must have a control. White noise, random labels, shuffled labels, random activations, same-size random circuits, background-region patches, text-only/image-only baselines, and random adapter directions should fail visibly.

## Finish message

When finishing a task, report:

- changed files;
- original ARENA preservation status;
- student-facing improvements;
- signature result;
- controls and baselines;
- play cells;
- tests run;
- limitations left.
