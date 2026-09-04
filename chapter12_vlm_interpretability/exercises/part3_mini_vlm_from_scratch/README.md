# [12.3] Mini VLM from Scratch Verification Assets

This section is a learner-facing MiniVLM lab, not a report wrapper. Students
implement the frozen RGB-plus-occupancy patch encoder, detached visual-token
cache, learned connector, visual-prefix insertion, causal self-attention block,
complete two-layer decoder, answer loss and metrics, bounding-box mapping,
token replacement, exact toy oracle, training loop, modality controls, and
causal patching path in the notebooks. The architecture is introduced in
dependency order rather than supplied as a complete class before Exercise 1.

The next release report must rerun the complete 260-step training experiment on CUDA.
Acceptance requires held-out muted-style VQA to beat text-only, image-only, and
shuffled-visual controls, and requires object-token counterfactual patches to
flip both color and shape answers while background and same-size random-region
patches preserve the clean answer. Full-sequence patches must reproduce the
counterfactual run.

The visible signature panel and layer-position heatmap live in the chapter's
`instructions/assets` directory. The exact toy oracle is reported separately
from the learned MiniVLM result. The checked-in report predates the causal
decoder rewrite and must not be treated as evidence for the new architecture.
