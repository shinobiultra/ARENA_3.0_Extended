# [12.3] Mini VLM from Scratch Verification Assets

This section is a learner-facing MiniVLM lab, not a report wrapper. Students
implement the visual-token cache, multimodal sequence, VQA metrics,
bounding-box mapping, token replacement, exact toy oracle, trained model, and
causal patching path in the notebooks.

The release report reruns the complete 260-step training experiment on CUDA.
Acceptance requires held-out muted-style VQA to beat text-only, image-only, and
shuffled-visual controls, and requires object-token counterfactual patches to
flip both color and shape answers while background and same-size random-region
patches preserve the clean answer. Full-sequence patches must reproduce the
counterfactual run.

The visible signature panel and layer-position heatmap live in the chapter's
`instructions/assets` directory. The exact toy oracle is reported separately
from the learned MiniVLM result.
