# [12.5] Object Hallucination and Modality Arbitration

This section keeps its historical `part2_object_hallucination_arbitration`
directory so shared chapter configuration does not change, but the existing chapter
plan assigns the topic to **12.5**. The learner-facing files use that number.

The CPU lab uses an explicit pixel-and-prompt ground-truth VLM. Its inference API
accepts only image tensors and prompt tokens. Labels, conditions, bounding boxes,
and split metadata are excluded from model inference. The benchmark crosses four
colors with two shapes in development, validation, and test renders; every prior
sign is balanced across true and false labels.

Students implement prompt encoding, benchmark construction, image-to-patch geometry,
behavior metrics, cache capture, token activation replacement, the full controlled
counterfactual patch sweep, and validation-only layer selection. The signature result
reports joint/conflict accuracy and hallucination rate, then shows object patches
flipping untouched test answers while two distinct, equal-size, pairwise-disjoint
regions outside target and distractor geometry have zero effect.

The optional pinned-Qwen path is a visual-counterfactual preflight. It uses
neutral-prior image pairs, not actual hallucination repair cases. Its gates cover
behavior plus donor-directed object patching, background, random, and full-visual
controls; passing them does not distinguish the four failure hypotheses. Full-visual patching is a positive control; intermediate-layer
equality with the donor output is not required after visual and text states have mixed.

The eight answer dropdowns now complete the learner without missing support
functions. Provided geometry, metric, and experiment helpers are visible alongside
the stubs. Real preflight checks reject wrong checkpoint IDs/revisions, nonfinite
values, impossible rates, inconsistent denominators, incomplete controls, and
aggregates that disagree with raw margins and patch rows. These consistency checks
do not turn the preflight into a complete causal hallucination study.

## Runtime and evidence

- Learner time: 4-6 hours.
- Solved CPU execution: seconds; no GPU is used.
- CUDA release verification: expected 12-25 minutes, hard gate at 40 minutes and
  24 GB peak allocation.
- The checked-in CPU smoke fixture and notebook outputs are executed evidence.
- No real-Qwen report is accepted. Hallucination repair and stage-specific diagnosis
  must be implemented and reviewed before this candidate can satisfy that scope.

Run the section tests with:

```bash
CUDA_VISIBLE_DEVICES='' uv run pytest -q \
  chapter12_vlm_interpretability/exercises/part2_object_hallucination_arbitration/tests.py
```

Rebuild sources and then refresh the evidence hashes with:

```bash
CUDA_VISIBLE_DEVICES='' uv run python \
  chapter12_vlm_interpretability/exercises/part2_object_hallucination_arbitration/build_notebooks.py
CUDA_VISIBLE_DEVICES='' uv run python \
  chapter12_vlm_interpretability/exercises/part2_object_hallucination_arbitration/build_notebooks.py --finalize
```
