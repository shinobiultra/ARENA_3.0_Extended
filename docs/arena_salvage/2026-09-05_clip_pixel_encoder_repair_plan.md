# Complete the CLIP/SigLIP encoder learning path

## Verified gap

The roadmap at lines 3566–3630 requires image and text encoders, projection
heads, normalized embeddings, temperature, contrastive learning, zero-shot
classification, retrieval, and typographic controls. The current 12.1 learner
has 47 cells. Its training exercise, `train_toy_clip_projectors`, receives
`toy_batch.image_features` and `toy_batch.text_features`. The shared helper
`arena_ext/vlm_interpretability.py` builds those engineered features and trains
only two linear projection matrices. Rendering an image grid beside that
exercise does not teach pixel and token encoders from scratch.

The real CLIP/SigLIP extraction and activation-patching path remains useful.
Preserve it, and retain the small projector exercise only as an introductory
baseline. This is work within the existing priority pilot, not a new topic.
Original ARENA remains untouched; its CNN/ResNet and transformer implementation
progressions are the teaching comparators.

## Required learner result

Students should build a complete tiny dual encoder accepting RGB images and
caption token IDs, train both encoders and projection heads, and see a held-out
retrieval matrix and zero-shot class predictions. Compare CLIP and SigLIP
objectives on the same examples and architecture, with matched seeds and
controls. Keep the official-model comparison at the end. Do not claim a large
checkpoint reproduction from the small synthetic training exercise.

Use distinct rendered instances for train and evaluation, varying position,
size and background without embedding semantic labels in image features.
Captions go through an explicit vocabulary and trainable token embeddings.
Handle duplicate semantic positives in batch construction and retrieval metrics;
do not penalize an otherwise correct image merely for selecting a different
instance with the same caption. State which semantic classes and templates are
shared and which are held out.

Core implementations must live in the notebook: pixel patch/CNN encoder,
caption encoder and padding/EOS handling, projection and normalization,
learnable scale/bias, both losses, training loop, and retrieval/classification.
Use exact small forward and gradient fixtures, tests that both towers actually
learn, pixel perturbations, padding invariance, and broken implementations that
cannot pass by returning label-derived feature vectors.

The visible comparison needs shuffled pairing, blank/noisy image, wrong-caption,
and untrained/frozen-encoder baselines, plus measured seed variation. Add a
play cell that changes pixels or caption words and immediately redraws retrieval.
Provide expected outputs, help, complete answers, interpretation and limitations.
Verify the completed learner from its own dropdowns in a fresh kernel.

The roadmap also requires typographic attacks. Inspect the existing real CLIP
path and design a controlled word-overlay dataset with placement/background and
irrelevant-word controls. Keep a negative attack result if measured; synthetic
encoders without OCR training do not establish robustness of pretrained CLIP.
Do not silently count this requirement as done after adding encoder training.

## Implementation and verification scope

Own the existing 12.1 section sources/tests, paired notebooks, a readable
canonical lesson/builder, its instruction page, figures, expected outputs and
scope metadata. Keep unrelated shared helpers and original lessons intact.
Inspect the full paired learner/solution and original comparator before edits.
Re-run the real section evidence after any scientific change; the earlier
readability manifest refresh is only cosmetic evidence and cannot cover it.
Update roadmap statuses only to the extent the actual learner and measured
controls satisfy each requirement.

## Completed 2026-09-05

The gap is repaired in the existing 12.1 section: twenty exercises and 52
completed-learner code cells run in a fresh kernel. Core encoder, objective,
training, retrieval and attack metrics are visible and editable; all legacy
answer dropdowns are now pasteable. Forty section tests pass. The actual
288-image held-out experiment and 48-scene pretrained typography experiment
are documented in the section expected outputs and latest HANDOFF entry.
Both encoder roadmap rows are now TAUGHT; typography is only PREFLIGHT_ONLY
because causal feature localization and mitigation are still absent.

The independent reference report is regenerated after scientific changes,
including the original pinned-model controls. Preservation, style, hard
ladders and readability pass; the full repository test suite remains at
415 passes and one artifact-hygiene failure for untracked required payloads.
No original ARENA content, release threshold for an existing experiment,
staging state, commit, workflow or subagent policy was changed.
