# [12.1] CLIP, SigLIP, and VLM Controls

Start with the exercise notebook, or use `lesson.md` as the readable source.
The paired solution contains the same complete method implementations and
outputs from a fresh kernel completed solely from the learner's own answers.
Original ARENA chapters 0–4 remain unchanged.

The progression has twenty exercises. A short engineered-feature projector
warm-up leads to a complete RGB CNN, causal caption block, explicit EOS pooling,
normalized projections, both losses, semantic retrieval and optimizer loop.
Both towers train on actual pixels and token IDs. The final gallery has 288
fresh instances of twelve familiar classes, separate from 576 training images.
Three matched model seeds compare full training with frozen towers and shuffled
pairing; blank/noise pixels and wrong-color captions test the evidence source.

The visible results are training curves, class-mean retrieval matrices, all-seed
control plots, actual correct/incorrect predictions, and editable pixel/caption
play cells. Fresh-instance classification uses class descriptions without a
separate classifier, but does not claim unseen-class generalization.

Pinned CLIP and SigLIP retain their two-scene embedding and activation-patching
preflight, including object/background/same-size random/full-sequence controls.
The learner reconstructs official model logits from its extracted embeddings.
Qwen's two-scene generation check remains behavioral evidence only. A separate
48-scene pretrained CLIP experiment uses zero-shot class descriptions and ten
word-overlay conditions, with matched occlusion and semantic controls. Its
margin changes and targeted flips are computed by the learner. A background
word card never overlaps the object; a blank object card controls for covering
part of the shape. Feature localization and mitigation of typography remain
further work.

## Reproduce

From the repository root:

```bash
uv run python chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/build_notebooks.py
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run python chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/execute_learner.py
uv run pytest -q chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/tests.py chapter12_vlm_interpretability/exercises/part1_clip_siglip_vlm_controls/pixel_tests.py
uv run python scripts/run_extension_verification_reports.py --section 12.1
```

Building resets notebook outputs; execute afterwards. The default learner uses
CUDA if available. `ARENA_CLIP_CPU_ONLY=1` explicitly skips the real checkpoints;
it still executes both tiny-encoder objectives and all controls. Set thread
counts as above for predictable CPU overhead on this machine. The fresh kernel
retains its completed notebook at `/tmp/clip_completed_learner.ipynb`, including
partial outputs if a cell fails.

`pixel_clip.py` contains readable reference bodies copied into the lesson;
only rendering, data containers and plotting are imported as plumbing.
`pixel_tests.py` contains independent forward/gradient/optimizer fixtures,
semantic and causal tests, deliberately broken implementations, and answer
completion checks. The learner passes its own functions to those tests.
`typography.py` supplies rendering and batching, while the notebook shows the
comparison loop and implements the attack metric. `solutions.py` recomputes
supporting reference evidence through the release harness.

`artifacts.lock.yml` pins weights, data seeds and the expanded evidence inputs.
Reports support the measured claims; the notebook never loads a stored report
to manufacture its experiment outputs. This is a controlled model-organism
lesson, not a natural-image benchmark or a full-scale CLIP/SigLIP reproduction.
