# [12.1] CLIP, SigLIP, and VLM Controls Verification Assets

Generated support files for the roadmap verification contract.

- `artifacts.lock.yml` pins the current synthetic VLM ladder plus real CLIP,
  SigLIP, Qwen2.5-VL rendered-shape controls, hidden visual-token activation
  patching controls, and clothing-geometry controls.
- `12.1_CLIP_SigLIP_and_VLM_Controls_exercises.ipynb` contains the learner
  stubs and visible tests.
- `12.1_CLIP_SigLIP_and_VLM_Controls_solutions.ipynb` runs the reference
  solution checks and committed report assertions.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records concrete retrieval,
  grounding, activation-patching, and generation acceptance metrics.

The current GPU report verifies pinned CLIP and SigLIP retrieval plus
object-region counterfactual patching on rendered red-square / blue-circle
controls against both background and same-size random-region controls, and
pinned CLIP/SigLIP hidden visual-token activation patching at
`vision_model.embeddings`. Object-token activation patches must flip the
contrastive answer, background and same-size random-token patches must preserve
it, and full visual-sequence patches must match the corrupt visual sequence.
The report also verifies pinned Qwen2.5-VL 3B generation on the same
counterfactual controls and a deterministic clothing garment/color/style
geometry ladder with spurious text-prior and seeded permuted-label controls.
