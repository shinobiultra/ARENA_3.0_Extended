# [12.3] Mini VLM from Scratch

Build the complete RGB-to-word path: learned patch embeddings, a bidirectional
vision transformer, a connector and a causal word decoder. Train word-level
questions and multiword captions, then test which image activations change the
actual generated words. RGB/occupancy tokens and an exact additive game provide
short warm-ups before the learned model.

The readable `lesson.md` builds the paired notebooks and full instruction page.
Nineteen exercise groups contain the core methods, immediate independent tests,
expected outputs, help and complete working solutions. Three executable play
activities inspect patch boundaries, edit question words and move interventions
across stages. The default experiments run on CPU in about five to ten minutes;
allow two to three study days for the exercises and interpretation.

The signature results are generated in the notebook:

- Learning curves for all three seeds and all four training/control modes.
- Frozen familiar/palette/compositional evaluations, with every seed visible.
- Actual word/caption galleries: familiar success and unseen-combination failure.
- Stage-by-region patching with object, background, unrestricted random,
  all-visual and full-sequence conditions; all random draws remain visible.

The reference learner averages 89.6% exact generation on new familiar solid
images and 85.8% on their muted versions. Captions on the three excluded
color–shape combinations have zero exact match across all three seeds. This
is retained as an observed learning failure, not repaired by tuning final data.
A contextualized background token can carry object information. After the final
decoder block, visual-only patches cannot change word states, while full-sequence
replacement reproduces the donor generation. Every conditional flip figure
states its joint-correct endpoint denominator.

```bash
uv run python chapter12_vlm_interpretability/exercises/part3_mini_vlm_from_scratch/build_notebooks.py --check
uv run python chapter12_vlm_interpretability/exercises/part3_mini_vlm_from_scratch/build_notebooks.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run python -u chapter12_vlm_interpretability/exercises/part3_mini_vlm_from_scratch/execute_learner.py --scope full --output-directory /tmp/arena-word-vlm-new-full-run --publish
```

Choose a new output directory for each execution. The full route trains all
12 models afresh from the learner's own answers. Numerical-only execution
skips trained-model experiments and cannot publish a full result.
`word_evidence.py` verifies source/notebook/figure provenance and recomputes all
metrics from actual generated words. `solutions.run_gpu_test()` separately
trains four fresh seed-0 CUDA models, freezes on development, and measures
final generation and 95 stage/region conditions; it does not relabel saved CPU
results as a CUDA run.

The published learner completed all 34 code cells, twelve fresh fits and the
full experiments with unchanged source; the exact runtime is recorded in
`expected_outputs/learner_execution.json`. Its raw-token metrics and seven
published figure hashes validate. All 43 section tests pass.
`verification_report.json` records the separate CUDA check and its source
freshness. Candidate runs and scoped earlier learner snapshots remain in
`expected_outputs/word_candidates`; weight files are not included. Original
ARENA chapters remain unchanged.
