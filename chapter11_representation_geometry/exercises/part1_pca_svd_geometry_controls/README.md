# [11.1] PCA, SVD, and Geometry Controls

Start with `11.1_PCA_SVD_and_Geometry_Controls_exercises.ipynb`. The solution notebook
includes actual outputs. The lesson asks whether recognizing a calendar word,
recovering chronological geometry and changing an answer are the same claim.

Eleven exercises teach PCA/SVD, circular regression, calendar-order permutations,
kNN, trustworthiness/continuity, Procrustes alignment, minimum-norm edits,
mean ablation, native capture, semantic token alignment and full-vocabulary scoring.
Thirteen figures connect exact mathematical controls to actual Mistral-7B calendar
interventions. Four play cells vary exact-model angles, nonlinear views, continuous
real-model coordinates and an individual real prompt.

From the repository root:

```bash
uv run python -m pytest chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/tests.py -q
uv run python chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/execute_learner.py --scope full --output-directory ~/.cache/arena_extended/my-geometry-study
```

Choose a new output directory for each run. Numerical work takes seconds on CPU;
the complete recorded notebook took 384 seconds and 13.62 GiB on the RTX 5090
Laptop GPU, excluding the first checkpoint download. `--scope foundations` skips
pretrained-model calls explicitly. `ARENA_GEOMETRY_MODEL_CACHE` may select an
existing Hugging Face cache; the checkpoint revision stays pinned.

`lesson.md` is the editable source. The builder completes exercise stubs only
from their own solution dropdowns. `publish_study.py --learner-directory ...`
checks that actual executed scientific code matches and publishes measured
outputs. The report runner independently reloads the model, refits training
probes and reproduces 6,656 fixed-batch outcomes across all 32 draws of each null.

The 579 new-template prompts, all 5,634 donor pairs, 120 nonlinear fits and
negative results are retained in `expected_outputs/measured_geometry`. Model
weights stay in the model cache. Raw activation tensors stay in the named local
learner run and can be regenerated. Archived development and prior Pythia material
is provenance, not current course evidence.

The result is partial causal evidence on two small task families. Permuted calendar
order can also work, mean ablation contributes to the effect, class-disjoint
accuracy drops, and t-SNE is explicitly transductive. Years, cities, geographic
coordinates and historical-time interventions remain separate unfinished work.

CPU least squares uses an explicit SVD driver. The initial pivoted-QR run and its failed fresh comparison are archived in `own_full_release_v1` and `reproduction_failure_v1`; tiny solver differences crossed BF16 rounding boundaries. The corrected full run retains all original scientific choices.

The corrected full own-answer run and fresh CUDA comparison pass. All 6,656
checked outcomes across 208 condition batches match exactly, including all null
draws. The complete 71-cell notebook contains 35 executed code cells, sixteen
immediate numerical tests and thirteen inspected figures. The final formatting
pass preserves the exact Python syntax trees; it does not change the experiment.

A documentation-only refresh can reuse a completed native comparison by setting
`ARENA_GEOMETRY_REPRODUCTION_MANIFEST` to its archived `manifest.json`.
This checks the archive, every native source, the current own-answer Python,
and all 6,656 actual reproduced rows. Any scientific change rejects reuse.
The report explicitly says when CUDA evidence was reused and gives its hashes;
omitting the variable runs a new model comparison. The final prose refresh uses
`expected_outputs/fresh_cuda_release_v3/manifest.json`. An intervening CUDA-memory
failure before any compared examples remains in `fresh_cuda_resource_failure_v4`.
