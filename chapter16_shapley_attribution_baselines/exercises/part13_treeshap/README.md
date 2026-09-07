# 16.13 TreeSHAP from First Principles

The editable course source is [lesson.md](lesson.md). `build_notebooks.py` creates a paired exercise
notebook and own-answer solution notebook in a new draft directory; it does not publish them into
the course tree.

Allow 4–6 hours for self-study. Nine implementation exercises progress from exact tree routing and
coalition values to Shapley contraction, path extension and unwinding, recursive TreeSHAP, ensemble
linearity, empirical interventional games and an intentionally non-Shapley Saabas control. Every
exercise has an immediate test, expected output, help, interpretation and an exact following
`<summary>Solution</summary>`. The complete fitting/evaluation and mechanism-diagnostic source is
inserted visibly into both notebooks from its current AST.

The signature result combines a real held-out prediction waterfall with row-by-feature attribution
heatmaps. Seventeen figures and two editable plays connect the algorithm to a controlled
repeated-feature tree, correlated missing-feature games, all fitted models and seeds,
finite-background sensitivity, held-out deletion orderings and two runtime axes.

## Frozen study

The study fits decision trees, random forests and gradient-boosted trees at three predeclared seeds
using clean and shuffled targets: 18 fresh fits and 486 component trees. It uses all 1,030 UCI
Concrete Compressive Strength rows. Identical seven-ingredient recipes remain together across 758
training, 153 explanation and 119 audit rows. The primary empirical interventional game uses 64
intact training donors selected by a source-ID hash. Rankings and the background are saved before
the audit split is loaded.

The first complete native run took 12.68 seconds on the development CPU. Its audit mean-squared
errors were:

| Predictor | Audit MSE (MPa²) |
|---|---:|
| Training-target mean | 244.6794 |
| Clean decision tree, all three seeds | 86.3754 |
| Clean random forest, seeds 0–2 | 55.6805, 53.8209, 53.7356 |
| Clean gradient boost, all three seeds | 58.1941 |
| Shuffled-target fits | 244.2969–277.8811 |

The repeated clean tree and boost seeds are reproducibility checks: under these deterministic
settings they make identical full predictions. They are not three independent uncertainty estimates.
Forest predictions vary because bootstrap sampling varies.

For the default audit recipe (source row 1), the target is 61.89 MPa and the primary forest predicts
57.8987 MPa. Its interventional base is 35.5558 MPa. Contributions in input order are `[16.5993,
-1.9227, 0.0389, 2.5455, -0.5817, -0.1425, 1.3857, 4.4203]` MPa. The path-dependent base is 36.5205
MPa. These explain a fixed prediction relative to two specified missing-feature games; they are not
causal effects of changing recipe quantities.

The float64 exhaustive oracle agrees with recursive TreeSHAP for all 256 coalitions, both games,
every explanation/audit row and the first clean seed of all three families; the maximum
audit-coordinate error is `2.49e-14`. Path SHAP, interventional SHAP and Saabas happen to give
identical mean-absolute rankings for the clean forest and boost in this run, so their audit deletion
curves overlap. The clean tree's interventional order differs only at the three-feature prefix. The
deletion panel is therefore a model/game diagnostic, not evidence that TreeSHAP beats Saabas.

The runtime result is also a useful negative control. On balanced depth-2…8 trees, the readable
Python recurrence is slower than the vectorized exhaustive oracle throughout (roughly 100 ms versus
2 ms for eight depth-eight queries). It wins on the separate fixed-depth dummy sweep only when the
enumerator is forced to visit coalitions for 12 and 15 supplied features. Polynomial scaling does
not imply a lower runtime at these small sizes and constant factors.

## An informative failed reference control

The frozen expectation that unmodified SHAP 0.49.1 interventional output would meet `atol=2e-5,
rtol=2e-6` was falsified. Its compiled adapter narrows learned float64 thresholds to float32. In the
primary-forest example this misroutes two explanation rows, causing about 0.4902 MPa local
reconstruction/parity error. Every affected model-row pair, raw array, error and failed flag is
retained across the full comparison; the tolerance is not increased.

Across the full run, raw vector parity fails on 469 of 4,896 model-row pairs and raw prediction
reconstruction fails on 18, with maxima 0.376032 and 0.616826 MPa respectively. The globally
threshold-corrected reference reaches maximum local and reconstruction errors of `2.825e-6` and
`3.142e-6` MPa. Separately, the raw compiled reference agrees with exhaustive enumeration of its
rounded-threshold game to `1.456e-6` across all 816 exact-model rows, identifying the source of its
disagreement with the original fitted models.

A separately labeled compatibility reference floors every compiled-explainer threshold to the
greatest float32 value no larger than the original. This preserves sklearn routing for every
float32-representable query, and it meets the original strict tolerance. The fitted models and
student implementation keep their original thresholds. The proof and observed split are documented
in [reference_boundary_addendum.md](reference_boundary_addendum.md).

## Run locally

Use the repository's existing `uv` environment. Data downloads on first use from the byte-pinned UCI
source. No GPU is needed. Give every execution a new output directory.

```bash
uv run python chapter16_shapley_attribution_baselines/exercises/part13_treeshap/build_notebooks.py --check
uv run python -m pytest -q \
  chapter16_shapley_attribution_baselines/exercises/part13_treeshap/treeshap_methods_tests.py \
  chapter16_shapley_attribution_baselines/exercises/part13_treeshap/treeshap_reference_tests.py
uv run python \
  chapter16_shapley_attribution_baselines/exercises/part13_treeshap/execute_learner.py \
  --output-directory /tmp/treeshap-own-new
CUDA_VISIBLE_DEVICES='' uv run python \
  chapter16_shapley_attribution_baselines/exercises/part13_treeshap/reproduce_native.py \
  --learner-directory /tmp/treeshap-own-new \
  --output-directory /tmp/treeshap-fresh-new
```

The full protocol, source-row IDs, hashes, model settings and background ordering are in
[protocol.md](protocol.md), [protocol_freeze.json](protocol_freeze.json),
[data_manifest.json](data_manifest.json) and [author_reference.json](author_reference.json). Primary
reading is Lundberg, Erion & Lee, [*Consistent Individualized Feature Attribution for Tree
Ensembles*](https://arxiv.org/abs/1802.03888), Section 3 and Algorithms 1–2, with the current [SHAP
`TreeExplainer`
documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
