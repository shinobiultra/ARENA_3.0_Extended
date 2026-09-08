# 16.14 DeepSHAP: When Does Fast Attribution Match Exact SHAP?

The editable source is [lesson.md](lesson.md). It generates a 114-cell learner notebook with 70
code cells, nine student exercises, 17 visible figures and two editable play cells. Each exercise
has an immediate independent test, expected output, help, interpretation and an exact following
`<summary>Solution</summary>`. The own-answer definitions are AST-matched to
`deepshap_methods.py`.

The verified course surfaces are:

- [exercise notebook][exercise-notebook];
- [solution notebook][solution-notebook];
- [instruction page][instruction-page].

`publish_study.py` generated all three from the exact learner-v3 and reproduction-v3 evidence after
browser review passed for 42 dropdowns, 34 rendered equations and all 17 figures.

## What the learner builds

Students implement paired forward states, ReLU Rescale slopes, modified multiplier propagation,
single-reference contributions, background averaging, intact-donor coalition values, exact
Shapley contraction, bounded RevealCancel groups and ranked replacement curves. Training, raw-unit
folding, model reconstruction, explanation and evaluation remain visible. Only mechanical tensor,
shape, identity and cloning checks are imported as named plumbing.

The study uses all 1,030 UCI Concrete rows in a recipe-disjoint 758/153/119
train/explanation/audit split. It fits three clean and three shuffled-target float64 ReLU networks
plus a ridge control. The empirical game uses 64 intact training donors and all 256 feature
coalitions. Every model and all 272 held-out rows remain in the comparison.

The threshold and `min` circuits show the central limitation: DeepSHAP can be complete while
differing from exact feature Shapley, and equivalent network representations can receive different
Rescale allocations. Affine, ridge, dummy, constant, near-zero epsilon, mean-reference,
gradient-times-delta, shuffled-target, background and random-order controls make that claim
falsifiable.

## Executed result

The complete own-answer learner-v3 run passed in `17.277692300` seconds on CPU. It executed 114
cells, all seven fresh fits, nine completed exercises, 17 embedded figures and both editable plays.
A separate fresh reproduction refit all seven models and matched 842 deterministic arrays covering
5,887,125 elements plus 603 scalars exactly, with maximum error zero.

Clean-model audit MSE was `31.04`–`36.58` MPa², versus `99.65` for ridge, `244.68` for the
training-target mean and `452.79`–`643.67` for shuffled-target controls. Across all 272 held-out
rows, the clean MLPs' mean absolute DeepSHAP-minus-exact coordinate errors were `1.415`, `1.585`
and `1.943` MPa even though maximum completeness residuals stayed near `1e-13` MPa. Ridge agreed
with exact Shapley to floating-point precision, and the student Rescale implementation agreed with
installed `shap.DeepExplainer` to `1.71e-13` MPa.

The signature figures retain all signed coordinates and show performance controls, all-row parity,
error heatmaps, completeness beside exactness, background sensitivity and globally ranked
replacement curves. The fixed source row has prediction `70.6716` MPa and base `35.5483` MPa;
its coarse-aggregate attribution is `3.3873` MPa under exact Shapley and `6.9269` MPa under
DeepSHAP. The row was fixed before audit evaluation rather than selected for a large discrepancy.

The supported method is deliberately narrow: a scalar-output `nn.Sequential` chain of distinct
CPU-float64 `nn.Linear` and `nn.ReLU` modules. Empirical donor replacement is a model computation
on observed recipes, not evidence of a physical causal effect. Completeness and independent-library
parity establish correct DeepSHAP execution; neither establishes exact feature-game agreement.

## Reproduce from source

Use the repository `uv` environment. The output directories below must not already exist.

```bash
section="chapter16_shapley_attribution_baselines/exercises/part14_deepshap"

uv run python "${section}/build_notebooks.py" --check
uv run python -m pytest -q \
  "${section}/deepshap_methods_tests.py" \
  "${section}/deepshap_study_tests.py" \
  "${section}/deepshap_contract_tests.py" \
  "${section}/deepshap_reference_tests.py"

learner_output="$HOME/.cache/arena_extended/deepshap/learner-readme-v1"
fresh_output="$HOME/.cache/arena_extended/deepshap/reproduction-readme-v1"

uv run python "${section}/execute_learner.py" \
  --output-directory "$learner_output"
uv run python "${section}/reproduce_native.py" \
  --learner-directory "$learner_output" \
  --output-directory "$fresh_output"
```

Primary reading is Lundberg and Lee, [*A Unified Approach to Interpreting Model
Predictions*](https://arxiv.org/abs/1705.07874), and Shrikumar, Greenside and Kundaje,
[*Learning Important Features Through Propagating Activation
Differences*](https://arxiv.org/abs/1704.02685). The data come from Yeh,
[*Concrete Compressive Strength*](https://doi.org/10.24432/C5PK67). The exact protocol, frozen
source identities, background ordering and provenance are recorded in `protocol.md`,
`protocol_freeze.json`, `data_manifest.json` and `author_reference.json`.

[exercise-notebook]:
  16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP_exercises.ipynb
[solution-notebook]:
  16.14_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP_solutions.ipynb
[instruction-page]:
  ../../instructions/pages/14_[16.14]_DeepSHAP_When_Does_Fast_Attribution_Match_Exact_SHAP.md
