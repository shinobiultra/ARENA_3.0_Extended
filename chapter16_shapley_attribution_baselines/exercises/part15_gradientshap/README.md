# 16.15 Expected Gradients: Which Expectation Does GradientSHAP Estimate?

The editable source is [lesson.md](lesson.md). It builds a 126-cell learner notebook with 79 code
cells, nine student exercises, 20 visible figures and two editable play cells. Each exercise has an
immediate independent test, expected output, help, interpretation and an exact following
`<summary>Solution</summary>`. The own-answer definitions are AST-matched to
`gradientshap_methods.py`.

The verified course surfaces are:

- [exercise notebook][exercise-notebook];
- [solution notebook][solution-notebook];
- [instruction page][instruction-page];
- [published execution evidence](expected_outputs/README.md).

`publish_study.py` generated the notebook pair, instruction page, 20 namespaced figures and evidence
records from the exact learner-v3 and reproduction-v3 artifacts after browser review passed.

## What the learner builds

Students implement deterministic per-source donor/path schedules, interpolation, input gradients,
per-draw Expected Gradient contributions, ordinary plug-in Monte Carlo standard errors,
full-background Expected Gradients, Gauss-Legendre path quadrature, Gaussian-noise contributions
and ranked coalition replacement curves. Model fitting, raw-unit folding, reconstruction,
explanation, comparison, ranking and evaluation remain visible in short notebook cells. Only named
tensor, shape, identity, persistence, package-reference and plotting plumbing is imported.

The study uses all 1,030 UCI Concrete rows in a recipe-disjoint 758/153/119
train/explanation/audit split. It refits three clean and three shuffled-target float64 ReLU networks
plus a ridge control, checks all 21 known 16.14 model-state hashes, and explains all 272 held-out
rows for all seven models. The audit split is loaded only after the fitted and prepared explanation
state has been persisted.

The controls separate several questions that package names can blur. Finite Monte Carlo error is
measured against approximate 512-node path quadrature and 64 independent sampling schedules;
quadrature drift is reported separately. Exact 256-coalition feature Shapley, DeepSHAP and endpoint
gradient-times-delta probe different allocation rules. Five empirical backgrounds, three Gaussian
noise conditions, shuffled-target models, ridge, the training mean, 100 random replacement orders
and path/hybrid distances expose sensitivity to the predictor, baseline support and evaluation
recipe. Same-draw adapters compare the student estimator with SHAP 0.49.1 and Captum 0.9.0 while
retaining each package's actual draw convention.

## Executed result

The own-answer learner-v3 run passed in `60.045227675` seconds on CPU. It executed all 126 cells,
all seven fresh fits, nine completed exercises, 20 embedded figures and both editable plays. A
separate CPU process refit all seven models without cached fitted inputs and matched 1,442 arrays
covering 464,237,288 elements, 2,730 records and 872 scalars exactly, with maximum absolute error
zero.

Clean-model audit MSE was `31.04`–`36.58` MPa², versus `99.65` for ridge, `244.68` for the
training-target mean and `452.79`–`643.67` for shuffled-target controls. Across all 272 held-out
rows, mean absolute 512-node-quadrature-minus-exact-Shapley coordinate differences were
`1.268`–`2.002` MPa for the clean MLPs and `4.311`–`4.934` MPa for the shuffled controls; ridge
agreed to floating-point precision. Mean absolute K=2048-Monte-Carlo-minus-quadrature differences
were `0.214`–`0.235` MPa for the clean MLPs, `0.617`–`0.659` MPa for shuffled controls and
`0.079` MPa for ridge. These are observed discrepancies, not acceptance tolerances.

The 64-schedule diagnostic gave descriptive K=2048 coordinate coverage rates of `0.9477`,
`0.9488` and `0.9507` for `clean_0`, `shuffled_0` and ridge using estimate ± 1.96 ordinary SE.
Same-draw reference checks passed over 3,808 package vectors and 30,464 coordinates: maximum
student-versus-SHAP error was `5.68e-14` MPa and maximum student-versus-Captum error was zero.

The signature figures retain every signed coordinate and show performance, convergence,
quadrature drift, method disagreement, background/noise sensitivity, replicate coverage,
path/hybrid distance and globally ranked audit replacement curves. The fixed audit row was frozen
before audit evaluation: `clean_0` predicts `70.6716` MPa from a `35.5483` MPa background mean;
its coarse-aggregate coordinate is `3.3873` MPa under exact feature Shapley, `6.9269` MPa under
DeepSHAP, `5.8999` MPa under K=2048 Expected Gradients and `6.4185` MPa under 512-node
quadrature.

The supported student-gradient path assumes scalar-output, row-independent CPU-float64 models; the
native graphs are the declared affine/ReLU chains. Gaussian noise changes the explained prediction
target. Donor replacement, interpolation and noise are model computations on observational recipes,
not physical interventions or causal ingredient effects. Nearest-training-row distance is a bounded
geometric diagnostic, not a density estimate.

## Reproduce from source

Use the repository `uv` environment. Both output directories must be new.

```bash
section="chapter16_shapley_attribution_baselines/exercises/part15_gradientshap"

uv run python "${section}/build_notebooks.py" --check
uv run python -m pytest -q "${section}/gradientshap_contracts.py"

learner_output="$HOME/.cache/arena_extended/gradientshap/learner-readme-v1"
fresh_output="$HOME/.cache/arena_extended/gradientshap/reproduction-readme-v1"

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run python "${section}/execute_learner.py" \
  --output-directory "$learner_output"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run python "${section}/reproduce_native.py" \
  --learner-directory "$learner_output" \
  --output-directory "$fresh_output"
```

Primary reading is Erion et al., [*Improving performance of deep learning models with axiomatic
attribution priors and expected gradients*](https://arxiv.org/abs/1906.10670), Sundararajan, Taly
and Yan, [*Axiomatic Attribution for Deep
Networks*](https://proceedings.mlr.press/v70/sundararajan17a.html), and the pinned source evidence
for SHAP 0.49.1 `GradientExplainer` and Captum 0.9.0 `GradientShap`. The data come from Yeh,
[*Concrete Compressive Strength*](https://doi.org/10.24432/C5PK67). Exact protocol, RNG, source,
split and freeze identities are recorded in [protocol.md](protocol.md),
[protocol_freeze.json](protocol_freeze.json), [data_manifest.json](data_manifest.json) and
[author_reference.json](author_reference.json).

[exercise-notebook]:
  16.15_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate_exercises.ipynb
[solution-notebook]:
  16.15_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate_solutions.ipynb
[instruction-page]:
  ../../instructions/pages/15_[16.15]_Expected_Gradients_Which_Expectation_Does_GradientSHAP_Estimate.md
