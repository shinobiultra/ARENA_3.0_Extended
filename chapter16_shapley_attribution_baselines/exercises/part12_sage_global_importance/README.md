# 16.12 SAGE: Which Features Help a Model Predict?

Use the [exercise notebook](16.12_SAGE_Global_Importance_exercises.ipynb),
[solutions](16.12_SAGE_Global_Importance_solutions.ipynb), or [instruction
page](../../instructions/pages/12_[16.12]_SAGE_Global_Importance.md). The editable source is
[lesson.md](lesson.md).

Allow 4–6 hours for self-study. Nine implementation exercises lead from a performance game to conditional
Gaussian restriction, exact and sampled SAGE, local-loss versus prediction explanations, and feature-order
evaluation on new concrete recipes. Eighteen figures and two editable plays expose nonlinear averaging,
correlations, negative importance, interactions, background/model/data variation and sampling bias. Every
taught method and all model/study functions are visible in the notebook, with immediate tests, expected
output, help and following solutions.

The CPU study freshly trains six small networks and one ridge regressor. It uses all 1030 UCI rows with recipe
groups kept separate across 758 training, 153 explanation and 119 audit rows. Rankings are saved before
loading audit data. The clean networks beat ridge/constant baselines; shuffled-target networks fail. Cement
and age receive large positive importance, while smaller attributions can be negative or change sign across
evaluation splits. These are predictive quantities for fixed models and marginal backgrounds, not causal
ingredient effects or universal optimal-model importance.

The source is [Covert, Lundberg & Lee (NeurIPS
2020)](https://papers.neurips.cc/paper_files/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html),
with [author code](https://github.com/iancovert/sage/tree/0e34e6c7e2a6b1666e39043e83b9929e68ee68f3). Data: Yeh
(1998), [UCI Concrete Compressive
Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), DOI 10.24432/C5PK67, CC BY
4.0. Full data/protocol choices and hashes are in [readable protocol](protocol_readable.md),
[data_manifest.json](data_manifest.json), and [protocol_freeze.json](protocol_freeze.json).

Run from the repository root with the existing uv environment. Data and two byte-pinned author reference files
download on first use. No GPU is needed. Use a new output directory for every run.

```bash
uv run python -m pytest -q chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance/sage_methods_tests.py chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance/sage_contract_tests.py
uv run python chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance/execute_learner.py --output-directory /tmp/sage-own-new
CUDA_VISIBLE_DEVICES='' uv run python chapter16_shapley_attribution_baselines/exercises/part12_sage_global_importance/reproduce_native.py --learner-directory /tmp/sage-own-new --output-directory /tmp/sage-fresh-new
CUDA_VISIBLE_DEVICES='' uv run python scripts/run_extension_verification_reports.py --section 16.12
```

The complete own-answer notebook took about 12 seconds on the development machine. A fresh native reproduction
took about four seconds and matched 464 tensors / 3,786,320 elements and 61 scalars exactly. This compares
every saved fit, prediction, local game, attribution, sampled path, ranking and default play. The independent
checks enumerate all 40,320 feature orderings, solve a constrained weighted regression, compare real NumPy
network predictions, and use the unmodified author imputer. Browser review checked all 32 dropdowns and 81
rendered math expressions.

The generic report command uses a legacy GPU-named entry point; `CUDA_VISIBLE_DEVICES=''` keeps the shared
runner's preliminary CUDA cleanup away from unrelated GPU jobs. The section itself trains and evaluates only
on CPU.

A failed development notebook exposed NumPy's initial-mask buffer reuse under this Jupyter runtime. The
independent permutation checker now uses explicit nonaliasing output storage and checks conservation for every
complete path. Its failure evidence is retained; the frozen experiment, data selection, fitted predictions and
scientific conclusions did not change.
