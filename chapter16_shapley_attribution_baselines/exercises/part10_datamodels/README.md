# [16.10] Datamodels

Predict a classifier's margin from its training-set membership, then test what
happens when the most supportive images are removed and the model is retrained.

Start with [the exercise notebook](16.10_Datamodels_exercises.ipynb). The
[executed solution notebook](16.10_Datamodels_solutions.ipynb) contains the same
experiments and all reference outputs. [The instruction page](../../instructions/pages/10_[16.10]_Datamodels.md)
and [readable lesson source](lesson.md) follow the same learner arc.

You implement ten functions across eight exercise groups, including subset
sampling, margins, Lasso, validation-only selection and matched deletions.
Fifteen figures show held-out predictions, per-query errors, sparsity, sample
complexity, image coefficients, distribution shifts and actual deletion effects.
Two editable experiments let you introduce an interaction or choose a real query
and retrain its counterfactual.

The full reference lesson runs on CPU in about sixteen seconds on the development
machine, with 10,753 independently converged classifier fits. Study takes several
hours. The first run downloads the pinned public UCI digit archive; subsequent
runs validate its cached bytes. Original ARENA chapters remain unchanged.

The classifier is small and deterministic. Datamodels are trained on the labeled
query margins, so the central held-out unit is the training subset, not the query
image. Most selected coefficients are dense. Same-size accuracy does not imply
accurate effects at other sizes; the lesson retains those errors and compares
against least squares, permuted responses and class-matched random removals.

From the repository root:

```bash
uv run python -m pytest -q chapter16_shapley_attribution_baselines/exercises/part10_datamodels/datamodel_methods_tests.py chapter16_shapley_attribution_baselines/exercises/part10_datamodels/datamodel_contract_tests.py
uv run python chapter16_shapley_attribution_baselines/exercises/part10_datamodels/execute_learner.py --output-directory /tmp/datamodels-own-run
uv run python chapter16_shapley_attribution_baselines/exercises/part10_datamodels/reproduce_native.py --learner-directory /tmp/datamodels-own-run --output-directory /tmp/datamodels-fresh-run
```

Choose new output directories for each run. See [the frozen scientific protocol](protocol.md)
and [source/split manifest](data_manifest.json) for exact data selection, seeds,
regularization, stopping criteria and scope. Reports support the notebook's
actual results; they are not the teaching artifact.
