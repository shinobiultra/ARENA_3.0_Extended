# 16.17 — One Query, Many Interactions

How accurately can you estimate every pair interaction when you can query only some coalitions?
Build SHAP-IQ's shared-query mean and KernelSHAP-IQ's sequential weighted regression, then compare
them on the same model responses. This continues the exact SII definition in section 16.3.

Start with [the exercises](16.17_One_Query_Many_Interactions_exercises.ipynb).
The [solutions](16.17_One_Query_Many_Interactions_solutions.ipynb) retain actual executed results;
the [instruction page](../../instructions/pages/17_[16.17]_One_Query_Many_Interactions.md) includes
the same questions, hints, solutions and figures. Allow roughly 2–3 hours, with extra time for the
nullspace derivation if it is new to you.

Eight exercises teach nine functions: mask design, shared SII weights, size-then-subset sampling,
importance-weighted estimates and SE, cached query receipts, constrained least squares, pair
residual design, finite kernels, and the sequential estimator. Each function has an immediate test.
Ten figures show shared contributions, sampling probabilities, the boundary limit, model controls,
signed pair estimates, convergence, sampling variance, unique queries and rank deficiency.
Two editable plays let you change a queried coalition or the model, recipe, seed, proposal and budget.

Run the notebook in the repository's `uv sync --frozen` environment. The complete study uses CPU
float64 with four threads. The validated learner took about 29 seconds and a sampled peak process-tree
RSS of 2.85 GB; timing depends on your machine. It writes about 0.95 GB to `estimator_outputs`.
The study requires a new empty `native` output directory: choose a new `OUTPUT_DIR` for a second
complete run. The two play cells can be rerun freely using the in-memory results.

The real study reproduces seven known Concrete predictors on all 272 held-out rows, with 64 intact
training donors. Every model, row, pair, replicate and budget is retained. Constants, additive games,
signed interactions, exact SII, zero-pair estimates, shuffled-target predictors and ridge provide
controls. An additive game can still have noisy finite SHAP-IQ estimates. Shuffled predictors can
have real interactions in their fitted functions.

The comparison measures **logical queries to a precomputed model-game oracle**. Constructing the
complete reference costs 487,424 coalition values and 31,195,136 donor predictions. This is not a
measured inference-speedup study, a new predictive discovery, a whole-paper replication, or evidence
of causal ingredient effects. The KernelSHAP-IQ implementation covers the proved pairwise case.
Sampling SE is conditional on the fixed model and replacement game.

Validation commands, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run pytest -q chapter16_shapley_attribution_baselines/exercises/part17_shapiq_estimators/estimator_methods_tests.py chapter16_shapley_attribution_baselines/exercises/part17_shapiq_estimators/estimator_reference_tests.py chapter16_shapley_attribution_baselines/exercises/part17_shapiq_estimators/estimator_study_tests.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python scripts/run_extension_verification_reports.py --section 16.17 --max-vram-gb 0
```

The registered report refits all models in a fresh process and compares every deterministic value
with the actual own-answer learner. The scientific design was fixed before native execution in
[protocol.md](protocol.md) and [data_manifest.json](data_manifest.json). Evidence paths and scope
are documented in [expected_outputs](expected_outputs/README.md). Visual and implementation
review was performed by the author while the authorized subagents were unavailable; it is not
an independent review. Original ARENA chapters remain unchanged.
