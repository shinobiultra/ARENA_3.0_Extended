# Datamodels: expected outputs and their provenance

## How this fixture was produced

The published outputs come from completing each exercise using only its own
following Solution dropdown and running all 31 code cells in a fresh CPU kernel.
The notebook trains every subset and counterfactual model itself. It does not
load saved classifier weights or datamodel coefficients to produce a figure.

`learner_execution.json` identifies that run, its seventeen source inputs, source
snapshots, executed notebook, fifteen figures and reviewed browser screenshots.
`native_summary.json` records the experiment size and independent checks.
`fresh_reproduction.json` identifies a separately reloaded and completely
retrained study. Every one of the three native tensor payloads is compared,
including masks, IDs, fits, coefficients, validation choices and all outcomes.
The normal verification command performs another complete native retraining.

## Trusted implementation

The student-facing implementations are checked against independent scalar-grid,
orthogonal-design, scikit-learn coordinate-descent and NumPy/SciPy classifier
oracles. Actual digit runs include six separately scheduled classifier refits,
ten independent SciPy optima and twelve independent native Lasso fits. Fixed-size
coefficient ambiguity is respected: compare fitted predictions and objectives
rather than demanding a unique coefficient representative from two Lasso solvers.

## Random seed

All data choices, subset seeds, three fractions, 16-point regularization paths,
five sample sizes, four deletion budgets and matched random-control seeds are
specified in [the frozen protocol](../protocol.md). The protocol freeze predates
the first digit fit. Query labels supply regression targets, while test-subset
responses cannot influence model selection or deletion masks.

## Allowed tolerances

Classifier gradient infinity norm is at most 1e-10; Lasso KKT violation is at
most 1e-8. Independent classifier parameter comparisons permit 1e-8 absolute
error. Native Lasso prediction parity permits 1e-5 absolute error and objective
parity 1e-7, with no claim of coefficient uniqueness. Fresh native reproduction
uses `rtol=1e-10` and `atol=1e-11` and rejects dtype, shape, missing-field or ID-order
changes. The recorded current run reproduces all 5,546,159 tensor elements and
327 scalar values with zero difference.

## When to regenerate

After a method, data, scientific protocol, source dependency or executable lesson
change, run the own-answer notebook and a separate complete native reproduction
again, inspect current figures and rendered math/dropdowns, then publish those
exact artifacts. Preserve earlier runs and their failures. Cosmetic formatting
must preserve executable ASTs, with any normalized docstring indentation noted.
Do not copy evidence from a prior source state or relabel partial output as a
complete native run.
