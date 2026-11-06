# [16.10] Influence-guided balanced deletion: does final query loss fall?

Does influence-selected balanced removal of eight training rows in the corrupted-label condition
reduce mean unregularized BCE on the 64 historical final queries relative to the full-data
optimum?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-16-10)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](16.10_Influence_Functions_and_TracIn_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](16.10_Influence_Functions_and_TracIn_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `logistic_losses`, `logistic_gradients`,
`weighted_objective`, `objective_derivatives`, `fit_weighted_logistic`, `upweight_influence`,
`class_matched_removals`.

- no deletion/full-data optimum
- clean-label matched condition
- all 32 class-matched random ranking refits
- ranking-LOO and gradient/pixel baselines
- converged refit residual and independent SciPy solver check
- loss and argmax accuracy shown together

## Evidence and limits

Retained historical completed-own-answer UCI digit logistic study; no fresh native execution was
performed during master migration. The original Influence publication was 16.9; canonical Data
Shapley16.9 is a distinct method/case. Published curves and rounded prose are historical
references; exact deletion arrays have not been located, and the recorded external tensor caches
are absent. Do not treat visual curve estimates as an exact machine observation.

- Final queries are historical/monitored, not untouched TEST.
- Single small regularized binary logistic model and fixed writer-separated source split; no
  large nonconvex-model or population data-cleaning claim.
- Group deletion is a finite refit, not the sum of local derivatives or an exact path/checkpoint
  identity.
- Budget 16 is already shown in the historical five-budget curve; its changed-condition forecast
  is exposed guided practice, not unseen transfer.
- Historical plots/prose support approximate visual comparison only. An exact numeric deletion
  target requires a separately declared and authorized re-execution unless a recoverable exact
  tensor source is established.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
