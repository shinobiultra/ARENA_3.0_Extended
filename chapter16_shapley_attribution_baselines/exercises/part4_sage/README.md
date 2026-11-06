# [16.4] SAGE: predictive information or model use?

For the second input, what is conditional SAGE minus marginal SAGE in squared-target units for
the frozen two-variable Gaussian population?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-16-4)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](16.4_SAGE_Global_Importance_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](16.4_SAGE_Global_Importance_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `gaussian_game`, `exact_shapley`, `coalition_masks`.

- marginal SAGE for the same model and population
- independent-input covariance control
- coalition efficiency against full-minus-empty game value

## Evidence and limits

Public historical reference outputs; no fresh full lesson execution or independent evaluation is
established by this workbook.

- Analytic linear Gaussian example; no sampled-data uncertainty or general nonlinear-model
  guarantee.
- The primary model does not directly use its second input; the transfer model does. Conditional
  predictive information does not by itself establish direct model use.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
