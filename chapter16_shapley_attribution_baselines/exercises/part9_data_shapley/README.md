# [16.9] In-run data credit: local approximation and learning rate

At one fixed parameter snapshot and minibatch, will second-order credit have lower error against
exact local coalition Shapley values than first-order credit at the predeclared learning rate?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-16-9)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](16.9_Data_Shapley_in_One_Training_Run_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](16.9_Data_Shapley_in_One_Training_Run_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `local_game_values`, `exact_local_shapley`,
`first_order_credit`, `second_order_credit`.

- exact local coalition enumeration
- first-order credit
- zero learning rate
- learning-rate sweep

## Evidence and limits

Exact finite local coalition game and Taylor approximations at a fixed snapshot; learned
trajectory results remain separate historical evidence.

- Local Taylor credit is not retraining Data Shapley and does not supply a full-trajectory value
  by itself.
- ReLU boundary crossings can invalidate smooth higher-order error expectations.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
