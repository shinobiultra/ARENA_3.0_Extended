# [7.3] Activation oracles: what survives the bottleneck?

Rank animal-answer accuracy of trained activation oracle, linear probe, text-only oracle and
shuffled-vector oracle, highest first, on the frozen examples.

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-7-3)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](7.3_Mini_Activation_Oracles_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](7.3_Mini_Activation_Oracles_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `evaluate_oracle`, `control_vectors`,
`accuracy_by_position`.

- linear probe
- text only
- shuffled vectors

## Evidence and limits

Historical saved model-run outputs are public references, not a newly reproduced run.

- A trained oracle may learn task correlations; decodability alone does not show the source
  model uses the decoded feature.
- Position transfer changes the state distribution and is guided practice.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
