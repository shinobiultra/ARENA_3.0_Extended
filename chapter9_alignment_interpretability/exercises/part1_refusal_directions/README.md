# [9.1] Refusal directions: readable versus causally effective

Will the direction/layer with the best readout separation also be the strongest causal refusal
edit under the frozen benign-KL budget?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-9-1)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](9.1_Refusal_Directions_and_Representation_Engineering_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](9.1_Refusal_Directions_and_Representation_Engineering_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `sweep_directions`, `select_direction`, `kl_divergence`.

- matched-norm random direction
- label-shuffled direction
- zero-dose edit
- benign prompts

## Evidence and limits

Historical saved model-run outputs are public references, not a newly reproduced run.

- A refusal metric is not a safety evaluation; decoding evidence alone does not establish a
  causal edit.
- Selection on development prompts does not make them independent evaluation.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
