# [7.4] Natural-language autoencoders: same words, different code?

Will shuffling the same four description words preserve transplant accuracy to within 5
percentage points of the unshuffled reconstruction?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-7-4)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](7.4_Mini_Natural_Language_Autoencoders_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](7.4_Mini_Natural_Language_Autoencoders_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `fve`, `transplant_accuracy`, `edit_and_patch`.

- unshuffled four words
- shuffled four words
- mean reconstruction

## Evidence and limits

Historical saved model-run outputs are public references, not a newly reproduced run.

- A language bottleneck can encode order-dependent shortcuts; fluent descriptions need not be
  faithful.
- A single edited example demonstrates a local intervention, not general semantic control.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
