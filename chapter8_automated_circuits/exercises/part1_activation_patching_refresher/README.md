# [8.1] Activation patching: direct effects and compensation

Will joint noising of heads 9.9, 9.6 and 10.0 have a smaller absolute total IOI logit-difference
effect than its absolute direct effect?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-8-1)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](8.1_Activation_Patching_Refresher_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](8.1_Activation_Patching_Refresher_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `total_and_direct_effect`, `head_direct_contributions`.

- clean run
- frozen downstream direct effect
- same-size random late-head triples

## Evidence and limits

Historical saved model-run outputs are public references, not a newly reproduced run.

- Direct and total effects are different interventions; sign cancellations and donor choice
  matter.
- A mean can conceal large positive and negative per-prompt effects.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
