# [8.5] Causal scrubbing: what do induction keys actually need?

What fraction of induction loss is recovered by the textbook q/k/v information hypothesis
relative to clean and everything-resampled anchors?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-8-5)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](8.5_Causal_Scrubbing_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](8.5_Causal_Scrubbing_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `loss_recovered`, `scrubbed_loss`, `per_position_resample`.

- nothing resampled
- everything resampled
- deliberately wrong dependency hypothesis

## Evidence and limits

Historical saved model-run outputs are public references, not a newly reproduced run.

- A high score tests a hypothesis under its resampling distribution, not a unique mechanistic
  explanation.
- Averaged loss recovery can hide positions that fail.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
