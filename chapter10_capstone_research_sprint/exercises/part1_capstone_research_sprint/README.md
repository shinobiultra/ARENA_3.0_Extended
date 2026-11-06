# [10.1] Capstone: basis dependence as a falsifiable claim

Under an orthogonal hidden-basis rotation with the readout transformed consistently, will logits
and direction-patch effects remain invariant within the frozen numerical tolerance?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-10-1)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](10.1_Capstone_Research_Sprint_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](10.1_Capstone_Research_Sprint_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `patch_coordinates`, `patch_direction`, `evaluate_patch`.

- unrotated logits
- rotated coordinate patch
- unrotated direction patch
- random matched-size coordinates

## Evidence and limits

Analytic change-of-basis claim with a learner small-model check; a saved digit-model run is
historical supporting evidence.

- Algebraic invariance assumes consistent transforms and equivalent floating-point operations.
- A small digit model is not evidence that any selected language-model circuit is basis
  invariant.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
