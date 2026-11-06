# [7.5] Predictive concept decoders: useful signal versus loud variance

With nuisance amplitude changed from (2,4) to (0.3,0.6), rank SAE and PCD concept-subspace
energy, highest first, using the same bottleneck and training budget.

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-7-5)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](7.5_Predictive_Concept_Decoders_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](7.5_Predictive_Concept_Decoders_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `sample_planted`, `train_bottleneck`, `subspace_energy`.

- nuisance amplitude (2,4)
- matched-budget SAE
- shuffled-label readout

## Evidence and limits

planted_toy: explicit synthetic concepts and nuisance directions with known subspaces.

- Planted orthogonal concept/nuisance subspaces are known by construction; this is not a
  discovery claim about GPT-2.
- Equal training steps do not guarantee either objective has converged.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
