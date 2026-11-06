# [7.2] Feature verbalizers: does an explanation predict activation?

Rank own-explanation, next-latent-explanation and noise-explanation simulation correlation with
the target latent, highest first, on the frozen scoring windows.

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-7-2)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](7.2_Feature_Verbalizers_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](7.2_Feature_Verbalizers_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `simulate_activations`, `simulation_scores`,
`permutation_null`.

- next latent explanation
- noise explanation
- permuted activation null

## Evidence and limits

Historical saved model-run outputs are public references, not a newly reproduced run.

- A simulator can share language priors with the explainer; correlation is not a causal
  explanation.
- Saved generations/scores are historical; fresh hosted inference is not launched by this
  workbook.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
