# [8.8] Compiled circuits: saturated gradients and planted structure

At attention-weight noise sigma=0.03, rank exact patching, EAP and 10-step EAP-IG by
planted-edge ROC AUC, highest first; declare a tie policy before scoring.

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-8-8)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](8.8_Benchmarking_Circuit_Discovery_on_Compiled_Transformers_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](8.8_Benchmarking_Circuit_Discovery_on_Compiled_Transformers_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `compile_induction`, `patching_scores`, `eap_scores`,
`eap_ig_scores`, `roc_auc`.

- random-ranking AUC 0.5 reference
- noise-free model
- task accuracy and log probability

## Evidence and limits

planted_toy: a hand-compiled transformer with an explicit planted circuit, not a naturally
discovered circuit.

- The known mask is planted structure; injected noise may create additional real dependencies.
- Circuit recovery and task behavior can diverge; AUC does not measure faithfulness.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
