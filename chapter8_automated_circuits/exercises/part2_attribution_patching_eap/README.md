# [8.2] EAP versus integrated gradients at a fixed circuit budget

At a budget of 1000 edges under ABC corruption, will EAP-IG have higher normalized circuit
faithfulness than EAP?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-8-2)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](8.2_Attribution_Patching_and_EAP_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](8.2_Attribution_Patching_and_EAP_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `eap_scores`, `eap_ig_scores`, `faithfulness_curve`.

- EAP top 1000
- same-size random circuits
- full and empty graph

## Evidence and limits

Historical saved model-run outputs are public references, not a newly reproduced run.

- Edge rankings depend on corruption and metric; a top-k graph is not automatically complete or
  minimal.
- The fixed 1000-edge comparison cannot establish a universal best attribution method.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
