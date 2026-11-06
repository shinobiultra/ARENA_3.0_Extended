# [8.4] ACDC: does greedy context beat isolated edge effects?

At ACDC threshold tau=0.1, rank ACDC, same-size top-single-edge and same-size random circuits by
normalized induction faithfulness, highest first.

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-8-4)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](8.4_ACDC_and_Circuit_Metrics_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](8.4_ACDC_and_Circuit_Metrics_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `acdc`, `single_edge_effects`, `circuit_faithfulness`.

- same-size top single-edge
- same-size random circuits
- full and empty graph

## Evidence and limits

Fresh full CUDA notebook reproduction on 2026-10-16; exact cached model/data identities, source
hashes, outputs and runtime are recorded in
docs/mastery/evidence/pilot_fresh_execution_20261016/8.4/publication.json and its execution
receipt. Preserved historical runs remain separate. Public results support guided practice, not
untouched evaluation.

- Greedy pruning depends on ordering, donor and threshold; a faithful circuit may retain
  redundancy.
- This real-model exercise has no complete known ground-truth circuit.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
