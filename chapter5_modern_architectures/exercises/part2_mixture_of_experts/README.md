# [5.2] Balanced routes and expert necessity

In the recorded three-seed aux+z synthetic MoE study, is the mean accuracy drop from ablating
each regime's dev-preferred expert greater than the mean drop from ablating the other experts?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-5-2)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](5.2_Mixture_of_Experts_from_Scratch_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](5.2_Mixture_of_Experts_from_Scratch_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `topk_router`, `compute_capacity_state`, `dispatch_tokens`,
`combine_expert_outputs`, `SparseMoE`, `routing_by_regime`, `causal_expert_intervention_table`.

- intact fixed model and exhaustive all-expert ablations
- dev-only preferred selection; final data for measurement
- paired aux+z, aux-only, z-only and neither
- equal-width dense model with explicit padded-work and parameter mismatch
- delexicalized context probe versus shuffled labels
- coherent expert permutation with infinite capacity
- all-expert forcing compared with ablation; no equivalence claim

## Evidence and limits

Recorded synthetic CPU study and exact retained JSON values, not a released-model study.
Historical outputs and source AST are retained; no independent full-own-answer source-hashed
execution receipt exists. The fresh20-cell CPU mechanics run excludes all training and cannot
establish the scientific result.

- Historical/monitored synthetic data, not untouched TEST.
- High task accuracy and balanced hard loads do not establish expert necessity or positive
  specialization.
- Three-seed aggregate and representative seed7 intervention table have different aggregation
  levels.
- Original positive-specialization release predicate remains rejected; canonical5.8 is now a
  different topic.
- Dense baseline is equal-width, not parameter- or executed-work-matched.
- Forcing transfer was already measured and published; it is exposed guided practice.
- Released-small-MoE routing in the broader5.2 brief remains outside this migration.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
