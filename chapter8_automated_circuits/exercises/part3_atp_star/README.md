# [8.3] AtP*: accuracy under an exact-verification budget

Rank AtP, QK-fix-only, mean-absolute aggregation and full AtP* by exact verification count
needed for full recall of nodes with |exact effect| > 0.1; lowest count first.

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-8-3)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](8.3_AtP_from_Scratch_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](8.3_AtP_from_Scratch_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `exact_node_effects`, `atp_estimates`, `qk_fix_estimates`,
`nodes_for_full_recall`.

- plain AtP
- QK-only ablation
- mean-absolute-only ablation

## Evidence and limits

Historical saved model-run outputs are public references, not a newly reproduced run.

- The exact target set depends on the frozen task/metric; verification savings need not transfer
  across corruption.
- Component improvements can interact and should not be added independently.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
