# [16.2] KernelSHAP: what does an equal coalition budget buy?

Across the frozen 50 seeds, what is mean maximum-player absolute KernelSHAP error for
paired-kernel sampling minus the corresponding mean for uniform sampling?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-16-2)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](16.2_KernelSHAP_and_PartitionSHAP_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](16.2_KernelSHAP_and_PartitionSHAP_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `sample_interior_coalitions`, `kernelshap_values`,
`shapley_from_dividends`, `partition_shap_values`.

- full-table exact KernelSHAP
- uniform sampling at the same coalition budget
- independent dividend oracle
- singleton, aligned and misaligned partitions with exact Owen allocation
- 100 random balanced partitions with seeds 0 through 99
- endpoint-preserving shuffled interior values with seeds 100 through 149

## Evidence and limits

The corrected finite-game notebook passed a fresh bounded CPU execution on 2026-10-16; its
source, runtime and outputs are recorded separately from the preserved historical originals and
neural-model report. This fixed monitored synthetic game is guided practice, not independent
evaluation.

- Finite six-player game; it does not establish accuracy for realistic model masking, data
  backgrounds or learned feature hierarchies.
- The paired strategy combines kernel-proportional sampling, complement pairing and kernel row
  weighting; the comparison neither isolates pairing alone nor establishes an unbiased generic
  estimator. Error bands show seed standard deviations.
- The stable rank-aware solver repair is separately recorded; old historical outputs are not
  evidence of execution of changed numerical code.
- Partition/Owen allocation and ordinary Shapley are different games when the grouping cuts
  interactions.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
