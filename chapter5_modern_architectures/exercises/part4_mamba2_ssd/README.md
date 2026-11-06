# [5.4] SSD: how much of the final read comes from recurrent carry?

For the exact three-token CPU fixture X=[2,3,5], all transitions 0.5, B=C=1, zero initial state
and carry_scale=1, what is final output(normal carry) minus final output(carry_scale=0)? Report
the scalar output difference, not a relative error or accuracy.

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-5-4)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](5.4_Mamba-2_and_State-Space_Duality_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](5.4_Mamba-2_and_State-Space_Duality_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `segment_sums`, `semiseparable_matrix`, `ssd_recurrent`,
`ssd_quadratic`, `ssd_chunked`.

- carry_scale=0 preserves current writes while removing previous-token memory
- quadratic SSD and chunked SSD reproduce the normal carry=1 operator
- chunk sizes 1,2,4 include nondivisible padding and Q greater than L; final states must agree
- future matrix entries are zero, diagonal weights are one

## Evidence and limits

Fresh bounded exact CPU operator play and 12 immediate checks executed as the exact six new
cells with minimum visible-definition closure; actual outputs, counts, origins and plot are
receipted. All 21 complete historical code-cell objects remain identical, including old model
results and CUDA precision findings. No fresh model study, optimizer, training, full notebook,
checkpoint inference or GPU execution occurred.

- One exact three-token scalar operator, not learned-model behavior or generalization.
- The default and changed cases are visible or tested, so preserve seen/unknown exposure and
  award no unseen score.
- A changed carry alters the operator; changing only chunk size changes its schedule. Do not
  compare quadratic/chunked normal-carry outputs to an ablated recurrence as a parity claim.
- The historical 1,024-example synthetic evaluation was repeatedly monitored and doubled-length
  evaluation is extrapolation evidence; neither is untouched TEST.
- Historical CUDA parity uses IEEE dot arithmetic and old pinned inputs; it is not a newly
  executed official-kernel throughput benchmark or learner acceptance.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
