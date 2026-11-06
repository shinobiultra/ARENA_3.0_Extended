# [16.6] SPEX: collisions versus decoder radius

For planted supports whose degree exceeds BCH correction radius t, will increasing bins alone
recover those supports while keeping t fixed?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-16-6)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](16.6_Sparse_Interaction_Discovery_with_SPEX_and_ProxySPEX_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](16.6_Sparse_Interaction_Discovery_with_SPEX_and_ProxySPEX_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `spex`, `predict_spectrum`.

- supports within radius t
- more bins at fixed t
- larger correction radius at matched queries

## Evidence and limits

planted_toy: known synthetic Fourier supports, explicit query accounting and decoder
assumptions.

- BCH radius bounds identify decoder capability; extra bins address collisions rather than
  increasing that radius.
- Planted sparse spectra are not proof that a language-model value function is sparse or that
  ProxySPEX attribution is exact.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
