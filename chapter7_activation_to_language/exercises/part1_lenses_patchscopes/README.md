# [7.1] Patchscopes: information or prompt prior?

At source layer 4 and target layer 1, what is capital top-1 accuracy(patched country state)
minus accuracy(text only), in percentage points?

Start with the [guided prediction workbook](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-7-1)
before viewing reference results. Record your forecast, confidence, reasoning and prior
exposure. Then compare the controls and commit a changed-case prediction.

Open [the exercise notebook](7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_exercises.ipynb) to implement the methods and use its
immediate tests. Follow the notebook's setup and runtime instructions in your ARENA
environment. Run the small play cells separately from any training or model-loading cells.
[The solutions notebook](7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_solutions.ipynb) contains worked code and available result evidence.

## Methods and controls

The visible implementation includes `get_source_vectors`, `run_patchscope`,
`patchscope_controls`.

- text only (no patch)
- random vector, matched norm
- wrong position (BOS)

## Evidence and limits

Fresh full CUDA notebook reproduction on 2026-10-16; exact cached model/data identities, source
hashes, outputs and runtime are recorded in
docs/mastery/evidence/pilot_fresh_execution_20261016/7.1/publication.json and its execution
receipt. Preserved historical runs remain separate. Public results support guided practice, not
untouched evaluation.

- Prompt priors can answer some examples without patched information; one-token filtering
  changes the cohort.
- The original historical GPT-2 run lacks an immutable revision. The fresh cached identity is
  recorded separately; repetition of public cases is not independent evaluation.

Reading a saved result and executing the experiment are different observations. Record
which you used, with its source and configuration. The workbook labels this as guided
practice; historical or already exposed results do not earn unseen-mastery credit.

[Integration and verification record](../../../docs/mastery/continuation_2026-10-16.md).
