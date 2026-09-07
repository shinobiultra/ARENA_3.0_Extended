# Measured QLoRA evidence

`reference_metrics.json` records the actual completed learner, filled from its
own solution dropdowns and run in a fresh kernel: 21 code cells, exact CPU
oracles, the three-seed CUDA comparison, and a separate precision/buffer probe.
The solved notebook retains the same executed outputs.

`smoke_test.json` contains the measured CPU mechanics only. Its local acceptance
flag is not course release acceptance. The actual CUDA behavior and memory gates
pass, but the unchanged BF16 parity gate fails (0.03125 versus 0.01), so the
candidate remains unreleased. No accepted verification report is registered.

The reference records actual pytest output, critical source/notebook hashes,
and both signature-image hashes. `test_reference_evidence_is_current` checks
these fingerprints; the generator runs all other semantic tests before writing
the new reference to avoid a circular freshness check.

Regenerate from the repository root:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 uv run python \
  chapter15_peft_misalignment/exercises/part3_qlora_from_scratch/generate_cpu_fixtures.py
```

Despite its historical filename, this runs CUDA whenever available. Use
`--cpu-only` for the limited mechanics path; CUDA is then explicitly absent.
Do not regenerate to conceal a failed check or weaken thresholds.

All behavior intervals use Student-t with df=2 across seeds 0, 1, and 2.
They measure seed variability on this fixed synthetic dataset, not uncertainty
on new tasks. Uniform int8 nested-scale arithmetic is a toy derivation;
materialized QLoRA uses bitsandbytes' dynamic nested codebook.
