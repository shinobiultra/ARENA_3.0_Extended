# [5.7] Mamba-2 and SSD Expected Outputs

## How this fixture was produced

`reference_metrics.json` freezes acceptance thresholds. It contains no sampled
predictions. The notebook produces new outputs from its own implementations;
the release harness runs strict CUDA and binds the current sources to its report.

**Trusted implementation:** the hand-computable recurrence `[2, 4, 6]`, the
official `mamba_ssm.modules.ssd_minimal` Listing 1, and independently loaded
Transformers Mamba-2 with all 219 checkpoint tensors mapped strictly.

**Random seed:** operator seed 7, gradient seed 17, recall initialization 0;
training seeds 0/1/2, held-out seed 100, doubled-length seed 101. The three
trained models now receive the same number of optimizer steps. Every ID
evaluation uses the requested training length; OOD doubles that length.

**Allowed tolerances:** float64 operators and gradients use `rtol=0`,
`atol=1e-11`; float32 cache comparisons use `rtol=2e-6`, `atol=2e-6`.
The existing released-model bounds remain max hidden error 0.002, mean hidden
error 5e-5, max logit error 5e-4, mean logit error 5e-5, with exact top tokens.
The fused-operator bound remains 2e-4. Both CUDA comparisons require IEEE
float32 dot products (`TRITON_F32_DEFAULT=ieee`), not Triton's default TF32.

**When to regenerate:** after changes to code, lesson prose, tests, locks,
dependencies, or experiment settings. Rebuild, execute the completed learner
notebook, run semantic tests, and then run the generic CUDA report harness.
Never change a tolerance merely to match a failed result.

These fixtures protect the learner-visible contract. Exact float64 operator
comparisons use `1e-11`; float32 block cache parity uses `2e-6`.

The delayed-recall result is accepted only when the true-label Mamba-2 model
reaches at least `0.95` held-out accuracy and each causal control remains at or
below `0.65`. Doubled-length evaluation must remain at least `0.65`; it is an
OOD stress test, not a perfect-generalization claim.

`reference_metrics.json` records thresholds, not a license to regenerate until
a failure disappears. `smoke_test.json` states the frozen claim boundary. CUDA
evidence belongs in a fresh `verification_report.json` produced by the parent
release harness after it runs both strict hooks.

The learner signature figure must be generated live from notebook values and
must show:

- all three independent SSD forms;
- parity with the official minimal SSD implementation;
- true-label, state-ablation, local-convolution, and shuffled-label results;
- doubled-length OOD accuracy;
- measured CPU timings labeled as educational rather than fused-kernel speed;
- full-matrix versus chunk-local decay storage.
- recurrent/quadratic/chunked gradient parity below `1e-11`;
- stacked-LM full/cache logits parity below `2e-6`;
- runnable Mamba-2, attention, and Mamba-1 blocks within 10% parameter count;
- strict mapping of 219 released tensors and comparison of all 25 hidden states;
- checkpoint logits below `5e-4` max error and matching top tokens on two prompts.
