# [5.5] Discrete Diffusion LMs: From Copying Digits to Generating Words

The 90-cell learner contains 23 implementation groups, twelve live figures and
four play cells. It retains the exact copy-grammar warm-up, derives finite
absorbing diffusion, then trains the same inspectable Transformer with
bidirectional or causal attention. Students implement noising, the weighted
loss, reverse probabilities, three samplers, attention, cached AR generation,
matched training, scoring and fair timing. Each group has immediate tests,
expected feedback, help and its own complete answer.

The text study trains twelve models: three seeds times diffusion/AR with correct
or within-task shuffled supervision. All receive 12,000 updates with identical
initialization and input minibatches within each seed. Tasks are parentheses,
addition, context-dependent QA and pinned TinyStories continuations, with a
training-only byte-level BPE vocabulary. All 24 targets start masked during
generation; the sampler never receives the answer length through validity.

All 126 final conditions remain: 64,512 generations, reveal-order, temperature
and block ablations, selected and last-update checkpoints, and untrained
controls. Cached-AR timing retains 78 conditions and 390 synchronized repeats.
Actual token diagrams, error galleries, uncertainty and commitment accompany
the task matrix. Primary diffusion averaged 57.8% addition accuracy versus 1.8%
for shuffled supervision; AR was stronger on valid parentheses (47.4% versus
21.1%). Both remained around 28–29% on QA, and story outputs remain weak. The
fixed QA example is wrong for both; the fact-edit play succeeds for both trained
models while its shuffled control fails. No fluent-language or full-paper claim.

From the repository root, run the own-answer numerical exercises with:

```bash
uv run python chapter5_modern_architectures/exercises/part5_diffusion_language_models/execute_learner.py --scope numerical --output-directory /tmp/diffusion-numerical
```

Use `--scope full` and a new output directory for all actual training and final
experiments. First execution builds the pinned data; set
`ARENA_DIFFUSION_DATA_DIRECTORY` to reuse a verified directory. The observed
candidate took about 7.3 minutes on the RTX 5090 Laptop GPU; first download and
other hardware costs vary. Every checkpoint, generation and source is retained.

`publish_study.py --learner-directory RUN --data-directory DATA` validates and
copies that actual execution to paired notebooks and the page. Then run:

```bash
uv run python scripts/run_extension_verification_reports.py --section 5.5
```

The normal report retains the copy and DiffusionGemma checks and freshly refits
four text models at seed zero, regenerating 42 configurations and 21,504 examples.
It compares selected/endpoint weights and actual words/traces. Missing local
learner/data directories are regenerated; changed-source evidence is rejected.
The notebook itself creates its visible results; reports support that evidence.

The previous copy-only publication and failed candidates are archived in
`expected_outputs`. The pinned DiffusionGemma NVFP4 artifact still proves one
isolated-vLLM generation. Released-model denoising-step hidden-state capture and
patching remain unfinished, and the main uv runtime is unchanged.
