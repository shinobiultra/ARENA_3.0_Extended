# [6.1] SAE Variants

This section is a CPU-runnable comparison of ReLU-L1, TopK, gated, and
JumpReLU sparse autoencoders on exact planted sparse ground truth.

The learner implements the planted generator, all four encoder rules, the
variant objectives, dictionary recovery, held-out AUC, and causal steering and
ablation. Those functions drive a fresh four-variant training run and a six-panel
signature figure. The figure includes a train-mean reconstruction baseline, true
latent L0, random-decoder recovery, shuffled labels, orthogonal random steering,
and an equal-coefficient random-direction ablation.

Run the CPU tests with:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q \
  chapter6_sparse_feature_methods/exercises/part1_sae_variants/tests.py
```

`solutions.run_gpu_test(max_vram_gb=24.0)` remains the real-model escalation.
It loads pinned `EleutherAI/pythia-70m-deduped` final-token hidden states and
trains a width-256 TopK-16 SAE. The parent runner owns that CUDA job; its report
is supporting evidence and is not loaded as the learner-facing result. On the
pinned run, reconstruction and intervention controls pass, but no individual
SAE feature cleanly separates the labels on train or on topic-disjoint held-out prompts. The
report preserves this as a negative result rather than claiming a clean
semantic feature.
