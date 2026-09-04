# Expected outputs for [15.2]

`smoke_test.json` is the exact CPU-checkable contract. It proves LoRA matrix
orientation, numerical rank, merge parity, model revision, and the 20-example
disjoint-template split without pretending to be the scientific result.

`reference_metrics.json` records both the bounded CPU preflight and the accepted
96-step real-model CUDA result. The committed `verification_report.json` binds
the exact run to the learner surface by content hash. Expected behavior is:

- rank-1 LoRA, rank-4 LoRA, and full finetuning beat the frozen base;
- random-label and same-norm random low-rank controls remain near chance;
- LoRA update spectra are more concentrated than full finetuning;
- dominant adapter-induced activation-direction ablation hurts target behavior
  more than a matched random direction;
- unrelated TinyStories next-token NLL is reported and bounded, not omitted.

The solution notebook's short CPU run remains a pipeline preflight. It is not
substituted for the full CUDA signature result.
