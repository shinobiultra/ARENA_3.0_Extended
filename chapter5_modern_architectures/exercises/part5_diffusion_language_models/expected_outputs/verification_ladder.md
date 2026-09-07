# 5.5 verification ladder provenance

## How this fixture was produced

The own-answer learner runs the copy warm-up and twelve matched text fits,
freezes development choices, generates 126 final conditions, performs 78 timing
conditions and executes the three new plays. `text_learner_execution.json` binds
its exact source inventory, paired notebooks, page, raw rows and twelve figures.
`completed_text_study.json.gz` retains all outcomes; `study_evidence.py`
recomputes scores, checkpoint choices, complete denominators and source hashes.

## Trusted implementation

The finite-loss test enumerates the absorbing forward chain and independently
constructs its Bayes posterior, bound and every gradient. Explicit head-by-head
attention checks values/gradients. Two manually derived clipped AdamW updates
check both objectives. Cached AR agrees with uncached generation, preserves
per-row EOS and reduces input work. Adversarial samplers expose wrong visible
logits and erased greedy confidence. Scoring retains special-token errors and
valid alternate strings. Audit mutations reject missing runs, duplicate failures,
relabeled words/scores and changed sampling configurations.

## Random seed

Training uses 0, 1 and 2. Private batch, noise and shuffle generators add 100,
200 and 300. Fixed development noising uses seed 1729 and three repetitions.
Final generation uses seed 1000; timing resets seed 990 per repeat. Three
initializations share one data split; they are not independent datasets.

## Allowed tolerances

Token IDs, strings, EOS, correctness and inventories use rtol=0, atol=0. The
float64 finite-bound/loss oracle uses atol=1e-12. Explicit attention/cache uses
atol=1e-11, rtol=1e-11, with per-test gradient tolerances. Independent AdamW
uses atol=1e-10, rtol=1e-9 in float64, avoiding amplified float32 roundoff on
mathematically invisible key-bias gradients. Fresh CUDA weights/traces permit
atol=1e-4, rtol=0; generated tokens and scores must still match exactly. The
candidate had zero errors. Latency is measured rather than matched to an old value.

## When to regenerate

After a method, data, protocol or executed-source change, rerun the full
own-answer learner into a new directory, inspect all twelve figures, run the
semantic tests, publish actual outputs, then run the normal section-5.5 report.
The report freshly fits and generates on CUDA. Keep failed candidates and
strong controls. Independent release/report orchestration is outside the
explicit learner input inventory and receives its own report source hashes;
this separation never permits replacing a student method.
