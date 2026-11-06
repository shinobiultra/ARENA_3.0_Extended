# 9.5 Course ROME-like Rank-One Model Editing

**Core question:** Can one affine MLP projection receive an exact course ROME-like rank-one write, and what does a controlled CPU experiment actually establish about that write?

This section separates four claims which the earlier version conflated:

1. **Algebra:** the learner update is checked against an independently formulated constrained linear-system oracle, has rank one, and makes the actual biased `dense_4h_to_h` module hit the requested output.
2. **Rewrite behavior:** `Rome > Paris`, Rome top-1, and full-target likelihood are reported separately for 20 held-out paraphrases.
3. **Qualified locality:** 24 candidate facts are filtered by the unedited model's top-1 answer. Factual retention is reported only on the 14 candidates the pinned model actually knows.
4. **Localization and selectivity:** all layers and a wrong subject position receive norm-matched updates; 16 adversarial prompts are separated into target generalization, context override, non-target relations, and legitimate Rome-target strata.

The pinned CPU result is a negative localization case: multiple norm-matched layers and the wrong position can produce the requested continuation. The experiment also exposes one repetitive Louvre continuation, which is a localized generation failure rather than evidence about general fluency. It must not be described as an official ROME reproduction or a selective knowledge edit.

## Scope and missing parity

The implementation teaches the rank-one write equation and a small-model value optimization loop. Official ROME parity is not established because this section does not reproduce:

- Wikipedia-derived covariance statistics;
- released random-prefix key and value contexts;
- the released ROME value objective and its full optimization protocol.

The value optimizer's preservation KL is surfaced as a result, including an explicit warning when it is large.

## Experimental separation

- Eight optimization prompts and 20 held-out paraphrases have explicit IDs, wording families, split labels, and provenance.
- Twenty-four course-authored covariance contexts estimate key geometry only; they are not a replacement for Wikipedia covariance statistics.
- A disjoint set of 24 candidate facts is base-qualified before any factual-locality denominator is constructed.
- Activation patching uses three noise scales by three independently sampled directions, for nine genuinely distinct corruption conditions. It is not presented as causal-tracing protocol parity.
- Controls include all six norm-matched edit layers, a norm-matched wrong-position edit, norm-matched inconsistent shuffled-city targets, and five seeded same-rank/same-norm random updates.
- The multi-token `New York` playground is scoring-only: it changes the evaluator target, not the fixed Louvre-to-Rome update.

`build_notebooks.py` synchronizes the instruction page and both notebooks.
Function implementations are extracted from `solutions.py`, so visible answers
and tested code stay identical. Final experiments receive an explicit
`LearnerFunctions` bundle and execute the learner-written scorers, evaluator,
summarizer, patching sweep, and update builder. Provided row formatters are also
present in the learner; thirteen answer dropdowns complete all 32 code cells.

The current CPU learner execution gives 20/20 Rome-over-Paris preferences but
17/20 Rome top-1 responses. The edit retains 11/14 base-qualified facts; the eight
leakage-sensitive probes increase Rome probability by 0.448 on average. Five
norm-matched layers and the wrong-position control pass the pairwise threshold.
The applied affine write error is 2.97e-6. Final preservation KL is 0.729, measured
at the returned optimizer vector after all twelve steps. These results support
the negative localization lesson and leave selective editing unestablished.

Immediate checks now call learner functions for noise conditions, qualification,
and repetition. They also verify complete biased MLP captures, restoration after
an evaluation exception, and independently scored final multi-token loss/KL.

`release_report.py` provides the parent release process with a closed, hashable list of section-owned inputs and an exact CPU runner contract. Shared registration remains parent-owned and is intentionally not modified here.

## CPU verification

```bash
CUDA_VISIBLE_DEVICES='' uv run pytest -q chapter9_alignment_interpretability/exercises/part5_rome_model_editing/tests.py
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg uv run jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
  chapter9_alignment_interpretability/exercises/part5_rome_model_editing/9.5_ROME_Model_Editing_solutions.ipynb
```

The notebook does not require a GPU.
