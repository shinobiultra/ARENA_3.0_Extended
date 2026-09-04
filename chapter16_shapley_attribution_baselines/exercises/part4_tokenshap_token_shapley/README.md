# [16.4] TokenSHAP and TokenShapley

This section is an exact-first learner lab, not a verification-report wrapper.
Students define a position-preserving token coalition game, implement exact and
permutation-sampled Shapley values, and test the attribution against methods that
fail for known reasons.

The five-player organism contains:

- a necessary `transfer` token;
- the redundant pair `approve` / `allow`;
- an `urgent` x `transfer` interaction;
- the distractor `Please`.

The visible signature result compares exact Shapley with leave-one-out and a
recency-only control, measures sampling error over 40 seeds, verifies that a
random semantic token receives zero credit, and shows that sampling converges
to the wrong explanation after coalition values are shuffled within mask-size
buckets. It also shows that subword grouping changes the attribution. A
separate anomaly hunt proves that perfectly correlated observations cannot
identify off-manifold coalition values.

Core methods remain in both notebooks. `solutions.py` supplies reference answers
and preserves the existing release-only CUDA entry point. The CUDA preflight is
supporting implementation evidence; it is not the lesson or an LLM-faithfulness
claim.

Run the CPU checks with:

```bash
uv run pytest -q chapter16_shapley_attribution_baselines/exercises/part4_tokenshap_token_shapley/tests.py
uv run jupyter nbconvert --execute --to notebook --inplace \
  chapter16_shapley_attribution_baselines/exercises/part4_tokenshap_token_shapley/16.4_TokenSHAP_and_TokenShapley_solutions.ipynb \
  --ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=python3
```

Expected learner landmarks:

- exact Shapley: `[0, 0.3333, 0.3333, 1.5, 6.8333]`, sum `9.0`;
- leave-one-out: `[0, 0, 0, 3, 9]`, sum `12.0`;
- redundancy/synergy finite differences: `-2.0` / `+3.0`;
- 1024-sample error over 40 seeds: mean `0.0539`, p90 `0.0904`;
- random-token control: `[0, 0.3333, 0.3333, 0, 5.3333]`, total `6.0`;
- shuffled-value semantic max error: `5.6667` despite unchanged endpoints;
- grouped word credit `6.8333` versus subword-credit sum `7.6667`;
- correlated-support observed difference `0`, off-manifold difference `2`,
  attribution difference `0.6667`.
