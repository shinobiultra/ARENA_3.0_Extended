# 9.1 verification ladder provenance

## How this fixture was produced

The full own-answer learner completed all 26 code cells, 15 development candidates, 85
final conditions and six live figures. `learner_execution.json` records its source
hashes and full execution. `completed_refusal_study.json.gz` stores the actual rows;
`direction_evidence.py` recomputes their metrics and frozen selection. The neighboring
README explains data splits, scopes and file contents. `fresh_cuda_release_v1/` archives
the separate 3,640-example and 65-direction CUDA regeneration with original/compressed
hashes.

## Trusted implementation

Immediate tests in `direction_method_tests.py` use analytic mean and projection
matrices, a translated-distribution threshold counterexample, tied-score AUROC,
independently reconstructed training permutations, native decoder-input captures and an
independent weight-projection oracle on a disposable tiny Qwen. Persistent cached
generation must agree with uncached generation, preserve unrelated hooks and parameters,
and handle per-example EOS. These are invoked on student function bodies. Three
selection/paired-denominator tests and five raw-evidence tests supplement the twelve
immediate method tests. All twenty pass; no source import alone establishes validity.

## Random seed

Null ensembles use seed 901 with all 32 draws; the first eight per sampler receive final causal evaluation. The disposable numerical model uses seed 12; the permutation test uses seed 1. The real model and dataset are revision-pinned, generation is greedy, and all fifteen candidates are selected exclusively on development before final evaluation.

## Allowed tolerances

Exact identities, token IDs, strings, marker flags, example IDs and inventory use
rtol=0, atol=0. Recomputed float metrics use rtol=0, atol=1e-6. Fresh FP32 vectors and
scores use rtol=0, atol=1e-4; this release's maximum observed errors are zero. The
immediate tests state their own dtype-appropriate `torch.testing.assert_close`
tolerances. No numeric tolerance permits changing a generated outcome or dropping a
strong null.

## When to regenerate

After a method, data, protocol or learner-source change, rebuild and execute the full
learner with `execute_learner.py --scope full --publish` into a new output directory,
run the section tests, inspect all six figures and run the separate 9.1 CUDA report.
Preserve previous failed candidates. Marker absence is not safety or correctness; the
selected model and dataset do not constitute the paper's full replication suite.
