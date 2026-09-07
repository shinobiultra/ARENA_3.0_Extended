# [7.2] Feature Verbalizers

This section asks one question: does a layer-3 Pythia-70m direction deserve the
verbalizer "the subject noun is plural"?

Students implement a transparent mean-difference decoder, exact component
replacement, real residual caching, executable semantic hypotheses, matched
baselines, the exhaustive `2**12` matched-pair label-swap null, real transformer
hooks, and separate distributions of 64 coefficient-matched orthogonal
interventions. A stronger causal control uses 256 directions from fixed-seed
within-pair training-label swaps, with identical per-example edit sizes.
The signature performs both singular-to-plural and
plural-to-singular edits. The play cell applies the same real hook to
learner-chosen nouns.

The result is intentionally honest about its weak points. Target held-out
accuracy is `1.0`, but 548 of 4,096 pair-preserving null fits also reach `1.0`
(`p=0.134`, p95 `1.0`). Decoding alone is therefore not significant. The suffix
rule is perfect on regular and nonce nouns and fails only on irregulars. Ten
held-out nouns are multi-token; the exact-ID lookup baseline uses their final
tokenizer IDs and has zero train/held-out ID overlap.

The real path loads pinned `EleutherAI/pythia-70m` offline on CUDA 13.2. The
number edit changes the downstream `are - is` margin by `8.48` against a
64-direction p95 of `1.09`. The reverse edit changes it by `9.62` against its
own matched-null p95 of `1.85`.

`verification_report.json` is supporting release evidence only. Neither the
notebooks nor their figure load results from it. Original ARENA is unchanged.

The stronger paired causal null gives forward/reverse p-values about 0.016 and
0.008. Validation evidence tables expose irregular-plural counterexamples to
the suffix explanation before the held-out evaluation. Candidate descriptions
are human-written executable predicates; automatic free-form verbalization is
still outside this lab's demonstrated scope.

Build from `build_notebooks.py`, execute the completed learner in a fresh
kernel, then run the tests and `scripts/run_extension_verification_reports.py
--section 7.2`. Both CUDA and explicit CPU paths use the same pinned model.
