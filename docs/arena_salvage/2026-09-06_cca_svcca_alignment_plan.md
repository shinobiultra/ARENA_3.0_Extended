# CCA, SVCCA and Procrustes learner

Selected topic: `representation.cca_svcca_procrustes` from the existing roadmap. This adds no topic beyond the user's selection. There is no existing 11.2 learner to preserve; use the original 1.1 notebook's question, mathematical explanation, shape contracts, help, implementation, direct tests and expected-result arc. Chapter 11.1 supplies prerequisites, with original chapters unchanged.

Current learner failure: the topic is unimplemented. Procrustes occurs only as a reducer-seed diagnostic; no notebook derives canonical coordinates, fits SVCCA, measures held-out alignment or asks whether alignment transports a causal change.

Target section claim: students can fit two representations into common coordinates, distinguish training agreement from held-out predictive agreement, and test whether a fitted map transfers an activation edit. CCA's invariance does not identify a unique mechanistic interpretation.

Signature result to create: a canonical-correlation spectrum with correctly paired, row-permuted and independent-noise controls; a width-to-sample sweep where noise obtains perfect training correlation and fails on new rows; cross-layer GPT-2 IOI alignment followed by actual original/donor/transported next-token distributions and intervention-distance curves. Both successes and failures remain visible.

Student-implemented functions: covariance and rank handling, SVD whitening, paired CCA, held-out canonical correlations, variance-threshold SVCCA, training-only orthogonal Procrustes, truncated transport of differences through canonical coordinates, paired logit-distribution scores. Methods must live in the notebook, not a hidden reference import.

Controls/baselines: independent views, shuffled row correspondence, high-width/small-sample overfit, held-out rows/templates/words, training-only fitting, equal-rank PCA/Procrustes and ridge predictors, same-norm random edits, zero edit, direct target-layer donor. Compare the actual donor distribution, not a fabricated desired report.

Play cells: change the dimension/sample ratio; discard a low-variance causal feature; vary canonical rank and inspect one transported real-model edit. Fit choices on development rows, retaining all candidates. Both models run from persistent cache on CPU. Pythia development gave weak donor behavior with generic next words and is retained as a negative; GPT-2 IOI development produced visible actual name changes. The GPT-2 protocol was frozen before final model access.

Files to modify: a new part2_cca_svcca_alignment directory, its readable lesson, paired notebooks, tests, CPU own-answer runner, instruction page and append-only chapter navigation after the complete learner runs. Do not modify the queued 11.1 scientific inputs while its GPU pipeline waits/runs.

Tests to run: exact covariance whitening and rank-deficient cases; known canonical correlation oracles; independent affine transformations and orthogonal-only Procrustes behavior; held-out sample leakage control; high-dimensional independent-noise counterexample; exact transported intervention and unrelated-coordinate preservation; native-model no-op and donor parity; full own-answer CPU learner; inspect every figure; original preservation and applicable course checks.

Sources: Raghu et al., SVCCA (2017), https://arxiv.org/abs/1706.05806v2 and authors' https://github.com/google/svcca ; Kornblith et al. (2019), https://arxiv.org/abs/1905.00414 for the sample-dimension/invariance failure mode. Implement the mathematics transparently; do not add another similarity method to this section.
