# [10.1] Expected Outputs

These fixtures describe the CPU-executable exact XOR-direction study used in
the learner notebook. They are review aids, not the source of the notebook's
scientific result.

The smoke-test fixture records the preregistered claim and decision thresholds.
The reference-metrics fixture records the deterministic CPU reference values.
The solution notebook recomputes every metric from model-organism activations
before building its figures.

The fixed seeds are 7 for the orthogonal activation rotation, 11 for 128
shuffled-label fits, 13 for 256 isotropic random patch directions, and 123 for
the activation-noise stress test.

Exact structural identities use float64 and atol 1e-10. Behavioral gates use
the explicit preregistered thresholds because a later CUDA repetition may differ
at negligible floating-point boundaries.
