The first normal report failed after successful fresh CUDA execution because its summary
tried to read story exact-answer accuracy, a metric intentionally undefined for
open-ended continuation. All 21,504 CUDA examples and all checkpoint/trace comparisons
passed. This archive retains the failed report, successful raw CUDA evidence and source
snapshot. The aggregation fix separates story reference diagnostics from objectively
scored task means. No model or sampler was tuned in response. Weights remain in the
named run directory; their hashes are retained.
