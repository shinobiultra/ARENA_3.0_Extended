# Data Shapley root-test migration

The maintained teaching source is `infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_9.py`, generated into `chapter16_shapley_attribution_baselines/exercises/part9_data_shapley/`. The staged deletion of the former `part7_data_shapley_in_one_training_run` prototype remains intact. No compatibility lesson or deleted report assembly has been restored.

The root tests imported the deleted 16.7 prototype. This migration targets current methods directly and retains the old mathematical guarantees. The four legacy linear-game helper
tests continue to target `arena_ext.shapley_attribution`: their selected-subset mean-square game is distinct from the fixed-original-denominator logistic/SGD games in 16.9. Passing
either does not establish equality of those games.

| Old root guarantee / API | Current target | Retained assertion |
| --- | --- | --- |
| Four-example exact helpful/harmful signal; MC 512 samples; first-order correlation; singleton/empty utilities | Existing `arena_ext.shapley_attribution` public helpers | Original exact vector, deletion/addition booleans, MC error < 0.08, correlation 1, original singleton utilities remain unchanged |
| `random_data_attribution_failure_smoke_test` | 16.9 `exact_data_shapley` over the original one-step utility; `first_order_credit` over existing public linear-gradient scores | Same seed 13 and tensor draw order, harmful/helpful index changes, maximum absolute signal correlation <= 0.25 |
| `label_shuffled_attribution_failure_smoke_test` | Same current generic exact/first-order APIs | Same label permutation `(0, 3, 2, 1)`, harmful index moves, exact and first-order correlation <= 0 |
| `runtime_overhead_smoke_test(repeats=8)` | Direct `perf_counter` measurement of full utility, current generic exact enumeration, current first-order allocation | Untimed warmup plus eight repeats, each elapsed time > 0; no hardware-dependent speedup assertion |
| `mlp_forward`, `mlp_factors`, `per_example_gradients` | `mlp_forward`, `backprop_factors`, `per_example_grads` | Independent `nn.Sequential`/autograd oracles for one-, two-, three-layer networks; weights, biases, activations, unreduced loss gradients |
| `ghost_dot_products` | Same name in current lesson | Full autograd dot-product parity, unequal batch sizes, bias-only nonzero contribution, mismatched layer-count rejection |
| `sgd_step`, masked accountant update | `sgd_step`, `sgd_train(keep=...)` | Fixed original batch denominator, independently weighted autograd update, empty/all-masked update inert, input parameters unchanged |
| `local_coalition_values` | `local_game_values` | Every local coalition compared against independent autograd SGD + validation loss; no mutation |
| `batch_shapley` | `exact_local_shapley` | Seed-9 random 16-coalition game with nonzero empty utility, all 24 orderings, efficiency; malformed 7-entry table rejection plus empty/singleton/matrix rejection |
| `first_order_scores`, `second_order_scores` | `first_order_credit`, `second_order_credit` | Scaling and exact quadratic-loss oracle including self/cross terms; ghost-vector product parity |
| `run_accountant` | `run_in_run_shapley` | Per-step and aggregate exact credit telescope; all method aggregates equal per-step sums; credit absent outside each batch; ordinary SGD parameter parity; duplicate ID within a batch accumulates |
| `inrun_report.array`, `inrun_report.compare` | `arena_ext.legacy_shapley_evidence` | Exact tensor dtype/shape/value checks; container types and dict order; only 23 explicit historical timing paths may vary, each finite/positive; unknown timing changes rejected |

The numeric comparator is retained as a small production evidence utility, separate from test fixtures. Its caller supplies the precise timing inventory. The test uses the
historical three seeds `16701, 16702, 16703`, fourteen timing rows and six training times. This preserves a numeric comparison contract, **not** the deleted report's scientific
acceptance, source-hash validation, 49-run inventory, data-manifest validation, or notebook-execution certificate. No current receipt is fabricated from historical evidence.

## Narrow method repair and execution boundary

The tests initially exposed two genuine regressions in 16.9: Python `zip` silently truncated unequal factor-layer lists, and `exact_local_shapley` silently ignored the tail of a non-power-of-two table. The master now rejects these malformed inputs with `ValueError`; valid-input computations are unchanged. The complete-table guard also requires a one-dimensional table for at least one player.

Before regeneration, both notebooks and the master were archived under `docs/mastery/evidence/data_shapley_guard_migration/`, with original-byte SHA-256 values and code/output
inventories in `before.json`. The historical solution notebook had 56 executed code cells and 43 output-bearing cells. Those outputs belong to the pre-guard code hash. Regeneration
correctly cleared canonical execution outputs because the code changed; thirteen static expected-figure outputs remain, with zero executed code cells. Historical snapshots preserve
the prior execution and its provenance. The lead subsequently authorized a bounded full offline CPU execution. The canonical solution notebook now has all 56 code cells executed
and thirteen freshly generated PNG figures. `execution.json` records source hashes, runtime, output hashes, timing-only differences, and the initial sandbox kernel-start failure.
The intermediate `after.json` intentionally retains the pre-execution generated state.

Root regression tests use the offline existing environment and tiny CPU fixtures. Independent strict root assertions retain historical `atol=rtol=1e-12` for gradients, ghost
products, weighted SGD, local-game values, quadratic allocations, and trajectory telescoping. Broader lesson callable tests supplement these strict checks; their looser defaults do
not replace the historical guarantees. A `+1e-8` ghost-product mutation is explicitly rejected. No model downloads, GPU calls or cluster jobs were involved. Tests and solution
execution are distinct evidence layers; neither establishes human learner acceptance.

Validation: the first targeted run produced 19 passes and two expected guard failures. After the narrow master repair and regeneration, `UV_CACHE_DIR=/tmp/arena-mastery-uv
MPLCONFIGDIR=/tmp/arena-mastery-mpl uv run --no-sync --offline pytest -q tests/test_data_shapley.py tests/test_inrun_data_shapley.py` initially passed 24 tests in 1.70 seconds.
After independent review restored the explicit strict historical tolerance checks and added the mutation regression, it passed **28 tests in 1.52 seconds**. Compressed snapshots
were decompressed and checked against every recorded SHA-256; `git diff --check` passed for the edited tests, master and generated Python.

## Authorized canonical CPU execution

The exact `build.py 16.9 --execute --timeout 120` run used `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
and the existing offline uv environment. Jupyter initially failed before any cell because the sandbox prohibited local sockets; the same bounded command then ran with authorized
local socket access. The successful build reported 30 seconds; execution metadata totals 26.759528 cell seconds, with a maximum cell of 19.446485 seconds. All 56 code cells
executed without error and thirteen PNG figures were regenerated from actual outputs.

Master SHA-256: `2d7fc7851976eecd9e4efed88b53e93bfa8181409e946341d01b3bd1f009bfb8`. Executed solution notebook SHA-256: `043c7a35beaa0ee6dda7044e53ff803d367d72009472774743a0949754b70847`. Full source and figure hashes appear in `execution.json`.

All scientific printed text and rich tables match the historical execution. Only the ghost timing benchmark, an elapsed-time line, and the seconds column of the costs table differ.
Twelve of thirteen embedded PNG figures are byte-identical; only the measured timing figure changed. The toy mislabelled credit remains -0.3278 and full utility 0.5082. Seed 0
first-order credit still overallocates 1.5547 against actual gain 0.6111; second-order total remains 0.6089. Shuffled-label validation loss rises from 0.6805 to 1.5256. Removing
twelve lowest shuffled-credit rows worsens test loss by +0.0336, +0.0184, +0.0282 across the three seeds, whereas the corresponding first-order selections improve it by -0.0631,
-0.0696, -0.0843. These are rounded observed outputs, not a new general data-pruning claim.
