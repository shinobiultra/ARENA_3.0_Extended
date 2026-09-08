# Evidence for the complete 16.9 learner

The learning result is the executed notebook and its eighteen plots and image
galleries. These files support that result; they do not replace it.

`learner_execution.json` identifies the exact source inputs, completed own-answer
notebook, published pair/page and figures. `native_summary.json` records counts,
convergence, reference comparisons, retained sign errors and fixed gallery IDs.
`fresh_reproduction.json` identifies a separate process that reloads the pinned
UCI source bytes and recomputes every model, SGD state, query prediction,
attribution and deletion policy. The normal report runner creates a new such
run; it does not count a cached fit as retraining.

The published own-answer run is `own-full-v3` under
`/home/bitwise/.cache/arena_extended/data_attribution`. It executes 80 cells,
including 40 code cells, eleven exercise groups, thirteen student functions and
eighteen figures. The default plays include a step-size counterexample and
33 actual query-specific deletion refits. All 42 dropdowns open; 43 math
expressions render without errors. Eight current browser views and every figure
were visually inspected, including byte-identical previously inspected figures.

Every complete native run retains 930 independently converged reference fits,
six separate-versus-batched refit checks, ten independent NumPy/SciPy optima and
six full SGD trajectories (30720 updates). The data and settings were frozen
before digit fitting. Ranking queries alone choose the 48 policies per label
condition; all 240 policy/budget refits per condition finish before final
measurement. The later play is explicitly exploratory and uses a ranking query.

Fresh reproduction compares 1048 tensors containing 16726855 elements and 7272
non-tensor values. Every value matched exactly in the checked runs. Fifteen
native `.pt` payloads per run retain parameters, fitting histories, every
prediction, all random policies and the query play. These payloads remain in
the persistent cache; read-back archives list exact paths, sizes and hashes.

`full_own_answer_v3`, `fresh_native_v3` and `release_gate_v1` preserve release
checkpoints. Earlier `own-full-v1` layouts had an overlapping colorbar and
cramped gallery labels; the outputs and review are retained separately.
The v2 layout corrections and v3 code formatting changed no scientific settings.
The original mathematical foundation and ranking-only development remain under
`development_evidence`, with source snapshots rather than rewritten history.

Known scientific limits remain visible: influence has 54/8192 clean and 34/8192
corrupted wrong-sign final pairs despite excellent rankings; Taylor path sums
can greatly overestimate observed loss reduction; checkpoint choice can degrade
loss or accuracy. This small binary classifier does not establish large-network
attribution accuracy, exact Data Shapley or full-paper replication.

## How this fixture was produced

Small quadratic and logistic fixtures use explicit values in the visible test
module. Real digit files are downloaded from the SHA-256-pinned UCI archive,
with source IDs and selected tensors bound by `data_manifest.json`.

## Trusted implementation

Independent autograd derivatives, NumPy/SciPy stationary optima, complete affine
parameter gradients and separately restarted refits check the learner methods.
Fresh reproduction restarts every prescribed fit and SGD trajectory; it compares
the complete output structures, including all source IDs and counterfactuals.

## Random seed

Native order seeds are 0, 1 and 2. Matched random deletion seeds are 1000 through
1031. Source-row ordering and the deliberate corruption use fixed SHA-256 ordering,
not an outcome-selected random draw. See `native_protocol.md` for every setting.

## Allowed tolerances

Native refits require maximum absolute gradient at most 1e-10. Separate SciPy
parameter comparisons require maximum error below 1e-8. Fresh tensor comparisons
use rtol=1e-10 and atol=1e-11; observed maximum difference is zero. The unit tests
state their independent-oracle tolerances beside each assertion. No tolerance
is chosen to hide a scientific wrong-sign or checkpoint-ranking failure.

## When to regenerate

Regenerate the own-answer run, render/review, fresh reproduction and published
artifacts after changing any frozen input, method or experiment code. Regenerate
the normal report after changing its declared source inputs. Preserve earlier
runs and their failure evidence; use a new output directory for each run.
