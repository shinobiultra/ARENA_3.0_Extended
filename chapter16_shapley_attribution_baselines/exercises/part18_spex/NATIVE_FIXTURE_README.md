# Execution evidence for SPEX and ProxySPEX

The notebooks teach the methods through visible implementations and experiments.
These files support their provenance. Start with `DATA_ACCESS.md` for the portable
notebook workflow and the separate requirements for full model reproduction.

## How this fixture was produced

`run_native.py` evaluates the prospectively fixed word-mask experiment on GPT-2.
Its source, model revision, mask design, prompts, budgets and controls are bound
by `native_freeze_runtime_v2.json`. A separate cold process evaluates the same
experiment without receiving the first process's responses or fitted states.
The supervised comparison checks every retained scientific array and container.

`native_learner.export_native_fixture` exports exact saved training and audit
responses from the completed study. `publish_study.py seal-fixture` binds the
compressed and decoded data to its native execution receipt. Both notebooks use
that shared file to refit selected real and shuffled conditions with the student's
own functions. The all-prompt population curves are recorded study comparisons;
the notebook does not repeat the entire experiment or query GPT-2 again.

`learner_execution.json` identifies the actual executed notebook, figures and
portable replays. `native_execution.json`, `fresh_native_execution.json` and
`fresh_reproduction.json` preserve the two model runs and their comparison.
When concurrent scheduling is used, `scheduling_supervision.json` additionally
covers the complete second run, wait and comparison under their joint resource
limits. An inner success cannot override an outer failure. Exact scheduling
source, prospective supplement and independent approval accompany that receipt.

The review archive includes portable learner data and bounded supporting files.
The full scientific arrays, oracle database and fitted states remain in the
physically inventoried external study. Absolute historical paths are provenance,
not public download links. The repository supports the default portable replay;
it does not contain every dependency of the historical full run.

## Trusted implementation

Immediate tests compare the visible functions with known Boolean games, explicit
Fourier sums, signed decoding cases and exact tree-path spectra. The full study
retains the declared author SPEX, author ProxySPEX and maintained ProxySPEX
reference stages, including errors, timeouts and numerical disagreements.
Reference success and predictive superiority are not publication conditions.
`MODULE_API.md`, `native_protocol.md` and the saved reference outcomes disclose
implementation differences; a reference label alone does not establish parity.

## Random seed

The experiment fixes estimator seeds 0, 1 and 2 at budgets 456, 912 and 1,824.
`native_query_plan.py` and its recorded manifest fix the ordered masks, grouped
folds and response permutations. Shuffled-response permutations use
`PCG64(3300 + seed)`. All 32 prompts and all 1,152 method/seed/budget/label
conditions remain in the experiment. Editable notebook choices are exploratory
and do not replace its predeclared cases or recorded population results.

## Allowed tolerances

Cold reproduction requires exact discrete values, shapes and container structure.
Logit comparisons use `atol=1e-5, rtol=1e-6`; estimator predictions and portable
replay comparisons use `atol=1e-8, rtol=1e-7`. Exact matches and nonzero differences
within tolerance are reported separately. Tolerances do not impose a minimum
prediction quality, successful language-model answer or reference-stage success.
Immediate analytic tests retain their case-specific tolerances in the visible
test modules.

## When to regenerate

Use new output directories when executed sources, scientific inputs or the
environment change. Preserve earlier receipts and failures. A changed scientific
design requires a new prospective decision before inspecting its outcomes.
Never replace historical hashes to make an old execution appear current.

After complete model execution and cold reproduction, export and seal the
portable fixture, execute both complete own-answer notebooks, inspect all figures
and play results, and assemble reports through `native_report.py`. The legacy
GPU smoke/report writer is not this experiment's entrypoint. Installation follows
independent review of the exact publication candidate. See `DATA_ACCESS.md` and
`PUBLICATION_API.md` for the required inputs and commands.
