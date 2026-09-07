# [8.3] ACDC and Circuit Metrics Expected Outputs

These fixtures freeze the results learners should see after completing the section. They describe the section's actual teaching surface, not a generic report schema.

## Exact toy result

The toy graph has ten named candidate edges and a declared eight-edge ground-truth circuit. One-shot insertion assigns zero recovery to every edge because the primary mechanism contains a product gate and the backup mechanism needs two edges. Iterative greedy deletion at threshold `0.10` removes only the two matched decoys, retains all eight ground-truth edges, and reaches recovery `1.0`.

The threshold sweep has two interpretable cliffs: at `0.17` it removes the two-edge backup path and falls to recovery `5/6`; at `0.84` it removes the six-edge primary path. Pairwise completeness is required because single-edge addition cannot reveal the omitted two-edge backup interaction.

All `10 choose 8 = 45` same-size circuits are enumerated. The recovered circuit ranks first with exact empirical `p = 1/45`; the best wrong circuit reaches only `5/6`. Its recovery remains exactly `1.0` in all three held-out toy regimes.

## Pinned real-model result

The CPU solution run loads the pinned TransformerLens `gelu-1l` checkpoint and patches named slices of `blocks.0.attn.hook_result`. Greedy deletion at threshold `0.05` retains `L0H0`, `L0H4`, and `L0H6`, recovering `0.946` of the clean-corrupt logit-difference gap. This set ranks first among all 56 three-head sets and stays above `0.75` recovery on three held-out prompt pairs.

Patching the clean MLP output alone reaches `0.991` recovery. This is deliberately presented as a downstream bottleneck control: it does not show that the upstream heads are irrelevant, and this component warm-up alone does not establish an edge-level IOI or greater-than circuit.

## Tolerances

The analytic toy checks use `rtol=1e-9` and `atol=1e-9`. The pinned float32 real-model acceptance thresholds are declared in `artifacts.lock.yml`; exact observed CPU values are recorded in `reference_metrics.json` for review and regression diagnosis.

## How this fixture was produced

`solutions.run_toy_study()` evaluates the explicit ten-edge graph and all 45
same-size circuits. `run_transformerlens_component_study(device="cpu")`
produces the pinned CPU checkpoint comparison; the strict CUDA report repeats
that real model study independently.

## Trusted implementation

The graph's declared eight ground-truth edges and analytic product/backup
mechanisms determine the toy answer. Exhaustive enumeration supplies an exact
same-size null independently of the greedy deletion order. Real-model evidence
uses actual named TransformerLens head slices, recomputes the downstream MLP,
and enumerates all 56 three-head sets; the warm-up result is component-level evidence. The full edge learner below
adds separate graph/intervention/search experiments.

## Random seed

The toy and real nulls enumerate their finite candidate sets; no Monte Carlo
seed chooses a favorable circuit. The real study uses the pinned checkpoint
and fixed prompt pairs documented in `artifacts.lock.yml`. Compare the held-out
pairs separately from the primary discovery prompt.

## Allowed tolerances

Use the `rtol=1e-9, atol=1e-9` analytic contract above. Float32 checkpoint
interventions use the lock's declared recovery thresholds. Exact circuit counts,
rank, edge identities, and selected head names are discrete checks.

## When to regenerate

Regenerate after changing a graph edge, intervention semantics, search order,
threshold, prompt split, checkpoint revision, or control. Run the immediate
oracles and the complete learner notebook before refreshing report hashes;
never substitute a cached report for a fresh circuit search.

## Expanded full additive-edge learner

The sections above describe the retained exact and GELU-1L warm-ups. The main
learner now builds all 32,491 legal GPT-2 additive residual edges, including
Q/K/V, MLP-input and final-output receivers. An active edge transmits the
current hybrid sender, and a deleted edge its matched corrupt reference.
Independent native-hook/slow-oracle tests and full/empty endpoint checks support
those semantics. EAP and input-interpolated EAP-IG scores also match pinned
unedited author code; the small course datasets are not complete paper benchmarks.

Each of four IOI thresholds starts a fresh search from the full graph:

| KL tolerance | Edges | Development recovery | Development accuracy | Development KL |
|---|---:|---:|---:|---:|
| .05 | 32 | −.2347 | 7/32 | 2.8059 |
| .004 | 383 | .5226 | 24/32 | .7706 |
| .001 | 737 | .8815 | 27/32 | .4088 |
| .0005 | 955 | .7426 | 26/32 | .2730 |

The fixed criteria require recovery ≥.85, accuracy within .10 of clean 31/32
(and thus at least 28/32), and KL ≤.5. None qualifies. The IOI holdout stays
unopened; the highest-recovery 737-edge candidate is diagnosed on development
only. All four failures remain visible. The fourth threshold was an explicitly
recorded development-only amendment after the original three failed.

Greater-than development instead freezes ACDC 78, EAP 300, task-loss EAP-IG 100
and KL-loss EAP-IG 300 edges. On 64 changed-wording prompts their recovery is
about .8318, .8952, .8586 and .9090. Preserve ACDC's drop below the development
floor. The metric uses full-vocabulary probability mass over later versus
earlier-or-equal two-digit suffixes, not probability conditioned on numeric
outputs.

Each inspected circuit retains both sets of 32 uniform/receiver-indegree-matched
nulls. Every retained ACDC edge is deleted individually, and 72 full-versus-
circuit subset probes test bounded completeness. Negative attribution curves,
all-name/repeated-subject corruption differences and failures remain evidence.

`execute_learner.py --scope full --publish` executes all own answers, every
fresh search and the full experiments before writing `learner_execution.json`
and `completed_edge_study.json.gz`. Partial/cached-candidate validations are
explicitly scoped under `edge_candidates`; none may replace the full run.
`edge_evidence.py` checks exact source and notebook provenance, then freshly
remeasures the IOI development diagnostic, four greater-than transfer circuits
and all 320 nulls on CUDA. The full 44-code-cell learner and separate recheck
now pass, with all 56 semantic tests passing. The release explicitly continues
its own interrupted checkpoint chain; `own_full_release_v2` preserves both
execution records, exact sources and all owned checkpoint contexts and traces.
No independent candidate archive was substituted for an owned search.
