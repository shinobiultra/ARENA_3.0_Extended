# Complete the real edge-level ACDC pilot

## Current learner failure

The paired 8.3 notebooks each have 32 cells. Their nonlinear named-graph
interventions and sequential deletion are useful. The real-model stage instead
patches entire GELU-1L head-result tensors from a clean cache into corrupt runs,
and searches eight whole heads on one continuation contrast. That cannot teach
individual sender-to-receiver deletion or reproduce a GPT-2 IOI/greater-than
circuit. Its README is older still and describes residual-position pruning.

Inspection includes the paired code and page, the original ARENA 1.4.1 IOI
path-patching/metric exercises, the ACDC paper and its upstream sender/receiver
hooks. Keep the original course unchanged and retain the useful toy warm-up.

## Target section claim

Students implement additive residual edges between upstream components and
individual Q/K/V inputs, MLP inputs and the final residual output. A retained
edge transmits the **current hybrid** sender, while a removed edge transmits
its matched corrupt reference. Layer normalization and downstream computation
are rerun. Define the full legal candidate graph before search; do not replace
it with a preselected set of known IOI heads.

The roadmap explicitly asks for both IOI and greater-than, ACDC and EAP-IG,
size/faithfulness comparisons, matched random controls and shifted prompts.
Start by validating the reusable full-graph engine and running real IOI ACDC;
keep the remaining requirements explicit until they are actually implemented.
The existing component-level result must not be relabeled as a full replication.

## Signature results

- A discovered real component-edge graph labeled by sender and receiver port.
- Threshold versus retained edges, KL divergence and held-out task recovery.
- Per-edge deletion/minimality effects and bounded completeness diagnostics.
- Distributions of equal-size random circuits, plus shifted-template results.
- Concrete prompts, correct answers and competing model predictions.

## Student implementations

Legal graph construction, residual message reconstruction, hook installation
and cleanup, hybrid edge execution, fixed-label task metrics, sequential
reverse-topological deletion with recomputation, graph reachability, threshold
sweeps, matched baselines and interpretable circuit metrics. Loading, dataset
generation and plotting can be provided. Core methods remain visible in the
notebook and receive immediate independent tests and working answer dropdowns.

## Controls and numerical checks

Full retained graph matches the untouched clean model; no retained edges
matches the untouched corrupt model. Single-edge changes affect only the
chosen receiver port, including attention Q/K/V and MLP inputs before their
normalization. Compare a vectorized implementation with an independent slow
oracle on a small real TransformerLens model and reject frozen-clean sender,
wrong-port, zero-substitution and leaked-hook implementations. Validate padded
end positions, fixed answer labels and matched corrupt sequences.

Use separate discovery, development and held-out prompts. Cache corrupt
activations only as the intervention baseline. A search trace must show each
candidate measured in the current graph. Disconnected branches may be removed
without evaluation only when reachability proves they cannot affect output.
Report KL and task behavior separately; neither small KL nor a local minimality
check proves complete mechanism recovery. Enumerating or sampling bounded
omitted subsets is a diagnostic, not universal completeness.

## Play cells

Change the threshold or an individual sender/receiver edge and redraw task
predictions and the graph. Inspect a held-out failure and compare a matching
random circuit. Preserve all measured negative results and seed/order effects.

## Files and verification

Own the existing 8.3 section sources, new readable lesson/builder/executor,
paired notebooks, tests, instruction page, figures, expected outputs and scope
metadata. Do not modify original ARENA code or unrelated experiment sources.
Run numerical and semantic tests, an actual completed learner from its own
answers, the full CUDA evidence harness, source/figure manifests, preservation,
style, readability and appropriate repository checks. Do not upgrade roadmap
requirements on the strength of a partial engine or a narrowed candidate set.

## Candidate selection protocol before held-out evaluation

Discovery uses 32 IOI examples at seed 1. Development uses 32 independently
generated examples at seed 2 from the same six template families. The final
holdout uses 64 examples at seed 3 from the next six original IOI templates.
Do not score the holdout until a candidate and selection reason are frozen.

The first exploratory threshold, 0.05 KL damage per deletion, completed all
legal candidates and disconnected-branch eliminations. It left 32 edges and
gave **negative discovery recovery (-0.277)**. Preserve it as a failure rather
than discarding it from the visible comparison. This observation motivates
the smaller thresholds 0.004 and 0.001. Each is a fresh full-graph search;
changing a cutoff on old one-shot scores is not an ACDC sweep.

Choose the smallest circuit among these completed candidates with development
recovery at least 0.85, pairwise accuracy no more than 0.10 below the clean
model, and KL(clean || circuit) at most 0.5. If none meets these criteria, keep
the failed comparison explicit and revise only with development data. Record
the actual decision before a held-out run. The planned uniform and receiver-
indegree-matched nulls each contain 32 random legal circuits, seed 701; keep
every sample. Random results do not enter the threshold selection criterion.

## Development-only amendment after the three IOI thresholds

The threshold 0.001 run completed 20,584 edge trials in 3,442.1 seconds and
retained 737 edges. Discovery recovery is 1.023, with 32/32 positive IO–S
margins. On the independent development set it gets recovery 0.881523 and
KL 0.408796, but only 27/32 correct name preferences. The clean model gets
31/32; the predeclared maximum 0.10 accuracy drop requires at least 28/32.
Thus **none of the original three candidates qualifies**. Do not round away
this failure or weaken the accuracy criterion.

Before opening any IOI held-out prompt, add one fresh full-graph search at
threshold **0.0005**, keeping discovery/development seeds, counts, templates,
corruption, metric, greedy order and selection criteria unchanged. Choose the
smallest qualifying candidate among all four tested thresholds. This is an
adaptive development-only extension of the threshold sweep, not an originally
preplanned fourth threshold. Preserve all four results and this distinction
in the learner. If it also fails, keep the IOI holdout unused and revise the
discovery protocol rather than presenting a failing candidate as qualified.

## Preserve the failed IOI selection as a learner result

The fourth threshold completed 23,032 trials, retaining 955 edges. Development
recovery is 0.7426, correct name preferences 26/32, and KL 0.2730. It also fails;
tighter local deletion tolerance is not a monotone guarantee of transfer.
The four-threshold study therefore ends with **no qualified IOI circuit**.
Keep the IOI holdout unopened. Do not weaken the criteria or keep extending
thresholds until a favorable result appears.

The lesson can now complete its bounded experiment: four fresh IOI searches
with an explicit failed selection, and the already frozen greater-than ACDC,
EAP and EAP-IG circuits with changed-wording transfer. For IOI anomaly hunting,
inspect the maximum-development-recovery candidate (0.001, 737 edges) on the
same development prompts. This is a diagnostic choice, not a qualified circuit
or independent transfer evidence. Show all 32+32 random nulls, edge deletions
and completeness probes, and label every figure with its development scope.
The execution/evidence harness must handle failed selection without opening
the holdout or aborting before students can study the failure. A future
discovery-protocol experiment is an extension activity, not an unreported
continuation of this selection sweep.

## Interrupted full learner continuation

The first full own-answer kernel was interrupted after 4,299.5 seconds during
IOI threshold .001, with its last atomic checkpoint at 9,700 decisions. The
original error saved in the executed notebook is KeyboardInterrupt; nbclient
then raised a secondary cleanup AttributeError. All three processes ended.
Before editing execution plumbing, all 24 bound source files were checked
against the failed execution and copied verbatim to its `source_snapshot`.

The new executor accepts only that learner's own interrupted checkpoint chain.
It rejects method failures, changed lesson/method sources, ambiguous contexts,
modified checkpoint bytes and mismatched live losses. Only three infrastructure
files changed: executor, checkpoint wrapper and its tests. The residual engine,
student method code, data, thresholds and selection rules did not change. The
new full kernel will rerun all cells and diagnostics while continuing actual
owned traces. Three orchestration tests pass, including identical uninterrupted
versus resumed decisions and rejection of changed methods. This is explicitly
recorded as owned interrupted continuation, not a newly repeated search or an
imported candidate result. Run it in a separate process session so the notebook
is not coupled to an interactive terminal's lifetime.
