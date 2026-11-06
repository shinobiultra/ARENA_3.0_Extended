# Independent source/protocol review — 8.7 blank graph prediction companion

**Decision: CHANGES REQUIRED before adopting this exact candidate.** This is a bounded source/record/DEV review, not permission to run the target, publish, or claim scientific or human acceptance.

Reviewed manifest SHA256: `a0d3d316d181f189f57978f5eabb6ddd11f63cd9283d8b1f6294f5763b1d0c4a`.
Reviewed lesson SHA256: `7681f24d8f44039614f3b77567bf22f20597dee852890ca91b8b72e185ae2bb3`.
Reviewed tests SHA256: `5994b272ed44938450f62694c85ed9104db7a17616a8e1ea1022980feeffcd3c`.
Frozen author directory: `/tmp/arena-graph-blind-build-20261020`.

The reference implementation agrees, by source comparison, with the historical 8.7 scalar metric, signed exact effects, joint-mask midpoint IG, threshold ordering and six-decimal score storage, stable maximum-product path search/depth, full-background single-edge minimality, denominator, random nonselected nonzero pool, and reversed masks. The problem is meaningful independent checking of learner implementations, plus one narrow assembly-envelope validation gap.

## R1 — P1: The immediate DEV tests cannot detect an incorrect IG method

Location: `mastery_hub/units/graph_prediction/lesson.py:711` (score practice fixture), `:745` (assembly practice), `:435` (future run only validates IG shape/finite values before downstream self-consistency).

The only output-reaching practice route has two edges. On that route, `integrated_scores(graph, steps) = exact_edge_effects(graph)/2` and a midpoint implementation that unconditionally sets `steps=1` both reproduce every current expected score. Both incorrect implementations pass **all five** current practice stages. The future run checks the resulting IG matrix's shape/support and uses those same scores for graph selection/path validation; it does not independently check that the required 16-step attribution was computed. Consequently a learner can receive green practice feedback and an observed receipt for a different selection method. This does not require malicious code: using exact drops as a shortcut or unintentionally hard-coding a debug step count suffices.

Reproducer: `dev_mutant_proof.py`; measured results in `dev_mutant_proof.json`. The reference passes a separate anonymous three-edge chain with four midpoint steps. Each attribution must be `63/64 = .984375`, derived from `M(alpha)=3 alpha^3` and each gate derivative `3 alpha^2`. The two incorrect methods return `1.5` and `.75`. These are DEV values only.

Repair: add an immediate distinct three-or-more-edge DEV fixture, a hand-derived midpoint oracle, and regression mutants for half-exact and ignored-step-count implementations. Exercise at least two distinct step counts and preserve the signed test. Do not inspect the target or change its weights, method, gates, or selection.

## R2 — P2: The denominator and minimality exercises have degenerate expected values

Location: `mastery_hub/units/graph_prediction/lesson.py:732` and `:745`.

Every actual practice graph has a zero corrupt anchor and equal drops along its one output-reaching route. The fabricated gate-boundary case also has zero corrupt and one drop. Replacing `D=M_full-M_corrupt` with `D=M_full`, or replacing `min(path_drops)` with `max(path_drops)`, passes **all five** current practice stages. The final receipt validator would reject inconsistent assembled ratios, but only after the separately authorized target attempt; the immediate teaching checks should catch these errors before a precious target attempt is spent.

The independent anonymous parallel-route fixture in the same proof has full/corrupt/path-only/path-removed metrics `5,1.25,3,1` and full-background path drops `[4,3,3]`. Its correct ratios are `7/15,16/15,4/5`. The reference matches; both incorrect assembly methods disagree. This fixture is independently derived from path contributions `3+1+1`, not copied from reference output. It is never sealed or exported as target evidence.

Repair: use distinct parallel-route and nonzero-corrupt DEV checks with unequal per-edge drops; test the three path criteria separately, including a valid rejection. Retain zero-denominator rejection and the inclusive boundaries. Consider a negative denominator fixture to ensure signs are preserved, with clear interpretation.

## R3 — P2: Student assembly can overwrite recorder-owned metadata

Location: `mastery_hub/units/graph_prediction/lesson.py:476–483`.

`expected_assembly` is checked only as a subset. Then `obs = {'kind': ..., 'elapsed_seconds': measured_elapsed, **assembled}` gives the student's dictionary precedence over both recorder-owned fields. An extra `elapsed_seconds: 0.0` is therefore accepted by the observation schema and replaces the captured wall time; an otherwise correct assembly is not rejected for returning unknown fields. The independent deadline still exists, so this is a provenance/schema issue rather than a timeout bypass.

Repair: require the assembly dictionary's exact declared key set before constructing the envelope, and add recorder-owned fields after validation. Reject extra `kind`/`elapsed_seconds` keys instead of silently accepting or discarding them. A pure distinct-DEV mapping regression suffices; no target run is needed.

## Checks performed and positive findings

- Inspected the actual lesson, complete test source, generator, protocol, generated notebook, design record, and historical source before execution.
- Re-ran only the 57 authored source/record/generator/distinct-DEV tests: **57 passed in 1.80s**; log retained as `pytest_57.log`.
- Root explicitly authorized the added bounded distinct-DEV mutation proof. It used **0.694954757 CPU seconds** under a ten-second CPU limit. Four incorrect implementations evade the existing fixtures; independent oracles distinguish all four. The reference passes those oracles.
- The shipped notebook has no target results or filled forecast. Core functions are legible editable source strings with immediate fixture tests, help and postcommit solution access. Freeze/commit parse definitions and hash literal float32 bytes without evaluating the target. Defaults/decorators/annotations that could execute at definition time are rejected.
- Separate record schema, canonical sealed text, parent linkage, foreign-record rejection, malformed/nonfinite rejection, exact case identity, file/source/runtime hashes, private exclusive export, and failure envelopes are present.
- The optional reveal/run calls are commented out and ordered after commitment. Repeated attempts have an explicit process-local guard; the documented limitations correctly distinguish that from cross-process enforcement or human authority.
- Stored replay and plotting do not instantiate the target or invoke student methods or path search. Plots use stored scores, graph/path, ratios and both margins. Source/metric verification is not mislabeled as human mastery.
- The generator's preservation tests cover current executed, edited, filled, stale executed, malformed, duplicate JSON, invalid UTF-8 and unknown metadata notebooks; `--check` is read-only. Its CLI path is fixed to the additive companion.
- Scientific labels remain a public-derived fixed synthetic diagnostic, TRAIN none / DEV exposed / TEST none. No target success, human unseen status, independent generalization or scientific acceptance is asserted.

## Boundaries and handoff

The reviewer did not instantiate or evaluate the seven-node target, compute its selected path or predicate, read any target answer, run a GPU/cluster/model/download, modify author files, or modify repository files. All reviewer outputs are in this directory. Whole-repository preservation remains the root's integration responsibility.

The requested repair changes DEV checks and assembly validation only. Keep this initial review and mutation proof unchanged. A repaired snapshot needs a new manifest and bounded rereview; source review does not authorize a target attempt. The reference method and fixed target design need no outcome-dependent alteration.

Original ARENA preservation: reviewer made no edits to original chapters or any repository file. Student-facing improvement still needed: discriminating immediate feedback before commitment. Signature result remains an unexecuted stored-receipt plotting route. Controls and play cells are present; no target behavior or human learner walkthrough was accepted.
