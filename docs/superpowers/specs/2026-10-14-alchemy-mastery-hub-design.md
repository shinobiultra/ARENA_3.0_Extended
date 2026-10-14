# ARENA Mastery Hub: Deep Learning Alchemy Companion Design

**Date:** 2026-10-14 (Asia/Tokyo)  
**Status:** Proposed design; documentation only, not an implemented course.  
**Entry point:** [MASTERY-HUB.md](../../../MASTERY-HUB.md)  
**Execution plan:** [Implementation plan](../plans/2026-10-14-alchemy-mastery-hub.md)

## 1. Intent and success

The user wants ARENA plus the existing extension to become a mastery hub, with Deep Learning Alchemy integrated as an original external resource and a substantially richer companion layer for intuition, interactive visualization, guided experimentation, prediction, verification, and research transfer. The added value must come from doing and understanding more, not rewriting resources that are already excellent.

Success is a learner completing an end-to-end loop without hidden machinery: stating an assumption, predicting a changed experiment before seeing it, manipulating meaningful controls, implementing the core method, checking it against a reference, explaining a failed hypothesis, and predicting a genuinely different case. It is not a promise that more sliders automatically produce better learning.

The requested delivery now is a plan and a pull request. Product implementation, training runs, and calendar changes are out of scope for this PR.

## 2. Project-wide constraints

- Original ARENA baseline: `f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0`; preserve original lessons, notebooks, and assets byte-for-byte.
- Planning baseline: `458a7b3ac327af538abac3400fa0fe8c64b146f9` on `codex/arena-frontier-lab-extension`; target the extension branch, not `main`.
- This planning PR adds only `MASTERY-HUB.md` and the two documents under `docs/superpowers/`; it changes no existing files.
- Preserve the existing lockfile and dependency pins; no dependency upgrades or cloud services are authorized by this plan.
- Use the repository's existing Python, Streamlit, Plotly, Jupyter, NumPy, and test tooling; validate compatibility at implementation time rather than assuming the environment works.
- The pilot must work on CPU without downloads, accounts, API keys, or paid compute; later GPU paths are optional, preflighted, and bounded by the existing 24 GB VRAM ceiling.
- Keep Stanford materials canonical and upstream; do not copy handouts, quizzes, code, images, or answer keys without checking their license and retaining required attribution.
- Never substitute generated curves, mocked measurements, historical receipts, or toy results for fresh real-model evidence.
- No calendar events, reminders, or recurring automations before the agreed release is built, verified, and accepted by the learner.

The [preservation contract](../../original_preservation_contract.md) and [AGENTS.md](../../../AGENTS.md) still apply. The instruction to plan this companion is not permission to ignore unfinished learner-experience work elsewhere. This PR makes no claim that earlier pilots are complete and does not change their priorities or status.

## 3. Integration choice

**Recommended: additive companion.** A new `mastery_hub/` entry point links to original materials and hosts the guided learning workflow. A small `arena_ext/mastery/` package contains reproducible simulation, recording, and display support. Notebooks expose the method being taught instead of hiding it in this package.

A bookmark-only index is insufficient because it does not provide the requested experimentation or assessment. A wholesale course rewrite or standalone new web platform would duplicate excellent content and introduce unnecessary maintenance. Reuse the existing local-first stack and add narrowly scoped components.

For navigation, keep the top-level `MASTERY-HUB.md` as the discoverable repository entry point. The future app links out to the original ARENA and Stanford sources and into companion units. Do not repurpose original chapter numbers or edit the original README merely to advertise a planned feature.

## 4. Source inventory and curriculum map

The official [CS 312 homepage](https://deep-learning-alchemy.github.io/) was checked on 2026-10-14. It identifies the course as Stanford Fall 2026, emphasizes experiment prediction, and labels its assignment topics tentative. At this snapshot its schedule links assignments 1–3; later assignment releases lie ahead. This is a source snapshot, not a promise that a link or syllabus remains unchanged.

The following short topic labels come from that homepage. Every proposed visualization and transfer exercise is our companion design, not a claim about the contents of an unreleased Stanford assignment.

| Upstream topic | Proposed original companion | Existing ARENA connection |
| --- | --- | --- |
| Basics: hyperparameter tuning and scaling | Budgeted sweep explorer; predict interpolation versus extrapolation; inspect all seeds and failed runs | Fundamentals; Chapter 17 training dynamics |
| Optimization 1: hyperparameter invariants | Quadratic optimizer pilot, update-scale comparisons, controlled width-transfer experiments | Fundamentals; Chapters 5 and 17 |
| Optimization 2: loss landscapes | Curvature/noise explorer; parameterization and projection counterexamples | Chapters 11 and 17 |
| Architecture 1: stability across depth/time | Activation and gradient propagation; residual/normalization ablations | Original transformer material; Chapter 5 |
| Architecture 2: extending model capacity | Capacity and compute-matched architecture comparisons | Chapters 5 and 11 |
| Generalization: data-efficient algorithms | Data-budget and distribution-shift explorer | Chapters 11 and 12 |
| Exotic: unspecified | Pending upstream definition; no invented syllabus or placeholder completion claim | Candidate capstone connection only |

The implementation must record source URL, attribution, date checked, release status, and code revision when a repository is actually used. Separate `source_status` from `companion_status`: a published source does not mean a companion has been built. Inventory any other previously supplied resources from actual links before mapping them; do not silently invent that inventory.

## 5. Mandatory learning contract

Every published companion has the following sequence, with advanced material behind progressive disclosure rather than a wall of text.

1. **Orient:** one concrete question, prerequisites, a visual mental model, a worked numerical example, and the original source link.
2. **Predict:** a directional, ranking, or numeric prediction plus confidence and a brief causal explanation, recorded before the result is shown.
3. **Explore:** meaningful sliders/selectors, pause/step/reset, repeatable seeds, overlays, and explicit accounting of what is fixed versus changed.
4. **Implement:** learner-written core functions in the notebook, with sub-function tests, helpful failure messages, and expandable hints/solutions.
5. **Verify:** an independent reference or exact oracle, positive and negative controls, a baseline, and a visible signature result.
6. **Falsify:** a prompt to break the current explanation; include an assumption-violating condition or a confound that changes the conclusion.
7. **Transfer:** unseen parameter regimes or code diffs, followed by a MechInterp/VLM connection where scientifically justified.
8. **Reflect and return:** explain the mismatch, revise the mental model, save the next concrete action, and revisit an unseen case later.

AI assistance may help with scaffolding, debugging, visualization, and Socratic hints. Challenge mode must not supply an answer or full solution before the learner commits their own prediction. Do not turn the course into watching an agent perform every experiment.

## 6. Pilot: optimizer dynamics that can be seen and checked

**Core question:** When do changes to curvature, learning rate, momentum, and gradient noise change optimization from convergence to oscillation or divergence?

Use the explicitly labeled analytic toy `L(x) = 0.5 * x.T @ H @ x`, where `H = R diag(lambda_min, lambda_max) R.T` is positive definite. In float64, start from `x0` and `v0 = 0`, draw independent seeded Gaussian gradient noise when enabled, then update `v_next = beta * v + H @ x + noise` and `x_next = x - eta * v_next`. This is the unnormalized heavy-ball convention; do not silently add a `(1 - beta)` factor.

Controls: both eigenvalues, rotation, starting point, learning rate, momentum, noise standard deviation, seed, and step count. Give each an explanation, reset-to-baseline action, and displayed units or conventions.

Visible results: contour trajectory with synchronized step selection; loss versus step with a log option that handles zero explicitly; displacement/update norm; and the prediction beside the observed result. Show the deterministic stability boundary only for the model and assumptions to which it applies. A 2-D toy is not a transformer loss landscape.

Exact checks derived from this recurrence: with zero noise and zero momentum, strict asymptotic stability requires `0 < eta < 2 / lambda_max`. The upper equality permits a non-decaying oscillatory eigendirection. With `0 <= beta < 1`, each heavy-ball eigendirection has characteristic polynomial `r^2 - (1 + beta - eta * lambda) * r + beta`; use its roots as an independent stability oracle. Noise does not inherit a claim of exact convergence to zero.

Required controls include `eta = 0` (no movement), zero initial state with zero noise (fixed optimum), a deliberately unstable step, and rotation of the coordinate frame with corresponding initial-state rotation (same loss trajectory). State assumptions beside every signature plot. Stop divergent runs honestly instead of clipping a curve into apparent convergence.

Two modes are separate: free exploration and an unassisted prediction challenge. An already revealed configuration cannot later be scored as unseen. A local self-study interface is not tamper-proof exam software; say so explicitly.

## 7. Interaction, records, and progress

Store immutable attempt records with schema version, unit/challenge identifiers, canonical configuration fingerprint, source/code revision, seeds, prediction, confidence, reasoning, observed result, control outcomes, and evidence class. Evidence classes are `analytic_toy`, `recorded_real_run`, and `live_real_run`; a replayed result is not a freshly computed training run.

Changing any question-defining setting before reveal invalidates the pending prediction. After reveal, retain the completed record and start a new attempt. Browser reruns and reloads must not erase a committed prediction or silently relabel an explored case as held out. JSON export is strict: no NaN/Infinity, executable objects, credentials, or arbitrary output paths.

For binary forecasts, use the Brier score `(p - y)^2`; ranking and numeric tasks report their own error measures instead of being forced into a binary metric. Display per-attempt evidence and sample count. Do not infer broad calibrated expertise from a handful of predictions, and do not treat a negative scientific result as a failed learner.

Default progress remains private and local, outside Git. Export Markdown suitable for a research notebook plus portable JSON only on explicit action. No remote analytics or automatic upload. Support saved progress, a single next action, optional short/standard/deep study modes, keyboard-operable controls, and explanatory text/data tables alongside plots. These are product requirements, not medical claims about the learner.

## 8. Compute and scientific validity

The live toy responds locally. Actual training requires an explicit Run action, a displayed configuration, cancellation, and a user-selected compute budget; moving a visualization slider must not start GPU work. Cache keys include code/data/model revisions, seeds, optimizer convention, and the complete relevant configuration.

A small real-model bridge follows the pilot. Freeze the organism, data split, metric, seed set, and budget before looking at results. Separate two legitimate questions: changing one component under a fixed recipe, and comparing methods after equal-budget retuning. Do not confound these questions or silently retune only the preferred method.

A ready real-model lesson includes readable code, a reproducible artifact, all completed/failed conditions, a small seed sweep with seed-level plots, a held-out condition, and a clearly stated scope. A small language-model result is not a replication of the course's larger organism unless the protocol actually matches. No cloud billing or paid service signup is implied.

## 9. Build and release boundaries

- **M0 — planning:** the three additive documents in this PR.
- **M1 — pilot:** the catalog, app, optimizer simulator, challenge state machine, notebook, export/replay, and tests form one complete learning loop.
- **M2 — empirical core:** a real-model bridge plus original companions for basics/scaling and the two optimization units.
- **M3 — curriculum extension:** architecture and generalization companions, matched to verified upstream releases; missing source material is reported as pending.
- **M4 — research transfer:** causal-versus-correlational interpretability challenges and an independently reviewable capstone.
- **M5 — accepted release and study handoff:** intended release scope is explicitly checked against M1–M4, the learner completes a walkthrough, and the remaining pending upstream scope is visible.

The pilot alone does not mean the mastery hub is complete. Each later lab needs its own focused design/review and must meet the same learning contract; do not expand the topic list merely because a generic widget framework exists.

A recurring study schedule is a post-release action. After release acceptance, inspect the learner's current calendar, propose a sustainable workload and session types, resolve clashes, and obtain approval of concrete blocks before writing events. No schedule is created, and no future background work is promised, in this planning PR.
