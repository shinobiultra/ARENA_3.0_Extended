# Deep Learning Alchemy Mastery Hub Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` when supported, or `superpowers:executing-plans` for inline execution, to implement this plan task-by-task. Checkbox steps are future work, not completed evidence. Read the spec and current repository instructions first.

**Goal:** Add an original, prediction-first interactive companion around the preserved ARENA and Deep Learning Alchemy resources, then establish a study routine only after an accepted learning release.

**Architecture:** An additive `mastery_hub/` learning entry point uses small pure components under `arena_ext/mastery/`. The app and notebooks share experiment definitions and record formats, while the notebook keeps the taught method visible and student-implementable. External resources remain canonical upstream.

**Tech stack:** Existing repository Python/uv environment, NumPy, Streamlit, Plotly, Jupyter/nbformat, pytest; PyTorch only for the later real-model bridge. Do not add a separate frontend stack.

**Spec:** [Design and learning contract](../specs/2026-10-14-alchemy-mastery-hub-design.md).  
**Planning baseline:** `458a7b3ac327af538abac3400fa0fe8c64b146f9`.  
**Status:** Planning-only PR. No app, lab, notebook, training evidence, or calendar integration is implemented by these documents.

## Global constraints

- Original ARENA baseline: `f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0`; preserve original lessons, notebooks, and assets byte-for-byte.
- Planning baseline: `458a7b3ac327af538abac3400fa0fe8c64b146f9` on `codex/arena-frontier-lab-extension`; target the extension branch, not `main`.
- This planning PR adds only `MASTERY-HUB.md` and the two documents under `docs/superpowers/`; it changes no existing files.
- Preserve the existing lockfile and dependency pins; no dependency upgrades or cloud services are authorized by this plan.
- Use the repository's existing Python, Streamlit, Plotly, Jupyter, NumPy, and test tooling; validate compatibility at implementation time rather than assuming the environment works.
- The pilot must work on CPU without downloads, accounts, API keys, or paid compute; later GPU paths are optional, preflighted, and bounded by the existing 24 GB VRAM ceiling.
- Keep Stanford materials canonical and upstream; do not copy handouts, quizzes, code, images, or answer keys without checking their license and retaining required attribution.
- Never substitute generated curves, mocked measurements, historical receipts, or toy results for fresh real-model evidence.
- No calendar events, reminders, or recurring automations before the agreed release is built, verified, and accepted by the learner.

## Review focus

1. A widget rerun, reload, or config edit must not score an obsolete prediction: covered by Task 3 state-transition tests and Task 5 UI tests.
2. Zero loss, invalid numbers, extreme curvature, and divergence must remain honest and exportable: covered by Task 2 numeric tests and Task 4 record validation.
3. Missing network/source material must not make the CPU pilot unusable or promote a placeholder: covered by Task 1 catalog tests and Task 5 offline smoke.
4. A wrong-but-confident answer, a scientific negative result, or an exposed challenge must not become a false mastery badge: covered by Tasks 3 and 8.
5. Progress export must not expose private notes automatically or execute imported content: covered by Task 4 strict schema, filename, and no-network tests.

## File ownership and boundaries

All paths below are **proposed implementation paths**, not claims that the files already exist.

| Path | Responsibility |
| --- | --- |
| `mastery_hub/app.py` | Local Streamlit entry point: source links, next action, explore/challenge UI |
| `mastery_hub/catalog.json` | Versioned resource/unit metadata, prerequisites, honest readiness and source status |
| `mastery_hub/README.md` | Exact launch, offline use, attribution, privacy, and troubleshooting |
| `arena_ext/mastery/catalog.py` | Catalog parsing and semantic validation |
| `arena_ext/mastery/quadratic.py` | Pure reference simulator and independent stability oracle |
| `arena_ext/mastery/challenges.py` | Prediction/reveal state machine and separate score types |
| `arena_ext/mastery/records.py` | Canonical fingerprints, private persistence, safe export/import, replay metadata |
| `arena_ext/mastery/plots.py` | Plotly figures derived from actual simulation traces |
| `mastery_hub/units/u02_optimization/lesson.py` | Readable canonical lesson cells, student stubs, visible solutions, tests, and play prompts |
| `mastery_hub/units/u02_optimization/exercises.ipynb` | Generated learner notebook; no filled-in answers in exercise cells |
| `mastery_hub/units/u02_optimization/solutions.ipynb` | Generated solution notebook with reproducible visible results |
| `scripts/build_mastery_notebooks.py` | Additive nbformat generator with a check-only mode; no changes to original generators |
| `tests/mastery/` | New focused numerical, state, catalog, record, UI, and notebook tests |
| `docs/mastery/` | Source snapshots, release checks, reviewed evidence, and later focused designs |

Create package `__init__.py` files only as needed. Do not refactor unrelated extension helpers. Keep generated user progress outside the checkout by default.

## Dependency and review order

Tasks 1–6 build M1. Task 2 is independent of Task 1; Task 3 consumes Task 2's configuration conventions; Task 4 consumes Task 3's records; Task 5 integrates Tasks 1–4; Task 6 verifies the notebook and whole pilot. Task 7 starts only after the pilot passes a learner walkthrough. Task 8 starts only after the required empirical foundations exist. Task 9 is the release gate and subsequent scheduling handoff.

For each engineering task: write the named tests, run them and confirm the intended failure, implement the smallest change, rerun the tests, review the diff, and commit only that task's files. Existing failing tests must be recorded separately rather than silently repaired or relabeled as this task's result.

### Task 1 — Source catalog and honest navigation

**Files:** create `mastery_hub/catalog.json`, `mastery_hub/README.md`, `arena_ext/mastery/catalog.py`, `tests/mastery/test_catalog.py`, and `docs/mastery/source_inventory.md`.

**Interfaces:** `load_catalog(path: Path) -> list[dict[str, object]]` validates schema version 1 and returns unit records. Each record has `id`, `title`, `source_url`, `checked_on`, `source_status`, `companion_status`, `prerequisites`, `original_paths`, and nullable `entrypoint`. Source statuses are `published`, `pending_upstream`, or `link_only`; companion statuses are `planned`, `prototype`, `validated`, or `learner_ready`.

- [ ] Write `test_catalog_statuses_are_independent`: a published source with a planned companion remains planned and has no runnable entrypoint.
- [ ] Write `test_catalog_rejects_unknown_ids_cycles_and_missing_paths`: reject duplicate IDs, prerequisite cycles, unknown statuses, and a declared ready entrypoint that does not exist.
- [ ] Write `test_pending_source_has_no_fabricated_release`: pending upstream units have no claimed assignment download or ready local lesson.
- [ ] Run `uv run pytest -q tests/mastery/test_catalog.py`; confirm the expected missing-implementation failures.
- [ ] Implement validation and populate only verified original links. Record the seven short upstream topic labels from the design; do not mirror Stanford files. Keep unverified additional resources out of the catalog.
- [ ] Rerun the catalog tests, check local links, and commit the catalog/source-inventory slice.

**Acceptance:** a learner can identify the canonical original, the prerequisite route, and whether a companion actually exists. Network availability is not required to parse or navigate the local catalog.

### Task 2 — Exact, inspectable optimizer reference

**Files:** create `arena_ext/mastery/quadratic.py` and `tests/mastery/test_quadratic.py`.

**Interfaces:** frozen `QuadraticConfig(lambda_min, lambda_max, angle_deg, x0, eta, beta, noise_std, seed, steps)`; `simulate(config: QuadraticConfig) -> QuadraticTrace`; `spectral_radius(config: QuadraticConfig) -> float`. `QuadraticTrace` contains the initial state, accepted step states, losses, update norms, and `stop_reason` (`completed`, `loss_cap`, or `nonfinite`). Use float64, a local NumPy generator, and the exact recurrence in spec section 6.

Validation: all numeric parameters finite; `0 < lambda_min <= lambda_max`; angle in `[0, 180)`; two finite coordinates; `0 <= eta`; `0 <= beta < 1`; `0 <= noise_std`; integer seed in `[0, 2**32)`; integer steps in `[1, 1000]`. Set a documented loss cap of `1e12`; preserve finite states and record the reason when a proposed update exceeds the cap or becomes nonfinite.

- [ ] Write `test_gd_one_step`: diagonal eigenvalues `(1, 10)`, `x0=(1, 1)`, `eta=0.1`, `beta=0`, and no noise produce `x1=(0.9, 0)` and `loss1=0.405` within `1e-12` absolute tolerance.
- [ ] Write `test_heavy_ball_convention`: the same case with `beta=0.9` produces `x2=(0.72, -0.9)` within `1e-12`; catch normalized-momentum and update-order mistakes.
- [ ] Write `test_stability_boundary`: for eigenvalues `(1, 10)` and `beta=0`, spectral radius is below 1 at `eta=0.19`, equal to 1 at `0.20`, and above 1 at `0.21`; derive the oracle from the polynomial, not the simulated plot.
- [ ] Add named tests for zero learning rate, a stationary noiseless optimum, same-seed replay, different-seed noisy paths, rotation invariance with rotated `x0`, invalid parameters, finite JSON-compatible divergence traces, and no global RNG mutation.
- [ ] Run `uv run pytest -q tests/mastery/test_quadratic.py`; confirm failures, implement the reference and independent oracle, then rerun to passing.
- [ ] Commit the numerical slice with assumptions and tolerance rationale beside the tests.

**Acceptance:** the simulator is parameter-driven and every displayed loss is recomputable from the recorded state. A numerical failure is visible, never converted into a successful run.

### Task 3 — Prediction-first challenge state machine

**Files:** create `arena_ext/mastery/challenges.py` and `tests/mastery/test_challenges.py`.

**Interfaces:** `new_attempt(challenge_id: str, config_fingerprint: str) -> dict[str, object]`; `commit_prediction(attempt, prediction, confidence: float, reasoning: str) -> dict[str, object]`; `reveal(attempt, config_fingerprint: str, observed) -> dict[str, object]`; `brier(p: float, y: int) -> float`. Transitions return new records; they do not mutate completed history. Attempt states are `uncommitted`, `committed`, `revealed`, and `invalidated`.

- [ ] Write `test_reveal_requires_prediction_and_matching_config`: uncommitted reveal and a changed fingerprint both fail with a useful message; a matched committed attempt reveals once.
- [ ] Write `test_completed_attempt_is_immutable_and_duplicate_reveal_is_idempotent`: reruns return the same result without creating a second score.
- [ ] Write `test_explored_case_is_not_unseen`: previously exposed challenge/config pairs are tagged practice, never included in unseen accuracy.
- [ ] Write `test_brier_reference_values`: `(1,1)` scores 0, `(0,1)` scores 1, and `(0.5,0)` scores 0.25. Reject confidence outside `[0,1]`, nonfinite values, and outcomes outside `{0,1}`.
- [ ] Define held-out templates before tuning examples: change curvature regime, starting direction, noise, or update convention; record the distinction between interpolation and extrapolation. Ranking/numeric tasks keep separate metrics.
- [ ] Run `uv run pytest -q tests/mastery/test_challenges.py`; confirm failures, implement the transitions and scores, rerun, and commit.

**Acceptance:** an answer is committed before the relevant result is revealed. No confidence badge is awarded from a single correct guess; counts and actual evidence remain visible.

### Task 4 — Private records, export, and replay

**Files:** create `arena_ext/mastery/records.py`, `tests/mastery/test_records.py`, and `docs/mastery/record_schema.md`.

**Interfaces:** `fingerprint(config: dict[str, object]) -> str` uses SHA-256 of canonical sorted JSON; `export_attempt(attempt: dict[str, object], destination: Path) -> Path`; `load_attempt(path: Path) -> dict[str, object]`; `replay(record: dict[str, object]) -> QuadraticTrace`. Schema version 1 stores the spec section 7 fields and the completed state from Task 3. Config includes every scientific input and an explicit simulator version.

Default persistence is a dedicated directory under the user's home data directory, not a tracked repository path. An export action is explicit and local; filenames are generated from validated identifiers. Import accepts strict JSON only, with a 1 MiB limit, known schema versions, validated fields, and no NaN/Infinity. Use atomic writes; refuse accidental overwrite unless explicitly requested.

- [ ] Write tests for key-order-independent fingerprints, changed scientific fields producing changed fingerprints, JSON round-trip, schema-version rejection, and unknown evidence-class rejection.
- [ ] Write `test_replay_matches_seed_and_config`: replay a saved noisy toy trace at `1e-12` tolerance in the same numerical environment; a simulator-version mismatch is explicit, not silently accepted.
- [ ] Write tests rejecting path traversal identifiers, nonfinite JSON, oversized imports, missing required fields, and executable/pickled content. Test interrupted-write behavior.
- [ ] Write a no-network test and verify that loading/exporting does not automatically upload notes, metrics, or files.
- [ ] Run `uv run pytest -q tests/mastery/test_records.py`; confirm failures, implement, rerun, and commit the persistence slice.

**Acceptance:** a learner can resume, export, and reproduce an attempt without publishing private progress or fabricating a new run.

### Task 5 — Interactive pilot app and honest plots

**Files:** create `mastery_hub/app.py`, `arena_ext/mastery/plots.py`, and `tests/mastery/test_app.py`; update only the new `mastery_hub/README.md` and pilot catalog entry.

**Interfaces:** `build_figures(trace: QuadraticTrace, config: QuadraticConfig) -> dict[str, object]` returns named Plotly figures for trajectory, loss, and update norms. The app consumes Tasks 1–4; it must not implement a second optimizer.

- [ ] Write UI tests for changing each scientific control, step selection, reset, switching explore/challenge modes, stale-prediction invalidation, and restoration after a rerun.
- [ ] Write `test_plot_values_equal_trace`: plot arrays match the simulator's actual states/losses; log display handles exact zero with a visible annotation, not invented loss values.
- [ ] Write tests for an unavailable external source, an offline CPU session, and an empty progress store. The app must still open the local pilot.
- [ ] Use Streamlit's supported testing facilities after verifying them against the locked environment; keep pure state tests independent of the UI test adapter.
- [ ] Run `uv run pytest -q tests/mastery/test_app.py`; confirm failures, implement the UI, then rerun.
- [ ] Launch `uv run streamlit run mastery_hub/app.py` and manually exercise stable, oscillatory, noisy, and divergent cases. Check keyboard operation, readable small-window layout, labels/units, comparison overlays, and text/data alternatives to plots.
- [ ] Record browser smoke evidence without calling it proof of scientific validity; commit the app slice.

**Acceptance:** controls change actual computation, assumptions are visible, and one complete prediction-to-replay loop works. Sliders never launch cloud or GPU jobs.

### Task 6 — ARENA-style notebook and M1 learner gate

**Files:** create the three proposed `mastery_hub/units/u02_optimization/` files, `scripts/build_mastery_notebooks.py`, `tests/mastery/test_notebook_contract.py`, and `docs/mastery/pilot_review.md`.

**Interfaces:** `lesson.py` exposes `build_cells(kind: str) -> list[dict[str, object]]` for `kind` equal to `exercises` or `solutions`. The generator writes only the two new notebooks and supports `--check`. Notebook core functions implement the Hessian, momentum update, simulation loop, and stability reasoning visibly; imports from `arena_ext/mastery/` are permitted for checks/records/plotting, not to replace student implementation.

- [ ] Write tests that generated notebooks are valid, deterministic, and confined to their new output paths; `--check` fails on drift without writing files.
- [ ] Write contract tests for a worked example, immediate sub-function tests, expected outputs, expandable hints/solutions, explicit assumptions, controls, a play cell, and an unseen transfer question.
- [ ] Add learner exercises in increasing difficulty: hand-compute one update; implement the recurrence; predict the stability boundary; test noise/rotation; construct a counterexample; compare notebook results with the reference.
- [ ] Run `uv run pytest -q tests/mastery/test_notebook_contract.py`; confirm failures, implement the canonical source and generator, and rerun.
- [ ] Run `uv run python scripts/build_mastery_notebooks.py --check` and execute the solution notebook using Jupyter's existing execution path. Execute learner cells with supplied reference answers separately; do not call deliberately unfilled stubs a runnable solution.
- [ ] Run `uv run pytest -q tests/mastery` and the existing preservation/config checks listed in Task 9. Record exact commands, revision, environment, and outcomes.
- [ ] Conduct a learner walkthrough: can a person predict, manipulate, implement, verify, explain a failed control, and resume without a developer's help? Keep status below `learner_ready` until both the tests and walkthrough pass.
- [ ] Commit the notebook and review slice; only then begin M2.

**Acceptance:** one genuinely useful companion, not merely a framework. This gate does not certify a whole course or a real-model result.

### Task 7 — Real-model bridge and published-unit companions (M2)

**Files:** create focused designs under `docs/mastery/designs/`, run protocols under `docs/mastery/protocols/`, and new unit directories `u01_basics/` and `u03_landscapes/`; extend `u02_optimization/` with a separately identified real-model bridge.

- [ ] Read the then-current official releases and repository model helpers. Freeze a small organism, data revision/split, metrics, seed set, controls, compute limit, and negative-result policy in a reviewed protocol before training. Do not invent Stanford protocol details.
- [ ] Specify a CPU verification path and optional GPU training path with explicit Run/cancel controls, a maximum wall-time/step budget, and the existing VRAM preflight. Cloud compute remains opt-in and outside this plan's authorization.
- [ ] **Basics/scaling:** predict a held-out budget/size setting; implement clean budgeted sweeps; show loss versus both training steps and compute/tokens. Include a deliberately misleading comparison that mixes budgets, then repair it. Distinguish fit interpolation from extrapolation.
- [ ] **Optimization invariants:** compare changes under fixed settings separately from equal-budget retuning. Show update/weight norms; record parameterization and optimizer convention. Add width-transfer, decay, or newer-optimizer comparisons only after their precise definitions and source revisions are verified.
- [ ] **Landscapes:** compare a chosen 2-D slice with directional curvature and a reparameterized or differently oriented slice. Show how an attractive projection can mislead; do not treat sharpness as a universal generalization oracle.
- [ ] For every new unit, write a focused design and file-level test plan before implementation. Required test families: deterministic replay, held-out split integrity, equal-budget accounting, baseline/control execution, stale-cache rejection, correct evidence-class display, and notebook learner-surface checks.
- [ ] Run actual small seed sweeps, retain failures/negative findings, execute solution notebooks, and have an independent reviewer inspect the scientific claim and learner experience before readiness promotion.

**Acceptance:** actual reproducible small-model evidence alongside the toy, with scope stated honestly. No requirement to reproduce a larger external training run on inadequate hardware, and no replacement with fabricated curves.

### Task 8 — Architecture, generalization, and research transfer (M3–M4)

**Files:** create focused unit designs and then `mastery_hub/units/u04_stability/`, `u05_capacity/`, `u06_generalization/`, and `research_transfer/`; update only new catalog entries and companion documents.

- [ ] Recheck upstream releases before attributing any assignment-specific content. Keep the unspecified exotic unit `pending_upstream`; do not hold the quality of completed units hostage to a guessed syllabus.
- [ ] **Stability:** predict how a normalization/residual change affects activation and gradient propagation. Plot layerwise norms and measured training behavior; compare both a fixed recipe and explicitly budgeted retuning. Tests must detect unintended changes to the supposedly fixed settings.
- [ ] **Capacity:** compare architecture/component ablations with parameter, token, and compute accounting. Make the prediction question explicit about its comparison budget; include a baseline and a budget-confounded counterexample.
- [ ] **Generalization:** manipulate data quantity or a defined distribution shift; use disjoint evaluation data, repeatable seeds, and leakage/shuffle controls. Do not assume an unreleased upstream modality protocol.
- [ ] **MechInterp transfer:** connect to existing lens, patching, SAE, geometry, and VLM lessons through their original links. Build prediction questions around real measured interventions, same-size/random controls, and held-out prompts. Separate probe correlation from causal intervention and apparent feature clarity from validated behavior.
- [ ] **Capstone:** require a pre-experiment claim, held-out prediction, controlled run, visible uncertainty/negative results, a reproducible artifact, and an explanation of which assumption failed. Judge the reasoning and evidence, not whether the hypothesis won.
- [ ] Before each implementation, write its focused file-level design/test slice. Run the same source, prediction-state, compute-budget, notebook, numerical/reference, and learner gates as the pilot, plus experiment-specific validity checks.

**Acceptance:** deep transfer exercises exist and work; they are not just links, static images, copied explanations, or agent-generated answers.

### Task 9 — Release verification and then study scheduling (M5)

**Files:** create `docs/mastery/release_checklist.md` and `docs/mastery/study_handoff.md`; update readiness only in the new catalog after evidence review. Changes to existing course registries/configuration require a separately reviewed integration diff and are not assumed here.

- [ ] Record the intended release scope and independently check every in-scope unit against the learning contract. List pending upstream material separately; never call the whole hub complete because M1 passes.
- [ ] Run the new focused suite, notebook generation/execution checks, and relevant existing gates. Existing commands known at the planning baseline are:

```bash
uv run pytest -q tests/mastery
uv run python scripts/build_mastery_notebooks.py --check
uv run python scripts/audit_original_arena_preservation.py
uv run python scripts/build_merged_config.py --check
```

The first two commands depend on future implementation. The latter two already exist. Recheck CLI contracts before execution, and run additional applicable learner/evidence/CI gates without editing original lessons to force a pass. Do not report GPU or full-suite success unless those runs actually occurred.

- [ ] Verify the full Git diff against the accepted baseline: original course files unchanged; no weights, credentials, private progress, or accidental dependency changes; no unsupported readiness claims.
- [ ] Have the learner complete a fresh challenge and a resume/replay walkthrough. Capture rough edges and fix them before acceptance. An author self-review is not an independent reviewer approval.
- [ ] Only after the agreed release is accepted, inspect the current calendar and ask the learner to approve a concrete study cadence. A draft workload may combine original reading, guided experiment sessions, an unseen prediction/retrieval session, and a periodic transfer/capstone session; it must fit actual availability rather than impose the university timetable.
- [ ] Preserve existing commitments, account for timezone, use next-action links in blocks, and establish a smaller fallback session instead of accumulating catch-up debt. Create events/reminders only through an available calendar/task integration after concrete scheduling approval.

**Acceptance:** release evidence and limitations are inspectable, the learner has accepted the usable workflow, and any study schedule is a separate confirmed action. This plan itself creates no events or background monitoring.

## Current PR verification boundary

For this documentation proposal, verify the remote diff contains exactly three added Markdown files and no changes to the existing tree. Check relative document links and consistency between the spec and plan. The future application tests and training runs above have not been executed and must not be reported as passing.
