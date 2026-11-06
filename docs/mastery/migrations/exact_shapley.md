# Exact Shapley canonical master migration — 2026-10-16

## Plan before editing

**Current learner failure:** the full finite-game lesson is published under the
retired `part1_exact_shapley_ground_truth_games` route while configuration expects
`part1_exact_shapley`. Its method is already visible, but its checks import the old
prototype library and its reference-rich result has no explicit forecast/reflection
boundary. The existing CPU play is valid and should be preserved.

**Concrete question / target claim:** when a player's marginal effect depends on
its predecessor coalition, does exact Shapley conserve the four-player game's
known interaction credit while full-context leave-one-out fails conservation?

**Signature result:** retain the original three-panel figure: dividend oracle /
exact Shapley / leave-one-out bars; interaction-scale overcount curve against
Shapley efficiency and the additive control; predecessor-context marginals against
the weighted mean. Preserve all existing scientific code, prose and saved outputs.

**Student-implemented core:** coalition enumeration/validation, dividend game and
independent oracle, weighted marginal rows, exact Shapley, efficiency/symmetry/
dummy tests, permutation parity, leave-one-out and interaction-scale sweep.
Immediate tests and generated solution dropdowns must remain usable.

**Controls/baselines:** analytic dividend allocation, independent permutation
enumeration, additive scale zero, symmetric conjunction, dummy player and
leave-one-out failure. Replace only prototype-dependent test plumbing with local
independent finite-game oracles; do not change the taught methods or game.

**Play:** preserve editable `PLAY_INTERACTION_SCALE` and `PLAY_PLAYER` and the
analytic parity check. Add specific forecast/reflection prompts before existing
reveals, including negative interaction scale and the player-1 sign crossing.
Reference exposure remains guided practice, never unseen mastery.

**Files owned:** new `part1_exact_shapley` section, `master_16_1.py` and generated
master notebook, canonical page/media, the focused migration test, this note and
`docs/mastery/evidence/exact_shapley_master_migration/`. Root owns configuration
and checkpoint integration. All old section/page/assets stay physically unchanged.

**Verification:** archive/hash all pre-existing lesson sources and assets; compare
every original notebook code AST and all historical outputs/counts with explicit
setup/output-path routing exceptions only; complete every student stub from its
own solution; check safe import and independent tiny CPU game oracles; inspect the
retained signature PNG. Build without `--execute`; do not train, download, load a
pretrained model, initialize CUDA or regenerate the historical scientific report.

Historical reference retention is distinct from fresh scientific execution and
learner acceptance. Retain any failed checks and unresolved scope gaps below.

## Implemented teaching surface

The canonical route is `chapter16_shapley_attribution_baselines/exercises/part1_exact_shapley/`.
`infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_1.py`
generates the canonical notebooks, module and
`instructions/pages/01_[16.1]_Exact_Shapley_on_Ground-Truth_Games.{md,py}`.
The generated master notebook is present but ignored by the repository's existing
`master_*.ipynb` rule; the readable Python master is the authoring source.

The exercise notebook has 63 cells, 12 code cells, seven exercise groups and nine
method-specific forecast/reflection pairs. All fifteen student functions have an
immediate semantic check and exactly one working generated answer dropdown after
their cell. The historical worked formulations also remain in separately named
dropdowns before each exercise. The full original explanations, controls, play,
limitations and readings survive; the added navigation does not replace them.

Forecasts cover complete-table counts, baseline invariance, predecessor weighting,
symmetry/dummy/efficiency, 24-ordering parity, leave-one-out credit, the interaction
sweep, all three signature panels and the editable play. Every pair asks for
confidence, causal reasoning, exposure, a discrepancy explanation and a changed
scientific condition. Reference-rich guided practice earns no unseen score.

No attribution implementation is imported from `arena_ext` in the canonical
master, generated module or local semantic tests. The local tests retain the
original test names and replace prototype-dependent references with hand-worked
tables, explicit additive weights and independent conservation/permutation facts.
The old exercise-only CUDA wrapper remains an unexecuted supporting cell and
delegates to the physically unchanged historical neural-game implementation.

## Preservation and exact adaptation record

The pre-edit archive contains twenty safe regular members and is sealed with the
existing review-bundle contract. Its SHA-256 is
`f1a4d5fb2fc4a544830265c179546642b9ef8c841ecb79d3f187e5b1c2e51e89`.
Sixteen old lesson/page/asset files remain physically unchanged, as do their
original historical receipts. Four source dependencies were also captured to
preserve the historical neural/release path. There were no nested archives in the
capture. All fifteen historical report-input hashes matched physical bytes at
capture; this establishes identity, not a fresh scientific run.

All eleven original solution code ASTs, nonblank code lines, execution counts
and outputs survive. Exactly three notebook setup adaptations are normalized
when comparing the first cell: the section string, local test import and
`assets_dir` routing to canonical `media/`. Mathematical definitions, dividends,
players, constants, plotting calls, checks and play settings are unchanged.
The original exercise-only release wrapper is appended verbatim as code cell 12
with no execution count or output. Python-only root discovery uses `__file__` and
runtime calls are guarded with `MAIN`; the converter removes those guards from
notebook code. Importing the module from an unrelated directory preserves RNG
state and gradient state, produces no output/files and makes no CUDA query.

The saved signature PNG and notebook inline PNG are distinct historical renders:

| Render | SHA-256 |
| --- | --- |
| Saved instruction asset copied into canonical media | `5532bf813ea5cc97d8baa28bfc8f0a2b2a6578d49d50335e48b8c0015e10a485` |
| Inline notebook output, also embedded under the student figure cell | `c237d97ac139a0c0bc2a4a6b07c6c098609e195b44d5222bf88922601e723e54` |

Both images and the two supporting SVG assets are byte-identically preserved.
The three-panel inline render was visually inspected: oracle/exact bars coincide,
the additive point anchors zero surplus, the leave-one-out curve reaches nine at
scale two, and context variation is visible beside the additive control.
The historical interpretation said “including negative credit” although all four
scale-one bars are positive. The original sentence remains, followed by a clear
qualification and a negative-scale play instruction; no negative reference bar
is claimed.

Prose display adaptations are confined to converter-compatible exercise-rating
markup, wrapping, navigation, the initial objectives heading, media routing and
the separately named historical worked formulations. The original scientific
code and explanations remain inspectable. No original ARENA chapters were edited.

## Root-owned guided case proposal

For root's checkpoint integration, use the declared four-player game at baseline
`0.3`, scale `1`, with no learned model or dataset split. Ask the learner to
forecast exact Shapley `[1.9, 0.6, 1.05, 0.55]` and total `4.1`, leave-one-out
`[3.5, 1.9, 2.2, 1.0]` and total `8.6`, and surplus `4.5`. Require the pair/triple
dividend-count explanation and observation of additive-zero, analytic-oracle and
permutation controls. Changed condition: scale `-0.5` predicts surplus `-2.25`;
player 1 changes sign at scale `0.25`. A new seed is not transfer.

The code fingerprint must bind all declared dividends, baseline, player count,
scale and selected player. Separate evidence class `exact_finite_game`, execution
kind `retained_historical_reference` for saved notebook outputs, and learner
exposure `seen`/`unknown` as appropriate. The 28 tiny CPU play checks are fresh
engineering checks of explicit finite games, not a fresh neural experiment.
Root owns checkpoint/config/workbook changes; none were made in this lane.

## Verification and remaining gates

The existing environment was used without synchronization, downloads or CUDA:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/arena-audit-compat-uv \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
uv run --no-sync --offline python infrastructure/extension/build.py 16.1

PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/arena-audit-compat-uv \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-sync --offline python -m pytest -q -p no:cacheprovider \
  tests/mastery/test_exact_shapley_master_migration.py
```

The focused suite passed **18 tests** in 5.47 seconds: archive/physical retention,
all-cell AST/output parity, every own-answer completion, prose/guided practice,
all figure bytes, inert import, ten local independent student oracles, signed-scale
CPU play over seven scales and four players, and rejection of broken answers.
An ordinary subsequent build retained all outputs through the canonical same-code
check. No notebook kernel or full scientific experiment was launched.

Initial converter attempts exposed three formatting integration issues: the old
objectives heading, textual exercise ratings and a non-isolated bonus header.
They were corrected without changing scientific code; failed logs remain in the
evidence directory. The first focused test run found one whitespace-sensitive
prose assertion; the final check normalizes Markdown wrapping. Its failed log is
retained. These were engineering failures, not failed scientific experiments.

Remaining gates: independent review, root-owned checkpoint integration and learner
acceptance. Any fresh neural-game/CUDA reproduction and empirical generalization
remain outside this migration. Historical report acceptance fields are preserved
source records and do not mark the canonical migration or the full course ready.
