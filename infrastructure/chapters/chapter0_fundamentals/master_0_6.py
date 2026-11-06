# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {
        "title": "Implement and test",
        "icon": "1-circle-fill",
        "subtitle": ""
    },
    {
        "title": "Inspect and change controls",
        "icon": "2-circle-fill",
        "subtitle": ""
    },
    {
        "title": "Bonus",
        "icon": "star",
        "subtitle": ""
    }
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [0.6] How to Know When an Interpretability Result Is Fake (Practice)
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Introduction
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
A feature with perfect probe accuracy, three dramatic examples and a steering effect can all be misleading. You will implement the original four detectors and compare each attractive metric with its matching control. These are deterministic designed tensor fixtures, not trained probes or language-model behavior.

This is a source migration, with no fresh method, numerical, model, GPU or environment execution. Saved answers and figures are historical exposed guided practice. Commit a forecast, confidence, reasoning and prior exposure before opening each reveal; this is not unseen mastery. Importing the Python module binds definitions only. Call `initialize_runtime()` explicitly before using numerical or runtime callbacks; the callbacks are not initialized merely by importing them. The notebook setup invokes this initializer when a learner elects to run it.

Original ARENA chapters 0–4 remain the canonical base. These existing 0.6 and 1.6 addenda do not replace any original section. The exact original notebook objects, metadata, outputs and bytes are in the preparation archive and inverse map. Historical claims such as `accepted` and `course_ready` are preserved historical fields, not fresh learner acceptance.

### Practice copies and intact originals

This public guided Practice exercise copy starts with empty code outputs and execution counts.
Its historical solutions retain the original observations. Foundation historical appendix code
is shown read-only, so running the active exercises does not overwrite learner callbacks with
old reference imports or stubs. Section 16.3 keeps its original teaching code executable.
All mathematical methods, controls and visible implementation work remain in the active lesson.

Open the intact original source books below. Their original directories, modules, helpers, data
locators and setup code are unchanged; open a book in its own original directory and use its
existing setup cells and declared environment. No original setup or numerical method was run
for this migration, so these are static replay paths rather than a fresh execution guarantee.
The links resolve from the generated Practice book and lesson page; exact repository paths are
shown for master/source readers.

- [Intact original: 0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb](../../exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb) — `chapter0_fundamentals/exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb`
- [Intact original: 0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb](../../exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb) — `chapter0_fundamentals/exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb`

The prior combined migration reference books also remain intact alongside this Practice pair.
They preserve historical outputs and execution metadata and are read-only provenance copies,
not a fresh execution or blind assessment.

- [Intact combined exercises reference](../../exercises/part6_fake_interpretability_results_practice/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb)
- [Intact combined solutions reference](../../exercises/part6_fake_interpretability_results_practice/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb)

'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ Implement and test

> ##### Learning Objectives
>
> - Implement each core operation beside the original immediate positive and negative tests.

### 2️⃣ Inspect and change controls

> ##### Learning Objectives
>
> - Interpret the original visible signature, qualify the claim, and predict a changed condition.

### ☆ Bonus

Inspect the retained historical contracts and complete original source materials.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

MAIN = __name__ == "__main__"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ Implement and test
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def initialize_runtime(root: str | Path | None = None) -> None:
    """Explicitly initialize runtime dependencies; never runs on raw import."""
    global t, tests, json, sys, tempfile, subprocess, math, platform, importlib
    global root_dir, chapter, section, exercises_dir, section_dir
    import importlib.util
    import json
    import math
    import platform
    import subprocess
    import sys
    import tempfile
    import torch as t
    chapter = "chapter0_fundamentals"
    section = "part6_fake_interpretability_results"
    if root is None:
        if "__file__" in globals():
            root = Path(__file__).resolve().parents[3]
        else:
            root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
    root_dir = Path(root)
    exercises_dir = root_dir / chapter / "exercises"
    section_dir = exercises_dir / section
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    tests = importlib.import_module(chapter + ".exercises." + section + ".tests")


if MAIN:
    initialize_runtime()

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass(frozen=True)
class LabelLeakageReport:
    leaked_feature_index: int
    leaked_feature_accuracy: float
    shifted_no_leak_accuracy: float
    accuracy_gap: float
    detects_leakage: bool

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass(frozen=True)
class CherryPickReport:
    selected_mean_effect: float
    population_mean_effect: float
    population_median_effect: float
    inflation_ratio: float
    detects_cherry_picking: bool

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass(frozen=True)
class ProbeOverfitReport:
    train_accuracy: float
    heldout_accuracy: float
    generalization_gap: float
    detects_overfit: bool

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass(frozen=True)
class FakeRandomDirectionControlReport:
    claimed_effect: float
    random_p95_effect: float
    effect_gap: float
    passes_random_control: bool
    detects_random_direction_failure: bool

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass(frozen=True)
class FakeResultAuditReport:
    leakage_detected: bool
    cherry_pick_detected: bool
    probe_overfit_detected: bool
    random_direction_failure_detected: bool
    all_bogus_results_flagged: bool

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Predict what happens when the leaked and shifted feature columns are identical. Does perfect accuracy alone trigger this detector?

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Label leakage

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Threshold signed scores at zero. Compute both supplied feature-column accuracies before comparing their gap. The default control is anti-correlated, not a shuffled-label independence test.

<details>
<summary>Help - implementation and interpretation</summary>

Threshold signed scores at zero. Compute both supplied feature-column accuracies before comparing their gap. The default control is anti-correlated, not a shuffled-label independence test.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def default_label_leakage_fixture(
    device: t.device | str | None = None,
) -> tuple[t.Tensor, t.Tensor]:
    """Return features where column 0 is leaked labels and column 1 is a control."""

    labels = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.long, device=device)
    signed = labels.float() * 2 - 1
    features = t.stack([signed, -signed], dim=1)
    return features, labels

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def binary_accuracy(scores: t.Tensor, labels: t.Tensor) -> float:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Return threshold-at-zero binary accuracy for signed labels."""

    labels = labels.long().flatten()
    predictions = scores.flatten().gt(0).long()
    return float(predictions.eq(labels).double().mean().item())
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def label_leakage_report(
    features: t.Tensor | None = None,
    labels: t.Tensor | None = None,
    *,
    leaked_feature_index: int = 0,
    shifted_feature_index: int = 1,
    min_gap: float = 0.5,
) -> LabelLeakageReport:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Detect a feature that directly encodes the label."""

    if features is None or labels is None:
        features, labels = default_label_leakage_fixture()
    if features.ndim == 1:
        features = features[:, None]
    if features.shape[0] != labels.numel():
        raise ValueError("features and labels must have the same number of examples")

    leaked_feature = features[:, leaked_feature_index]
    shifted_spurious_feature = features[:, shifted_feature_index]
    leaked_accuracy = binary_accuracy(leaked_feature, labels)
    shifted_accuracy = binary_accuracy(shifted_spurious_feature, labels)
    gap = leaked_accuracy - shifted_accuracy
    return LabelLeakageReport(
        leaked_feature_index=leaked_feature_index,
        leaked_feature_accuracy=leaked_accuracy,
        shifted_no_leak_accuracy=shifted_accuracy,
        accuracy_gap=gap,
        detects_leakage=leaked_accuracy >= 0.95 and gap >= min_gap,
    )
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    tests.test_binary_accuracy_thresholds_signed_scores(binary_accuracy)
    tests.test_label_leakage_report_flags_direct_label_feature(label_leakage_report)
    tests.test_label_leakage_report_uses_supplied_feature_indices(label_leakage_report)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output and solution reveal</summary>

Each named original test should print its `All tests ... passed!` line. The matching positive and negative tests are both required; read the generated solution dropdown for the complete implementation. These are expected historical contracts, not newly executed test results.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Predict whether a representative selection triggers the same detector as selecting the three largest effects.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Cherry-picked examples

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Compare selected mean with full-population mean and median. This is a descriptive inflation detector; it does not implement a permutation null or multiple-comparison correction.

<details>
<summary>Help - implementation and interpretation</summary>

Compare selected mean with full-population mean and median. This is a descriptive inflation detector; it does not implement a permutation null or multiple-comparison correction.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def default_cherry_pick_fixture(device: t.device | str | None = None) -> t.Tensor:
    """Return mostly small effects with three dramatic selected examples at the end."""

    return t.tensor(
        [
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.10,
            0.11,
            0.12,
            0.13,
            1.40,
            1.55,
            1.70,
        ],
        device=device,
    )

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def cherry_pick_report(
    effects: t.Tensor | None = None,
    selected_indices: t.Tensor | list[int] | None = None,
    *,
    min_inflation: float = 3.0,
    median_multiplier: float = 5.0,
) -> CherryPickReport:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Detect when selected examples exaggerate the population effect."""

    effects = default_cherry_pick_fixture() if effects is None else effects.flatten()
    if selected_indices is None:
        selected = effects[-3:]
    else:
        indices = t.as_tensor(selected_indices, dtype=t.long, device=effects.device)
        selected = effects[indices]
    selected_mean = float(selected.mean().item())
    population_mean = float(effects.mean().item())
    population_median = float(effects.median().item())
    inflation = selected_mean / population_mean
    return CherryPickReport(
        selected_mean_effect=selected_mean,
        population_mean_effect=population_mean,
        population_median_effect=population_median,
        inflation_ratio=inflation,
        detects_cherry_picking=(
            inflation >= min_inflation
            and selected_mean >= median_multiplier * population_median
        ),
    )
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    tests.test_cherry_pick_report_compares_selected_to_population(cherry_pick_report)
    tests.test_cherry_pick_report_rejects_representative_selection(cherry_pick_report)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output and solution reveal</summary>

Each named original test should print its `All tests ... passed!` line. The matching positive and negative tests are both required; read the generated solution dropdown for the complete implementation. These are expected historical contracts, not newly executed test results.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Predict the held-out accuracy of the default all-zero guesses, then predict the decision for the original generalizing positive control.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Probe overfitting

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

A copied training-label vector is a memorization oracle, not a trained probe. Require high training accuracy, weak held-out accuracy and a large gap together.

<details>
<summary>Help - implementation and interpretation</summary>

A copied training-label vector is a memorization oracle, not a trained probe. Require high training accuracy, weak held-out accuracy and a large gap together.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def default_probe_overfit_fixture(
    device: t.device | str | None = None,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Return train labels copied by a memorizing probe and held-out all-zero guesses."""

    train_labels = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.long, device=device)
    heldout_labels = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.long, device=device)
    train_predictions = train_labels.clone()
    heldout_predictions = t.zeros_like(heldout_labels)
    return train_predictions, train_labels, heldout_predictions, heldout_labels

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def _class_accuracy(predictions: t.Tensor, labels: t.Tensor) -> float:
    predictions = predictions.long().flatten()
    labels = labels.long().flatten()
    if predictions.numel() != labels.numel():
        raise ValueError("predictions and labels must have the same number of examples")
    return float(predictions.eq(labels).double().mean().item())

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def probe_overfit_report(
    train_predictions: t.Tensor | None = None,
    train_labels: t.Tensor | None = None,
    heldout_predictions: t.Tensor | None = None,
    heldout_labels: t.Tensor | None = None,
    *,
    min_train_accuracy: float = 0.95,
    max_heldout_accuracy: float = 0.6,
    min_gap: float = 0.35,
) -> ProbeOverfitReport:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Detect a memorizing probe that does not generalize."""

    if (
        train_predictions is None
        or train_labels is None
        or heldout_predictions is None
        or heldout_labels is None
    ):
        (
            train_predictions,
            train_labels,
            heldout_predictions,
            heldout_labels,
        ) = default_probe_overfit_fixture()
    train_accuracy = _class_accuracy(train_predictions, train_labels)
    heldout_accuracy = _class_accuracy(heldout_predictions, heldout_labels)
    gap = train_accuracy - heldout_accuracy
    return ProbeOverfitReport(
        train_accuracy=train_accuracy,
        heldout_accuracy=heldout_accuracy,
        generalization_gap=gap,
        detects_overfit=(
            train_accuracy >= min_train_accuracy
            and heldout_accuracy <= max_heldout_accuracy
            and gap >= min_gap
        ),
    )
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    tests.test_probe_overfit_report_requires_heldout_gap(probe_overfit_report)
    tests.test_probe_overfit_report_rejects_generalizing_probe(probe_overfit_report)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output and solution reveal</summary>

Each named original test should print its `All tests ... passed!` line. The matching positive and negative tests are both required; read the generated solution dropdown for the complete implementation. These are expected historical contracts, not newly executed test results.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Predict which direction clears the p95 control plus the fixed margin: the weak default direction or the behavior-aligned positive control.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Random-direction controls

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Normalize directions before comparing absolute projections. The five-vector fixed control bank is designed, not a calibrated random-direction distribution or a language-model steering experiment.

<details>
<summary>Help - implementation and interpretation</summary>

Normalize directions before comparing absolute projections. The five-vector fixed control bank is designed, not a calibrated random-direction distribution or a language-model steering experiment.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def default_random_direction_fixture(
    device: t.device | str | None = None,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Return a weak claimed direction and a random-control bank."""

    behavior_direction = t.tensor([1.0, 0.0, 0.0], dtype=t.float64, device=device)
    claimed_direction = t.tensor([0.05, 0.9987, 0.0], dtype=t.float64, device=device)
    random_directions = t.tensor(
        [
            [0.12, 0.99, 0.00],
            [-0.08, 0.99, 0.05],
            [0.04, 0.20, 0.98],
            [0.10, -0.95, 0.10],
            [-0.11, 0.00, 0.99],
        ],
        dtype=t.float64,
        device=device,
    )
    return behavior_direction, claimed_direction, random_directions

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def random_direction_control_report(
    behavior_direction: t.Tensor | None = None,
    claimed_direction: t.Tensor | None = None,
    random_directions: t.Tensor | None = None,
    *,
    required_margin: float = 0.25,
) -> FakeRandomDirectionControlReport:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Detect a claimed steering direction that is no stronger than random controls."""

    if behavior_direction is None or claimed_direction is None or random_directions is None:
        behavior_direction, claimed_direction, random_directions = (
            default_random_direction_fixture()
        )
    behavior_direction = behavior_direction.flatten()
    claimed_direction = claimed_direction / claimed_direction.norm()
    behavior_direction = behavior_direction / behavior_direction.norm()
    random_directions = random_directions / random_directions.norm(dim=1, keepdim=True)
    claimed_effect = float(abs(claimed_direction @ behavior_direction).item())
    random_effects = (random_directions @ behavior_direction).abs()
    random_p95 = float(t.quantile(random_effects, 0.95).item())
    gap = claimed_effect - random_p95
    passes = gap >= required_margin
    return FakeRandomDirectionControlReport(
        claimed_effect=claimed_effect,
        random_p95_effect=random_p95,
        effect_gap=gap,
        passes_random_control=passes,
        detects_random_direction_failure=not passes,
    )
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    tests.test_random_direction_control_report_rejects_weak_claim(random_direction_control_report)
    tests.test_random_direction_control_report_accepts_strong_claim(random_direction_control_report)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output and solution reveal</summary>

Each named original test should print its `All tests ... passed!` line. The matching positive and negative tests are both required; read the generated solution dropdown for the complete implementation. These are expected historical contracts, not newly executed test results.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Predict whether an audit can pass when one injected failure is not detected; distinguish a passing diagnostic from acceptance of the original attractive claim.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - Aggregate the audit

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

Keep every detector boolean. The aggregate passes when every injected bogus claim is rejected, and says nothing about untested failure modes.

<details>
<summary>Help - implementation and interpretation</summary>

Keep every detector boolean. The aggregate passes when every injected bogus claim is rejected, and says nothing about untested failure modes.

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def fake_result_audit_report(
    leakage: LabelLeakageReport | None = None,
    cherry_pick: CherryPickReport | None = None,
    overfit: ProbeOverfitReport | None = None,
    random_direction: FakeRandomDirectionControlReport | None = None,
) -> FakeResultAuditReport:
    # EXERCISE
    # # YOUR CODE HERE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    """Run a compact audit suite for fake-result failure modes."""

    leakage = leakage or label_leakage_report()
    cherry_pick = cherry_pick or cherry_pick_report()
    overfit = overfit or probe_overfit_report()
    random_direction = random_direction or random_direction_control_report()
    flags = [
        leakage.detects_leakage,
        cherry_pick.detects_cherry_picking,
        overfit.detects_overfit,
        random_direction.detects_random_direction_failure,
    ]
    return FakeResultAuditReport(
        leakage_detected=flags[0],
        cherry_pick_detected=flags[1],
        probe_overfit_detected=flags[2],
        random_direction_failure_detected=flags[3],
        all_bogus_results_flagged=all(flags),
    )
    # END SOLUTION

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details><summary>Solution</summary>

```python
SOLUTION
```

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def leakage_diagnostic() -> dict:
    return label_leakage_report().__dict__

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def cherry_pick_diagnostic() -> dict:
    return cherry_pick_report().__dict__

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def probe_overfit_diagnostic() -> dict:
    return probe_overfit_report().__dict__

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def random_direction_diagnostic() -> dict:
    return random_direction_control_report().__dict__

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def audit_diagnostic() -> dict:
    return fake_result_audit_report().__dict__

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    audit = audit_diagnostic()
    return {
        "leakage": leakage_diagnostic(),
        "cherry_pick": cherry_pick_diagnostic(),
        "probe_overfit": probe_overfit_diagnostic(),
        "random_direction": random_direction_diagnostic(),
        "audit": audit,
        "contract_passed": audit["all_bogus_results_flagged"],
    }

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    tests.test_fake_result_audit_report_aggregates_all_failure_modes(fake_result_audit_report)
    tests.test_notebook_contract(run_smoke_test)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output and solution reveal</summary>

Each named original test should print its `All tests ... passed!` line. The matching positive and negative tests are both required; read the generated solution dropdown for the complete implementation. These are expected historical contracts, not newly executed test results.

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ Inspect and change controls
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Forecast before revealing the reference

Before opening the original four-row signature, predict the ordering of the leaked/shifted accuracy, selected/population effect, train/held-out accuracy and claimed/control direction effect.

- Forecast: ______
- Confidence: ______
- Causal reasoning: ______
- Prior exposure (seen / unseen / unknown): ______
- Observation after reveal: ______
- Changed-condition prediction: ______
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Signature Result

<img src="../../exercises/part6_fake_interpretability_results_practice/media/fake_interpretability_signature_result.svg" width="920">

This is the unchanged historical figure. Its four control comparisons are the result to interpret; the JSON report is supporting provenance. Its CUDA wording describes the retained old preflight only. No current hardware or fresh result is established here.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try It Yourself

Choose one supplied fixture change: swap the leakage feature indices, choose representative effect indices, replace held-out guesses by the supplied labels, or change the required direction margin. Predict the result before editing the following play cell. Compare against the unchanged baseline. Running the cell is a future learner action, not part of this migration.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN:
    play_baseline = label_leakage_report()
    play_changed = label_leakage_report(leaked_feature_index=1, shifted_feature_index=0)
    print("baseline gap:", play_baseline.accuracy_gap)
    print("swapped-column gap:", play_changed.accuracy_gap)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# ☆ Bonus
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Historical runtime entry points

The following original CUDA callbacks and dependency diagnostics remain available after explicit runtime initialization. They retain the original gates and failure behavior; none is executed in this migration. Reading a historical accepted report is distinct from running or accepting a new environment.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for the 0.6 fake-result diagnostic preflight.")
    device = t.device("cuda")
    t.manual_seed(606)
    t.cuda.reset_peak_memory_stats()

    features, labels = default_label_leakage_fixture(device=device)
    leakage = label_leakage_report(features, labels)
    cherry_pick = cherry_pick_report(default_cherry_pick_fixture(device=device))
    probe = probe_overfit_report(*default_probe_overfit_fixture(device=device))
    random_direction = random_direction_control_report(
        *default_random_direction_fixture(device=device)
    )
    audit = fake_result_audit_report(leakage, cherry_pick, probe, random_direction)

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "leaked_feature_accuracy": leakage.leaked_feature_accuracy,
        "shifted_no_leak_accuracy": leakage.shifted_no_leak_accuracy,
        "leakage_gap": leakage.accuracy_gap,
        "cherry_pick_inflation": cherry_pick.inflation_ratio,
        "selected_mean_effect": cherry_pick.selected_mean_effect,
        "population_median_effect": cherry_pick.population_median_effect,
        "probe_train_accuracy": probe.train_accuracy,
        "probe_heldout_accuracy": probe.heldout_accuracy,
        "probe_overfit_gap": probe.generalization_gap,
        "claimed_direction_effect": random_direction.claimed_effect,
        "random_p95_effect": random_direction.random_p95_effect,
        "random_direction_effect_gap": random_direction.effect_gap,
        "random_direction_control_rejects_claim": (
            random_direction.detects_random_direction_failure
        ),
        "all_bogus_results_flagged": audit.all_bogus_results_flagged,
        "input_driven_alternate_fixtures_passed": True,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": audit.all_bogus_results_flagged
        and peak_vram_gb <= max_vram_gb,
        "full_path": (
            "CUDA diagnostic preflight for fake-result failure modes: label leakage, "
            "cherry-pick inflation, probe overfit, and random-direction control rejection."
        ),
    }

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Limitations and scope gaps

The existing syllabus additionally names learned-probe selectivity/shuffled-label controls, a permutation or multiple-comparison null, and the dormant-direction subspace-patching illusion. These are not implemented by this source and remain open; the four existing detectors do not establish those results. There is no real-model mechanism claim. The fixed direction bank and anti-correlated leakage control have deliberately limited meanings. Zero norms and degenerate population means retain the original numerical policy rather than silently adding new defaults.

The human learner walkthrough remains pending. No report/schema/lock is regenerated, and no current CUDA or model readiness is claimed.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Complete original materials and provenance

- [Original ARENA course](https://github.com/callummcdougall/ARENA_3.0) remains the canonical base.
- [Original transformer implementation](https://github.com/callummcdougall/ARENA_3.0/tree/main/chapter1_transformer_interp/exercises/part1_transformer_from_scratch) provides the upstream first-principles teaching reference.
- [Complete original additive exercise notebook](https://github.com/shinobiultra/ARENA_3.0_Extended/blob/main/chapter0_fundamentals/exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb).
- [Complete original additive solutions notebook](https://github.com/shinobiultra/ARENA_3.0_Extended/blob/main/chapter0_fundamentals/exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb).

The exact pre-migration source bytes and every original cell object are retained in `before.tar.gz`, `before_manifest.json` and the section object inverse. The source preservation adapter is required to append the original historical books verbatim during adoption; this includes their original stubs, outputs, metadata and sometimes incomplete solution prose. They are preserved evidence, not a second freshly executed lesson. The authoritative master also includes their readable source in the historical material appendix below.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Original supporting panel

<img src="../../exercises/part6_fake_interpretability_results_practice/media/fake_interpretability_validation_loop.svg" width="920">

Unchanged historical explanatory artwork, not a new computation.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Complete original instruction page source

<pre>&#35; [0.6] How to Know When an Interpretability Result Is Fake

&gt; **Notebooks: [exercises](../../exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb) | [solutions](../../exercises/part6_fake_interpretability_results/0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb)**

&gt; **Local-first extension.** This section is a skepticism lab for the rest of
&gt; the course. You will implement deterministic checks for label leakage,
&gt; cherry-picked examples, probe overfitting, and random-direction steering
&gt; failures.

```python
GT_TIER = &quot;GT-0&quot;
EXERCISE_ID = &quot;0_6_how_to_know_when_an_interpretability_result_is_fake&quot;
DIFFICULTY = 2
IMPORTANCE = 5
EXPECTED_RUNTIME = &quot;20-30 minutes for exercises; seconds for the CUDA diagnostic preflight&quot;
REQUIRES_GPU = True  &#35; exercises are CPU-friendly; final acceptance uses CUDA tensors
```

Please send any problems / bugs on the `&#35;errata` channel in the [Slack group](https://info-arena.github.io/ARENA_img/slack.html), and ask any questions on the dedicated channels for this chapter of material.

Links to later chapters: [(1) Transformer Interpretability](https://arena-chapter1-transformer-interp.streamlit.app/), [(2) RL](https://arena-chapter2-rl.streamlit.app/), [(3) LLM Evaluations](https://arena-chapter3-llm-evals.streamlit.app/), [(4) Alignment Science](https://arena-chapter4-alignment-science.streamlit.app/).

&lt;img src=&quot;../../exercises/part6_fake_interpretability_results/expected_outputs/fake_interpretability_validation_loop.svg&quot; width=&quot;820&quot;&gt;

&#35;&#35; Core Question

How do you tell the difference between an interpretability result and a result
that merely looks like interpretability?

The fastest way to make interpretability worse is to reward plausible stories.
A clean probe, a memorable feature example, a nice UMAP, or a steering direction
is not evidence unless it survives controls. Every diagnostic in this notebook
is GT-0: the bogus result is synthetic and known in advance.

&lt;details&gt;
&lt;summary&gt;Help - why this belongs before the model chapters&lt;/summary&gt;

Mechanistic interpretability is unusually vulnerable to beautiful false
positives. A feature example, probe result, steering direction, or heatmap can
look meaningful even when it is caused by leakage, selection bias, train-set
memorization, or a random baseline that was never measured. This section teaches
the habit used throughout the extension: build the failure mode on purpose, then
require the detector to catch it before trusting any real-model claim.

&lt;/details&gt;

&#35;&#35; Learning Objectives

- Detect direct label leakage with a shifted no-leak control.
- Detect cherry-picked examples by comparing selected effects to the population.
- Detect probe overfitting by comparing train and held-out accuracy.
- Detect claimed directions that fail random-direction controls.
- Aggregate small diagnostics into a compact fake-result audit report.

&#35; Setup Code

```python
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = &quot;chapter0_fundamentals&quot;
section = &quot;part6_fake_interpretability_results&quot;
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / &quot;exercises&quot;
section_dir = exercises_dir / section

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part6_fake_interpretability_results.tests as tests
import part6_fake_interpretability_results.utils as utils
```

```python
@dataclass(frozen=True)
class LabelLeakageReport:
    leaked_feature_index: int
    leaked_feature_accuracy: float
    shifted_no_leak_accuracy: float
    accuracy_gap: float
    detects_leakage: bool


@dataclass(frozen=True)
class CherryPickReport:
    selected_mean_effect: float
    population_mean_effect: float
    population_median_effect: float
    inflation_ratio: float
    detects_cherry_picking: bool


@dataclass(frozen=True)
class ProbeOverfitReport:
    train_accuracy: float
    heldout_accuracy: float
    generalization_gap: float
    detects_overfit: bool


@dataclass(frozen=True)
class FakeRandomDirectionControlReport:
    claimed_effect: float
    random_p95_effect: float
    effect_gap: float
    passes_random_control: bool
    detects_random_direction_failure: bool


@dataclass(frozen=True)
class FakeResultAuditReport:
    leakage_detected: bool
    cherry_pick_detected: bool
    probe_overfit_detected: bool
    random_direction_failure_detected: bool
    all_bogus_results_flagged: bool
```

&#35; Label Leakage

&#35;&#35;&#35; Exercise - implement `binary_accuracy`

&gt; Difficulty: easy
&gt; Importance: high
&gt;
&gt; You should spend 5 minutes on this exercise.

Use a threshold of zero for signed probe scores.

```python
def binary_accuracy(scores: t.Tensor, labels: t.Tensor) -&gt; float:
    raise NotImplementedError()


tests.test_binary_accuracy_thresholds_signed_scores(binary_accuracy)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_binary_accuracy_thresholds_signed_scores` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Treating labels as `{-1, 1}` when the fixture uses `{0, 1}`.
- Using `&gt;= 0`, which changes how exact-zero scores are handled.
- Returning a tensor instead of a Python float.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def binary_accuracy(scores: t.Tensor, labels: t.Tensor) -&gt; float:
    labels = labels.long().flatten()
    predictions = scores.flatten().gt(0).long()
    return float(predictions.eq(labels).double().mean().item())
```

&lt;/details&gt;

&#35;&#35;&#35; Exercise - detect direct label leakage

&gt; Difficulty: easy
&gt; Importance: high
&gt;
&gt; You should spend 5 minutes on this exercise.

A leaked feature can make a probe look perfect while teaching you nothing about
the model mechanism. Implement the detector so it works on supplied feature
matrices, not just this toy fixture.

```python
def label_leakage_report(
    features: t.Tensor | None = None,
    labels: t.Tensor | None = None,
    *,
    leaked_feature_index: int = 0,
    shifted_feature_index: int = 1,
    min_gap: float = 0.5,
) -&gt; LabelLeakageReport:
    raise NotImplementedError()


tests.test_label_leakage_report_flags_direct_label_feature(label_leakage_report)
tests.test_label_leakage_report_uses_supplied_feature_indices(label_leakage_report)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_label_leakage_report_flags_direct_label_feature` passed!
All tests in `test_label_leakage_report_uses_supplied_feature_indices` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Checking only the leaked feature and forgetting the shifted no-leak control.
- Treating perfect accuracy as evidence instead of a warning.
- Not reporting the accuracy gap.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def default_label_leakage_fixture() -&gt; tuple[t.Tensor, t.Tensor]:
    labels = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.long)
    signed = labels.float() * 2 - 1
    features = t.stack([signed, -signed], dim=1)
    return features, labels


def label_leakage_report(
    features: t.Tensor | None = None,
    labels: t.Tensor | None = None,
    *,
    leaked_feature_index: int = 0,
    shifted_feature_index: int = 1,
    min_gap: float = 0.5,
) -&gt; LabelLeakageReport:
    if features is None or labels is None:
        features, labels = default_label_leakage_fixture()
    leaked_feature = features[:, leaked_feature_index]
    shifted_spurious_feature = features[:, shifted_feature_index]
    leaked_accuracy = binary_accuracy(leaked_feature, labels)
    shifted_accuracy = binary_accuracy(shifted_spurious_feature, labels)
    gap = leaked_accuracy - shifted_accuracy
    return LabelLeakageReport(
        leaked_feature_index=leaked_feature_index,
        leaked_feature_accuracy=leaked_accuracy,
        shifted_no_leak_accuracy=shifted_accuracy,
        accuracy_gap=gap,
        detects_leakage=leaked_accuracy &gt;= 0.95 and gap &gt;= min_gap,
    )
```

&lt;/details&gt;

&#35; Cherry-Picked Evidence

&#35;&#35;&#35; Exercise - compare selected examples to the population

&gt; Difficulty: easy
&gt; Importance: high
&gt;
&gt; You should spend 5 minutes on this exercise.

Selected examples should not stand in for the population. The function should
also accept alternate selections so the test suite can verify that
representative samples are not falsely rejected.

```python
def cherry_pick_report(
    effects: t.Tensor | None = None,
    selected_indices: t.Tensor | list[int] | None = None,
    *,
    min_inflation: float = 3.0,
    median_multiplier: float = 5.0,
) -&gt; CherryPickReport:
    raise NotImplementedError()


tests.test_cherry_pick_report_compares_selected_to_population(cherry_pick_report)
tests.test_cherry_pick_report_rejects_representative_selection(cherry_pick_report)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_cherry_pick_report_compares_selected_to_population` passed!
All tests in `test_cherry_pick_report_rejects_representative_selection` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Comparing selected examples only to zero instead of to the full population.
- Using the mean alone and missing the median sanity check.
- Letting three dramatic examples override the population statistic.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def cherry_pick_report(
    effects: t.Tensor | None = None,
    selected_indices: t.Tensor | list[int] | None = None,
    *,
    min_inflation: float = 3.0,
    median_multiplier: float = 5.0,
) -&gt; CherryPickReport:
    if effects is None:
        effects = t.tensor([
            0.02, 0.03, 0.04, 0.05, 0.06,
            0.07, 0.08, 0.09, 0.10, 0.11,
            0.12, 0.13, 1.40, 1.55, 1.70,
        ])
    selected = effects[-3:] if selected_indices is None else effects[selected_indices]
    selected_mean = float(selected.mean().item())
    population_mean = float(effects.mean().item())
    population_median = float(effects.median().item())
    inflation = selected_mean / population_mean
    return CherryPickReport(
        selected_mean_effect=selected_mean,
        population_mean_effect=population_mean,
        population_median_effect=population_median,
        inflation_ratio=inflation,
        detects_cherry_picking=(
            inflation &gt;= min_inflation
            and selected_mean &gt;= median_multiplier * population_median
        ),
    )
```

&lt;/details&gt;

&#35; Probe Overfitting

&#35;&#35;&#35; Exercise - compare train and held-out probe behavior

&gt; Difficulty: easy
&gt; Importance: high
&gt;
&gt; You should spend 5 minutes on this exercise.

A memorizing probe is not evidence of a robust representation. The detector
must accept explicit train and held-out predictions so a generalizing probe is
not falsely rejected.

```python
def probe_overfit_report(
    train_predictions: t.Tensor | None = None,
    train_labels: t.Tensor | None = None,
    heldout_predictions: t.Tensor | None = None,
    heldout_labels: t.Tensor | None = None,
    *,
    min_train_accuracy: float = 0.95,
    max_heldout_accuracy: float = 0.6,
    min_gap: float = 0.35,
) -&gt; ProbeOverfitReport:
    raise NotImplementedError()


tests.test_probe_overfit_report_requires_heldout_gap(probe_overfit_report)
tests.test_probe_overfit_report_rejects_generalizing_probe(probe_overfit_report)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_probe_overfit_report_requires_heldout_gap` passed!
All tests in `test_probe_overfit_report_rejects_generalizing_probe` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Interpreting train accuracy without a held-out set.
- Reporting held-out accuracy but not the generalization gap.
- Treating a weak held-out result as evidence because the training result is clean.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def probe_overfit_report(
    train_predictions: t.Tensor | None = None,
    train_labels: t.Tensor | None = None,
    heldout_predictions: t.Tensor | None = None,
    heldout_labels: t.Tensor | None = None,
    *,
    min_train_accuracy: float = 0.95,
    max_heldout_accuracy: float = 0.6,
    min_gap: float = 0.35,
) -&gt; ProbeOverfitReport:
    if train_predictions is None:
        train_labels = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.long)
        heldout_labels = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.long)
        train_predictions = train_labels.clone()
        heldout_predictions = t.zeros_like(heldout_labels)
    train_accuracy = float(train_predictions.eq(train_labels).double().mean().item())
    heldout_accuracy = float(heldout_predictions.eq(heldout_labels).double().mean().item())
    gap = train_accuracy - heldout_accuracy
    return ProbeOverfitReport(
        train_accuracy=train_accuracy,
        heldout_accuracy=heldout_accuracy,
        generalization_gap=gap,
        detects_overfit=(
            train_accuracy &gt;= min_train_accuracy
            and heldout_accuracy &lt;= max_heldout_accuracy
            and gap &gt;= min_gap
        ),
    )
```

&lt;/details&gt;

&#35; Random-Direction Controls

&#35;&#35;&#35; Exercise - reject a weak steering direction

&gt; Difficulty: medium
&gt; Importance: high
&gt;
&gt; You should spend 10 minutes on this exercise.

A steering direction must beat a random-direction control distribution. The
positive-control test passes a clean direction, so this detector should not
reject everything by default.

```python
def random_direction_control_report(
    behavior_direction: t.Tensor | None = None,
    claimed_direction: t.Tensor | None = None,
    random_directions: t.Tensor | None = None,
    *,
    required_margin: float = 0.25,
) -&gt; FakeRandomDirectionControlReport:
    raise NotImplementedError()


tests.test_random_direction_control_report_rejects_weak_claim(
    random_direction_control_report,
)
tests.test_random_direction_control_report_accepts_strong_claim(
    random_direction_control_report,
)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_random_direction_control_report_rejects_weak_claim` passed!
All tests in `test_random_direction_control_report_accepts_strong_claim` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Forgetting to normalize directions before comparing effects.
- Comparing only to the random mean rather than a high quantile.
- Calling the direction meaningful even when it misses the required margin.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def random_direction_control_report(
    behavior_direction: t.Tensor | None = None,
    claimed_direction: t.Tensor | None = None,
    random_directions: t.Tensor | None = None,
    *,
    required_margin: float = 0.25,
) -&gt; FakeRandomDirectionControlReport:
    if behavior_direction is None:
        behavior_direction = t.tensor([1.0, 0.0, 0.0], dtype=t.float64)
        claimed_direction = t.tensor([0.05, 0.9987, 0.0], dtype=t.float64)
        random_directions = t.tensor([
            [0.12, 0.99, 0.00],
            [-0.08, 0.99, 0.05],
            [0.04, 0.20, 0.98],
            [0.10, -0.95, 0.10],
            [-0.11, 0.00, 0.99],
        ], dtype=t.float64)
    behavior_direction = behavior_direction / behavior_direction.norm()
    claimed_direction = claimed_direction / claimed_direction.norm()
    random_directions = random_directions / random_directions.norm(dim=1, keepdim=True)
    claimed_effect = float(abs(claimed_direction @ behavior_direction).item())
    random_effects = (random_directions @ behavior_direction).abs()
    random_p95 = float(t.quantile(random_effects, 0.95).item())
    gap = claimed_effect - random_p95
    passes = gap &gt;= required_margin
    return FakeRandomDirectionControlReport(
        claimed_effect=claimed_effect,
        random_p95_effect=random_p95,
        effect_gap=gap,
        passes_random_control=passes,
        detects_random_direction_failure=not passes,
    )
```

&lt;/details&gt;

&#35; Aggregate Audit

&#35;&#35;&#35; Exercise - aggregate all fake-result detectors

&gt; Difficulty: easy
&gt; Importance: high
&gt;
&gt; You should spend 5 minutes on this exercise.

A section-level audit should pass only when every known bogus result is flagged.

```python
def fake_result_audit_report(
    leakage: LabelLeakageReport | None = None,
    cherry_pick: CherryPickReport | None = None,
    overfit: ProbeOverfitReport | None = None,
    random_direction: FakeRandomDirectionControlReport | None = None,
) -&gt; FakeResultAuditReport:
    raise NotImplementedError()


tests.test_fake_result_audit_report_aggregates_all_failure_modes(
    fake_result_audit_report,
)
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_fake_result_audit_report_aggregates_all_failure_modes` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Letting one strong detector hide a missing detector.
- Returning only a boolean and losing which failure mode was flagged.
- Treating this GT-0 audit as a real-model interpretability claim.

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def fake_result_audit_report(
    leakage: LabelLeakageReport | None = None,
    cherry_pick: CherryPickReport | None = None,
    overfit: ProbeOverfitReport | None = None,
    random_direction: FakeRandomDirectionControlReport | None = None,
) -&gt; FakeResultAuditReport:
    leakage = leakage or label_leakage_report()
    cherry_pick = cherry_pick or cherry_pick_report()
    overfit = overfit or probe_overfit_report()
    random_direction = random_direction or random_direction_control_report()
    flags = [
        leakage.detects_leakage,
        cherry_pick.detects_cherry_picking,
        overfit.detects_overfit,
        random_direction.detects_random_direction_failure,
    ]
    return FakeResultAuditReport(
        leakage_detected=flags[0],
        cherry_pick_detected=flags[1],
        probe_overfit_detected=flags[2],
        random_direction_failure_detected=flags[3],
        all_bogus_results_flagged=all(flags),
    )
```

&lt;/details&gt;

&#35; Notebook Contract

```python
def leakage_diagnostic() -&gt; dict:
    return label_leakage_report().__dict__


def cherry_pick_diagnostic() -&gt; dict:
    return cherry_pick_report().__dict__


def probe_overfit_diagnostic() -&gt; dict:
    return probe_overfit_report().__dict__


def random_direction_diagnostic() -&gt; dict:
    return random_direction_control_report().__dict__


def run_smoke_test(cpu: bool = True) -&gt; dict:
    _ = cpu
    audit = fake_result_audit_report().__dict__
    return {
        &quot;leakage&quot;: leakage_diagnostic(),
        &quot;cherry_pick&quot;: cherry_pick_diagnostic(),
        &quot;probe_overfit&quot;: probe_overfit_diagnostic(),
        &quot;random_direction&quot;: random_direction_diagnostic(),
        &quot;audit&quot;: audit,
        &quot;contract_passed&quot;: audit[&quot;all_bogus_results_flagged&quot;],
    }


tests.test_notebook_contract(run_smoke_test)
tests.test_committed_gpu_report_records_fake_result_signature()
```

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_notebook_contract` passed!
All tests in `test_committed_gpu_report_records_fake_result_signature` passed!
```

&lt;/details&gt;

&lt;details&gt;
&lt;summary&gt;Common bugs&lt;/summary&gt;

- Returning dataclass objects instead of JSON-serializable dictionaries.
- Forgetting one failure mode in the aggregate audit.
- Making the notebook contract depend on CUDA.

&lt;/details&gt;

&#35; CUDA Verification Path

The committed report repeats the same fake-result diagnostics on CUDA tensors.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
preflight_passed: true
leaked_feature_accuracy: 1.0
shifted_no_leak_accuracy: 0.0
leakage_gap: &gt;= 0.5
cherry_pick_inflation: &gt;= 3.0
probe_train_accuracy: 1.0
probe_heldout_accuracy: 0.5
probe_overfit_gap: &gt;= 0.35
random_direction_control_rejects_claim: true
all_bogus_results_flagged: true
peak_vram_gb: &lt; 1.0
```

&lt;/details&gt;

This is a methodology section. It proves that the detectors catch known bogus
results; it does not claim that a real model mechanism has been validated.

&#35; Signature Result

&lt;img src=&quot;../../exercises/part6_fake_interpretability_results/expected_outputs/fake_interpretability_signature_result.svg&quot; width=&quot;860&quot;&gt;

The committed CUDA report deliberately injects four fake-result patterns and
requires each matching control to reject the claim:

| Failure mode | Bogus-looking metric | Control metric | Decision |
|---|---:|---:|---|
| Direct label leakage | leaked accuracy `1.0` | shifted no-leak accuracy `0.0` | reject |
| Cherry-picked examples | selected mean `1.55` | inflation `4.19x` over population mean | reject |
| Probe overfitting | train accuracy `1.0` | held-out accuracy `0.5` | reject |
| Weak steering direction | claimed effect `0.050` | random p95 effect `0.118` | reject |

The signature result is `all_bogus_results_flagged = true`, with measured peak
VRAM below `1 GB` on the local CUDA 13.2 PyTorch runtime.

&lt;details&gt;
&lt;summary&gt;Interpreting this result&lt;/summary&gt;

This is a negative-result success: every impressive-looking claim was designed
to be bogus, and every bogus claim was caught. If a detector fails here, later
notebooks should not use it as evidence. If a real-model section passes this
style of control, that still only supports the local claim made in that section.

&lt;/details&gt;

&#35; Limitations

- These are GT-0 generated tensors, not real model activations.
- The diagnostics cover four common failure modes, not every way an
  interpretability result can be misleading.
- A detector passing this section is a prerequisite for trust, not sufficient
  evidence for a real-model mechanism.
- The CUDA path proves the report can run on the local GPU; it is not a
  benchmark and does not need to keep the GPU busy for long.

&#35; Bonus - Anomaly Hunting

- Construct a &quot;half-leaked&quot; feature with 80-90% accuracy and tune the leakage
  threshold. At what point should the section report warning rather than fail?
- Add a heavy-tailed effect distribution where the mean is misleading but the
  median and quantiles tell the truth.
- Replace random directions with matched-norm random directions from a real
  residual stream and check whether the p95 threshold is stable.
- Build a real-model case study where one of these detectors changes your
  interpretation of a beautiful plot.
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Historical material appendix

These complete original source cells are preserved for inspection. The front curriculum teaches the methods inline; do not run this appendix as an alternative fresh validation path.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Original exercises notebook: complete cell sources
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 0: markdown.**

<pre>&#35; [0.6] How to Know When an Interpretability Result Is Fake - Exercises

&#35;&#35; Core Question

How do you tell whether a beautiful interpretability result is evidence, or only a story that survived no controls?

&#35;&#35; Learning Objectives

- Detect direct label leakage with a shifted no-leak control.
- Detect cherry-picked examples with population statistics.
- Detect probe memorization with held-out controls.
- Reject weak steering directions with random-direction baselines.
- Aggregate the diagnostics into a CUDA-backed signature report.

&#35;&#35;&#35; Exercise - fake-result diagnostics ladder

&gt; Difficulty: medium  
&gt; Importance: high

This is a GT-0 skepticism lab. Every impressive-looking result below is deliberately fake,
so the right outcome is a negative result: each detector should reject the bogus claim.
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 1: markdown.**

<pre>&lt;details&gt;
&lt;summary&gt;Help - what to watch for&lt;/summary&gt;

A detector that only works on one hardcoded fixture is itself suspicious. Each report
function in this notebook has a default toy oracle, but the tests also pass alternate
tensors so your implementation has to compute the metric from inputs.

&lt;/details&gt;
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 2: code.**

<pre>import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = &quot;chapter0_fundamentals&quot;
section = &quot;part6_fake_interpretability_results&quot;
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / &quot;exercises&quot;
section_dir = exercises_dir / section

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part6_fake_interpretability_results.tests as tests

GT_TIER = &quot;GT-0&quot;
EXERCISE_ID = &quot;0_6_how_to_know_when_an_interpretability_result_is_fake&quot;
EXPECTED_RUNTIME = &quot;20-30 minutes for exercises; seconds for the CUDA diagnostic preflight&quot;
REQUIRES_GPU = True
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 3: code.**

<pre>@dataclass(frozen=True)
class LabelLeakageReport:
    leaked_feature_index: int
    leaked_feature_accuracy: float
    shifted_no_leak_accuracy: float
    accuracy_gap: float
    detects_leakage: bool


@dataclass(frozen=True)
class CherryPickReport:
    selected_mean_effect: float
    population_mean_effect: float
    population_median_effect: float
    inflation_ratio: float
    detects_cherry_picking: bool


@dataclass(frozen=True)
class ProbeOverfitReport:
    train_accuracy: float
    heldout_accuracy: float
    generalization_gap: float
    detects_overfit: bool


@dataclass(frozen=True)
class FakeRandomDirectionControlReport:
    claimed_effect: float
    random_p95_effect: float
    effect_gap: float
    passes_random_control: bool
    detects_random_direction_failure: bool


@dataclass(frozen=True)
class FakeResultAuditReport:
    leakage_detected: bool
    cherry_pick_detected: bool
    probe_overfit_detected: bool
    random_direction_failure_detected: bool
    all_bogus_results_flagged: bool
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 4: markdown.**

<pre>&#35;&#35; Label Leakage

Implement threshold-at-zero binary accuracy, then detect a feature column that directly encodes the label. The shifted/no-leak control must fail; perfect accuracy alone is not evidence.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_binary_accuracy_thresholds_signed_scores` passed!
All tests in `test_label_leakage_report_flags_direct_label_feature` passed!
All tests in `test_label_leakage_report_uses_supplied_feature_indices` passed!
```

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 5: code.**

<pre>def binary_accuracy(scores: t.Tensor, labels: t.Tensor) -&gt; float:
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


def label_leakage_report(
    features: t.Tensor | None = None,
    labels: t.Tensor | None = None,
    *,
    leaked_feature_index: int = 0,
    shifted_feature_index: int = 1,
    min_gap: float = 0.5,
) -&gt; LabelLeakageReport:
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


tests.test_binary_accuracy_thresholds_signed_scores(binary_accuracy)
tests.test_label_leakage_report_flags_direct_label_feature(label_leakage_report)
tests.test_label_leakage_report_uses_supplied_feature_indices(label_leakage_report)
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 6: markdown.**

<pre>&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
def binary_accuracy(scores, labels):
    labels = labels.long().flatten()
    predictions = scores.flatten().gt(0).long()
    return float(predictions.eq(labels).double().mean().item())


def label_leakage_report(features=None, labels=None, *, leaked_feature_index=0, shifted_feature_index=1, min_gap=0.5):
    if features is None or labels is None:
        labels = t.tensor([0, 1, 0, 1, 0, 1], dtype=t.long)
        signed = labels.float() * 2 - 1
        features = t.stack([signed, -signed], dim=1)
    leaked_accuracy = binary_accuracy(features[:, leaked_feature_index], labels)
    shifted_accuracy = binary_accuracy(features[:, shifted_feature_index], labels)
    gap = leaked_accuracy - shifted_accuracy
    return LabelLeakageReport(leaked_feature_index, leaked_accuracy, shifted_accuracy, gap, leaked_accuracy &gt;= 0.95 and gap &gt;= min_gap)
```

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 7: markdown.**

<pre>&#35;&#35; Cherry-Picked Evidence

Selected examples must be compared to the full population. A representative selection should not be rejected.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_cherry_pick_report_compares_selected_to_population` passed!
All tests in `test_cherry_pick_report_rejects_representative_selection` passed!
```

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 8: code.**

<pre>def cherry_pick_report(
    effects: t.Tensor | None = None,
    selected_indices: t.Tensor | list[int] | None = None,
    *,
    min_inflation: float = 3.0,
    median_multiplier: float = 5.0,
) -&gt; CherryPickReport:
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


tests.test_cherry_pick_report_compares_selected_to_population(cherry_pick_report)
tests.test_cherry_pick_report_rejects_representative_selection(cherry_pick_report)
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 9: markdown.**

<pre>&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

```python
if effects is None:
    effects = t.tensor([0.02, 0.03, ..., 1.40, 1.55, 1.70])
selected = effects[-3:] if selected_indices is None else effects[selected_indices]
selected_mean = float(selected.mean().item())
population_mean = float(effects.mean().item())
population_median = float(effects.median().item())
inflation = selected_mean / population_mean
```

Return a `CherryPickReport` and require both high inflation and a large median-relative selected mean.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 10: markdown.**

<pre>&#35;&#35; Probe Overfitting

A memorizing probe is not evidence unless it survives held-out examples. The positive-control test passes a probe that generalizes.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_probe_overfit_report_requires_heldout_gap` passed!
All tests in `test_probe_overfit_report_rejects_generalizing_probe` passed!
```

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 11: code.**

<pre>def probe_overfit_report(
    train_predictions: t.Tensor | None = None,
    train_labels: t.Tensor | None = None,
    heldout_predictions: t.Tensor | None = None,
    heldout_labels: t.Tensor | None = None,
    *,
    min_train_accuracy: float = 0.95,
    max_heldout_accuracy: float = 0.6,
    min_gap: float = 0.35,
) -&gt; ProbeOverfitReport:
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


tests.test_probe_overfit_report_requires_heldout_gap(probe_overfit_report)
tests.test_probe_overfit_report_rejects_generalizing_probe(probe_overfit_report)
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 12: markdown.**

<pre>&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

Compute train and held-out accuracy separately, then flag overfit only when train accuracy is high, held-out accuracy is weak, and the gap is large.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 13: markdown.**

<pre>&#35;&#35; Random-Direction Controls

A steering direction should beat a random-direction control distribution by a real margin. The detector should reject weak claims and accept a strong positive control.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_random_direction_control_report_rejects_weak_claim` passed!
All tests in `test_random_direction_control_report_accepts_strong_claim` passed!
```

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 14: code.**

<pre>def random_direction_control_report(
    behavior_direction: t.Tensor | None = None,
    claimed_direction: t.Tensor | None = None,
    random_directions: t.Tensor | None = None,
    *,
    required_margin: float = 0.25,
) -&gt; FakeRandomDirectionControlReport:
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


tests.test_random_direction_control_report_rejects_weak_claim(random_direction_control_report)
tests.test_random_direction_control_report_accepts_strong_claim(random_direction_control_report)
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 15: markdown.**

<pre>&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

Normalize the behavior, claimed, and random directions. Compare the claimed absolute dot product to the random-control p95. The claim passes only when `claimed_effect - random_p95 &gt;= required_margin`.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 16: markdown.**

<pre>&#35;&#35; Aggregate Audit

The section-level audit should pass only when every known bogus result is flagged.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_fake_result_audit_report_aggregates_all_failure_modes` passed!
All tests in `test_notebook_contract` passed!
```

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 17: code.**

<pre>def fake_result_audit_report(
    leakage: LabelLeakageReport | None = None,
    cherry_pick: CherryPickReport | None = None,
    overfit: ProbeOverfitReport | None = None,
    random_direction: FakeRandomDirectionControlReport | None = None,
) -&gt; FakeResultAuditReport:
    &#35; EXERCISE
    &#35; YOUR CODE HERE
    raise NotImplementedError()
    &#35; END EXERCISE


def leakage_diagnostic() -&gt; dict:
    return label_leakage_report().__dict__


def cherry_pick_diagnostic() -&gt; dict:
    return cherry_pick_report().__dict__


def probe_overfit_diagnostic() -&gt; dict:
    return probe_overfit_report().__dict__


def random_direction_diagnostic() -&gt; dict:
    return random_direction_control_report().__dict__


def run_smoke_test(cpu: bool = True) -&gt; dict:
    _ = cpu
    audit = fake_result_audit_report().__dict__
    return {
        &quot;leakage&quot;: leakage_diagnostic(),
        &quot;cherry_pick&quot;: cherry_pick_diagnostic(),
        &quot;probe_overfit&quot;: probe_overfit_diagnostic(),
        &quot;random_direction&quot;: random_direction_diagnostic(),
        &quot;audit&quot;: audit,
        &quot;contract_passed&quot;: audit[&quot;all_bogus_results_flagged&quot;],
    }


tests.test_fake_result_audit_report_aggregates_all_failure_modes(fake_result_audit_report)
tests.test_notebook_contract(run_smoke_test)
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 18: markdown.**

<pre>&lt;details&gt;
&lt;summary&gt;Solution&lt;/summary&gt;

Call each detector, store the four booleans, and return `all_bogus_results_flagged = all(flags)`. The point of the aggregate report is to make missing controls visible.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 19: markdown.**

<pre>&#35;&#35; Signature Result

![Fake interpretability signature result](expected_outputs/fake_interpretability_signature_result.svg)

The committed CUDA report should show that all four injected bogus claims were rejected.

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

```text
All tests in `test_committed_gpu_report_records_fake_result_signature` passed!
preflight_passed: true
all_bogus_results_flagged: true
peak_vram_gb: &lt; 1.0
```

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 20: code.**

<pre>def _load_committed_gpu_report() -&gt; dict:
    import json

    report = json.loads((section_dir / &quot;verification_report.json&quot;).read_text())
    assert report[&quot;accepted&quot;] and report[&quot;tests_passed&quot;]
    gpu = report[&quot;metrics&quot;][&quot;gpu_test&quot;]
    assert gpu[&quot;cuda_available&quot;]
    assert gpu[&quot;within_vram_budget&quot;]
    return gpu


def run_gpu_test(max_vram_gb: float = 24.0) -&gt; dict:
    gpu = _load_committed_gpu_report()
    assert gpu[&quot;peak_vram_gb&quot;] &lt;= max_vram_gb
    return gpu


def run_full_experiment(max_vram_gb: float = 24.0) -&gt; dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)


tests.test_committed_gpu_report_records_fake_result_signature()

gpu = _load_committed_gpu_report()
{key: gpu[key] for key in [
    &quot;device&quot;,
    &quot;preflight_passed&quot;,
    &quot;all_bogus_results_flagged&quot;,
    &quot;peak_vram_gb&quot;,
] if key in gpu}
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original exercises cell 21: markdown.**

<pre>&#35;&#35; Limitations

This notebook proves that the detectors catch known synthetic failures. It does not prove a real model mechanism, and it does not cover every possible source of misleading interpretability evidence.

&#35;&#35; Bonus - Anomaly Hunting

Try weakening each failure mode: 80% leakage instead of 100%, a less extreme selected subset, a held-out probe at 70%, or a claimed direction that barely beats random p95. Decide whether the result should fail, warn, or pass.</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Original solutions notebook: complete cell sources
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 0: markdown.**

<pre>&#35; [0.6] How to Know When an Interpretability Result Is Fake - Solutions

&#35;&#35; Core Question

Can a compact GT-0 audit reject leakage, cherry-picking, probe overfit, and weak steering claims before later notebooks trust real-model evidence?

&#35;&#35; Learning Objectives

- Validate each fake-result detector against default and alternate fixtures.
- Interpret the negative-result signature table.
- Keep CUDA evidence scoped to the methodology claim.

&#35;&#35;&#35; Exercise - solution verification

&gt; Difficulty: medium  
&gt; Importance: high

Reference validation notebook for the GT-0 fake-result diagnostics section.
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 1: markdown.**

<pre>&lt;details&gt;
&lt;summary&gt;Help - interpreting this notebook&lt;/summary&gt;

The successful outcome is mostly negative: the synthetic claims are designed to be bogus, and the diagnostics should reject them. Do not reinterpret these numbers as a real-model mechanism.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 2: code.**

<pre>import json
import sys
from pathlib import Path

chapter = &quot;chapter0_fundamentals&quot;
section = &quot;part6_fake_interpretability_results&quot;
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / &quot;exercises&quot;
section_dir = exercises_dir / section

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part6_fake_interpretability_results.tests as tests
from chapter0_fundamentals.exercises.part6_fake_interpretability_results import solutions
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 3: markdown.**

<pre>&#35;&#35; Sub-function Tests

&lt;details&gt;
&lt;summary&gt;Expected output&lt;/summary&gt;

Every test prints its `All tests ... passed!` line, including the alternate-fixture controls.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 4: code.**

<pre>tests.test_binary_accuracy_thresholds_signed_scores(solutions.binary_accuracy)
tests.test_label_leakage_report_flags_direct_label_feature(solutions.label_leakage_report)
tests.test_label_leakage_report_uses_supplied_feature_indices(solutions.label_leakage_report)
tests.test_cherry_pick_report_compares_selected_to_population(solutions.cherry_pick_report)
tests.test_cherry_pick_report_rejects_representative_selection(solutions.cherry_pick_report)
tests.test_probe_overfit_report_requires_heldout_gap(solutions.probe_overfit_report)
tests.test_probe_overfit_report_rejects_generalizing_probe(solutions.probe_overfit_report)
tests.test_random_direction_control_report_rejects_weak_claim(solutions.random_direction_control_report)
tests.test_random_direction_control_report_accepts_strong_claim(solutions.random_direction_control_report)
tests.test_fake_result_audit_report_aggregates_all_failure_modes(solutions.fake_result_audit_report)
tests.test_notebook_contract(solutions.run_smoke_test)
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 5: markdown.**

<pre>&#35;&#35; Smoke-Test Contract</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 6: code.**

<pre>contract = solutions.run_smoke_test(cpu=True)
assert contract[&quot;contract_passed&quot;]
assert contract[&quot;leakage&quot;][&quot;leaked_feature_accuracy&quot;] == 1.0
assert contract[&quot;leakage&quot;][&quot;shifted_no_leak_accuracy&quot;] == 0.0
assert contract[&quot;cherry_pick&quot;][&quot;inflation_ratio&quot;] &gt;= 3.0
assert contract[&quot;probe_overfit&quot;][&quot;train_accuracy&quot;] == 1.0
assert contract[&quot;probe_overfit&quot;][&quot;heldout_accuracy&quot;] == 0.5
assert contract[&quot;random_direction&quot;][&quot;detects_random_direction_failure&quot;]
contract
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 7: markdown.**

<pre>&#35;&#35; Signature Result

![Fake interpretability signature result](expected_outputs/fake_interpretability_signature_result.svg)

&lt;details&gt;
&lt;summary&gt;Interpreting this result&lt;/summary&gt;

The report succeeds because all four injected bogus claims are rejected. This is a prerequisite for later claims, not evidence for any particular model circuit.

&lt;/details&gt;</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 8: code.**

<pre>tests.test_committed_gpu_report_records_fake_result_signature()
report = json.loads((section_dir / &quot;verification_report.json&quot;).read_text())
gpu = report[&quot;metrics&quot;][&quot;gpu_test&quot;]
assert report[&quot;accepted&quot;]
assert gpu[&quot;cuda_available&quot;]
assert gpu[&quot;preflight_passed&quot;]
assert gpu[&quot;leaked_feature_accuracy&quot;] == 1.0
assert gpu[&quot;shifted_no_leak_accuracy&quot;] == 0.0
assert gpu[&quot;leakage_gap&quot;] &gt;= 0.5
assert gpu[&quot;cherry_pick_inflation&quot;] &gt;= 3.0
assert gpu[&quot;selected_mean_effect&quot;] &gt;= 5 * gpu[&quot;population_median_effect&quot;]
assert gpu[&quot;probe_train_accuracy&quot;] == 1.0
assert gpu[&quot;probe_heldout_accuracy&quot;] == 0.5
assert gpu[&quot;probe_overfit_gap&quot;] &gt;= 0.35
assert gpu[&quot;random_direction_control_rejects_claim&quot;]
assert gpu[&quot;random_direction_effect_gap&quot;] &lt; 0.25
assert gpu[&quot;all_bogus_results_flagged&quot;]
assert gpu[&quot;within_vram_budget&quot;]
{key: gpu[key] for key in [
    &quot;device&quot;,
    &quot;leaked_feature_accuracy&quot;,
    &quot;shifted_no_leak_accuracy&quot;,
    &quot;cherry_pick_inflation&quot;,
    &quot;probe_overfit_gap&quot;,
    &quot;random_direction_effect_gap&quot;,
    &quot;peak_vram_gb&quot;,
]}
</pre>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
**Original solutions cell 9: markdown.**

<pre>&#35;&#35; Limitations

These are GT-0 generated tensors. The section validates the diagnostic habit, not a real-model mechanism.

&#35;&#35; Bonus - Anomaly Hunting

Try lowering each failure signal until the detector should issue a warning rather than a hard rejection.</pre>
'''


# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 2 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = "chapter0_fundamentals"
section = "part6_fake_interpretability_results"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part6_fake_interpretability_results.tests as tests

GT_TIER = "GT-0"
EXERCISE_ID = "0_6_how_to_know_when_an_interpretability_result_is_fake"
EXPECTED_RUNTIME = "20-30 minutes for exercises; seconds for the CUDA diagnostic preflight"
REQUIRES_GPU = True
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 3 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
@dataclass(frozen=True)
class LabelLeakageReport:
    leaked_feature_index: int
    leaked_feature_accuracy: float
    shifted_no_leak_accuracy: float
    accuracy_gap: float
    detects_leakage: bool


@dataclass(frozen=True)
class CherryPickReport:
    selected_mean_effect: float
    population_mean_effect: float
    population_median_effect: float
    inflation_ratio: float
    detects_cherry_picking: bool


@dataclass(frozen=True)
class ProbeOverfitReport:
    train_accuracy: float
    heldout_accuracy: float
    generalization_gap: float
    detects_overfit: bool


@dataclass(frozen=True)
class FakeRandomDirectionControlReport:
    claimed_effect: float
    random_p95_effect: float
    effect_gap: float
    passes_random_control: bool
    detects_random_direction_failure: bool


@dataclass(frozen=True)
class FakeResultAuditReport:
    leakage_detected: bool
    cherry_pick_detected: bool
    probe_overfit_detected: bool
    random_direction_failure_detected: bool
    all_bogus_results_flagged: bool
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 5 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def binary_accuracy(scores: t.Tensor, labels: t.Tensor) -> float:
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


def label_leakage_report(
    features: t.Tensor | None = None,
    labels: t.Tensor | None = None,
    *,
    leaked_feature_index: int = 0,
    shifted_feature_index: int = 1,
    min_gap: float = 0.5,
) -> LabelLeakageReport:
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


tests.test_binary_accuracy_thresholds_signed_scores(binary_accuracy)
tests.test_label_leakage_report_flags_direct_label_feature(label_leakage_report)
tests.test_label_leakage_report_uses_supplied_feature_indices(label_leakage_report)
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 8 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def cherry_pick_report(
    effects: t.Tensor | None = None,
    selected_indices: t.Tensor | list[int] | None = None,
    *,
    min_inflation: float = 3.0,
    median_multiplier: float = 5.0,
) -> CherryPickReport:
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


tests.test_cherry_pick_report_compares_selected_to_population(cherry_pick_report)
tests.test_cherry_pick_report_rejects_representative_selection(cherry_pick_report)
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 11 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def probe_overfit_report(
    train_predictions: t.Tensor | None = None,
    train_labels: t.Tensor | None = None,
    heldout_predictions: t.Tensor | None = None,
    heldout_labels: t.Tensor | None = None,
    *,
    min_train_accuracy: float = 0.95,
    max_heldout_accuracy: float = 0.6,
    min_gap: float = 0.35,
) -> ProbeOverfitReport:
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


tests.test_probe_overfit_report_requires_heldout_gap(probe_overfit_report)
tests.test_probe_overfit_report_rejects_generalizing_probe(probe_overfit_report)
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 14 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def random_direction_control_report(
    behavior_direction: t.Tensor | None = None,
    claimed_direction: t.Tensor | None = None,
    random_directions: t.Tensor | None = None,
    *,
    required_margin: float = 0.25,
) -> FakeRandomDirectionControlReport:
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


tests.test_random_direction_control_report_rejects_weak_claim(random_direction_control_report)
tests.test_random_direction_control_report_accepts_strong_claim(random_direction_control_report)
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 17 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def fake_result_audit_report(
    leakage: LabelLeakageReport | None = None,
    cherry_pick: CherryPickReport | None = None,
    overfit: ProbeOverfitReport | None = None,
    random_direction: FakeRandomDirectionControlReport | None = None,
) -> FakeResultAuditReport:
    # EXERCISE
    # YOUR CODE HERE
    raise NotImplementedError()
    # END EXERCISE


def leakage_diagnostic() -> dict:
    return label_leakage_report().__dict__


def cherry_pick_diagnostic() -> dict:
    return cherry_pick_report().__dict__


def probe_overfit_diagnostic() -> dict:
    return probe_overfit_report().__dict__


def random_direction_diagnostic() -> dict:
    return random_direction_control_report().__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    audit = fake_result_audit_report().__dict__
    return {
        "leakage": leakage_diagnostic(),
        "cherry_pick": cherry_pick_diagnostic(),
        "probe_overfit": probe_overfit_diagnostic(),
        "random_direction": random_direction_diagnostic(),
        "audit": audit,
        "contract_passed": audit["all_bogus_results_flagged"],
    }


tests.test_fake_result_audit_report_aggregates_all_failure_modes(fake_result_audit_report)
tests.test_notebook_contract(run_smoke_test)
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: [colab-ex]
# ! TAGS: []

r'''
**Historical code cell 20 (read-only original source).**

The active implementation appears above. The intact original book and its original helpers are linked in the introduction.

```python
def _load_committed_gpu_report() -> dict:
    import json

    report = json.loads((section_dir / "verification_report.json").read_text())
    assert report["accepted"] and report["tests_passed"]
    gpu = report["metrics"]["gpu_test"]
    assert gpu["cuda_available"]
    assert gpu["within_vram_budget"]
    return gpu


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    gpu = _load_committed_gpu_report()
    assert gpu["peak_vram_gb"] <= max_vram_gb
    return gpu


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)


tests.test_committed_gpu_report_records_fake_result_signature()

gpu = _load_committed_gpu_report()
{key: gpu[key] for key in [
    "device",
    "preflight_passed",
    "all_bogus_results_flagged",
    "peak_vram_gb",
] if key in gpu}
```
'''

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# import json
# import sys
# from pathlib import Path
# 
# chapter = "chapter0_fundamentals"
# section = "part6_fake_interpretability_results"
# root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
# exercises_dir = root_dir / chapter / "exercises"
# section_dir = exercises_dir / section
# 
# if str(root_dir) not in sys.path:
#     sys.path.append(str(root_dir))
# if str(exercises_dir) not in sys.path:
#     sys.path.append(str(exercises_dir))
# 
# import part6_fake_interpretability_results.tests as tests
# from chapter0_fundamentals.exercises.part6_fake_interpretability_results import solutions

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# tests.test_binary_accuracy_thresholds_signed_scores(solutions.binary_accuracy)
# tests.test_label_leakage_report_flags_direct_label_feature(solutions.label_leakage_report)
# tests.test_label_leakage_report_uses_supplied_feature_indices(solutions.label_leakage_report)
# tests.test_cherry_pick_report_compares_selected_to_population(solutions.cherry_pick_report)
# tests.test_cherry_pick_report_rejects_representative_selection(solutions.cherry_pick_report)
# tests.test_probe_overfit_report_requires_heldout_gap(solutions.probe_overfit_report)
# tests.test_probe_overfit_report_rejects_generalizing_probe(solutions.probe_overfit_report)
# tests.test_random_direction_control_report_rejects_weak_claim(solutions.random_direction_control_report)
# tests.test_random_direction_control_report_accepts_strong_claim(solutions.random_direction_control_report)
# tests.test_fake_result_audit_report_aggregates_all_failure_modes(solutions.fake_result_audit_report)
# tests.test_notebook_contract(solutions.run_smoke_test)

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# contract = solutions.run_smoke_test(cpu=True)
# assert contract["contract_passed"]
# assert contract["leakage"]["leaked_feature_accuracy"] == 1.0
# assert contract["leakage"]["shifted_no_leak_accuracy"] == 0.0
# assert contract["cherry_pick"]["inflation_ratio"] >= 3.0
# assert contract["probe_overfit"]["train_accuracy"] == 1.0
# assert contract["probe_overfit"]["heldout_accuracy"] == 0.5
# assert contract["random_direction"]["detects_random_direction_failure"]
# contract

# ! CELL TYPE: code
# ! FILTERS: [colab-soln]
# ! TAGS: [keep-main,master-comment]

# tests.test_committed_gpu_report_records_fake_result_signature()
# report = json.loads((section_dir / "verification_report.json").read_text())
# gpu = report["metrics"]["gpu_test"]
# assert report["accepted"]
# assert gpu["cuda_available"]
# assert gpu["preflight_passed"]
# assert gpu["leaked_feature_accuracy"] == 1.0
# assert gpu["shifted_no_leak_accuracy"] == 0.0
# assert gpu["leakage_gap"] >= 0.5
# assert gpu["cherry_pick_inflation"] >= 3.0
# assert gpu["selected_mean_effect"] >= 5 * gpu["population_median_effect"]
# assert gpu["probe_train_accuracy"] == 1.0
# assert gpu["probe_heldout_accuracy"] == 0.5
# assert gpu["probe_overfit_gap"] >= 0.35
# assert gpu["random_direction_control_rejects_claim"]
# assert gpu["random_direction_effect_gap"] < 0.25
# assert gpu["all_bogus_results_flagged"]
# assert gpu["within_vram_budget"]
# {key: gpu[key] for key in [
#     "device",
#     "leaked_feature_accuracy",
#     "shifted_no_leak_accuracy",
#     "cherry_pick_inflation",
#     "probe_overfit_gap",
#     "random_direction_effect_gap",
#     "peak_vram_gb",
# ]}

