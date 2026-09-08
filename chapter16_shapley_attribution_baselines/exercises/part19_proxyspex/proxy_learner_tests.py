"""Immediate checks for grouped notebook exercises and plot input contracts."""

import numpy as np
import pytest

from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import proxy_methods as answers
from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex.proxy_methods_tests import cube, direct_character


def test_predict_spectrum(function=answers.predict_spectrum):
    masks = cube(4)
    spectrum = {(): .7, (0,): -.3, (1, 3): 1.2}
    expected = .7 - .3 * direct_character(masks, (0,)) + 1.2 * direct_character(masks, (1, 3))
    np.testing.assert_allclose(function(masks, spectrum), expected, atol=1e-13)
    np.testing.assert_array_equal(function(masks, {}), np.zeros(len(masks)))
    print("predict_spectrum passed: signed reconstruction and an empty spectrum.")


def test_predict_boosted_trees(function=answers.predict_boosted_trees):
    masks = cube(4)
    values = .3 + masks[:, 0] + 2 * masks[:, 0] * masks[:, 1]
    model = answers.fit_boosted_trees(masks, values, max_depth=2, n_estimators=4, min_samples_leaf=1)
    expected = np.full(len(masks), model["initial_value"])
    for tree in model["trees"]:
        expected += model["learning_rate"] * tree.predict(masks)
    np.testing.assert_allclose(function(model, masks), expected, atol=1e-13)
    with pytest.raises(ValueError):
        function(model, cube(3))
    print("predict_boosted_trees passed: all shrinkage-weighted trees plus the initial mean.")


def test_fit_proxy(function=answers.fit_proxy):
    masks = cube(5)
    values = np.arange(len(masks)) / 100 + .4 * direct_character(masks, (1, 3))
    candidates = ({"max_depth": 1, "n_estimators": 3, "min_samples_leaf": 1},
                  {"max_depth": 2, "n_estimators": 5, "min_samples_leaf": 1})
    namespace = function.__globals__
    original = namespace["fit_boosted_trees"]
    observations, models = [], []

    def observe(rows, responses, **parameters):
        observations.append((rows.copy(), responses.copy()))
        model = original(rows, responses, **parameters)
        models.append(model)
        return model

    namespace["fit_boosted_trees"] = observe
    try:
        result = function(masks, values, fold_seed=21, candidates=candidates, ridge_alphas=(.01, 1.0))
    finally:
        namespace["fit_boosted_trees"] = original
    assert len(observations) == 7
    for candidate in range(2):
        for fold, record in enumerate(result["folds"]):
            rows, responses = observations[candidate * 3 + fold]
            np.testing.assert_array_equal(rows, masks[record["train_indices"]])
            np.testing.assert_array_equal(responses, values[record["train_indices"]])
    selected = result["proxy_cv"]["selected_index"]
    for fold in range(3):
        spectrum = namespace["ensemble_to_fourier"](models[selected * 3 + fold])
        assert result["ridge_cv"]["fold_supports"][fold] == namespace["select_energy_support"](spectrum)["supports"]
    assert result["accounting"]["residual_tree_fits"] == 24 + candidates[selected]["n_estimators"]
    assert result["accounting"]["ridge_candidate_fits"] == 7
    assert result["accounting"]["oracle_calls_by_methods"] == 0
    expected = np.full(len(masks), result["model"]["initial_value"])
    for tree in result["model"]["trees"]:
        expected += result["model"]["learning_rate"] * tree.predict(masks)
    np.testing.assert_allclose(namespace["predict_spectrum"](masks, result["full_spectrum"]), expected, atol=1e-12)
    with pytest.raises(TypeError, match="audit_values"):
        function(masks, values, fold_seed=21, audit_values=np.ones(30))
    print("fit_proxy passed: observed fold isolation, fold-specific supports, exact fit counts and no audit argument.")


def test_native_plots_reject_unexecuted_payload():
    from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import proxy_plots
    with pytest.raises(ValueError, match="unexecuted"):
        proxy_plots.plot_native_stages(None, prompt_index=0, seed=0)
    with pytest.raises(ValueError, match="unexecuted"):
        proxy_plots.plot_native_spectrum({"results": []}, prompt_index=0, seed=0, budget=456)


def synthetic_native_plot_payload():
    """Small labeled interface fixture; these are analytic test values, never GPT-2 evidence."""
    from copy import deepcopy
    values = np.linspace(-1, 1, 8)
    results = []
    for prompt in range(2):
        for budget in (456, 912, 1824):
            for method in ("spex", "proxyspex"):
                for condition in ("real", "shuffled"):
                    offset = (456 / budget) * (.2 if condition == "real" else 1.)
                    predictions = {"final": values + offset} if method == "spex" else {
                        "proxy": values + offset * 2, "extracted": values + offset * 2,
                        "truncated": values + offset * 3, "refined": values + offset,
                    }
                    baselines = {"constant": np.zeros(8), "additive": values + .15}
                    metrics = {name: {"rmse": float(np.sqrt(np.mean((prediction - values)**2))),
                                      "nrmse": float(np.sqrt(np.mean((prediction - values)**2)) / values.std())}
                               for name, prediction in {**predictions, **baselines}.items()}
                    results.append({"prompt_index": prompt, "prompt_id": f"SYNTHETIC_TEST_{prompt}", "seed": 0,
                        "budget": budget, "method": method, "condition": condition, "metrics": metrics,
                        "audit_values": values.copy(), "predictions": predictions, "baseline_predictions": baselines,
                        "stage_errors": {"model_to_proxy_rmse": offset * 2, "proxy_to_extraction_rmse": 0.,
                                         "extraction_to_truncation_rmse": offset, "model_to_refinement_rmse": offset},
                        "spectrum": {(): .3, (0,): .2, (1, 3): -.1},
                        "query_receipt": {"raw_requested": budget, "unique_model_masks": budget - 1}})
    payload = {"TEST_FIXTURE_ONLY": True, "results": results,
               "prompts": [{"feature_names": ["test", "words", "are", "synthetic"],
                            "prompt": "SYNTHETIC TEST VALUES"} for _ in range(2)],
               "audit": {index: {"audit_values": values.copy(), "unmasked": np.asarray([index + .1]),
                                 "counterfactual": np.asarray([.5 - index])} for index in range(2)}}
    compact = deepcopy(payload)
    for row in compact["results"]:
        for key in ("predictions", "audit_values", "baseline_predictions"):
            del row[key]
    return payload, compact


def test_native_compact_figures_and_recomputed_refit():
    import matplotlib.pyplot as plt
    from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import proxy_plots as plots
    full, compact = synthetic_native_plot_payload()
    population = plots.plot_native_population(compact)
    expected = [.2 / np.std(np.linspace(-1, 1, 8)), .1 / np.std(np.linspace(-1, 1, 8)), .05 / np.std(np.linspace(-1, 1, 8))]
    np.testing.assert_allclose(population.axes[0].lines[0].get_ydata(), expected)
    assert "1/2 prompts" in population.axes[1].get_title()
    curves = plots.plot_native_stages(compact, prompt_index=0, seed=0, budget=456)
    np.testing.assert_allclose(curves.axes[1].lines[1].get_ydata(), [0., 0., 0.])
    # The replay supplies fresh proxy predictions; the saved SPEX comparator has only metrics.
    mixed = {**full, "results": [row for row in full["results"] if row["method"] == "proxyspex"],
             "recorded_comparators": [row for row in compact["results"] if row["method"] == "spex"]}
    refit = plots.plot_native_refit(mixed, prompt_index=0, seed=0, budget=456)
    np.testing.assert_allclose([bar.get_height() for bar in refit.axes[0].patches][:4], [.4, .6, .2, 1.])
    spectrum = plots.plot_native_spectrum(mixed, prompt_index=0, seed=0, budget=456)
    assert "saved comparator" in spectrum.axes[0].get_title()
    assert "your refit" in spectrum.axes[1].get_title()
    for figure in (population, curves, refit, spectrum):
        figure.canvas.draw()
        plt.close(figure)


def test_native_bridge_requires_own_function_namespace(monkeypatch):
    import sys
    from types import SimpleNamespace
    from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import build_notebooks, lesson_cells
    sources = build_notebooks.function_sources()
    namespace = dict(vars(answers))
    names = tuple(lesson_cells.TEST_CALLS)
    for name in names:
        exec(sources[name], namespace)
    student = SimpleNamespace(**{name: namespace[name] for name in names})
    observations = []

    def replay(evidence, **options):
        observations.append((evidence, options))
        return {"TEST_FIXTURE_ONLY": True}

    module = "chapter16_shapley_attribution_baselines.exercises.part18_spex.native_learner"
    from chapter16_shapley_attribution_baselines.exercises import part18_spex
    fake_module = SimpleNamespace(replay_condition=replay)
    monkeypatch.setitem(sys.modules, module, fake_module)
    monkeypatch.setattr(part18_spex, "native_learner", fake_module, raising=False)
    result = lesson_cells.run_native_bridge({"TEST_FIXTURE_ONLY": True}, proxy_namespace=student)
    assert result["TEST_FIXTURE_ONLY"]
    assert observations[0][1]["methods"] is student
    assert observations[0][1]["budget"] == 1824
    student.tree_to_fourier = answers.tree_to_fourier
    with pytest.raises(ValueError, match="same completed learner namespace"):
        lesson_cells.run_native_bridge({}, proxy_namespace=student)


def test_native_notebook_cells_replay_and_preserve_own_answers():
    from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex import build_notebooks, lesson_cells
    student, _ = build_notebooks.build_notebooks()
    completed = build_notebooks.complete_from_own_answers(student)
    implemented = [name for cell in completed["cells"] for name in cell.get("metadata", {}).get("arena", {}).get("exercise_functions", [])]
    assert len(implemented) == 13 and set(implemented) == set(lesson_cells.TEST_CALLS)
    native = lesson_cells.CELLS["native_study"] + lesson_cells.CELLS["native_play"]
    assert "load_native_evidence()" in native
    assert "run_native_bridge(" in native
    assert "run_study(" not in native and "UNEXECUTED" not in native and "NATIVE_EXECUTED" not in native
    assert "QUERY_BUDGET = 456" in native and "PROMPT_INDEX = 0" in native


def test_native_replay_receipt_rejects_leakage_or_changed_prediction(tmp_path):
    import json
    from copy import deepcopy
    from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex.build_notebooks import native_replay_receipt
    binding = {"fixture_sha256": "a" * 64, "native_execution_sha256": "b" * 64, "fresh_reproduction_sha256": "c" * 64}
    (tmp_path / "native_replay_binding.json").write_text(json.dumps(binding))
    stages = ("proxy", "extracted", "truncated", "refined", "constant", "additive")
    comparison = {"student_metrics": {stage: {"rmse": .5, "r2": None} for stage in stages},
                  "recorded_metrics": {stage: {"rmse": .5, "r2": None} for stage in stages},
                  "maximum_prediction_absolute_delta": {stage: 3e-12 for stage in stages},
                  "prediction_allclose": {stage: True for stage in stages},
                  "prediction_tolerance": {"atol": 1e-8, "rtol": 1e-7}}
    replay = {"status": "COMPLETE", "new_model_queries": 0, "new_physical_model_rows": 0,
              "audit_labels_received_by_fit": False, "method": "proxyspex", "prompt_index": 0, "seed": 0,
              "comparisons": [{**deepcopy(comparison), "condition": condition} for condition in ("real", "shuffled")]}

    def write(value):
        for name, budget in (("native_default_replay.json", 1824), ("native_play_replay.json", 456)):
            (tmp_path / name).write_text(json.dumps({**value, "budget": budget}))

    write(replay)
    receipt = native_replay_receipt(tmp_path, binding["fixture_sha256"])
    assert receipt["maximum_reproduced_metric_absolute_delta"] == 0
    assert receipt["maximum_reproduced_prediction_absolute_delta"] == 3e-12
    assert receipt["pointwise_prediction_checks"] == 24
    bad = deepcopy(replay)
    bad["audit_labels_received_by_fit"] = True
    write(bad)
    with pytest.raises(ValueError, match="training/audit boundary"):
        native_replay_receipt(tmp_path, binding["fixture_sha256"])
    bad = deepcopy(replay)
    bad["comparisons"][0]["student_metrics"]["refined"]["rmse"] = .6
    write(bad)
    with pytest.raises(ValueError, match="reconstructed predictions"):
        native_replay_receipt(tmp_path, binding["fixture_sha256"])
    # Equal aggregate errors cannot rescue different predictions or omitted stage checks.
    for field, replacement in (
        ("prediction_allclose", {}),
        ("prediction_allclose", {**comparison["prediction_allclose"], "refined": False}),
        ("maximum_prediction_absolute_delta", {**comparison["maximum_prediction_absolute_delta"], "proxy": float("nan")}),
        ("prediction_tolerance", {"atol": .1, "rtol": .1}),
    ):
        bad = deepcopy(replay)
        bad["comparisons"][0][field] = replacement
        write(bad)
        with pytest.raises(ValueError, match="pointwise prediction checks"):
            native_replay_receipt(tmp_path, binding["fixture_sha256"])
    bad = deepcopy(replay)
    bad["comparisons"] = bad["comparisons"][:1]
    write(bad)
    with pytest.raises(ValueError, match="both real and shuffled"):
        native_replay_receipt(tmp_path, binding["fixture_sha256"])
    bad = deepcopy(replay)
    bad["prompt_index"] = 3
    write(bad)
    with pytest.raises(ValueError, match="two declared native replay cases"):
        native_replay_receipt(tmp_path, binding["fixture_sha256"])


def test_page_export_preserves_figure_bytes_and_student_code(tmp_path):
    from chapter16_shapley_attribution_baselines.exercises.part19_proxyspex.build_notebooks import export_page, digest
    figures = tmp_path / "figures"
    figures.mkdir()
    figure = figures / "01_test_figure.png"
    figure.write_bytes(b"SYNTHETIC PAGE EXPORT BYTE FIXTURE; NOT AN IMAGE")
    notebook = {"cells": [{"cell_type": "markdown", "source": "# Synthetic export test"},
                           {"cell_type": "code", "source": "raise NotImplementedError", "outputs": []},
                           {"cell_type": "code", "source": "show_figure(figure, '01_test_figure')",
                            "outputs": [{"data": {"image/png": "SYNTHETIC"}}]}]}
    receipt = export_page(tmp_path, notebook)
    from pathlib import Path
    page = Path(receipt["path"]).read_text()
    assert "raise NotImplementedError" in page and "../assets/1619_01_test_figure.png" in page
    assert receipt["assets"]["1619_01_test_figure.png"] == digest(figure)
