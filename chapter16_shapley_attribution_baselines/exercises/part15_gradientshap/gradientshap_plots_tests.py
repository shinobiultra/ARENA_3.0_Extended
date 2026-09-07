"""Shape-only adapter tests for the seven future native GradientSHAP figures.

The fabricated values below exercise dimensions and presentation only. Every
payload and rendered title is explicitly marked as non-scientific test data.
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from . import gradientshap_plots as plots


def shape_only_payloads():
    """Return a tiny structurally exact payload with no scientific meaning."""
    rng = np.random.default_rng(161599)
    model_keys = plots.MODEL_KEYS
    budgets = np.array([8, 32, 128, 512, 2048], dtype=np.int64)
    rows = 5
    audit_rows = 4
    features = 8

    primary, all_rows = {}, {}
    for model_index, key in enumerate(model_keys):
        scale = 0.35 + 0.12 * model_index
        q512 = rng.normal(0, scale, (rows, features))
        q256 = q512 + rng.normal(0, 0.035 + model_index * 0.004, q512.shape)
        errors = np.array([0.45, 0.24, 0.13, 0.07, 0.035])[:, None, None]
        phi_prefix = q512[None] + rng.normal(size=(len(budgets), rows, features)) * errors
        se_prefix = np.abs(rng.normal(size=phi_prefix.shape)) * errors * 0.9 + 0.005
        exact = q512 + rng.normal(0, 0.09 + model_index * 0.01, q512.shape)
        deep = exact + rng.normal(0, 0.06, q512.shape)
        gradient = exact + rng.normal(0, 0.12, q512.shape)
        record = {
            "mc": {
                "source_ids": np.arange(rows, dtype=np.int64) + 100,
                "budgets": budgets.copy(),
                "phi_prefix": phi_prefix,
                "se_prefix": se_prefix,
                "base_full": np.array(22.0 + model_index),
            },
            "quadrature": {
                "mean_256": q256,
                "mean_512": q512,
                "drift": q512 - q256,
            },
            "baselines": {
                "phi_exact": exact,
                "phi_deep": deep,
                "phi_gradient": gradient,
            },
        }
        primary[key] = copy.deepcopy(record)
        all_rows[key] = record

    prepared = {
        "_shape_only_fixture": True,
        "primary": primary,
        "feature_names": plots.FEATURE_NAMES,
        "model_labels": dict(plots.MODEL_LABELS),
    }
    performance = {
        key: {"mse": np.array(35.0 + 11 * index)}
        for index, key in enumerate(model_keys)
    }
    performance["training_mean_mse"] = np.array(115.0)

    background_variants = {}
    for index, name in enumerate(
        ("primary64", "nested16", "nested128", "independent64_a", "independent64_b")
    ):
        q512 = all_rows["clean_0"]["quadrature"]["mean_512"] + index * 0.025
        q256 = q512 + rng.normal(0, 0.02 + index * 0.003, q512.shape)
        phi = q512 + rng.normal(0, 0.05 + index * 0.006, q512.shape)
        background_variants[name] = {
            "mc": {
                "base_full": np.array(22.0 + index * 0.8),
                "phi_prefix": np.repeat(phi[None], len(budgets), axis=0),
                "se_prefix": np.abs(rng.normal(0.045, 0.01, (len(budgets), rows, features))),
            },
            "quadrature": {"mean_256": q256, "mean_512": q512},
        }

    noise_variants = {
        "standard_normal": rng.normal(size=(rows, 6, features)),
        "source_ids": np.arange(rows, dtype=np.int64) + 100,
    }
    for sigma, name in ((0.0, "sigma_0"), (0.02, "sigma_0_02"), (0.10, "sigma_0_10")):
        noise_variants[name] = {
            "sigma_standardized": np.array(sigma),
            "clean_target_residual": rng.normal(2.4 * sigma, 0.025, rows),
            "smoothed_target_residual": rng.normal(0, 0.025, rows),
            "smoothed_target_se": np.full(rows, 0.012 + sigma * 0.1),
        }

    replacement_curves = {}
    method_offsets = {
        "exact": 0.00,
        "deep": 0.05,
        "gradient": -0.08,
        "eg_k2048": 0.03,
        "quadrature_512": 0.02,
    }
    for model_index, key in enumerate(model_keys):
        replacement_curves[key] = {}
        for method, offset in method_offsets.items():
            insertion = np.stack(
                [
                    18
                    + model_index
                    + np.linspace(0, 8 + row, features + 1)
                    + offset * np.arange(features + 1)
                    for row in range(audit_rows)
                ]
            )
            deletion = insertion[:, ::-1].copy()
            replacement_curves[key][method] = {
                "insertion": insertion,
                "deletion": deletion,
            }
    random_insertion = rng.normal(25, 2, (len(model_keys), audit_rows, 100, features + 1))
    random_insertion[..., 0] = 18 + np.arange(len(model_keys))[:, None, None]
    random_insertion[..., -1] = 28 + np.arange(len(model_keys))[:, None, None]
    replacement_curves["random"] = {
        "models": model_keys,
        "orders": np.stack([rng.permutation(features) for _ in range(100)]),
        "insertion": random_insertion,
        "deletion": random_insertion[..., ::-1].copy(),
    }

    evaluated = {
        "_shape_only_fixture": True,
        "performance": performance,
        "all_rows": all_rows,
        "all_row_split": np.array([0, 0, 0, 1, 1], dtype=np.int64),
        "background_variants": background_variants,
        "noise_variants": noise_variants,
        "sampling_replicates": {
            "source_ids": np.array([101, 102, 103], dtype=np.int64),
            "models": ("clean_0", "shuffled_0", "ridge"),
            "budgets": budgets.copy(),
            "replicate_seeds": np.arange(4, dtype=np.int64) + 161511,
            "coverage_rate": np.array(
                [
                    [0.78, 0.86, 0.91, 0.94, 0.95],
                    [0.76, 0.84, 0.90, 0.93, 0.94],
                    [0.80, 0.88, 0.92, 0.95, 0.96],
                ]
            ),
        },
        "distance_diagnostic": {
            "query_distance": rng.uniform(0.1, 1.2, rows),
            "donor_distance": np.zeros(3),
            "path": {"distance": rng.uniform(0, 1.5, 24)},
            "coalition": {"distance": rng.uniform(0, 2.3, 40)},
        },
        "replacement_curves": replacement_curves,
        "feature_names": plots.FEATURE_NAMES,
        "model_labels": dict(plots.MODEL_LABELS),
    }

    local_values = {
        "eg": np.array([1.8, -0.7, 0.4, 1.1, 0.3, -0.2, 0.5, 2.0]),
        "quadrature_512": np.array([1.7, -0.6, 0.5, 1.0, 0.4, -0.3, 0.6, 1.9]),
        "exact": np.array([1.5, -0.4, 0.8, 0.7, 0.5, -0.1, 0.9, 1.8]),
        "deep": np.array([1.6, -0.5, 0.7, 0.8, 0.4, -0.2, 0.8, 1.9]),
        "gradient": np.array([2.1, -0.9, 0.2, 1.4, 0.1, -0.5, 0.3, 2.2]),
    }
    local_curves = {
        key: {
            "insertion": np.linspace(21.0, 27.0 + index * 0.15, features + 1),
            "deletion": np.linspace(27.0 + index * 0.15, 21.0, features + 1),
        }
        for index, key in enumerate(local_values)
    }
    native_play = {
        "_shape_only_fixture": True,
        "selection": {
            "model_key": "clean_0",
            "split": "audit",
            "row_index": 0,
            "source_id": 1,
            "background_key": "primary64",
            "samples": 2048,
            "sigma_standardized": 0.0,
            "audit_reuse": "exploratory",
        },
        "prediction": np.array(27.0),
        "base": np.array(22.0),
        "phi_eg": local_values["eg"],
        "se_eg": np.linspace(0.04, 0.11, features),
        "phi_q512": local_values["quadrature_512"],
        "phi_exact": local_values["exact"],
        "phi_deep": local_values["deep"],
        "phi_gradient": local_values["gradient"],
        "local_replacement_curves": local_curves,
        "feature_names": plots.FEATURE_NAMES,
        "units": {"target": "MPa"},
    }
    return prepared, evaluated, native_play


def render_shape_only_adapter_figures(output_dir):
    """Render the seven explicitly marked adapter checks to a cache directory."""
    prepared, evaluated, native_play = shape_only_payloads()
    calls = (
        (plots.plot_native_overview, (prepared, evaluated)),
        (plots.plot_allrow_method_map, (evaluated,)),
        (plots.plot_fixed_waterfall, (native_play,)),
        (plots.plot_background_and_noise, (evaluated,)),
        (plots.plot_replicates_and_distance, (evaluated,)),
        (plots.plot_replacement_curves, (evaluated,)),
        (plots.plot_real_recipe_play, (native_play,)),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for (expected_name, stem), (function, arguments) in zip(
        plots.NATIVE_FIGURE_SPECS, calls, strict=True
    ):
        assert function.__name__ == expected_name, (
            "The cached adapter render must use the frozen native plot API order."
        )
        figure = function(*arguments)
        path = output_dir / f"{stem}.png"
        figure.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def test_native_figure_api_and_titles_are_explicitly_shape_only():
    prepared, evaluated, native_play = shape_only_payloads()
    calls = (
        (plots.plot_native_overview, (prepared, evaluated), 3),
        (plots.plot_allrow_method_map, (evaluated,), 21),
        (plots.plot_fixed_waterfall, (native_play,), 2),
        (plots.plot_background_and_noise, (evaluated,), 4),
        (plots.plot_replicates_and_distance, (evaluated,), 2),
        (plots.plot_replacement_curves, (evaluated,), 2),
        (plots.plot_real_recipe_play, (native_play,), 2),
    )
    for (expected_name, _), (function, arguments, minimum_axes) in zip(
        plots.NATIVE_FIGURE_SPECS, calls, strict=True
    ):
        assert function.__name__ == expected_name, "The frozen plot API name changed."
        figure = function(*arguments)
        assert isinstance(figure, Figure), f"{expected_name} must return a Matplotlib figure."
        assert len(figure.axes) >= minimum_axes, f"{expected_name} dropped a declared control panel."
        title_is_marked = (
            figure._suptitle is not None
            and "SHAPE-ONLY ADAPTER FIXTURE" in figure._suptitle.get_text()
        )
        assert title_is_marked, (
            "Fabricated adapter data must never be displayed as a native scientific result."
        )
        plt.close(figure)


def test_native_plots_do_not_mutate_retained_payload_arrays():
    prepared, evaluated, native_play = shape_only_payloads()
    before = copy.deepcopy((prepared, evaluated, native_play))
    calls = (
        (plots.plot_native_overview, (prepared, evaluated)),
        (plots.plot_allrow_method_map, (evaluated,)),
        (plots.plot_fixed_waterfall, (native_play,)),
        (plots.plot_background_and_noise, (evaluated,)),
        (plots.plot_replicates_and_distance, (evaluated,)),
        (plots.plot_replacement_curves, (evaluated,)),
        (plots.plot_real_recipe_play, (native_play,)),
    )
    for function, arguments in calls:
        plt.close(function(*arguments))

    def assert_same(actual, expected, path="payload"):
        if isinstance(expected, dict):
            assert tuple(actual) == tuple(expected), f"{path} keys changed during rendering."
            for key in expected:
                assert_same(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (tuple, list)):
            assert len(actual) == len(expected), f"{path} length changed during rendering."
            for index, (actual_item, expected_item) in enumerate(
                zip(actual, expected, strict=True)
            ):
                assert_same(actual_item, expected_item, f"{path}[{index}]")
        elif isinstance(expected, np.ndarray):
            assert np.array_equal(actual, expected), f"{path} values changed during rendering."
        else:
            assert actual == expected, f"{path} changed during rendering."

    assert_same((prepared, evaluated, native_play), before)


def test_native_labels_expose_units_counts_and_diagnostic_limits():
    prepared, evaluated, native_play = shape_only_payloads()
    figures = (
        plots.plot_native_overview(prepared, evaluated),
        plots.plot_allrow_method_map(evaluated),
        plots.plot_fixed_waterfall(native_play),
        plots.plot_background_and_noise(evaluated),
        plots.plot_replicates_and_distance(evaluated),
        plots.plot_replacement_curves(evaluated),
        plots.plot_real_recipe_play(native_play),
    )
    text = []
    for figure in figures:
        text.extend(
            [
                figure._suptitle.get_text(),
                *(axis.get_xlabel() for axis in figure.axes),
                *(axis.get_ylabel() for axis in figure.axes),
                *(item.get_text() for item in figure.texts),
            ]
        )
    flattened = " ".join("\n".join(text).split())
    for required in (
        "MPa",
        "7 models",
        "held-out rows",
        "q512 is an approximate",
        "not a population claim",
        "not density",
        "not an attribution superiority gate",
        "exploratory",
    ):
        assert required in flattened, f"Native figures lost required learner-visible text: {required}"
    for figure in figures:
        plt.close(figure)
