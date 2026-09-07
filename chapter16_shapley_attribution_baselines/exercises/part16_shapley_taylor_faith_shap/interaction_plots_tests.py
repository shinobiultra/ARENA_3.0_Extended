"""Shape, semantic-label, and mutation tests for interaction figures."""

from __future__ import annotations

import copy
import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from . import interaction_diagnostics as diagnostics
from . import interaction_methods as methods
from . import interaction_plots as plots


matplotlib.use("Agg")

FEATURE_NAMES = (
    "Cement",
    "Slag",
    "Fly ash",
    "Water",
    "Superplasticizer",
    "Coarse aggregate",
    "Fine aggregate",
    "Age",
)
MODEL_KEYS = (
    "clean_0",
    "clean_1",
    "clean_2",
    "shuffled_0",
    "shuffled_1",
    "shuffled_2",
    "ridge",
)
PAIR_INDICES = np.asarray(list(itertools.combinations(range(8), 2)), dtype=np.int64)
PAIR_MASKS = np.asarray([(1 << a) | (1 << b) for a, b in PAIR_INDICES], dtype=np.int64)
SINGLE_MASKS = np.asarray([1 << index for index in range(8)], dtype=np.int64)


def _exact() -> dict:
    result = diagnostics.diagnose_mobius_games(methods.mobius_transform)
    result = diagnostics.diagnose_shapley_taylor(
        result, methods.shapley_taylor_interactions
    )
    result = diagnostics.diagnose_faith_design(
        result, methods.faith_design_matrix, methods.faith_kernel_weights
    )
    result = diagnostics.diagnose_faith_solver(
        result, methods.faith_shap_interactions, methods.faith_shap_closed_form
    )
    return diagnostics.diagnose_reconstruction(
        result, methods.reconstruct_game, methods.faithfulness_error
    )


def _native_fixture() -> tuple[dict, dict, dict, dict]:
    """Return explicitly labeled schema-only arrays, never scientific data."""

    rng = np.random.default_rng(160016)
    rows = 6
    st = np.zeros((7, rows, 256), dtype=np.float64)
    faith = np.zeros_like(st)
    sii = np.zeros((7, rows, 28), dtype=np.float64)
    for model in range(6):
        for row in range(rows):
            pair = (model + 1) * (row + 1) * np.linspace(-0.015, 0.025, 28)
            sii[model, row] = 1.2 * pair
            st[model, row, PAIR_MASKS] = pair
            faith[model, row, PAIR_MASKS] = 1.1 * pair
            st[model, row, SINGLE_MASKS] = np.linspace(-0.08, 0.08, 8)
            faith[model, row, SINGLE_MASKS] = np.linspace(-0.06, 0.1, 8)
    # Ridge is a real additive zero-pair control in the future native payload.
    st[6, :, PAIR_MASKS] = 0
    faith[6, :, PAIR_MASKS] = 0
    sii[6] = 0

    rmse = np.abs(rng.normal(size=(2, 7, rows, 9))) * 0.08
    rmse[..., [0, -1]] = 0
    kernel = np.abs(rng.normal(size=(2, 7, rows))) * 0.05
    kernel[1] = kernel[0] * 0.5
    residual = np.zeros((2, 7, rows, 256), dtype=np.float64)
    full_synergy = rng.normal(scale=0.2, size=(7, rows, 28))
    full_synergy[6] = 0
    direct_four = np.zeros((7, rows, 28, 4), dtype=np.float64)
    direct_four[..., 3] = full_synergy

    primary = {
        "donor_source_ids": np.arange(64, dtype=np.int64),
        "sii_pairs": sii,
        "stii_coefficients": st,
        "fsii_coefficients": faith,
        "pair_values": np.stack(
            (sii, st[..., PAIR_MASKS], faith[..., PAIR_MASKS]), axis=2
        ),
        "rmse_by_size": rmse,
        "kernel_mse": kernel,
        "residual": residual,
        "full_context_synergy": full_synergy,
    }
    background_variants = {}
    background_order = (
        "primary64",
        "nested16",
        "nested128",
        "independent64_a",
        "independent64_b",
    )
    for index, name in enumerate(background_order):
        background_variants[name] = {
            "stii_coefficients": st[0] + index * 0.004,
            "fsii_coefficients": faith[0] - index * 0.003,
        }

    evaluated = {
        "evidence_kind": "schema_fixture",
        "model_keys": MODEL_KEYS,
        "feature_names": FEATURE_NAMES,
        "target_unit": "MPa",
        "pair_indices": PAIR_INDICES,
        "source_ids": np.asarray([0, 70, 87, 1, 12, 17]),
        "row_split": np.asarray([0, 0, 0, 1, 1, 1]),
        "recipe_group_ids": np.arange(rows, dtype=np.int64),
        "all_rows": {
            "primary64": primary,
            "background_variants": background_variants,
        },
        "performance": {"audit": {"mse": np.linspace(4, 64, 7)}},
        "pair_capture": {
            "capture_curves": np.tile(np.linspace(0, 1, 29), (7, 3, 1)),
            "random_capture_curves": np.tile(np.linspace(0, 1, 29), (7, 256, 1)),
        },
        "random_controls": {
            "within_size": {
                "observed": {
                    "top3_share": np.asarray([0.3, 0.35, 0.4]),
                    "peak_to_mean": np.asarray([2.0, 2.5, 3.0]),
                },
                "null": {
                    "top3_share": np.tile([0.2, 0.25, 0.3], (128, 1)),
                    "peak_to_mean": np.tile([1.2, 1.3, 1.4], (128, 1)),
                },
            },
        },
        "direct_full_context": {
            "four_values": direct_four,
            "lookup_difference": np.zeros((7, rows, 28)),
        },
    }

    donor_ids = np.concatenate(([173], np.arange(1000, 1063)))
    donor_four = np.column_stack(
        (
            np.zeros(64),
            np.zeros(64),
            np.zeros(64),
            np.linspace(-0.3, 0.3, 64),
        )
    )
    empirical = {
        "pair": (0, 1),
        "context_mask": 252,
        "donor_source_ids": donor_ids,
        "four_values_by_donor": donor_four.T,
    }
    native_play = {
        "evidence_kind": "schema_fixture",
        "fixed_row": {
            "model_key": "clean_0",
            "source_id": 0,
            "background_key": "primary64",
            "donor_count": 64,
            "order": 2,
            "stii_coefficients": st[0, 0],
            "fsii_coefficients": faith[0, 0],
            "sii_pairs": sii[0, 0],
            "pair_indices": PAIR_INDICES,
            "target_unit": "MPa",
            "ridge_pair_max_abs": 0.0,
        },
        "empirical_background_average": empirical,
    }
    overview = evaluated
    context_play = {
        "evidence_kind": "schema_fixture",
        "model_key": "clean_0",
        "source_id": 0,
        "background_key": "primary64",
        "pair": (0, 1),
        "context_mask": 252,
        "four_values": donor_four.mean(axis=0),
        "four_values_by_donor": donor_four.T,
        "donor_source_ids": donor_ids,
        "pair_values": np.asarray([0.08, 0.1, 0.12]),
        "method_keys": ("sii", "stii", "fsii"),
        "target_unit": "MPa",
        "interprets_model_input_intervention_only": True,
    }
    return evaluated, native_play, overview, context_play


def _dividend_play() -> dict:
    exact = _exact()
    row = exact["game_names"].index("pure_four")
    return {
        "n_players": 4,
        "order": 2,
        "members": (0, 1, 2, 3),
        "value": 1.0,
        "game_values": exact["game_values"][row],
        "explained": {
            "stii_coefficients": exact["stii"][row],
            "fsii_coefficients": exact["fsii"][row],
            "ordinary_shapley": exact["ordinary_shapley"][row],
            "sii_pairs": exact["sii_pairs"][row],
            "pair_indices": np.asarray(exact["pair_indices"]),
            "reconstruction": exact["reconstruction"][:, row],
            "kernel_mse": exact["kernel_mse"][:, row],
        },
    }


def _figure_text(fig) -> str:
    artists = list(fig.texts)
    for axis in fig.axes:
        artists.extend(axis.texts)
        artists.extend(axis.get_xticklabels())
        artists.extend(axis.get_yticklabels())
        artists.extend([axis.xaxis.label, axis.yaxis.label, axis.title, axis._left_title, axis._right_title])
        legend = axis.get_legend()
        if legend is not None:
            artists.extend(legend.get_texts())
    return "\n".join(item.get_text() for item in artists if hasattr(item, "get_text"))


def test_all_eleven_approved_plot_interfaces_return_figures_and_close_cleanly():
    evaluated, native_play, overview, context_play = _native_fixture()
    exact = _exact()
    calls = [
        (plots.plot_exact_allocation_ledger, (exact,)),
        (plots.plot_reconstruction_reason, (exact,)),
        (plots.plot_native_game_and_controls, (overview,)),
        (plots.plot_fixed_row_interaction_graph, (native_play, FEATURE_NAMES)),
        (plots.plot_allrow_pair_heatmaps, (evaluated, FEATURE_NAMES)),
        (plots.plot_heldout_pair_order_capture, (evaluated, FEATURE_NAMES)),
        (plots.plot_order_two_reconstruction, (evaluated,)),
        (plots.plot_background_changes_the_game, (evaluated, FEATURE_NAMES)),
        (plots.plot_contextual_pair_interventions, (evaluated, native_play, FEATURE_NAMES)),
        (plots.plot_dividend_play, (_dividend_play(),)),
        (plots.plot_donor_context_play, (context_play, FEATURE_NAMES)),
    ]
    figures = []
    try:
        for function, arguments in calls:
            figure = function(*arguments)
            figures.append(figure)
            assert hasattr(figure, "savefig")
            assert figure.axes
    finally:
        for figure in figures:
            plt.close(figure)


def test_exact_ledger_labels_fractions_efficiency_and_sii_separation():
    figure = plots.plot_exact_allocation_ledger(_exact())
    try:
        text = _figure_text(figure)
        for label in ("1/6", "-1/5", "3/10", "1/4", "1/3"):
            assert label in text
        assert "Do not add these as one order-2 decomposition" in text
        assert "6 × 1/6 = 1" in text
    finally:
        plt.close(figure)


def test_exact_ledger_rejects_swapped_or_fabricated_method_values():
    exact = _exact()
    exact["stii"] = exact["stii"].clone()
    row = exact["game_names"].index("pure_four")
    exact["stii"][row, exact["pair_masks"]] = 3 / 10
    with pytest.raises(AssertionError):
        plots.plot_exact_allocation_ledger(exact)


def test_reconstruction_figure_states_kernel_specific_objective_and_control():
    figure = plots.plot_reconstruction_reason(_exact())
    try:
        text = _figure_text(figure)
        assert "kernel-specific" in text
        assert "degree-2 control" in text
        assert "empty/full endpoints" in text
    finally:
        plt.close(figure)


def test_schema_fixture_native_figures_are_visibly_watermarked():
    evaluated, native_play, overview, context_play = _native_fixture()
    calls = [
        (plots.plot_native_game_and_controls, (overview,)),
        (plots.plot_fixed_row_interaction_graph, (native_play, FEATURE_NAMES)),
        (plots.plot_allrow_pair_heatmaps, (evaluated, FEATURE_NAMES)),
        (plots.plot_heldout_pair_order_capture, (evaluated, FEATURE_NAMES)),
        (plots.plot_order_two_reconstruction, (evaluated,)),
        (plots.plot_background_changes_the_game, (evaluated, FEATURE_NAMES)),
        (plots.plot_contextual_pair_interventions, (evaluated, native_play, FEATURE_NAMES)),
        (plots.plot_donor_context_play, (context_play, FEATURE_NAMES)),
    ]
    for function, arguments in calls:
        figure = function(*arguments)
        try:
            assert "SCHEMA FIXTURE — NO SCIENTIFIC RESULT" in _figure_text(figure)
        finally:
            plt.close(figure)


def test_native_overview_requires_units_all_models_and_counts():
    _, _, overview, _ = _native_fixture()
    broken = copy.deepcopy(overview)
    del broken["target_unit"]
    with pytest.raises(KeyError, match="target_unit"):
        plots.plot_native_game_and_controls(broken)
    broken = copy.deepcopy(overview)
    broken["model_keys"] = broken["model_keys"][:6]
    with pytest.raises(ValueError, match="7"):
        plots.plot_native_game_and_controls(broken)


def test_fixed_graph_enforces_content_blind_row_model_background_and_scale_labels():
    _, native_play, _, _ = _native_fixture()
    figure = plots.plot_fixed_row_interaction_graph(native_play, FEATURE_NAMES)
    try:
        text = _figure_text(figure)
        assert "source 0" in text
        assert "primary64" in text
        assert "64 donors" in text
        assert "MPa" in text
        assert "pair sii only" in text.lower()
    finally:
        plt.close(figure)
    broken = copy.deepcopy(native_play)
    broken["fixed_row"]["source_id"] = 1
    with pytest.raises(ValueError, match="source 0"):
        plots.plot_fixed_row_interaction_graph(broken, FEATURE_NAMES)


def test_allrow_heatmap_requires_every_pair_model_and_audit_split():
    evaluated, _, _, _ = _native_fixture()
    figure = plots.plot_allrow_pair_heatmaps(evaluated, FEATURE_NAMES)
    try:
        text = _figure_text(figure)
        assert "all 7 models" in text
        assert "3 audit rows" in text
        assert "pair sii only" in text.lower()
        assert "mean absolute" in text
    finally:
        plt.close(figure)
    broken = copy.deepcopy(evaluated)
    broken["all_rows"]["primary64"]["sii_pairs"] = broken["all_rows"]["primary64"]["sii_pairs"][..., :-1]
    with pytest.raises(ValueError, match="every pair"):
        plots.plot_allrow_pair_heatmaps(broken, FEATURE_NAMES)


def test_capture_figure_requires_all_random_curves_and_labels_noncausal_target():
    evaluated, _, _, _ = _native_fixture()
    figure = plots.plot_heldout_pair_order_capture(evaluated, FEATURE_NAMES)
    try:
        text = _figure_text(figure)
        assert "all 256 random orders shown" in text
        assert "model-input diagnostic" in text
        assert "128 within-size structure controls" in text
    finally:
        plt.close(figure)
    broken = copy.deepcopy(evaluated)
    broken["pair_capture"]["random_capture_curves"] = broken[
        "pair_capture"
    ]["random_capture_curves"][:, :255]
    with pytest.raises(ValueError, match="256"):
        plots.plot_heldout_pair_order_capture(broken, FEATURE_NAMES)


def test_reconstruction_figure_enforces_faith_objective_and_exact_endpoints():
    evaluated, _, _, _ = _native_fixture()
    figure = plots.plot_order_two_reconstruction(evaluated)
    try:
        text = _figure_text(figure)
        assert "optimizes this declared objective" in text
        assert "exact endpoints" in text
        assert "Ridge" in text
        assert "MPa squared" in text
    finally:
        plt.close(figure)
    broken = copy.deepcopy(evaluated)
    broken["all_rows"]["primary64"]["kernel_mse"][1] = (
        broken["all_rows"]["primary64"]["kernel_mse"][0] + 1
    )
    with pytest.raises(ValueError, match="exceeds"):
        plots.plot_order_two_reconstruction(broken)
    broken = copy.deepcopy(evaluated)
    broken["all_rows"]["primary64"]["residual"][0, 0, 0, 0] = 0.1
    with pytest.raises(ValueError, match="endpoints"):
        plots.plot_order_two_reconstruction(broken)


def test_background_figure_requires_frozen_order_and_disclaims_uncertainty():
    evaluated, _, _, _ = _native_fixture()
    figure = plots.plot_background_changes_the_game(evaluated, FEATURE_NAMES)
    try:
        text = _figure_text(figure)
        assert "changes the game" in text
        assert "not confidence replicates" in text
        assert "all singleton and pair terms" in text
    finally:
        plt.close(figure)
    broken = copy.deepcopy(evaluated)
    broken["all_rows"]["background_variants"] = dict(
        reversed(list(broken["all_rows"]["background_variants"].items()))
    )
    with pytest.raises(ValueError, match="frozen donor-bank order"):
        plots.plot_background_changes_the_game(broken, FEATURE_NAMES)


def test_context_figure_uses_last_minus_middle_terms_plus_first():
    evaluated, native_play, _, _ = _native_fixture()
    figure = plots.plot_contextual_pair_interventions(
        evaluated, native_play, FEATURE_NAMES
    )
    try:
        first_axis = figure.axes[0]
        points = first_axis.collections[0].get_offsets()
        np.testing.assert_allclose(points[:, 0], points[:, 1], rtol=0, atol=1e-12)
        text = _figure_text(figure)
        assert "not an interaction definition" in text
        assert "fixed donor 173" in text
    finally:
        plt.close(figure)


def test_donor_context_play_rejects_wrong_sign_or_donor_average():
    _, _, _, context_play = _native_fixture()
    figure = plots.plot_donor_context_play(context_play, FEATURE_NAMES)
    try:
        text = _figure_text(figure)
        assert "Donors can disagree" in text
        assert "Different context averages can all be valid" in text
        assert "exploratory model-input intervention" in text
    finally:
        plt.close(figure)
    broken = copy.deepcopy(context_play)
    broken["four_values"][3] += 1
    with pytest.raises(AssertionError):
        plots.plot_donor_context_play(broken, FEATURE_NAMES)


def test_dividend_play_keeps_sii_visually_separate():
    figure = plots.plot_dividend_play(_dividend_play())
    try:
        text = _figure_text(figure)
        assert "pair SII only" in text
        assert "SII remains a separate pair comparator" in text
        assert "maximum order 2" in text
    finally:
        plt.close(figure)
