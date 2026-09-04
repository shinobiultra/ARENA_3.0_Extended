import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.shapley_attribution import (
        exact_shapley_values,
        vlm_modality_game,
        vlm_modality_shap_report,
        vlm_region_game,
        vlm_region_shap_report,
    )


def test_vlm_modality_game_splits_image_text_synergy():
    values = vlm_modality_game(image_weight=1.0, text_weight=0.5, synergy_weight=2.0)

    shapley = exact_shapley_values(values, num_players=2)
    report = vlm_modality_shap_report()

    t.testing.assert_close(shapley, t.tensor([2.0, 1.5], dtype=t.float64))
    assert report.synergy == pytest.approx(2.0)
    assert report.detects_synergy
    assert report.satisfies_efficiency


def test_vlm_region_game_localizes_target_object_over_background():
    values = vlm_region_game()

    shapley = exact_shapley_values(values, num_players=3)
    report = vlm_region_shap_report()

    t.testing.assert_close(shapley, t.tensor([2.25, 0.0, 1.0], dtype=t.float64))
    assert report.target_region == "object"
    assert report.target_value == pytest.approx(2.25)
    assert report.max_background_value == pytest.approx(1.0)
    assert report.localizes_target
    assert report.satisfies_efficiency


def test_vlm_region_report_requires_target_region_name():
    with pytest.raises(ValueError, match="target_region"):
        vlm_region_shap_report(target_region="missing")
