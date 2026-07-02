from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import torch as t

from arena_ext import shapley_attribution as reference


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part5_vlm_modality_region_shap import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _gpu_report() -> dict:
    report = json.loads((_section_dir() / "verification_report.json").read_text())
    return report["metrics"]["gpu_test"]


def _assert_close(actual: float, expected: float, *, msg: str, atol: float = 1e-9) -> None:
    assert abs(float(actual) - float(expected)) <= atol, (
        f"{msg} Expected {expected}, got {actual}."
    )


def _assert_tensor_close(
    actual: t.Tensor,
    expected: t.Tensor,
    *,
    msg: str,
    atol: float = 1e-9,
) -> None:
    assert t.allclose(actual.double(), expected.double(), atol=atol, rtol=0.0), (
        f"{msg} Expected {expected.tolist()}, got {actual.tolist()}."
    )


def _assert_report_close(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} should expose the same fields as the independent reference."
    )
    for key, expected_value in expected_dict.items():
        actual_value = actual_dict[key]
        if isinstance(expected_value, t.Tensor):
            _assert_tensor_close(actual_value, expected_value, msg=f"{msg} field {key!r}")
        elif isinstance(expected_value, float):
            _assert_close(actual_value, expected_value, msg=f"{msg} field {key!r}")
        else:
            assert actual_value == expected_value, (
                f"{msg} field {key!r} should be {expected_value!r}, got {actual_value!r}."
            )


def test_exact_shapley_values_splits_two_player_synergy(
    exact_shapley_values: Callable | None = None,
):
    exact_shapley_values = exact_shapley_values or _solutions().exact_shapley_values
    values = {
        frozenset(): 0.0,
        frozenset({0}): 1.0,
        frozenset({1}): 0.5,
        frozenset({0, 1}): 3.5,
    }
    shapley = exact_shapley_values(values, num_players=2)
    expected = reference.exact_shapley_values(values, num_players=2)
    _assert_tensor_close(
        shapley,
        expected,
        msg="Exact modality Shapley should split the two-point synergy equally.",
    )
    assert shapley.tolist() == [2.0, 1.5], (
        "Image should receive its additive point plus half the synergy; text should "
        "receive its half-point plus the other half."
    )
    print("All tests in `test_exact_shapley_values_splits_two_player_synergy` passed!")


def test_shapley_efficiency_report_requires_complete_coalition_table(
    shapley_efficiency_report: Callable | None = None,
):
    shapley_efficiency_report = (
        shapley_efficiency_report or _solutions().shapley_efficiency_report
    )
    values = reference.vlm_modality_game()
    report = shapley_efficiency_report(values, num_players=2)
    expected = reference.shapley_efficiency_report(values, num_players=2)
    _assert_report_close(report, expected, msg="Modality efficiency report")
    assert report.satisfies_efficiency and report.total_value_delta == 3.5, (
        "Shapley efficiency should compare the attribution sum to full-minus-empty value."
    )
    try:
        shapley_efficiency_report({frozenset(): 0.0, frozenset({0}): 1.0}, num_players=2)
    except ValueError as exc:
        assert "missing" in str(exc), (
            "Incomplete coalition tables should fail with a missing-coalitions error."
        )
    else:
        raise AssertionError("Incomplete coalition tables should raise ValueError.")
    print(
        "All tests in `test_shapley_efficiency_report_requires_complete_coalition_table` passed!"
    )


def test_vlm_modality_game_contains_expected_image_text_coalitions(
    vlm_modality_game: Callable | None = None,
):
    vlm_modality_game = vlm_modality_game or _solutions().vlm_modality_game
    values = vlm_modality_game()
    expected = reference.vlm_modality_game()
    assert values == expected, (
        "The toy modality game should keep the exact image-only, text-only, and "
        "image-plus-text values from the course contract."
    )
    assert values[frozenset()] == 0.0, "The absent-image/absent-text baseline should be zero."
    assert values[frozenset({0, 1})] == 3.5, (
        "The full coalition should include additive image, additive text, and synergy terms."
    )
    print("All tests in `test_vlm_modality_game_contains_expected_image_text_coalitions` passed!")


def test_vlm_modality_shap_report_detects_synergy_and_efficiency(
    vlm_modality_shap_report: Callable | None = None,
):
    vlm_modality_shap_report = (
        vlm_modality_shap_report or _solutions().vlm_modality_shap_report
    )
    report = vlm_modality_shap_report()
    expected = reference.vlm_modality_shap_report()
    _assert_report_close(report, expected, msg="VLM modality SHAP report")
    assert report.detects_synergy and report.synergy == 2.0, (
        "The report should explicitly flag the two-point image/text synergy."
    )
    weak_report = vlm_modality_shap_report(synergy_weight=0.1, min_synergy=1.0)
    assert not weak_report.detects_synergy, (
        "A weak interaction should not pass the modality-synergy gate."
    )
    print("All tests in `test_vlm_modality_shap_report_detects_synergy_and_efficiency` passed!")


def test_vlm_region_game_keeps_background_as_negative_control(
    vlm_region_game: Callable | None = None,
):
    vlm_region_game = vlm_region_game or _solutions().vlm_region_game
    values = vlm_region_game()
    expected = reference.vlm_region_game()
    assert values == expected, (
        "The structured region game should match the object/OCR/background contract."
    )
    assert values[frozenset({1})] == 0.0, (
        "The background-only region is the negative control and should add no score."
    )
    assert values[frozenset({0, 2})] == 3.25, (
        "Object plus OCR should include object evidence, OCR evidence, and their interaction."
    )
    print("All tests in `test_vlm_region_game_keeps_background_as_negative_control` passed!")


def test_vlm_region_shap_report_localizes_object_region(
    vlm_region_shap_report: Callable | None = None,
):
    vlm_region_shap_report = vlm_region_shap_report or _solutions().vlm_region_shap_report
    report = vlm_region_shap_report()
    expected = reference.vlm_region_shap_report()
    _assert_report_close(report, expected, msg="VLM region SHAP report")
    assert report.region_names == ("object", "background", "ocr_text"), (
        "Region names should stay aligned with the attribution vector order."
    )
    assert report.region_values.tolist() == [2.25, 0.0, 0.9999999999999999], (
        "Object should receive the largest attribution, background should stay at zero, "
        "and OCR should get its additive plus interaction share."
    )
    assert report.localizes_target, (
        "The target object attribution should beat every non-target region by the margin."
    )
    print("All tests in `test_vlm_region_shap_report_localizes_object_region` passed!")


def test_modality_shap_smoke_test(modality_shap_smoke_test: Callable | None = None):
    modality_shap_smoke_test = modality_shap_smoke_test or _solutions().modality_shap_smoke_test
    result = modality_shap_smoke_test()
    assert result["modality_values"] == [2.0, 1.5], (
        "The notebook smoke contract should expose JSON-serializable modality values."
    )
    assert result["detects_synergy"] and result["satisfies_efficiency"], (
        "The modality smoke test should require both the synergy and efficiency checks."
    )
    print("All tests in `test_modality_shap_smoke_test` passed!")


def test_region_shap_smoke_test(region_shap_smoke_test: Callable | None = None):
    region_shap_smoke_test = region_shap_smoke_test or _solutions().region_shap_smoke_test
    result = region_shap_smoke_test()
    assert result["region_values"] == [2.25, 0.0, 0.9999999999999999], (
        "The notebook smoke contract should expose exact object/background/OCR values."
    )
    assert result["target_region"] == "object" and result["localizes_target"], (
        "The region smoke test should require object localization over non-target regions."
    )
    print("All tests in `test_region_shap_smoke_test` passed!")


def test_committed_gpu_report_records_real_clip_controls():
    gpu = _gpu_report()
    assert gpu["preflight_passed"], (
        "The committed 16.5 report should have accepted the pinned real CLIP preflight."
    )
    assert gpu["cuda_available"] and "RTX 5090" in gpu["device"], (
        "The report should record a real CUDA device rather than placeholder evidence."
    )
    assert gpu["model_id"] == "openai/clip-vit-base-patch32", (
        "The report should identify the pinned CLIP checkpoint used for real logits."
    )
    assert gpu["revision"] == "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", (
        "The report should pin the CLIP revision used for the real-logit preflight."
    )
    assert gpu["claim_scope"] == "pinned_real_clip_rendered_vlm_shap_preflight", (
        "The report should scope claims to deterministic rendered-image CLIP coalitions."
    )
    assert gpu["modality_satisfies_efficiency"], (
        "The real CLIP modality Shapley values should satisfy efficiency."
    )
    assert gpu["region_names"] == ["object", "background", "ocr_text"], (
        "The region report should keep object/background/OCR names aligned with values."
    )
    assert gpu["region_satisfies_efficiency"], (
        "The real CLIP region Shapley values should satisfy efficiency."
    )
    assert gpu["modality_synergy"] >= 2.0, (
        "Real CLIP modality coalitions should clear the configured synergy threshold."
    )
    assert gpu["object_margin"] >= 1.0, (
        "Real CLIP region attributions should put object evidence above non-object controls."
    )
    assert gpu["target_distractor_margin"] >= 2.0, (
        "The rendered red-square image should score the target caption above the distractor."
    )
    assert gpu["peak_vram_gb"] < 1.0 and gpu["within_vram_budget"], (
        "The pinned CLIP rendered-control preflight should stay inside the 24GB budget."
    )
    print("All tests in `test_committed_gpu_report_records_real_clip_controls` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["modality"]["detects_synergy"], (
        "The notebook contract should include the modality synergy gate."
    )
    assert result["modality"]["satisfies_efficiency"], (
        "The notebook contract should include the modality efficiency gate."
    )
    assert result["region"]["localizes_target"], (
        "The notebook contract should include the object-region localization gate."
    )
    assert result["region"]["satisfies_efficiency"], (
        "The notebook contract should include the region efficiency gate."
    )
    print("All tests in `test_notebook_contract` passed!")
