"""No-fit tests for the independent 16.16 exact-reference adapter."""

from __future__ import annotations

import copy
import inspect
import random

import numpy as np
import pytest
import torch

from . import interaction_methods as methods
from . import interaction_reference as reference
from .interaction_study import pairwise_sii


@pytest.fixture(scope="module", autouse=True)
def _four_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _numpy_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _game_from_dividends(n_players: int, dividends: dict[int, float]) -> torch.Tensor:
    values = torch.zeros(1 << n_players, dtype=torch.float64)
    for coalition in range(1 << n_players):
        values[coalition] = sum(
            value for term, value in dividends.items() if coalition & term == term
        )
    return values


def _student(values: torch.Tensor, n_players: int, order: int) -> tuple[torch.Tensor, ...]:
    mobius = methods.mobius_transform(values, n_players=n_players)
    return (
        methods.shapley_taylor_interactions(
            mobius, n_players=n_players, order=order
        ),
        methods.faith_shap_interactions(
            values, n_players=n_players, max_order=order
        ),
        pairwise_sii(values, n_players=n_players),
    )


def _snapshot() -> dict:
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.random.get_rng_state().clone(),
        "dtype": torch.get_default_dtype(),
    }


def _assert_snapshot(before: dict) -> None:
    assert random.getstate() == before["python"]
    assert _numpy_state_equal(np.random.get_state(), before["numpy"])
    assert torch.equal(torch.random.get_rng_state(), before["torch"])
    assert torch.get_default_dtype() == before["dtype"]


def test_reference_environment_is_exactly_source_pinned():
    environment = reference.reference_environment()
    assert environment["shapiq"] == "1.5.2"
    assert {
        name: item["sha256"] for name, item in environment["installed_sources"].items()
    } == reference.EXPECTED_SOURCE_SHA256
    assert environment["uv_lock"]["sha256"] == reference.EXPECTED_UV_LOCK_SHA256
    assert len(environment["adapter_source_sha256"]) == 64


def test_reference_has_no_student_method_import():
    source = inspect.getsource(reference)
    assert "import interaction_methods" not in source
    assert "from .interaction_methods" not in source
    assert "_direct_stii" in source and "_direct_fsii" in source
    assert 'exact("STII"' in source
    assert 'exact("FSII"' in source
    assert 'exact("SII"' in source


def test_pure_three_plus_four_preserves_exact_disagreement_and_efficiency():
    values = _game_from_dividends(4, {0: 1.25, 0b0111: 3.0, 0b1111: -2.0})[None]
    student = _student(values, 4, 2)
    before = tuple(item.clone() for item in (values, *student))
    result = reference.reference_check(
        values, *student, n_players=4, order=2, expected_game_count=1
    )
    assert result["status"] == "PASS"
    assert result["counts"]["direct_shapiq_exact_computers"] == 1
    assert result["counts"]["direct_shapiq_index_calls"] == 3
    stii = result["independent"]["stii"][0]
    fsii = result["independent"]["fsii_closed_form"][0]
    torch.testing.assert_close(
        stii[[1, 2, 4, 8]], torch.zeros(4, dtype=torch.float64), rtol=0, atol=0
    )
    torch.testing.assert_close(
        stii[[3, 5, 6]], torch.full((3,), 2 / 3, dtype=torch.float64), rtol=0, atol=1e-15
    )
    torch.testing.assert_close(
        stii[[9, 10, 12]], torch.full((3,), -1 / 3, dtype=torch.float64), rtol=0, atol=1e-15
    )
    torch.testing.assert_close(
        fsii[[1, 2, 4]], torch.full((3,), -0.1, dtype=torch.float64), rtol=0, atol=1e-15
    )
    torch.testing.assert_close(
        fsii[[8]], torch.tensor([0.4], dtype=torch.float64), rtol=0, atol=1e-15
    )
    torch.testing.assert_close(
        fsii[[3, 5, 6]], torch.full((3,), 0.9, dtype=torch.float64), rtol=0, atol=1e-15
    )
    torch.testing.assert_close(
        fsii[[9, 10, 12]], torch.full((3,), -0.6, dtype=torch.float64), rtol=0, atol=1e-15
    )
    assert stii[1:].sum().item() == pytest.approx(1.0, abs=1e-15)
    assert fsii[1:].sum().item() == pytest.approx(1.0, abs=1e-15)
    for actual, expected in zip((values, *student), before, strict=True):
        assert torch.equal(actual, expected), "The adapter mutated a caller array."


def test_seeded_orders_one_to_three_and_full_order_match_direct_packages():
    generator = torch.Generator().manual_seed(161610)
    cases = 0
    for n_players in range(3, 8):
        values = torch.randn(
            (3, 1 << n_players), generator=generator, dtype=torch.float64
        )
        for order in (1, 2, 3):
            result = reference.reference_check(
                values,
                *_student(values, n_players, order),
                n_players=n_players,
                order=order,
                expected_game_count=3,
            )
            assert result["status"] == "PASS"
            assert result["passes"]["shapiq_stii_minus_independent"] is True
            assert result["passes"]["shapiq_fsii_minus_independent"] is True
            assert result["passes"]["shapiq_sii_pairs_minus_independent"] is True
            cases += 3
        full = reference.reference_check(
            values[:1],
            *_student(values[:1], n_players, n_players),
            n_players=n_players,
            order=n_players,
            expected_game_count=1,
        )
        expected = methods.mobius_transform(values[:1], n_players=n_players)
        torch.testing.assert_close(full["independent"]["stii"], expected)
        torch.testing.assert_close(full["independent"]["fsii_closed_form"], expected)
    assert cases == 45


def test_operator_is_exact_on_every_basis_and_seeded_random_games():
    n_players = 5
    order = 3
    for kind, function in (
        ("STII", reference._direct_stii),
        ("FSII", reference._direct_fsii),
    ):
        operator = reference._operator(kind, n_players, order)
        eye = np.eye(1 << n_players, dtype=np.float64)
        direct = np.asarray([function(row, n_players, order) for row in eye])
        np.testing.assert_array_equal(operator, direct.T)
        random_games = np.random.Generator(np.random.PCG64(161611)).normal(
            size=(17, 1 << n_players)
        )
        centered = random_games - random_games[:, :1]
        observed = centered @ operator.T
        expected = np.asarray(
            [function(row, n_players, order) for row in random_games]
        )
        observed[:, 0] = random_games[:, 0]
        np.testing.assert_allclose(observed, expected, rtol=0, atol=5e-15)


def test_high_offset_games_are_baseline_centered_before_package_calls():
    rng = np.random.Generator(np.random.PCG64(161612))
    values = 1.0e8 + rng.normal(size=(5, 256))
    stii = np.asarray([reference._direct_stii(row, 8, 2) for row in values])
    fsii = np.asarray([reference._direct_fsii(row, 8, 2) for row in values])
    sii = np.asarray([reference._direct_sii_pairs(row, 8) for row in values])
    result = reference.reference_check(
        values, stii, fsii, sii, n_players=8, order=2, expected_game_count=5
    )
    assert result["status"] == "PASS"
    assert result["scope"]["baseline_centering"].startswith("Subtract v_empty")
    assert result["summaries"]["shapiq_stii_minus_independent"][
        "max_abs_over_centered_game_scale"
    ] < 1e-12
    assert result["summaries"]["shapiq_sii_pairs_minus_independent"][
        "max_abs_over_centered_game_scale"
    ] < 1e-12


def test_tolerances_keep_finite_endpoint_fsii_separate():
    values = _game_from_dividends(4, {0: -0.5, 0b1111: 1.0})[None]
    result = reference.reference_check(
        values, *_student(values, 4, 2), n_players=4, order=2
    )
    assert result["tolerances"]["scale_definition"] == (
        "max(1, max(abs(game_values - v_empty))) per game"
    )
    assert result["tolerances"]["strict"]["absolute_floor"] == 1e-10
    assert result["tolerances"]["strict"]["scale_coefficient"] == 1e-12
    assert result["tolerances"]["package_FSII_coordinates"]["absolute_floor"] == 1e-8
    assert result["tolerances"]["package_FSII_coordinates"]["scale_coefficient"] == 5e-9
    assert result["tolerances"]["package_FSII_efficiency"]["scale_coefficient"] == 3e-8
    assert result["passes"]["shapiq_fsii_minus_independent"] is True
    assert any("finite 1e8 endpoint penalty" in item for item in result["warnings"])


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_success_restores_all_rng_streams_and_default_dtype(monkeypatch, default_dtype):
    original = reference._package_three

    def mutating_package(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(3)
        torch.set_default_dtype(
            torch.float64 if torch.get_default_dtype() == torch.float32 else torch.float32
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(reference, "_package_three", mutating_package)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        random.seed(810)
        np.random.seed(811)
        torch.manual_seed(812)
        before = _snapshot()
        values = _game_from_dividends(3, {0: 0.2, 3: 0.7})[None]
        result = reference.reference_check(
            values, *_student(values, 3, 2), n_players=3, order=2
        )
        assert result["status"] == "PASS"
        _assert_snapshot(before)
    finally:
        torch.set_default_dtype(old_dtype)


def test_exception_restores_all_rng_streams_and_default_dtype(monkeypatch):
    def fail(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(3)
        torch.set_default_dtype(torch.float32)
        raise RuntimeError("injected package failure")

    monkeypatch.setattr(reference, "_package_three", fail)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        random.seed(820)
        np.random.seed(821)
        torch.manual_seed(822)
        before = _snapshot()
        values = _game_from_dividends(3, {7: 1.0})[None]
        with pytest.raises(RuntimeError, match="injected package failure"):
            reference.reference_check(
                values, *_student(values, 3, 2), n_players=3, order=2
            )
        _assert_snapshot(before)
    finally:
        torch.set_default_dtype(old_dtype)


def test_validation_rejects_dtype_shape_nonfinite_and_count_mismatches():
    values = torch.zeros((2, 8), dtype=torch.float64)
    student = _student(values, 3, 2)
    with pytest.raises(TypeError, match="CPU float64"):
        reference.reference_check(values.float(), *student, n_players=3, order=2)
    with pytest.raises(ValueError, match=r"2\*\*n_players"):
        reference.reference_check(values[:, :-1], *student, n_players=3, order=2)
    with pytest.raises(ValueError, match="Expected 3"):
        reference.reference_check(
            values, *student, n_players=3, order=2, expected_game_count=3
        )
    bad = values.clone()
    bad[0, 1] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        reference.reference_check(bad, *student, n_players=3, order=2)
    with pytest.raises(ValueError, match="SII pairs"):
        reference.reference_check(
            values, student[0], student[1], student[2][:, :-1], n_players=3, order=2
        )


def test_direct_package_instances_equal_game_count(monkeypatch):
    calls = 0
    original = reference._package_three

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reference, "_package_three", counted)
    generator = torch.Generator().manual_seed(161613)
    values = torch.randn((7, 16), generator=generator, dtype=torch.float64)
    result = reference.reference_check(
        values, *_student(values, 4, 2), n_players=4, order=2
    )
    assert calls == 7
    assert result["counts"]["direct_shapiq_exact_computers"] == 7
    assert result["counts"]["direct_shapiq_index_calls"] == 21
    assert result["counts"]["raw_package_returned_coordinates"] == 7 * 33


def test_payload_validator_recomputes_values_bounds_counts_and_hashes():
    values = torch.stack(
        (
            _game_from_dividends(4, {0: 0.3, 0b0111: 1.2}),
            _game_from_dividends(4, {0: -0.7, 0b1111: -0.8}),
        )
    )
    result = reference.reference_check(
        values, *_student(values, 4, 2), n_players=4, order=2
    )
    validation = reference.validate_reference_payload(result, expected_game_count=2)
    assert validation == {
        "status": "PASS",
        "games": 2,
        "checked_coordinates": 2 * (16 + 16 + 6 + 16 + 11 + 11 + 6 + 7),
        "student_full_axis_coordinates": 2 * (16 + 16 + 6),
        "independent_vs_package_relevant_coordinates": 2 * (11 + 11 + 6),
        "raw_package_returned_coordinates": 2 * (11 + 11 + 11),
        "raw_identity_arrays": 11,
        "counts_recomputed": 13,
        "bounds_recomputed": 14,
        "sources_rehashed": 6,
    }
    tampered = copy.deepcopy(result)
    tampered["errors"]["student_stii_minus_independent"][0, 1] += 1.0
    with pytest.raises(AssertionError, match="Stored reference error changed"):
        reference.validate_reference_payload(tampered, expected_game_count=2)


def test_validate_study_reference_assembles_all_2992_games_in_frozen_order(monkeypatch):
    captured = {}

    def fake_reference_check(game_values, stii, fsii, sii, **kwargs):
        captured["games"] = game_values.clone()
        captured["stii"] = stii.clone()
        captured["fsii"] = fsii.clone()
        captured["sii"] = sii.clone()
        captured["kwargs"] = kwargs
        return {
            "status": "PASS",
            "counts": {
                "games": len(game_values),
                "student_full_axis_coordinates": 1_615_680,
                "independent_vs_package_relevant_coordinates": 305_184,
                "raw_package_returned_coordinates": 332_112,
            },
        }

    monkeypatch.setattr(reference, "reference_check", fake_reference_check)
    monkeypatch.setattr(
        reference,
        "validate_reference_payload",
        lambda payload, expected_game_count=None: {
            "status": "PASS",
            "games": expected_game_count,
        },
    )
    monkeypatch.setattr(
        reference,
        "_validate_exact_suite",
        lambda prepared: {
            "fixed_unique_games": 8,
            "random_unique_games": 15,
            "random_game_order_cases": 45,
            "random_STII_plus_FSII_case_calls": 90,
            "boundary_order_n_cases": 5,
            "direct_shapiq_exact_computers": 58,
            "direct_shapiq_index_calls": 174,
            "student_full_axis_coordinates": 5_814,
            "independent_vs_package_relevant_coordinates": 2_890,
            "raw_package_returned_coordinates": 3_230,
        },
    )

    def block(rows: int, marker: float, *, model_axis: bool):
        leading = (7, rows) if model_axis else (rows,)
        return {
            "coalition_values": torch.full((*leading, 256), marker, dtype=torch.float64),
            "stii_coefficients": torch.full((*leading, 256), marker + 0.1, dtype=torch.float64),
            "fsii_coefficients": torch.full((*leading, 256), marker + 0.2, dtype=torch.float64),
            "sii_pairs": torch.full((*leading, 28), marker + 0.3, dtype=torch.float64),
            **({"model_keys": reference.MODEL_KEYS} if model_axis else {}),
        }

    evaluated = {
        "phase": "evaluated_after_audit",
        "audit_used_for_selection": False,
        "all_rows": {
            "primary64": block(272, 1.0, model_axis=True),
            "background_variants": {
                "primary64": block(272, 1.0, model_axis=False),
                "nested16": block(272, 2.0, model_axis=False),
                "nested128": block(272, 3.0, model_axis=False),
                "independent64_a": block(272, 4.0, model_axis=False),
                "independent64_b": block(272, 5.0, model_axis=False),
            },
        },
    }
    result = reference.validate_study_reference(
        {"phase": "prepared_before_audit"}, evaluated
    )
    assert captured["games"].shape == (2_992, 256)
    assert captured["stii"].shape == (2_992, 256)
    assert captured["fsii"].shape == (2_992, 256)
    assert captured["sii"].shape == (2_992, 28)
    assert captured["kwargs"] == {
        "n_players": 8,
        "order": 2,
        "expected_game_count": 2_992,
    }
    assert [item["name"] for item in result["branches"]] == [
        "primary64",
        *reference.BACKGROUND_VARIANT_KEYS,
    ]
    assert [item["games"] for item in result["branches"]] == [1_904, 272, 272, 272, 272]
    assert torch.equal(
        captured["games"][:, 0],
        torch.cat(
            [
                torch.full((1_904,), 1.0),
                torch.full((272,), 2.0),
                torch.full((272,), 3.0),
                torch.full((272,), 4.0),
                torch.full((272,), 5.0),
            ]
        ),
    )
    assert result["counts"]["primary_games"] == 1_904
    assert result["counts"]["nonprimary_clean0_games"] == 1_088


def test_finalize_revalidates_exact_suite_without_any_new_package_call(monkeypatch):
    generator = torch.Generator().manual_seed(161614)
    fixed_values = torch.randn((8, 16), generator=generator, dtype=torch.float64)
    fixed_reference = reference.reference_check(
        fixed_values, *_student(fixed_values, 4, 2), n_players=4, order=2
    )
    random_blocks = {}
    for n_players in range(3, 8):
        values = torch.randn(
            (3, 1 << n_players), generator=generator, dtype=torch.float64
        )
        order_blocks = {}
        for order in (1, 2, 3):
            order_blocks[order] = {
                "reference": reference.reference_check(
                    values,
                    *_student(values, n_players, order),
                    n_players=n_players,
                    order=order,
                )
            }
        order = n_players
        boundary_values = values[:1]
        random_blocks[n_players] = {
            "order_blocks": order_blocks,
            "boundary_order_n": {
                "replicate": 0,
                "order": order,
                "reference": reference.reference_check(
                    boundary_values,
                    *_student(boundary_values, n_players, order),
                    n_players=n_players,
                    order=order,
                ),
            },
        }
    prepared = {
        "phase": "prepared_before_audit",
        "exact_games": {
            "fixed": {"reference": fixed_reference},
            "random": random_blocks,
        },
    }
    empirical = {
        "status": "PASS",
        "counts": {
            "games": 2_992,
            "direct_shapiq_exact_computers": 2_992,
            "direct_shapiq_index_calls": 8_976,
            "student_full_axis_coordinates": 1_615_680,
            "independent_vs_package_relevant_coordinates": 305_184,
            "raw_package_returned_coordinates": 332_112,
        },
    }
    original_validate = reference.validate_reference_payload

    def validate_without_fake_arrays(payload, *, expected_game_count=None):
        if payload is empirical:
            assert expected_game_count == 2_992
            return {"status": "PASS", "games": 2_992}
        return original_validate(payload, expected_game_count=expected_game_count)

    monkeypatch.setattr(reference, "validate_reference_payload", validate_without_fake_arrays)

    def forbidden(*args, **kwargs):
        raise AssertionError("finalization attempted a new package call")

    monkeypatch.setattr(reference, "_package_three", forbidden)
    result = reference.finalize_study_reference(prepared, empirical)
    assert result["counts"]["direct_shapiq_exact_computers"] == 2_992
    assert result["counts"]["direct_shapiq_index_calls"] == 8_976
    assert result["counts"]["fixed_unique_exact_games"] == 8
    assert result["counts"]["random_unique_exact_games"] == 15
    assert result["counts"]["random_game_order_cases"] == 45
    assert result["counts"]["random_STII_plus_FSII_case_calls"] == 90
    assert result["counts"]["boundary_order_n_cases"] == 5
    assert result["counts"]["exact_direct_shapiq_exact_computers"] == 58
    assert result["counts"]["exact_direct_shapiq_index_calls"] == 174
    assert result["counts"]["all_reference_direct_shapiq_exact_computers"] == 3_050
    assert result["counts"]["all_reference_direct_shapiq_index_calls"] == 9_150
    assert result["counts"]["all_reference_student_full_axis_coordinates"] == 1_621_494
    assert result["counts"]["all_reference_independent_vs_package_relevant_coordinates"] == 308_074
    assert result["counts"]["all_reference_raw_package_returned_coordinates"] == 335_342
    assert empirical["counts"] == {
        "games": 2_992,
        "direct_shapiq_exact_computers": 2_992,
        "direct_shapiq_index_calls": 8_976,
        "student_full_axis_coordinates": 1_615_680,
        "independent_vs_package_relevant_coordinates": 305_184,
        "raw_package_returned_coordinates": 332_112,
    }, "Finalization must not mutate the already-computed empirical result."


def test_large_reference_arrays_survive_notebook_operand_elision():
    """A fresh kernel reproduces the Python 3.14/NumPy 1.x corruption path.

    Package outputs are fixed synthetic values here to isolate storage; the
    other reference tests cover actual pinned-package computations.
    """
    from pathlib import Path
    import nbformat
    from nbclient import NotebookClient

    source = '''
import numpy as np
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import interaction_reference as reference
n, order = 8, 2
masks = np.arange(256)
one = 3.0 + ((masks & 7) == 7).astype(np.float64)
terms = reference._relevant_masks(n, order)
sii_terms = reference._relevant_masks(n, 2)
st = reference._direct_stii(one, n, order)
fs = reference._direct_fsii(one, n, order)
sii = reference._direct_sii_pairs(one, n)
package_sii = np.zeros(len(sii_terms), dtype=np.float64)
for mask, value in zip(reference._pair_masks(n), sii):
    package_sii[np.flatnonzero(sii_terms == mask)[0]] = value

def synthetic_package(*args):
    return st[terms].copy(), fs[terms].copy(), package_sii.copy(), np.zeros(4, dtype=np.float64)

reference._package_three = synthetic_package
repeat = lambda value: np.broadcast_to(value, (2992, len(value))).copy()
payload = reference.reference_check(repeat(one), repeat(st), repeat(fs), repeat(sii), n_players=n, order=order, expected_game_count=2992)
assert payload['status'] == 'PASS'
for family, key in (('independent', 'fsii_kkt'), ('packages', 'shapiq')):
    if family == 'independent':
        actual = payload[family][key].numpy()
        expected = repeat(fs)
    else:
        actual = payload[family][key]['stii'].numpy()
        expected = repeat(st[terms])
    assert float(np.max(np.abs(np.subtract(actual, expected)))) < 1e-10
assert reference.validate_reference_payload(payload, expected_game_count=2992)['status'] == 'PASS'
print('All retained coefficients survived the notebook error calculations.')
'''
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    root = Path(__file__).resolve().parents[3]
    NotebookClient(notebook, kernel_name="python3", timeout=60,
                   resources={"metadata": {"path": str(root)}}).execute()
