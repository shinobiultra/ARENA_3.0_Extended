"""Immediate semantic tests for the ten Shapley interaction methods."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from . import interaction_methods as methods


def _game_from_dividends(n_players: int, dividends: dict[int, float]) -> Tensor:
    values = torch.zeros(2**n_players, dtype=torch.float64)
    for coalition in range(2**n_players):
        values[coalition] = sum(
            value
            for term, value in dividends.items()
            if coalition & term == term
        )
    return values


def _unanimity(n_players: int, term: int | None = None) -> Tensor:
    term = 2**n_players - 1 if term is None else term
    return torch.tensor(
        [float(mask & term == term) for mask in range(2**n_players)],
        dtype=torch.float64,
    )


def _ordinary_shapley(game: Tensor, n_players: int) -> Tensor:
    result = torch.zeros_like(game)
    result[0] = game[0]
    for player in range(n_players):
        bit = 1 << player
        for context in range(2**n_players):
            if context & bit:
                continue
            size = context.bit_count()
            weight = math.factorial(size) * math.factorial(n_players - size - 1)
            weight /= math.factorial(n_players)
            result[bit] += weight * (game[context | bit] - game[context])
    return result


def _discrete_derivative(game: Tensor, interaction: int, context: int) -> Tensor:
    result = torch.zeros_like(game[0])
    subset = interaction
    while True:
        sign = -1 if (interaction.bit_count() - subset.bit_count()) % 2 else 1
        result = result + sign * game[context | subset]
        if subset == 0:
            return result
        subset = (subset - 1) & interaction


def _shapley_taylor_derivative_reference(
    game: Tensor, n_players: int, order: int
) -> Tensor:
    result = torch.zeros_like(game)
    result[0] = game[0]
    full = 2**n_players - 1
    for interaction in range(1, 2**n_players):
        size = interaction.bit_count()
        if size < order:
            result[interaction] = _discrete_derivative(game, interaction, 0)
        elif size == order:
            complement = full ^ interaction
            context = complement
            while True:
                weight = order / (
                    n_players * math.comb(n_players - 1, context.bit_count())
                )
                result[interaction] += weight * _discrete_derivative(
                    game, interaction, context
                )
                if context == 0:
                    break
                context = (context - 1) & complement
    return result


def _numpy_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _permute_bitmask_axis(values: Tensor, permutation: tuple[int, ...]) -> Tensor:
    """Relabel old player i as permutation[i] on a complete-game axis."""
    result = torch.empty_like(values)
    for old_mask in range(2 ** len(permutation)):
        new_mask = sum(
            1 << new_player
            for old_player, new_player in enumerate(permutation)
            if old_mask & (1 << old_player)
        )
        result[..., new_mask] = values[..., old_mask]
    return result


def test_mobius_transform(fn=methods.mobius_transform):
    games = torch.stack(
        (
            _game_from_dividends(3, {0: 2.0}),
            _game_from_dividends(3, {0: -0.4, 1: 1.2, 2: -0.7, 4: 0.3}),
            _game_from_dividends(3, {3: 2.5}),
            _game_from_dividends(3, {7: -1.75}),
        )
    )
    before = games.clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    actual = fn(games, n_players=3)
    expected = torch.zeros_like(games)
    expected[0, 0] = 2.0
    expected[1, [0, 1, 2, 4]] = torch.tensor(
        [-0.4, 1.2, -0.7, 0.3], dtype=torch.float64
    )
    expected[2, 3] = 2.5
    expected[3, 7] = -1.75
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=1e-15,
        msg="The Möbius transform must isolate constant, additive, pair, and triple dividends.",
    )
    assert torch.equal(games, before), "The transform must not mutate the coalition table."
    assert actual.data_ptr() != games.data_ptr(), "Return a fresh coefficient tensor."
    assert random.getstate() == python_state
    assert _numpy_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    with pytest.raises(TypeError, match="CPU float64"):
        fn(games.float(), n_players=3)
    with pytest.raises(ValueError, match=r"2\*\*n_players"):
        fn(torch.zeros(7, dtype=torch.float64), n_players=3)
    with pytest.raises(ValueError, match="finite"):
        fn(torch.tensor([0.0, float("nan")], dtype=torch.float64), n_players=1)
    print("All tests in test_mobius_transform passed!")


def test_shapley_taylor_interactions(fn=methods.shapley_taylor_interactions):
    pure_three = torch.zeros(8, dtype=torch.float64)
    pure_three[-1] = 1.0
    pure_four = torch.zeros(16, dtype=torch.float64)
    pure_four[-1] = 1.0
    three = fn(pure_three, n_players=3, order=2)
    four = fn(pure_four, n_players=4, order=2)
    assert torch.equal(three[[1, 2, 4]], torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(
        three[[3, 5, 6]],
        torch.full((3,), 1 / 3, dtype=torch.float64),
        rtol=0,
        atol=1e-15,
        msg="A pure three-way dividend must split equally among its three retained pairs.",
    )
    assert torch.equal(four[[1, 2, 4, 8]], torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(
        four[torch.tensor([mask.bit_count() == 2 for mask in range(16)])],
        torch.full((6,), 1 / 6, dtype=torch.float64),
        rtol=0,
        atol=1e-15,
        msg="A pure four-way dividend must split by binom(4,2), not by player count.",
    )
    assert torch.equal(
        four[torch.tensor([mask.bit_count() > 2 for mask in range(16)])],
        torch.zeros(5, dtype=torch.float64),
    ), "Every coefficient above the requested order must be exactly zero."

    generator = torch.Generator().manual_seed(161602)
    game = torch.randn(16, generator=generator, dtype=torch.float64)
    mobius = methods.mobius_transform(game, n_players=4)
    order_one = fn(mobius, n_players=4, order=1)
    torch.testing.assert_close(
        order_one,
        _ordinary_shapley(game, 4),
        msg="Order-one Shapley-Taylor must be the ordinary Shapley allocation.",
    )
    torch.testing.assert_close(
        fn(mobius, n_players=4, order=4),
        mobius,
        rtol=0,
        atol=2e-15,
        msg="At full order, Shapley-Taylor must retain the complete Möbius expansion.",
    )
    for order in (1, 2, 3):
        torch.testing.assert_close(
            fn(mobius, n_players=4, order=order),
            _shapley_taylor_derivative_reference(game, 4, order),
            atol=4e-15,
            rtol=0,
            msg=(
                "The Möbius allocation must match the independent discrete-derivative "
                "formula with denominator binom(n-1, |T|)."
            ),
        )
    batch = torch.stack((mobius, -0.25 * mobius))
    torch.testing.assert_close(
        fn(batch, n_players=4, order=2)[1],
        -0.25 * fn(mobius, n_players=4, order=2),
        msg="Shapley-Taylor must be linear and preserve arbitrary leading batches.",
    )
    permutation = (2, 1, 0, 3)
    torch.testing.assert_close(
        fn(
            _permute_bitmask_axis(mobius, permutation),
            n_players=4,
            order=2,
        ),
        _permute_bitmask_axis(fn(mobius, n_players=4, order=2), permutation),
        rtol=0,
        atol=2e-15,
        msg="Relabeling players must relabel the interaction vector in the same way.",
    )
    for result, values in ((order_one, game), (fn(mobius, n_players=4, order=2), game)):
        torch.testing.assert_close(
            result.sum(),
            values[-1],
            msg="The intercept plus all nonempty Shapley-Taylor terms must equal v(full).",
        )

    no_dummy = torch.randn(8, generator=generator, dtype=torch.float64)
    with_dummy = torch.tensor(
        [no_dummy[mask & 7] for mask in range(16)], dtype=torch.float64
    )
    dummy_result = fn(
        methods.mobius_transform(with_dummy, n_players=4), n_players=4, order=2
    )
    assert torch.equal(
        dummy_result[torch.tensor([bool(mask & 8) for mask in range(16)])],
        torch.zeros(8, dtype=torch.float64),
    ), "Every term containing a dummy feature must be exactly zero."
    with pytest.raises(ValueError, match="1 through n_players"):
        fn(mobius, n_players=4, order=0)
    print("All tests in test_shapley_taylor_interactions passed!")


def test_faith_design_matrix(fn=methods.faith_design_matrix):
    design, terms = fn(n_players=3, max_order=2)
    assert torch.equal(terms, torch.tensor([0, 1, 2, 3, 4, 5, 6]))
    assert design.shape == (8, 7) and design.dtype == torch.float64
    assert torch.equal(design[0], torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.float64))
    assert torch.equal(design[-1], torch.ones(7, dtype=torch.float64))
    assert torch.equal(
        design[3], torch.tensor([1, 1, 1, 1, 0, 0, 0], dtype=torch.float64)
    ), "X[S,T] must indicate whether the term T is a subset of coalition S."
    with pytest.raises(ValueError, match="at least two"):
        fn(n_players=1, max_order=1)
    with pytest.raises(ValueError, match="1 through n_players"):
        fn(n_players=3, max_order=4)
    print("All tests in test_faith_design_matrix passed!")


def test_faith_kernel_weights(fn=methods.faith_kernel_weights):
    weights = fn(n_players=4)
    raw = torch.zeros(16, dtype=torch.float64)
    for mask in range(1, 15):
        size = mask.bit_count()
        raw[mask] = 3 / (math.comb(4, size) * size * (4 - size))
    expected = raw / raw.sum()
    assert torch.equal(weights, expected), (
        "Use the normalized per-coalition kernel; a cardinality probability per row "
        "would count binom(n,s) twice."
    )
    assert weights[0] == weights[-1] == 0
    assert weights.sum() == 1
    assert weights[1] == 2 * weights[3], (
        "For n=4, each size-one coalition gets twice each size-two coalition's weight."
    )
    with pytest.raises(ValueError, match="at least two"):
        fn(n_players=1)
    print("All tests in test_faith_kernel_weights passed!")


def test_faith_shap_interactions(fn=methods.faith_shap_interactions):
    pure_three = _unanimity(3)
    pure_four = _unanimity(4)
    three = fn(pure_three, n_players=3, max_order=2)
    four = fn(pure_four, n_players=4, max_order=2)
    torch.testing.assert_close(
        three[[1, 2, 4]],
        torch.full((3,), -1 / 6, dtype=torch.float64),
        atol=2e-14,
        rtol=0,
        msg="Faith-Shap adjusts pure-three singleton terms to exactly -1/6.",
    )
    torch.testing.assert_close(
        three[[3, 5, 6]],
        torch.full((3,), 1 / 2, dtype=torch.float64),
        atol=2e-14,
        rtol=0,
        msg="Faith-Shap assigns exactly 1/2 to each pure-three pair.",
    )
    torch.testing.assert_close(
        four[[1, 2, 4, 8]],
        torch.full((4,), -1 / 5, dtype=torch.float64),
        atol=2e-14,
        rtol=0,
        msg="Faith-Shap adjusts pure-four singleton terms to exactly -1/5.",
    )
    torch.testing.assert_close(
        four[torch.tensor([mask.bit_count() == 2 for mask in range(16)])],
        torch.full((6,), 3 / 10, dtype=torch.float64),
        atol=2e-14,
        rtol=0,
        msg="Faith-Shap assigns exactly 3/10 to each pure-four pair.",
    )
    assert torch.equal(
        four[torch.tensor([mask.bit_count() > 2 for mask in range(16)])],
        torch.zeros(5, dtype=torch.float64),
    ), "The constrained solve must scatter exact zeros above max_order."

    generator = torch.Generator().manual_seed(161604)
    game = torch.randn((2, 16), generator=generator, dtype=torch.float64)
    game_before = game.clone()
    order_one = fn(game, n_players=4, max_order=1)
    for row in range(2):
        torch.testing.assert_close(
            order_one[row],
            _ordinary_shapley(game[row], 4),
            atol=3e-14,
            rtol=0,
            msg="Order-one Faith-Shap must be ordinary Shapley.",
        )
    mobius = methods.mobius_transform(game, n_players=4)
    torch.testing.assert_close(
        fn(game, n_players=4, max_order=4),
        mobius,
        atol=3e-13,
        rtol=0,
        msg="At full order, exact endpoint-constrained Faith-Shap is Möbius inversion.",
    )
    shifted = fn(game + 7.0, n_players=4, max_order=2)
    original = fn(game, n_players=4, max_order=2)
    torch.testing.assert_close(shifted[..., 1:], original[..., 1:], atol=3e-13, rtol=0)
    torch.testing.assert_close(shifted[..., 0], original[..., 0] + 7, atol=3e-13, rtol=0)
    torch.testing.assert_close(
        original.sum(-1),
        game[..., -1],
        atol=3e-13,
        rtol=0,
        msg="The KKT constraints must impose v(empty) and v(full) as equalities.",
    )
    linear = fn(game[0] + game[1], n_players=4, max_order=2)
    torch.testing.assert_close(linear, original[0] + original[1], atol=4e-13, rtol=0)
    permutation = (2, 1, 0, 3)
    torch.testing.assert_close(
        fn(_permute_bitmask_axis(game, permutation), n_players=4, max_order=2),
        _permute_bitmask_axis(original, permutation),
        atol=4e-13,
        rtol=0,
        msg="Faith-Shap must be equivariant to a feature relabeling.",
    )
    assert torch.equal(game, game_before), "The constrained solve must not mutate its games."

    degree_two_game = _game_from_dividends(
        4, {0: 0.5, 1: 1.0, 2: -0.5, 4: 0.25, 3: 2.0, 10: -1.0}
    )
    degree_two_mobius = methods.mobius_transform(degree_two_game, n_players=4)
    torch.testing.assert_close(
        fn(degree_two_game, n_players=4, max_order=2),
        degree_two_mobius,
        atol=2e-13,
        rtol=0,
        msg="Faith-Shap must exactly recover a game whose Möbius degree is at most two.",
    )
    torch.testing.assert_close(
        methods.shapley_taylor_interactions(
            degree_two_mobius, n_players=4, order=2
        ),
        degree_two_mobius,
        rtol=0,
        atol=0,
        msg="The two efficient definitions must agree on the matched degree-two control.",
    )

    no_dummy = torch.randn(8, generator=generator, dtype=torch.float64)
    with_dummy = torch.tensor(
        [no_dummy[mask & 7] for mask in range(16)], dtype=torch.float64
    )
    dummy = fn(with_dummy, n_players=4, max_order=2)
    assert dummy[torch.tensor([bool(mask & 8) for mask in range(16)])].abs().max() < 2e-13
    with pytest.raises(TypeError, match="CPU float64"):
        fn(game.float(), n_players=4, max_order=2)
    print("All tests in test_faith_shap_interactions passed!")


def test_faith_shap_closed_form(fn=methods.faith_shap_closed_form):
    generator = torch.Generator().manual_seed(161605)
    for n_players in range(2, 7):
        games = torch.randn(
            (2, 2**n_players), generator=generator, dtype=torch.float64
        )
        mobius = methods.mobius_transform(games, n_players=n_players)
        before = mobius.clone()
        for order in range(1, min(3, n_players) + 1):
            closed = fn(mobius, n_players=n_players, max_order=order)
            solved = methods.faith_shap_interactions(
                games, n_players=n_players, max_order=order
            )
            torch.testing.assert_close(
                closed,
                solved,
                atol=2e-11,
                rtol=2e-12,
                msg="The primary-paper Möbius formula must agree with constrained WLS.",
            )
        assert torch.equal(mobius, before), "The closed form must not mutate its dividends."
    pure_four = fn(
        torch.nn.functional.one_hot(torch.tensor(15), 16).to(torch.float64),
        n_players=4,
        max_order=2,
    )
    torch.testing.assert_close(
        pure_four[[1, 3]],
        torch.tensor([-1 / 5, 3 / 10], dtype=torch.float64),
        rtol=0,
        atol=1e-16,
    )
    with pytest.raises(ValueError, match="1 through n_players"):
        fn(torch.zeros(8, dtype=torch.float64), n_players=3, max_order=0)
    print("All tests in test_faith_shap_closed_form passed!")


def test_reconstruct_game(fn=methods.reconstruct_game):
    generator = torch.Generator().manual_seed(161606)
    coefficients = torch.randn((2, 3, 16), generator=generator, dtype=torch.float64)
    expected = torch.empty_like(coefficients)
    for coalition in range(16):
        subsets = [term for term in range(16) if coalition & term == term]
        expected[..., coalition] = coefficients[..., subsets].sum(-1)
    actual = fn(coefficients, n_players=4)
    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-15,
        rtol=0,
        msg="Reconstruction must sum every retained term contained in each coalition.",
    )
    torch.testing.assert_close(
        methods.mobius_transform(actual, n_players=4),
        coefficients,
        atol=3e-15,
        rtol=0,
        msg="Möbius and subset-zeta transforms must be inverses.",
    )
    with pytest.raises(ValueError, match=r"2\*\*n_players"):
        fn(torch.zeros(15, dtype=torch.float64), n_players=4)
    print("All tests in test_reconstruct_game passed!")


def test_faithfulness_error(fn=methods.faithfulness_error):
    game = torch.tensor(
        [[0.0, 1.0, -1.0, 2.0, 0.5, 1.5, -0.5, 4.0]], dtype=torch.float64
    )
    zero_coefficients = torch.zeros_like(game)
    weights = methods.faith_kernel_weights(n_players=3)
    expected = (weights * game.square()).sum(-1)
    torch.testing.assert_close(
        fn(game, zero_coefficients, n_players=3),
        expected,
        rtol=0,
        atol=0,
        msg="Report the normalized interior per-coalition weighted squared error.",
    )
    endpoint_changed = game.clone()
    endpoint_changed[..., 0] = 1_000
    endpoint_changed[..., -1] = -1_000
    assert torch.equal(
        fn(game, zero_coefficients, n_players=3),
        fn(endpoint_changed, zero_coefficients, n_players=3),
    ), "Endpoint rows have zero kernel weight because equalities are imposed separately."

    generator = torch.Generator().manual_seed(161607)
    games = torch.randn((5, 16), generator=generator, dtype=torch.float64)
    mobius = methods.mobius_transform(games, n_players=4)
    st = methods.shapley_taylor_interactions(mobius, n_players=4, order=2)
    faith = methods.faith_shap_interactions(games, n_players=4, max_order=2)
    assert torch.all(
        fn(games, faith, n_players=4) <= fn(games, st, n_players=4) + 2e-13
    ), "Faith-Shap must minimize its declared kernel error over the constrained degree-two family."
    with pytest.raises(ValueError, match="same shape"):
        fn(games, faith[0], n_players=4)
    print("All tests in test_faithfulness_error passed!")


class _StatefulLinear(nn.Module):
    def __init__(self, *, raises: bool = False):
        super().__init__()
        self.linear = nn.Linear(3, 1, dtype=torch.float64)
        self.dropout = nn.Dropout()
        self.register_buffer("calls", torch.tensor(0.0, dtype=torch.float64))
        self.raises = raises
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64))
            self.linear.bias.fill_(0.25)

    def forward(self, x):
        random.random()
        np.random.random()
        torch.rand(())
        torch.set_default_dtype(torch.float32)
        self.calls.add_(1)
        if self.linear.weight.grad is not None:
            self.linear.weight.grad.add_(1)
        if self.raises:
            raise RuntimeError("declared forward failure")
        return self.linear(x).squeeze(-1)


def test_masked_model_game(fn=methods.masked_model_game):
    model = _StatefulLinear()
    model.train()
    model.dropout.eval()
    model.linear.weight.grad = torch.tensor([[0.4, -0.3, 0.2]], dtype=torch.float64)
    gradient_object = model.linear.weight.grad
    gradient_before = gradient_object.clone()
    buffer_before = model.calls.clone()
    x = torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]], dtype=torch.float64)
    bank = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 1.0]], dtype=torch.float64)
    x_before, bank_before = x.clone(), bank.clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()

    actual = fn(model, x, bank, coalition_batch_size=3)
    expected = torch.empty((2, 8), dtype=torch.float64)
    weight = model.linear.weight.detach()[0]
    bias = model.linear.bias.detach()[0]
    for row in range(2):
        for mask in range(8):
            predictions = []
            for donor in bank:
                hybrid = torch.tensor(
                    [x[row, feature] if mask & (1 << feature) else donor[feature] for feature in range(3)],
                    dtype=torch.float64,
                )
                predictions.append((hybrid * weight).sum() + bias)
            expected[row, mask] = torch.stack(predictions).mean()
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=0)
    torch.testing.assert_close(
        actual[:, 0],
        model.linear(bank).squeeze(-1).mean().expand(2),
        msg="The empty coalition must be the mean intact-donor prediction.",
    )
    torch.testing.assert_close(
        actual[:, -1],
        model.linear(x).squeeze(-1),
        msg="The full coalition must be the query prediction.",
    )
    torch.testing.assert_close(
        fn(model, x, bank[0], coalition_batch_size=1),
        fn(model, x, bank[0:1], coalition_batch_size=100),
        rtol=0,
        atol=0,
        msg="A vector reference is exactly a one-row donor bank.",
    )
    mobius = methods.mobius_transform(actual, n_players=3)
    assert mobius[:, torch.tensor([mask.bit_count() > 1 for mask in range(8)])].abs().max() < 1e-15
    assert model.training and not model.dropout.training
    assert model.linear.weight.grad is gradient_object
    assert torch.equal(model.linear.weight.grad, gradient_before)
    assert torch.equal(model.calls, buffer_before)
    assert torch.equal(x, x_before) and torch.equal(bank, bank_before)
    assert random.getstate() == python_state
    assert _numpy_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.get_default_dtype() == default_dtype

    raising = _StatefulLinear(raises=True).train()
    raising.linear.weight.grad = torch.ones_like(raising.linear.weight)
    raising_gradient = raising.linear.weight.grad
    raising_before = raising_gradient.clone()
    raising_buffer = raising.calls.clone()
    raising_python_state = random.getstate()
    raising_numpy_state = np.random.get_state()
    raising_torch_state = torch.random.get_rng_state().clone()
    raising_default_dtype = torch.get_default_dtype()
    with pytest.raises(RuntimeError, match="declared forward failure"):
        fn(raising, x, bank)
    assert raising.training and raising.linear.weight.grad is raising_gradient
    assert torch.equal(raising.linear.weight.grad, raising_before)
    assert torch.equal(raising.calls, raising_buffer)
    assert random.getstate() == raising_python_state
    assert _numpy_state_equal(np.random.get_state(), raising_numpy_state)
    assert torch.equal(torch.random.get_rng_state(), raising_torch_state)
    assert torch.get_default_dtype() == raising_default_dtype

    with pytest.raises(TypeError, match="CPU float64"):
        fn(nn.Linear(3, 1, dtype=torch.float32), x, bank)
    with pytest.raises(ValueError, match="same feature"):
        fn(model, x, torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="positive integer"):
        fn(model, x, bank, coalition_batch_size=0)

    class VectorOutput(nn.Module):
        def forward(self, value):
            return value

    with pytest.raises(ValueError, match="one scalar output"):
        fn(VectorOutput(), x, bank)
    print("All tests in test_masked_model_game passed!")


def test_contextual_pair_synergy(fn=methods.contextual_pair_synergy):
    game = _game_from_dividends(3, {0: 0.7, 3: 2.0, 7: 3.0})
    batch = torch.stack((game, -2 * game))
    before = batch.clone()
    contexts, synergy = fn(batch, (0, 1), n_players=3)
    assert torch.equal(contexts, torch.tensor([0, 4], dtype=torch.int64))
    torch.testing.assert_close(
        synergy,
        torch.tensor([[2.0, 5.0], [-4.0, -10.0]], dtype=torch.float64),
        rtol=0,
        atol=1e-15,
        msg="Use v(Tij)-v(Ti)-v(Tj)+v(T), preserving its sign in every context.",
    )
    unsorted = torch.tensor([4, 0], dtype=torch.int64)
    selected, selected_synergy = fn(batch, (1, 0), n_players=3, context_masks=unsorted)
    assert torch.equal(selected, contexts) and torch.equal(selected_synergy, synergy)
    assert torch.equal(unsorted, torch.tensor([4, 0], dtype=torch.int64))
    assert torch.equal(batch, before)
    full_context = torch.tensor([4], dtype=torch.int64)
    _, full = fn(game, (0, 1), n_players=3, context_masks=full_context)
    assert full.item() == 5.0

    class NonlinearModel(nn.Module):
        def forward(self, value):
            return 0.4 + value[:, 0] * value[:, 1] * (1 + value[:, 2])

    model = NonlinearModel()
    query = torch.tensor([[2.0, -1.0, 3.0]], dtype=torch.float64)
    reference = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    stored_game = methods.masked_model_game(model, query, reference)[0]
    _, stored_full = fn(
        stored_game,
        (0, 1),
        n_players=3,
        context_masks=full_context,
    )
    direct = {}
    for mask in (4, 5, 6, 7):
        hybrid = torch.where(
            torch.tensor([bool(mask & (1 << player)) for player in range(3)]),
            query[0],
            reference,
        )
        direct[mask] = model(hybrid[None])[0]
    direct_full = direct[7] - direct[5] - direct[6] + direct[4]
    torch.testing.assert_close(
        stored_full[0],
        direct_full,
        rtol=0,
        atol=0,
        msg="Stored-game and direct four-forward full-context synergy must agree.",
    )
    with pytest.raises(ValueError, match="different"):
        fn(game, (1, 1), n_players=3)
    with pytest.raises(ValueError, match=r"\[0, n_players\)"):
        fn(game, (0, 3), n_players=3)
    with pytest.raises(ValueError, match="disjoint"):
        fn(game, (0, 1), n_players=3, context_masks=torch.tensor([1]))
    with pytest.raises(TypeError, match="CPU int64"):
        fn(game, (0, 1), n_players=3, context_masks=torch.tensor([0.0]))
    print("All tests in test_contextual_pair_synergy passed!")
