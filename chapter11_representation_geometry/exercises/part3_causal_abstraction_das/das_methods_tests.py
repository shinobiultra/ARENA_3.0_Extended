"""Independent mathematical and gradient oracles for the developing DAS lesson."""

import itertools
import math

import torch as t
import torch.nn.functional as F

from . import das_methods as reference


def test_high_level_mechanisms(m=reference):
    rows = t.tensor(list(itertools.product((0, 1), repeat=2)))
    base = rows.repeat_interleave(4, 0)
    donor = rows.repeat(4, 1)
    for variables in ((), (0,), (1,), (0, 1)):
        actual = m.high_level_interchange(base, donor, variables)
        expected = t.tensor(
            [[int(s[j] if j in variables else b[j]) for j in range(2)] for b, s in zip(base, donor)]
        )
        assert t.equal(actual, expected), "Intervene only on the specified high-level variables."
        assert t.equal(m.high_level_output(actual), expected[:, 0] * 2 + expected[:, 1]), (
            "Recompute the joint output after intervention."
        )
    assert t.equal(base, rows.repeat_interleave(4, 0)), "Do not mutate the base inputs."
    print("High-level intervention and output tests passed!")


def test_projector_matches_explicit_rotate_swap_unrotate(m=reference):
    g = t.Generator().manual_seed(41)
    rotation = t.linalg.qr(t.randn(7, 7, dtype=t.float64, generator=g)).Q
    base = t.randn(9, 7, dtype=t.float64, generator=g)
    donor = t.randn(9, 7, dtype=t.float64, generator=g)
    for rank in (0, 1, 3, 7):
        q = rotation[:, :rank]
        rotated = base @ rotation
        rotated[:, :rank] = (donor @ rotation)[:, :rank]
        expected = rotated @ rotation.T
        actual = m.subspace_interchange(base, donor, q)
        t.testing.assert_close(
            actual,
            expected,
            atol=1e-12,
            rtol=1e-12,
            msg="The compact projector must match rotate/swap/unrotate.",
        )
        t.testing.assert_close(
            actual @ q,
            donor @ q,
            atol=1e-12,
            rtol=1e-12,
            msg="Selected coordinates must equal the donor's.",
        )
        t.testing.assert_close(
            actual @ rotation[:, rank:],
            base @ rotation[:, rank:],
            atol=1e-12,
            rtol=1e-12,
            msg="The orthogonal complement must stay at the base values.",
        )
    t.testing.assert_close(
        m.subspace_interchange(base, base, rotation[:, :2]),
        base,
        atol=0,
        rtol=0,
        msg="A self-interchange must be an exact no-op.",
    )
    print("Zero/full-rank, donor coordinate and complement tests passed!")


def test_orthonormal_basis_limits_and_invalid_columns(m=reference):
    g = t.Generator().manual_seed(22)
    raw = t.randn(6, 3, dtype=t.float64, generator=g)
    q = m.orthonormal_columns(raw)
    t.testing.assert_close(
        q.T @ q,
        t.eye(3, dtype=q.dtype),
        atol=1e-12,
        rtol=1e-12,
        msg="Every retained basis column must be unit and orthogonal.",
    )
    t.testing.assert_close(
        q @ (q.T @ raw),
        raw,
        atol=1e-12,
        rtol=1e-12,
        msg="QR must preserve the supplied column span.",
    )
    assert (q.T @ raw).diagonal().gt(0).all(), "Use the positive-diagonal QR convention."
    assert m.orthonormal_columns(raw[:, :0]).shape == (6, 0), "Allow an empty subspace."
    for bad in (t.ones(6, 2, dtype=t.float64), t.full((6, 1), float("nan"))):
        try:
            m.orthonormal_columns(bad)
        except ValueError:
            continue
        raise AssertionError("Reject dependent or nonfinite alignment columns.")
    print("Orthonormal basis, zero-rank and invalid-column tests passed!")


def test_disjoint_interventions_commute_and_same_variable_overwrites(m=reference):
    g = t.Generator().manual_seed(62)
    rotation = t.linalg.qr(t.randn(5, 5, dtype=t.float64, generator=g)).Q
    base, first, second = [t.randn(4, 5, dtype=t.float64, generator=g) for _ in range(3)]
    a, b = rotation[:, :1], rotation[:, 1:3]
    ab = m.subspace_interchange(m.subspace_interchange(base, first, a), second, b)
    ba = m.subspace_interchange(m.subspace_interchange(base, second, b), first, a)
    t.testing.assert_close(
        ab, ba, atol=1e-12, rtol=1e-12, msg="Disjoint orthogonal interventions should commute."
    )
    twice = m.subspace_interchange(m.subspace_interchange(base, first, a), second, a)
    once = m.subspace_interchange(base, second, a)
    t.testing.assert_close(
        twice,
        once,
        atol=1e-12,
        rtol=1e-12,
        msg="A later intervention on the same variable overwrites it.",
    )
    print("Intervention composition tests passed!")


def test_qr_and_independent_finite_difference_gradient(m=reference):
    g = t.Generator().manual_seed(89)
    raw = t.randn(5, 2, dtype=t.float64, generator=g, requires_grad=True)
    base, donor = [t.randn(6, 5, dtype=t.float64, generator=g) for _ in range(2)]
    weights = t.randn(5, 3, dtype=t.float64, generator=g)
    target = t.tensor([0, 1, 2, 1, 0, 2])
    q = m.orthonormal_columns(raw)
    t.testing.assert_close(q.T @ q, t.eye(2, dtype=q.dtype), atol=1e-12, rtol=1e-12)
    loss = m.intervention_loss(m.subspace_interchange(base, donor, q) @ weights, target)
    grad = t.autograd.grad(loss, raw)[0]

    # Independent projector P = A(A.T A)^-1 A.T; no QR or taught CE routine.
    def oracle(a):
        projector = a @ t.linalg.solve(a.T @ a, a.T)
        return F.cross_entropy((base + (donor - base) @ projector) @ weights, target)

    epsilon = 1e-5
    finite = t.empty_like(raw)
    for i, j in itertools.product(range(5), range(2)):
        plus, minus = raw.detach().clone(), raw.detach().clone()
        plus[i, j] += epsilon
        minus[i, j] -= epsilon
        finite[i, j] = (oracle(plus) - oracle(minus)) / (2 * epsilon)
    t.testing.assert_close(
        grad,
        finite,
        atol=2e-8,
        rtol=2e-6,
        msg="The model-tail gradient must pass through the alignment.",
    )
    assert grad.norm() > 0.01, "This test should not pass with detached or zero gradients."
    print("QR and independent finite-difference gradient tests passed!")


def test_full_vocabulary_loss_and_empty_iia(m=reference):
    logits = t.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -998.0]], dtype=t.float64)
    targets = t.tensor([1, 0])
    t.testing.assert_close(m.intervention_loss(logits, targets), F.cross_entropy(logits, targets))
    assert m.interchange_accuracy(logits, targets) == 0.5, "Count exact output agreement."
    assert m.interchange_accuracy(logits, targets, t.tensor([False, True])) == 0.0, (
        "The selected second example is wrong; apply the mask to both predictions and targets."
    )
    assert math.isnan(m.interchange_accuracy(logits, targets, t.tensor([False, False]))), (
        "An empty subgroup has undefined accuracy, not 0% or 100%."
    )
    print("Stable loss and intervention-distribution metric tests passed!")


def test_predictive_unused_copy_fails_causal_interchange(m=reference):
    variables = t.tensor(list(itertools.product((0, 1), repeat=2)))
    base_vars = variables.repeat_interleave(4, 0)
    source_vars = variables.repeat(4, 1)

    def encode(v):
        return t.stack((2 * v[:, 0] - 1, 2 * v[:, 1] - 1, 2 * v[:, 0] - 1), -1).double()

    def read(h):
        classes = t.tensor(list(itertools.product((-1, 1), repeat=2)), dtype=h.dtype)
        return h[:, :2] @ classes.T

    base, source = encode(base_vars), encode(source_vars)
    target = m.high_level_output(m.high_level_interchange(base_vars, source_vars, (0,)))
    changed = base_vars[:, 0] != source_vars[:, 0]
    assert t.equal((base[:, 2] > 0).long(), base_vars[:, 0]), "The unused copy is a perfect probe."
    unused = m.subspace_interchange(base, source, t.eye(3, dtype=t.float64)[:, 2:3])
    causal = m.subspace_interchange(base, source, t.eye(3, dtype=t.float64)[:, :1])
    assert m.interchange_accuracy(read(unused), target, changed) == 0.0, (
        "Copying a predictive but unused feature cannot implement the high-level change."
    )
    assert m.interchange_accuracy(read(unused), target) == 0.5, (
        "Identity pairs explain the unused feature's 50% all-pair agreement."
    )
    assert m.interchange_accuracy(read(causal), target) == 1.0, (
        "Swapping the readout's used routing coordinate must implement every specified high-level change."
    )
    t.testing.assert_close(
        causal[:, 1],
        base[:, 1],
        atol=0,
        rtol=0,
        msg="The independent side decision must stay unchanged.",
    )
    print("Perfect-probe counterexample and causal oracle tests passed!")


def test_alignment_optimization_changes_subspace_and_freezes_model(m=reference):
    from .das_organism import make_organism, organism_pairs

    t.set_num_threads(2)
    organism = make_organism()
    train = organism_pairs(organism, repeats=8, seed=31)
    heldout = organism_pairs(organism, repeats=16, seed=73)
    weights_before = organism["readout"].clone()
    target = m.high_level_output(
        m.high_level_interchange(train["base_variables"], train["source_variables"], (0,))
    )
    fit = m.fit_alignment(
        train["base"],
        train["source"],
        target,
        lambda h: h @ organism["readout"],
        rank=1,
        steps=250,
        learning_rate=0.03,
        seed=1,
    )
    expected = m.high_level_output(
        m.high_level_interchange(heldout["base_variables"], heldout["source_variables"], (0,))
    )
    logits = (
        m.subspace_interchange(heldout["base"], heldout["source"], fit["basis"])
        @ organism["readout"]
    )
    assert m.interchange_accuracy(logits, expected) > 0.98, (
        "Learn the intervention on training nuisances and transfer to fresh ones."
    )
    assert fit["history"][-1]["loss"] < 0.1 * fit["history"][0]["loss"], (
        "Optimization must change the alignment and reduce intervention loss."
    )
    assert t.equal(weights_before, organism["readout"]), "DAS must leave model weights unchanged."
    assert organism["readout"].grad is None, "Freeze the low-level weights before training."
    assert not fit["basis"].requires_grad, "Return an independent saved final alignment."
    print("Alignment training, held-out intervention and frozen-model tests passed!")
