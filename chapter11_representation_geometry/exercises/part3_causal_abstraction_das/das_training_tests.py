"""Independent immediate checks for the native-model learner's core methods."""

import torch as t

from . import das_methods as m
from .das_organism import make_organism, organism_pairs, evaluate_alignment


def test_minibatch_alignment(fit=None):
    if fit is None:
        from .das_training_methods import fit_minibatch_alignment as fit
    organism = make_organism(seed=48)
    train = organism_pairs(organism, repeats=4, seed=714)
    heldout = organism_pairs(organism, repeats=16, seed=815)
    targets = m.high_level_output(
        m.high_level_interchange(train["base_variables"], train["source_variables"], (0,))
    )
    readout = organism["readout"].clone()
    calls = []

    def tail(indices, replacement):
        calls.append(indices.detach().clone())
        # Depend on recipient-specific context outside the replaced vector.
        # A missing or incorrectly sliced recipient index changes the target.
        side = train["base_variables"][indices, 1]
        logits = replacement @ organism["readout"]
        return logits + t.stack([1 - side, side, 1 - side, side], -1) * 0.05

    result = fit(
        train["base"],
        train["source"],
        targets,
        tail,
        rank=1,
        steps=180,
        batch_size=10,
        learning_rate=0.02,
        seed=2,
    )
    # The first epoch contains each of the 64 examples exactly once, including
    # a final smaller minibatch. This catches accidental dropped/duplicated rows.
    first_epoch = t.cat(calls[:7])
    t.testing.assert_close(first_epoch.sort().values, t.arange(len(targets)))
    assert [len(indices) for indices in calls[:7]] == [10, 10, 10, 10, 10, 10, 4]
    assert len(result["history"]) == 180
    assert evaluate_alignment(organism, heldout, result["basis"])["changed_iia"] > 0.97
    t.testing.assert_close(organism["readout"], readout, atol=0, rtol=0)
    assert organism["readout"].grad is None
    print("Minibatch alignment and recipient-context tests passed!")


def test_role_probe(fit=None):
    if fit is None:
        from .das_training_methods import fit_role_probe as fit
    generator = t.Generator().manual_seed(712)
    x = t.randn(41, 9, generator=generator, dtype=t.float64) + 2
    labels = (x[:, 3] > 2).long()
    ridge = 0.15
    probe = fit(x, labels, ridge)
    centered = x - x.mean(0)
    y = 2 * labels.double() - 1
    # Independent primal solve; implementation solves in sample space.
    coefficient = t.linalg.solve(
        centered.T @ centered + ridge * len(x) * t.eye(9, dtype=x.dtype),
        centered.T @ (y - y.mean()),
    )
    t.testing.assert_close(probe["coefficient"], coefficient, atol=1e-12, rtol=1e-12)
    t.testing.assert_close(
        probe["basis"] @ probe["basis"].T,
        t.outer(coefficient, coefficient) / coefficient.square().sum(),
    )
    shifted = fit(x + 12, labels, ridge)
    t.testing.assert_close(shifted["coefficient"], coefficient, atol=1e-12, rtol=1e-12)
    print("Observational probe tests passed!")


def test_ioi_intervention_contract():
    from .das_ioi_data import prompt_rows, interchange_pairs

    all_prompts = set()
    for split, count in (("train", 72), ("development", 48), ("final", 96)):
        rows = prompt_rows(split)
        assert len(rows) == count
        assert not all_prompts.intersection(row["prompt"] for row in rows)
        all_prompts.update(row["prompt"] for row in rows)
        pairs = interchange_pairs(rows)
        assert len(pairs) == 4 * count
        assert sum(p["changed_role"] and p["cross_name"] for p in pairs) == count
        for pair in pairs:
            base, source, cf = [rows[pair[key]] for key in ("base", "source", "counterfactual")]
            expected = base["a"] if source["role"] else base["b"]
            assert pair["target"] == cf["answer"] == expected
            assert (cf["a"], cf["b"]) == (base["a"], base["b"])
            assert base["prompt"][slice(*base["span"])] == (base["a"], base["b"])[base["role"]]
            if pair["cross_name"]:
                assert not ({base["a"], base["b"]} & {source["a"], source["b"]})
            elif not pair["changed_role"]:
                assert pair["base"] == pair["source"]
    print("High-level name-role intervention and split tests passed; no model execution!")
