"""Visible study orchestration, copied into the learner beside its experiments."""

from pathlib import Path

import torch


def map_training_data(train, methods, directory):
    seeds = list(range(161100, 161105))
    mapping = dict(ids=train["ids"], runs={}, statistics={}, windows={}, diagnosis={})
    for name, label_name, width in (
        ("clean", "y", 64),
        ("noisy", "noisy_y", 64),
        ("shuffled", "shuffled_y", 64),
        ("linear", "y", 0),
    ):
        labels = train[label_name].expand(5, -1)
        run = methods.train_ensemble(train["x"].expand(5, -1, -1), labels, seeds, width=width)
        mapping["runs"][name] = run
        mapping["statistics"][name] = {
            mode: methods.training_statistics(run[f"{mode}_logits"], labels)
            for mode in ("online", "end")
        }
    clean = mapping["runs"]["clean"]["end_logits"]
    labels = train["y"].expand(5, -1)
    mapping["windows"] = {
        "early": methods.training_statistics(clean[:10], labels),
        "late": methods.training_statistics(clean[-10:], labels),
        "repeated_final": methods.training_statistics(clean[-1:].expand_as(clean), labels),
    }
    budgets = torch.tensor([30, 100, 300, 600, 1000])
    for replicate in range(5):
        noisy_score = mapping["statistics"]["noisy"]["end"]["confidence"][replicate]
        clean_score = mapping["statistics"]["clean"]["end"]["confidence"][replicate]
        generator = torch.Generator().manual_seed(166100 + replicate)
        random_score = torch.rand(len(train["y"]), generator=generator)
        mapping["diagnosis"][str(replicate)] = {
            name: methods.retrieval_curve(score, train["changed"], budgets)
            for name, score in (
                ("noisy", noisy_score),
                ("clean", clean_score),
                ("random", random_score),
            )
        }
    mapping["diagnosis_budgets"] = budgets
    mapping["mapping_runs"] = 20
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(mapping, directory / "mapping.pt")
    return mapping


def make_selections(train, mapping, methods):
    stats = mapping["statistics"]["clean"]["end"]
    selections = {"full": torch.arange(len(train["y"])).expand(5, -1).clone()}
    for size_index, size in enumerate((300, 1000, 2000)):
        for region_index, region in enumerate(("easy", "hard", "ambiguous")):
            score = stats["variability" if region == "ambiguous" else "confidence"][0]
            selected = methods.select_indices(score, size, largest=region != "hard").sort().values
            name = f"{region}_{size}"
            selections[name] = selected.expand(5, -1).clone()
            selections[f"{name}_matched"] = torch.stack(
                [
                    methods.matched_random_indices(
                        train["y"],
                        selected,
                        161300 + 1000 * region_index + 10 * size_index + replicate,
                    )
                    for replicate in range(5)
                ]
            )
        selections[f"uniform_{size}"] = torch.stack(
            [
                torch.randperm(
                    len(train["y"]),
                    generator=torch.Generator().manual_seed(165000 + 10 * size_index + replicate),
                )[:size]
                .sort()
                .values
                for replicate in range(5)
            ]
        )
    return selections


def retrain_selections(train, mapping, methods, directory):
    selections = make_selections(train, mapping, methods)
    # Persist every selection before retraining, and retain all seeds afterwards.
    torch.save(selections, directory / "selections_before_refits.pt")
    fits = {}
    seeds = list(range(161200, 161205))
    for name, indices in selections.items():
        fits[name] = methods.train_ensemble(
            train["x"][indices], train["y"][indices], seeds, record=False
        )
    for source in ("ambiguous_1000", "ambiguous_1000_matched"):
        indices = selections[source]
        name = f"{source}_equal_updates"
        fits[name] = methods.train_ensemble(
            train["x"][indices], train["y"][indices], seeds, epochs=90, record=False
        )
        torch.testing.assert_close(fits[name]["loss"][:30], fits[source]["loss"], rtol=0, atol=0)
        selections[name] = indices.clone()
    result = dict(
        phase="ALL_SELECTIONS_AND_FRESH_TRAINING_COMPLETE",
        ids=train["ids"],
        selections=selections,
        fits=fits,
        heldout_predictions_computed=False,
        refits=sum(len(run["seeds"]) for run in fits.values()),
    )
    if result["refits"] != 120:
        raise ValueError("The complete selection study requires 120 fresh refits.")
    torch.save(result, directory / "prepared.pt")
    return result


def evaluate_study(mapping, prepared, evaluation, methods, directory):
    if (
        prepared["phase"] != "ALL_SELECTIONS_AND_FRESH_TRAINING_COMPLETE"
        or prepared["heldout_predictions_computed"]
    ):
        raise ValueError("Save every selection and fresh training run before evaluating holdouts.")
    output = dict(
        holdout_ids={name: split["ids"] for name, split in evaluation.items()},
        mapping={},
        refits={},
    )
    for category, runs in (("mapping", mapping["runs"]), ("refits", prepared["fits"])):
        for name, run in runs.items():
            output[category][name] = {
                split_name: methods.evaluate_models(run["parameters"], split["x"], split["y"])
                for split_name, split in evaluation.items()
            }
    output["evaluated_models"] = 140
    output["evaluation_used_for_selection"] = False
    torch.save(output, directory / "evaluated.pt")
    return output


def retraining_play(
    train, mapping, evaluation, methods, region="ambiguous", size=1000, seed=166000, epochs=30
):
    if region not in ("easy", "hard", "ambiguous"):
        raise ValueError("Choose easy, hard or ambiguous.")
    stats = mapping["statistics"]["clean"]["end"]
    score = stats["variability" if region == "ambiguous" else "confidence"][0]
    selected = methods.select_indices(score, size, largest=region != "hard").sort().values
    random = methods.matched_random_indices(train["y"], selected, seed + 1)
    indices = torch.stack([selected, random])
    run = methods.train_ensemble(
        train["x"][indices], train["y"][indices], [seed, seed], epochs=epochs, record=False
    )
    return dict(
        region=region,
        indices=indices,
        run=run,
        evaluation={
            name: methods.evaluate_models(run["parameters"], split["x"], split["y"])
            for name, split in evaluation.items()
        },
        exploratory_holdout_reuse=True,
    )
