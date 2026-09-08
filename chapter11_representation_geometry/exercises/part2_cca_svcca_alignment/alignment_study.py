"""Readable final alignment experiment; these bodies are provided in the learner."""

from pathlib import Path
import gzip, json
import torch as t
from . import alignment_methods as m
from .alignment_ioi_data import paired_rows, donor_indices
from .alignment_model import capture_rows, replace_rows
from .run_alignment_development import fit_candidates


def answer_record(
    rows, logits, donor_logits, tokenizer, condition, source_layer, target_layer, draw=None
):
    kl = m.distribution_kl(donor_logits, logits)
    top = logits.argmax(-1)
    donor_top = donor_logits.argmax(-1)
    output = []
    for i, row in enumerate(rows):
        expected = tokenizer.encode(" " + row["recipient"], add_special_tokens=False)
        changed = tokenizer.encode(" " + row["subject"], add_special_tokens=False)
        if len(expected) != 1 or len(changed) != 1:
            raise ValueError("Names must be single output tokens.")
        output.append(
            {
                "row_id": row["id"],
                "split": row["split"],
                "condition": condition,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "draw": draw,
                "seen_names": row["word_seen_in_training"],
                "kl_to_native_donor": float(kl[i]),
                "top_token_id": int(top[i]),
                "top_token_text": tokenizer.decode(int(top[i])),
                "native_donor_top_token_id": int(donor_top[i]),
                "native_top_agreement": bool(top[i] == donor_top[i]),
                "original_recipient_top": bool(top[i] == expected[0]),
                "changed_recipient_top": bool(top[i] == changed[0]),
                "recipient_logit": float(logits[i, expected[0]]),
                "subject_logit": float(logits[i, changed[0]]),
            }
        )
    return output


def run_alignment_study(model, tokenizer, protocol, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    train_rows = paired_rows("train")
    dev_rows = paired_rows("development")
    final_rows = paired_rows("final")
    if len({r["prompt"] for r in train_rows + dev_rows + final_rows}) != len(train_rows) + len(
        dev_rows
    ) + len(final_rows):
        raise ValueError("Prompt splits overlap.")
    train = capture_rows(model, tokenizer, train_rows, position_mode="prediction")
    development = capture_rows(model, tokenizer, dev_rows, position_mode="prediction")
    results = {
        "protocol": protocol,
        "rows": {"train": train_rows, "development": dev_rows, "final": final_rows},
        "pairs": {},
        "outcomes": [],
        "forward_rows": len(train_rows) + len(dev_rows),
        "final_opened": False,
        "complete": False,
    }
    fits = {}
    raw = {}
    raw_development = {}

    def save():
        (output / "alignment_study.json.gz").write_bytes(
            gzip.compress(json.dumps(results, separators=(",", ":")).encode(), mtime=0)
        )

    # Refit and display development first. Its already-frozen choices remain unchanged.
    for source_layer, target_layer in protocol["layer_pairs"]:
        key = f"{source_layer}_to_{target_layer}"
        x = train["activations"][:, source_layer].double()
        y = train["activations"][:, target_layer].double()
        candidates = fit_candidates(x, y)
        fits[key] = candidates
        source = development["activations"][:, source_layer].double()
        target = development["activations"][:, target_layer].double()
        donors = t.tensor(donor_indices(dev_rows))
        delta = source[donors] - source
        donor_logits = replace_rows(
            model, tokenizer, dev_rows, target_layer, target[donors], position_mode="prediction"
        )
        results["forward_rows"] += len(dev_rows)
        summaries = []
        raw_development[key] = {"native_donor": donor_logits}
        for candidate in candidates:
            logits = replace_rows(
                model,
                tokenizer,
                dev_rows,
                target_layer,
                target + delta @ candidate["map"],
                position_mode="prediction",
            )
            results["forward_rows"] += len(dev_rows)
            raw_development[key][candidate["name"]] = logits
            outcome = answer_record(
                dev_rows,
                logits,
                donor_logits,
                tokenizer,
                candidate["name"],
                source_layer,
                target_layer,
            )
            results["outcomes"].extend(outcome)
            summaries.append(
                {
                    "name": candidate["name"],
                    "method": candidate["method"],
                    "rank": candidate["effective_rank"],
                    "mean_kl": sum(r["kl_to_native_donor"] for r in outcome) / len(outcome),
                }
            )
        results["pairs"][key] = {
            "source_layer": source_layer,
            "target_layer": target_layer,
            "development": summaries,
            "final": [],
            "selected": protocol["selected"][key],
        }
        save()
        print(key, "development complete", flush=True)
    # First final-model access occurs only after all fitting and development work.
    final = capture_rows(model, tokenizer, final_rows, position_mode="prediction")
    results["final_opened"] = True
    results["forward_rows"] += len(final_rows)
    donors = t.tensor(donor_indices(final_rows))
    for pair_index, (source_layer, target_layer) in enumerate(protocol["layer_pairs"]):
        key = f"{source_layer}_to_{target_layer}"
        source = final["activations"][:, source_layer].double()
        target = final["activations"][:, target_layer].double()
        delta = source[donors] - source
        native = replace_rows(
            model, tokenizer, final_rows, target_layer, target[donors], position_mode="prediction"
        )
        noop = replace_rows(
            model, tokenizer, final_rows, target_layer, target, position_mode="prediction"
        )
        results["forward_rows"] += 2 * len(final_rows)
        if not t.equal(noop, final["logits"]):
            raise ValueError("Prediction-position no-op changed the actual baseline.")
        raw[key] = {"baseline": final["logits"], "no_op": noop, "native_donor": native}
        for condition, logits in raw[key].items():
            results["outcomes"].extend(
                answer_record(
                    final_rows, logits, native, tokenizer, condition, source_layer, target_layer
                )
            )
        for candidate in fits[key]:
            predicted = delta @ candidate["map"]
            measured = replace_rows(
                model,
                tokenizer,
                final_rows,
                target_layer,
                target + predicted,
                position_mode="prediction",
            )
            results["forward_rows"] += len(final_rows)
            raw[key][candidate["name"]] = measured
            outcome = answer_record(
                final_rows,
                measured,
                native,
                tokenizer,
                candidate["name"],
                source_layer,
                target_layer,
            )
            results["outcomes"].extend(outcome)
            summary = {
                "name": candidate["name"],
                "method": candidate["method"],
                "requested_rank": candidate["requested_rank"],
                "effective_rank": candidate["effective_rank"],
                "mean_kl": sum(r["kl_to_native_donor"] for r in outcome) / len(outcome),
                "native_top_agreement": sum(r["native_top_agreement"] for r in outcome)
                / len(outcome),
            }
            if "fit" in candidate:
                k = candidate["effective_rank"]
                summary["training_correlations"] = candidate["fit"]["correlations"][:k].tolist()
                summary["final_correlations"] = m.heldout_correlations(
                    source, target, candidate["fit"]
                )[:k].tolist()
            results["pairs"][key]["final"].append(summary)
        selected = next(c for c in fits[key] if c["name"] == protocol["selected"][key]["cca"])
        norm = (delta @ selected["map"]).norm(dim=-1, keepdim=True)
        for draw in range(protocol["null_draws"]):
            direction = t.randn(
                target.shape,
                generator=t.Generator().manual_seed(
                    protocol["null_seed"] + pair_index * 100 + draw
                ),
                dtype=t.float64,
            )
            replacement = target + direction / direction.norm(dim=-1, keepdim=True) * norm
            logits = replace_rows(
                model, tokenizer, final_rows, target_layer, replacement, position_mode="prediction"
            )
            results["forward_rows"] += len(final_rows)
            raw[key][f"random_{draw}"] = logits
            results["outcomes"].extend(
                answer_record(
                    final_rows,
                    logits,
                    native,
                    tokenizer,
                    "same_norm_random",
                    source_layer,
                    target_layer,
                    draw,
                )
            )
        save()
        print(key, "all final candidates and random controls complete", flush=True)
    t.save(
        {
            "train": train,
            "development": development,
            "final": final,
            "fits": fits,
            "logits": raw,
            "development_logits": raw_development,
        },
        output / "actual_tensors.pt",
    )
    results.update(complete=True, retained_outcomes=len(results["outcomes"]))
    save()
    return results, {
        "train": train,
        "development": development,
        "final": final,
        "fits": fits,
        "logits": raw,
        "development_logits": raw_development,
    }
