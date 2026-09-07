"""Development-only native Pythia experiment; never read final prompts here."""

from __future__ import annotations
import argparse, json, sys, time, hashlib
from pathlib import Path
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment import (
    alignment_methods as m,
)
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.alignment_data import (
    MODEL_ID,
    REVISION,
    paired_rows,
    donor_indices,
)
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.alignment_model import (
    capture_rows,
    replace_rows,
)


def fit_candidates(x, y):
    """Each candidate's parameters are learned from the paired training rows."""
    candidates = []
    cca = m.fit_cca(x, y, max_rank=32)
    svcca = m.fit_cca(x, y, variance_fraction=0.99)
    shuffled = m.fit_cca(
        x, y[t.randperm(len(y), generator=t.Generator().manual_seed(910))], max_rank=32
    )
    for requested in [2, 4, 8, 16]:
        for method, fit in [("cca", cca), ("svcca", svcca), ("shuffled_cca", shuffled)]:
            k = min(requested, len(fit["correlations"]))
            mapping = m.cca_difference_transport(
                t.eye(x.shape[1], device=x.device, dtype=x.dtype), fit, k
            )
            candidates.append(
                {
                    "name": f"{method}_rank{requested}",
                    "method": method,
                    "requested_rank": requested,
                    "effective_rank": k,
                    "map": mapping,
                    "fit": fit,
                }
            )
        px = m.svd_whitener(x, max_rank=requested)
        py = m.svd_whitener(y, max_rank=requested)
        k = min(px["retained_rank"], py["retained_rank"])
        bx = px["basis"][:, :k]
        by = py["basis"][:, :k]
        rigid = m.fit_procrustes((x - px["mean"]) @ bx, (y - py["mean"]) @ by)
        candidates.append(
            {
                "name": f"procrustes_rank{requested}",
                "method": "procrustes",
                "requested_rank": requested,
                "effective_rank": k,
                "map": bx @ rigid["rotation"] @ by.T,
            }
        )
    ridge = m.fit_ridge_transport(x, y, 0.1)
    candidates.append(
        {
            "name": "ridge_0.1",
            "method": "ridge",
            "requested_rank": None,
            "effective_rank": int(t.linalg.matrix_rank(ridge["weights"])),
            "map": ridge["weights"],
        }
    )
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--task", choices=["pythia", "gpt2_ioi"], default="pythia")
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment import (
        alignment_data,
        alignment_ioi_data,
    )

    data = alignment_ioi_data if args.task == "gpt2_ioi" else alignment_data
    model_id, revision = data.MODEL_ID, data.REVISION
    position_mode = "prediction" if args.task == "gpt2_ioi" else "word"
    pairs = [(2, 6), (4, 8), (6, 10)] if args.task == "gpt2_ioi" else [(0, 3), (1, 4), (2, 5)]
    source_names = [
        "alignment_methods.py",
        "alignment_data.py",
        "alignment_ioi_data.py",
        "alignment_model.py",
        "run_alignment_development.py",
    ]
    hashes = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in source_names
    }
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in source_names:
        (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())
    state = {
        "status": "RUNNING",
        "source_inputs": hashes,
        "scope": "train_and_development_only",
        "pairs": [],
        "model_id": model_id,
        "revision": revision,
        "task": args.task,
        "position_mode": position_mode,
        "device": "cpu",
        "dtype": "float32",
        "final_opened": False,
    }

    def save():
        (output / "development.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    started = time.time()
    t.set_num_threads(4)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=True,
                dtype=t.float32,
                attn_implementation="eager",
            )
            .cpu()
            .eval()
        )
        train_rows = data.paired_rows("train")
        dev_rows = data.paired_rows("development")
        train = capture_rows(model, tokenizer, train_rows, position_mode=position_mode)
        dev = capture_rows(model, tokenizer, dev_rows, position_mode=position_mode)
        t.save(
            {
                "train": train,
                "development": dev,
                "train_rows": train_rows,
                "development_rows": dev_rows,
            },
            output / "actual_captures.pt",
        )
        donors = t.tensor(data.donor_indices(dev_rows))
        all_fits = {}
        all_logits = {}
        for source_layer, target_layer in pairs:
            pair = f"{source_layer}_to_{target_layer}"
            x = train["activations"][:, source_layer].double()
            y = train["activations"][:, target_layer].double()
            source = dev["activations"][:, source_layer].double()
            target = dev["activations"][:, target_layer].double()
            delta = source[donors] - source
            native = replace_rows(
                model,
                tokenizer,
                dev_rows,
                target_layer,
                target[donors],
                position_mode=position_mode,
            )
            noop = replace_rows(
                model, tokenizer, dev_rows, target_layer, target, position_mode=position_mode
            )
            if not t.equal(noop, dev["logits"]):
                raise ValueError("Native no-op must exactly preserve original logits.")
            baseline_kl = m.distribution_kl(native, dev["logits"])
            record = {
                "source_layer": source_layer,
                "target_layer": target_layer,
                "baseline_kl_mean": float(baseline_kl.mean()),
                "no_op_max_error": 0.0,
                "candidates": [],
            }
            candidates = fit_candidates(x, y)
            all_fits[pair] = candidates
            all_logits[pair] = {"native_donor": native, "baseline": dev["logits"]}
            for candidate in candidates:
                replacement = target + delta @ candidate["map"]
                measured = replace_rows(
                    model,
                    tokenizer,
                    dev_rows,
                    target_layer,
                    replacement,
                    position_mode=position_mode,
                )
                all_logits[pair][candidate["name"]] = measured
                kl = m.distribution_kl(native, measured)
                saved = {
                    key: value for key, value in candidate.items() if key not in ("map", "fit")
                }
                saved.update(
                    kl_mean=float(kl.mean()),
                    kl_rows=kl.tolist(),
                    native_top_token_agreement=float(
                        measured.argmax(-1).eq(native.argmax(-1)).float().mean()
                    ),
                    predicted_delta_mse=float((replacement - target[donors]).square().mean()),
                    actual_top_token_ids=measured.argmax(-1).tolist(),
                )
                if "fit" in candidate:
                    k = candidate["effective_rank"]
                    saved["train_correlations"] = candidate["fit"]["correlations"][:k].tolist()
                    saved["development_correlations"] = m.heldout_correlations(
                        source, target, candidate["fit"]
                    )[:k].tolist()
                record["candidates"].append(saved)
                print(pair, candidate["name"], "KL", saved["kl_mean"], flush=True)
            eligible = [row for row in record["candidates"] if row["method"] != "shuffled_cca"]
            record["selected_candidate"] = min(
                eligible, key=lambda row: (row["kl_mean"], row["name"])
            )["name"]
            state["pairs"].append(record)
            save()
        t.save(all_fits, output / "training_fits.pt")
        t.save(all_logits, output / "actual_intervention_logits.pt")
        state.update(
            status="PASS",
            train_rows=len(train_rows),
            development_rows=len(dev_rows),
            forward_rows=len(train_rows) + len(dev_rows) + len(dev_rows) * 3 * 19,
        )
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["elapsed_seconds"] = time.time() - started
        state["changed_sources"] = [
            name
            for name, digest in hashes.items()
            if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != digest
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        save()


if __name__ == "__main__":
    main()
