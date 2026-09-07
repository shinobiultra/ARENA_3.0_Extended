"""CPU native DAS layer/position screening. Training and development only."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das import (
    das_ioi_data as data,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das import (
    das_methods as m,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.das_native import (
    capture_native,
    continue_native,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.das_training_methods import (
    fit_minibatch_alignment,
    fit_role_probe,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.run_native_development import (
    weight_digest,
)


def summarize(logits, pairs, rows, names, native_logits):
    targets = t.tensor([names[p["target"]] for p in pairs])
    base = t.tensor([p["base"] for p in pairs])
    source = t.tensor([p["source"] for p in pairs])
    cf = t.tensor([p["counterfactual"] for p in pairs])
    native_correct = native_logits.argmax(-1).eq(t.tensor([names[r["answer"]] for r in rows]))
    changed = t.tensor([p["changed_role"] for p in pairs])
    cross = t.tensor([p["cross_name"] for p in pairs])
    result = {}
    for name, mask in {
        "all": t.ones(len(pairs), dtype=t.bool),
        "changed": changed,
        "cross_changed": cross & changed,
        "cross_unchanged": cross & ~changed,
        "same_changed": ~cross & changed,
        "both_native_correct_cross_changed": native_correct[base]
        & native_correct[cf]
        & cross
        & changed,
    }.items():
        result[name] = dict(n=int(mask.sum()), iia=m.interchange_accuracy(logits, targets, mask))
    probabilities = logits.softmax(-1)
    result["target_probability"] = float(probabilities[t.arange(len(pairs)), targets].mean())
    result["cross_changed_target_probability"] = float(
        probabilities[t.arange(len(pairs)), targets][cross & changed].mean()
    )
    donor_answers = t.tensor([names[rows[i]["answer"]] for i in source])
    result["cross_donor_answer_probability"] = float(
        probabilities[t.arange(len(pairs)), donor_answers][cross].mean()
    )
    result["full_vocabulary_ce"] = float(m.intervention_loss(logits, targets))
    result["native_prompt_accuracy"] = float(native_correct.float().mean())
    result["top_token_ids"] = logits.argmax(-1).tolist()
    return result


@t.no_grad()
def evaluate_basis(model, cache, layer, mode, pairs, basis, batch_size=24, full_donor=False):
    activations = cache.activations(layer, mode)
    outputs = []
    for start in range(0, len(pairs), batch_size):
        selected = pairs[start : start + batch_size]
        base = t.tensor([p["base"] for p in selected])
        source = t.tensor([p["source"] for p in selected])
        replacements = (
            activations[source]
            if full_donor
            else m.subspace_interchange(activations[base], activations[source], basis)
        )
        outputs.append(continue_native(model, cache, layer, base, replacements, mode))
    return t.cat(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9])
    parser.add_argument(
        "--positions", nargs="+", choices=["subject", "end"], default=["subject", "end"]
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--random-draws", type=int, default=32)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_names = [
        "das_ioi_data.py",
        "das_methods.py",
        "das_native.py",
        "das_training_methods.py",
        "run_das_screen.py",
        "run_native_development.py",
    ]
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source_names}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in source_names:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    state = dict(
        status="RUNNING",
        scope="train_and_development_only",
        final_model_executed=False,
        source_inputs=hashes,
        model_id=data.MODEL_ID,
        revision=data.REVISION,
        device="cpu",
        dtype="float32",
        configuration={k: v for k, v in vars(args).items() if k != "output_directory"},
        candidates=[],
    )
    started = time.perf_counter()

    def save():
        state["elapsed_seconds"] = time.perf_counter() - started
        (output / "development.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    tensors, bases = {}, {}
    try:
        t.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(
            data.MODEL_ID, revision=data.REVISION, local_files_only=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = (
            AutoModelForCausalLM.from_pretrained(
                data.MODEL_ID,
                revision=data.REVISION,
                local_files_only=True,
                dtype=t.float32,
                attn_implementation="eager",
            )
            .cpu()
            .eval()
        )
        model.requires_grad_(False)
        weight_before = weight_digest(model)
        train_rows, dev_rows = data.prompt_rows("train"), data.prompt_rows("development")
        train_pairs, dev_pairs = (
            data.interchange_pairs(train_rows),
            data.interchange_pairs(dev_rows),
        )
        names = data.name_token_ids(tokenizer, train_rows)
        train = capture_native(model, tokenizer, train_rows, layers=tuple(args.layers))
        dev = capture_native(model, tokenizer, dev_rows, layers=tuple(args.layers))
        tensors.update(
            train_rows=train_rows,
            development_rows=dev_rows,
            train_pairs=train_pairs,
            development_pairs=dev_pairs,
            train_logits=train.logits,
            development_logits=dev.logits,
        )
        state["sizes"] = dict(
            train_prompts=len(train_rows),
            development_prompts=len(dev_rows),
            train_pairs=len(train_pairs),
            development_pairs=len(dev_pairs),
        )
        base_indices = t.tensor([p["base"] for p in train_pairs])
        source_indices = t.tensor([p["source"] for p in train_pairs])
        targets = t.tensor([names[p["target"]] for p in train_pairs])
        dev_base = t.tensor([p["base"] for p in dev_pairs])
        state["noop"] = summarize(dev.logits[dev_base], dev_pairs, dev_rows, names, dev.logits)
        state["native_counterfactual"] = summarize(
            dev.logits[t.tensor([p["counterfactual"] for p in dev_pairs])],
            dev_pairs,
            dev_rows,
            names,
            dev.logits,
        )
        save()

        def record(name, layer, mode, logits, extra):
            metrics = summarize(logits, dev_pairs, dev_rows, names, dev.logits)
            state["candidates"].append(
                dict(name=name, layer=layer, position=mode, metrics=metrics, **extra)
            )
            tensors[name] = logits
            save()
            print(
                name,
                "cross-changed IIA",
                metrics["cross_changed"],
                "CE",
                metrics["full_vocabulary_ce"],
                flush=True,
            )

        for layer in args.layers:
            for mode in args.positions:
                train_act = train.activations(layer, mode)
                tensors[f"train_activations_L{layer}_{mode}"] = train_act
                tensors[f"development_activations_L{layer}_{mode}"] = dev.activations(layer, mode)
                prefix = f"L{layer}_{mode}"
                full = evaluate_basis(model, dev, layer, mode, dev_pairs, None, full_donor=True)
                record(prefix + "_full_donor", layer, mode, full, dict(method="full_donor"))
                roles = t.tensor([r["role"] for r in train_rows])
                probe = fit_role_probe(train_act, roles)
                probe_logits = evaluate_basis(model, dev, layer, mode, dev_pairs, probe["basis"])
                dev_prediction = (
                    (dev.activations(layer, mode) - probe["x_mean"]) @ probe["coefficient"]
                    + probe["y_mean"]
                ) > 0
                probe_accuracy = float(
                    dev_prediction.eq(t.tensor([r["role"] for r in dev_rows])).float().mean()
                )
                record(
                    prefix + "_role_probe",
                    layer,
                    mode,
                    probe_logits,
                    dict(method="role_probe", observational_development_accuracy=probe_accuracy),
                )
                bases[prefix + "_role_probe"] = probe
                mean_difference = train_act[roles == 1].mean(0) - train_act[roles == 0].mean(0)
                mean_basis = (mean_difference / mean_difference.norm())[:, None]
                mean_logits = evaluate_basis(model, dev, layer, mode, dev_pairs, mean_basis)
                record(
                    prefix + "_mean_direction",
                    layer,
                    mode,
                    mean_logits,
                    dict(method="mean_direction", rank=1),
                )
                bases[prefix + "_mean_direction"] = {"basis": mean_basis}

                def tail(pair_indices, replacement):
                    return continue_native(
                        model, train, layer, base_indices[pair_indices], replacement, mode
                    )

                for rank in args.ranks:
                    for seed in args.seeds:
                        for shuffled in [False, True] if args.controls else [False]:
                            label = "shuffled" if shuffled else "das"
                            name = f"{prefix}_{label}_rank{rank}_seed{seed}"
                            objective = (
                                targets[
                                    t.randperm(
                                        len(targets),
                                        generator=t.Generator().manual_seed(400 + seed),
                                    )
                                ]
                                if shuffled
                                else targets
                            )
                            fit = fit_minibatch_alignment(
                                train_act[base_indices],
                                train_act[source_indices],
                                objective,
                                tail,
                                rank=rank,
                                steps=args.steps,
                                learning_rate=args.learning_rate,
                                seed=seed,
                            )
                            bases[name] = fit
                            logits = evaluate_basis(
                                model, dev, layer, mode, dev_pairs, fit["basis"]
                            )
                            record(
                                name,
                                layer,
                                mode,
                                logits,
                                dict(method=label, rank=rank, seed=seed, history=fit["history"]),
                            )
                            t.save(bases, output / "actual_bases.pt")
                    if args.controls:
                        for draw in range(args.random_draws):
                            name = f"{prefix}_random_rank{rank}_draw{draw}"
                            q = m.orthonormal_columns(
                                t.randn(
                                    train_act.shape[1],
                                    rank,
                                    generator=t.Generator().manual_seed(700 + draw),
                                )
                            )
                            bases[name] = {"basis": q}
                            logits = evaluate_basis(model, dev, layer, mode, dev_pairs, q)
                            record(
                                name,
                                layer,
                                mode,
                                logits,
                                dict(method="random", rank=rank, seed=700 + draw),
                            )
        state.update(
            status="PASS",
            weights_unchanged=weight_before == weight_digest(model),
            model_gradients_absent=all(p.grad is None for p in model.parameters()),
            weights_sha256=weight_before,
        )
        if not state["weights_unchanged"] or not state["model_gradients_absent"]:
            raise AssertionError("DAS must leave every GPT-2 parameter frozen.")
    except BaseException as error:
        state.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        t.save(tensors, output / "actual_logits_and_activations.pt")
        t.save(bases, output / "actual_bases.pt")
        state["changed_sources"] = [
            name
            for name, digest in hashes.items()
            if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        save()


if __name__ == "__main__":
    main()
