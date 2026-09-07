"""Fresh native reproduction of every frozen final DAS intervention.

The completed learner supplies the actual bases and visible intervention code.
This process reloads the pinned model, recaptures all final prompts and compares
every full-vocabulary output. It never uses a stored report as model evidence.
"""

from __future__ import annotations

import argparse
import ast
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
    das_protocol as protocol,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.build_notebooks import (
    lesson_cells,
    complete_learner,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.das_native import (
    capture_native,
    require_frozen,
    full_forward_intervention,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.run_native_development import (
    weight_digest,
)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_payload(saved):
    """Require every final condition, complete finite logits and valid subspaces."""
    names = {"No intervention", "Role probe", "Mean direction"}
    names |= {f"{kind}/seed{seed}" for kind in ("DAS", "Shuffled") for seed in protocol.SEEDS}
    names |= {f"Random/seed{seed}" for seed in protocol.RANDOM_SEEDS}
    if set(saved["final_bases"]) != names or set(saved["final_logits"]) != names | {
        "Whole donor",
        "Explicit counterfactual",
    }:
        raise ValueError("The learner must retain every seed, baseline and random control.")
    for name, logits in saved["final_logits"].items():
        if logits.shape != (len(saved["final_pairs"]), 50257) or not t.isfinite(logits).all():
            raise ValueError(f"Incomplete or invalid full-vocabulary logits: {name}")
    for name, basis in saved["final_bases"].items():
        rank = (
            0
            if name == "No intervention"
            else 1
            if name in ("Role probe", "Mean direction")
            else protocol.RANK
        )
        if basis.shape != (768, rank) or not t.isfinite(basis).all():
            raise ValueError(f"Incorrect subspace shape: {name}")
        t.testing.assert_close(basis.T @ basis, t.eye(rank), atol=1e-5, rtol=1e-5)


def validate_learner(directory):
    execution = json.loads((directory / "execution.json").read_text())
    if (
        execution["status"] != "PASS"
        or execution["changed_sources"]
        or not execution["pretrained_model_executed"]
    ):
        raise ValueError("Require a successful, unchanged, pretrained own-answer execution.")
    # Prose-only edits can be reviewed separately. Every executed code AST must
    # still match the current learner completed from its own solution dropdowns.
    notebook = json.loads((directory / "completed_learner.ipynb").read_text())
    current = complete_learner({"cells": lesson_cells()})

    def code_cells(book):
        return ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]

    actual, expected = code_cells(notebook), code_cells(current)
    if len(actual) != len(expected) or any(
        ast.dump(ast.parse(a)) != ast.dump(ast.parse(b)) for a, b in zip(actual, expected)
    ):
        raise ValueError("The completed learner does not match its current own-answer code.")
    for name, digest in execution["source_inputs"].items():
        if file_digest(directory / "source_snapshot" / name) != digest:
            raise ValueError(f"Altered execution source snapshot: {name}")
        if name != "lesson.md" and file_digest(HERE / name) != digest:
            raise ValueError(f"Execution input changed: {name}")
    tensor_path = directory / "figures" / "actual_native_tensors.pt"
    if file_digest(tensor_path) != execution["native_tensor_sha256"]:
        raise ValueError("The learner's actual native tensors changed.")
    saved = t.load(tensor_path, map_location="cpu", weights_only=True)
    rows, pairs = data.prompt_rows("final"), data.interchange_pairs(data.prompt_rows("final"))
    if saved["final_rows"] != rows or saved["final_pairs"] != pairs:
        raise ValueError("The learner's final intervention distribution changed.")
    selected = execution["native_summary"]["selected"]
    if selected != dict(layer=protocol.LAYER, position=protocol.POSITION, rank=protocol.RANK):
        raise ValueError("The learner did not use the frozen development-selected protocol.")
    validate_payload(saved)

    # Execute only these visible definitions, not the full learner a second time.
    methods = {"t": t, "require_frozen": require_frozen}
    wanted = {"subspace_interchange", "continue_native"}
    found = set()
    for source in actual:
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]), "<completed learner>", "exec"
                    ),
                    methods,
                )
                found.add(node.name)
    if found != wanted:
        raise ValueError("Required visible native intervention functions are missing.")
    return execution, saved, methods


def reproduce(learner, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = [
        "das_reproduction.py",
        "das_native.py",
        "das_ioi_data.py",
        "das_protocol.py",
        "run_native_development.py",
        "lesson.md",
        "build_notebooks.py",
    ]
    hashes = {name: file_digest(HERE / name) for name in inputs}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in inputs:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    state = dict(
        status="RUNNING",
        scope="fresh_native_all_final_interventions",
        native_model_executed=False,
        source_inputs=hashes,
        learner_directory=str(learner),
        device="cpu",
        dtype="float32",
        model_id=data.MODEL_ID,
        revision=data.REVISION,
        conditions=[],
        actual_intervention_rows=0,
    )
    start = time.perf_counter()

    def save():
        state["elapsed_seconds"] = time.perf_counter() - start
        (output / "reproduction.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        t.set_num_threads(4)
        execution, saved, methods = validate_learner(learner)
        state["learner_execution_sha256"] = file_digest(learner / "execution.json")
        state["learner_tensor_sha256"] = execution["native_tensor_sha256"]
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
        digest = weight_digest(model)
        if digest != saved["native_weights_sha256"]:
            raise ValueError("Fresh native model weights differ from the learner's checkpoint.")
        rows, pairs = saved["final_rows"], saved["final_pairs"]
        name_ids = data.name_token_ids(tokenizer, rows)
        target_ids = t.tensor([name_ids[pair["target"]] for pair in pairs])
        changed_cross = t.tensor([pair["changed_role"] and pair["cross_name"] for pair in pairs])
        cache = capture_native(model, tokenizer, rows, layers=(protocol.LAYER,))
        state["native_model_executed"] = True
        baseline_error = float((cache.logits - saved["native_final_logits"]).abs().max())
        if baseline_error > 3e-4 or not t.equal(
            cache.logits.argmax(-1), saved["native_final_logits"].argmax(-1)
        ):
            raise ValueError("Fresh native baseline does not reproduce the learner.")
        state["baseline_max_logit_error"] = baseline_error
        activation = cache.activations(protocol.LAYER, protocol.POSITION)
        actual_outputs = {}
        with (output / "actual_comparisons.jsonl").open("w") as comparison_file, t.no_grad():
            for name, expected in saved["final_logits"].items():
                outputs, hook_errors = [], []
                for first in range(0, len(pairs), 24):
                    batch = pairs[first : first + 24]
                    base = t.tensor([p["base"] for p in batch])
                    source = t.tensor([p["source"] for p in batch])
                    if name == "Explicit counterfactual":
                        result = cache.logits[t.tensor([p["counterfactual"] for p in batch])]
                    else:
                        replacement = (
                            activation[source]
                            if name == "Whole donor"
                            else methods["subspace_interchange"](
                                activation[base], activation[source], saved["final_bases"][name]
                            )
                        )
                        result = methods["continue_native"](
                            model, cache, protocol.LAYER, base, replacement, protocol.POSITION
                        )
                        if first == 0:
                            hook = full_forward_intervention(
                                model, cache, protocol.LAYER, base, replacement, protocol.POSITION
                            )
                            hook_errors.append(float((hook - result).abs().max()))
                            if hook_errors[-1] > 3e-4:
                                raise ValueError(f"Native full-forward hook mismatch: {name}")
                    outputs.append(result)
                actual = t.cat(outputs)
                errors = (actual - expected).abs().amax(-1)
                mismatches = actual.argmax(-1) != expected.argmax(-1)
                if errors.max() > 3e-4 or mismatches.any():
                    raise ValueError(f"Fresh final output mismatch: {name}")
                for index, pair in enumerate(pairs):
                    comparison_file.write(
                        json.dumps(
                            dict(
                                condition=name,
                                pair_id=pair["id"],
                                pair_index=index,
                                max_full_vocabulary_logit_error=float(errors[index]),
                                actual_top_token_id=int(actual[index].argmax()),
                                learner_top_token_id=int(expected[index].argmax()),
                            )
                        )
                        + "\n"
                    )
                actual_outputs[name] = actual
                correct = actual.argmax(-1) == target_ids
                cross_iia = float(correct[changed_cross].double().mean())
                reported = execution["native_summary"]["final_metrics"][name]["cross_changed"]
                if (
                    reported["n"] != int(changed_cross.sum())
                    or abs(reported["iia"] - cross_iia) > 1e-12
                ):
                    raise ValueError(
                        f"The learner's primary metric disagrees with actual native tokens: {name}"
                    )
                state["actual_intervention_rows"] += len(pairs)
                state["conditions"].append(
                    dict(
                        name=name,
                        rows=len(pairs),
                        max_logit_error=float(errors.max()),
                        top_token_mismatches=int(mismatches.sum()),
                        native_hook_max_logit_error=max(hook_errors, default=0),
                        cross_changed_iia=cross_iia,
                        cross_changed_count=int(changed_cross.sum()),
                    )
                )
                save()
        t.save(
            dict(
                final_rows=rows,
                final_pairs=pairs,
                native_logits=cache.logits,
                actual_logits=actual_outputs,
            ),
            output / "actual_fresh_native_tensors.pt",
        )
        state["actual_tensor_sha256"] = file_digest(output / "actual_fresh_native_tensors.pt")
        state["actual_comparisons_sha256"] = file_digest(output / "actual_comparisons.jsonl")
        state["native_tail_forward_rows"] = len(pairs) * (len(actual_outputs) - 1)
        state["cached_counterfactual_comparison_rows"] = len(pairs)
        state["total_native_forward_rows"] = len(rows) + (len(pairs) + 24) * (
            len(actual_outputs) - 1
        )
        state.update(
            status="PASS",
            weights_unchanged=weight_digest(model) == digest,
            model_gradients_absent=all(p.grad is None for p in model.parameters()),
            baseline_prompt_rows=len(rows),
            full_native_hook_rows=24 * (len(actual_outputs) - 1),
        )
        if not state["weights_unchanged"] or not state["model_gradients_absent"]:
            raise ValueError("Fresh native reproduction changed model weights or gradients.")
    except BaseException as error:
        state.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["changed_sources"] = [
            name for name, digest in hashes.items() if file_digest(HERE / name) != digest
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        save()
    print(json.dumps(state, indent=2))
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.learner_directory.resolve(), args.output_directory.resolve())
