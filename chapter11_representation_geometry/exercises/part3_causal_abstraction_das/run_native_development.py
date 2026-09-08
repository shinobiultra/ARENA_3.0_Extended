"""Native continuation checks, then optional train/development-only DAS runs.

Never construct or execute the final split from this development entry point.
Each invocation owns a new persistent output directory and snapshots its inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import torch as t
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das import (
    das_methods as m,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das import (
    das_ioi_data as data,
)
from chapter11_representation_geometry.exercises.part3_causal_abstraction_das.das_native import (
    capture_native,
    continue_native,
    full_forward_intervention,
    NativeCache,
)


def weight_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_native(model, tokenizer):
    """Actual pretrained logits, padding invariance and independent derivatives."""
    rows = data.prompt_rows("train")
    rows = [rows[i] for i in (0, 1, 4, 5, 24, 49)]
    cache = capture_native(model, tokenizer, rows, batch_size=6)
    indices = t.arange(len(rows))
    sources = t.tensor([1, 0, 3, 2, 5, 4])
    before = weight_digest(model)
    baseline = model(**cache.tokens, use_cache=False).logits[:, -1].detach()
    result = {
        "scope": "pretrained_native_numerical_checks_only",
        "rows": [row["id"] for row in rows],
        "final_opened": False,
        "capture_vs_full_max_error": float((cache.logits - baseline).abs().max()),
        "conditions": [],
    }
    t.testing.assert_close(cache.logits, baseline, atol=3e-5, rtol=3e-5)
    for layer in (3, 6, 9):
        for mode in ("subject", "end"):
            activation = cache.activations(layer, mode)
            for name, replacement in (("noop", activation), ("donor", activation[sources])):
                with t.no_grad():
                    tail = continue_native(model, cache, layer, indices, replacement, mode)
                    full = full_forward_intervention(
                        model, cache, layer, indices, replacement, mode
                    )
                error = float((tail - full).abs().max())
                t.testing.assert_close(tail, full, atol=3e-5, rtol=3e-5)
                if name == "noop":
                    t.testing.assert_close(tail, baseline, atol=3e-5, rtol=3e-5)
                result["conditions"].append(
                    dict(layer=layer, position=mode, intervention=name, max_logit_error=error)
                )

    # Single-row forward changes padding length. Word spans must still refer to
    # the repeated name, and each last-token distribution must remain the same.
    individual = t.cat(
        [capture_native(model, tokenizer, [row], layers=(6,)).logits for row in rows]
    )
    result["padding_invariance_max_error"] = float((individual - baseline).abs().max())
    t.testing.assert_close(individual, baseline, atol=8e-5, rtol=3e-5)

    # Double precision is used only for numerical differentiation. Training and
    # all ordinary GPT-2 evidence use the checkpoint's FP32 values.
    model.double()
    double_cache = capture_native(model, tokenizer, rows[:2], layers=(6,), batch_size=2)
    base = double_cache.activations(6, "subject")
    source = base.flip(0)
    ids = t.arange(2)
    targets = t.tensor(
        [tokenizer.encode(" " + row["b"], add_special_tokens=False)[0] for row in rows[:2]]
    )
    generator = t.Generator().manual_seed(621)
    raw = (t.randn(768, 2, dtype=t.float64, generator=generator) / 768**0.5).requires_grad_()
    direction = t.randn(raw.shape, dtype=raw.dtype, generator=generator)
    direction /= direction.norm()
    q = m.orthonormal_columns(raw)
    replacement = m.subspace_interchange(base, source, q)
    tail_loss = m.intervention_loss(
        continue_native(model, double_cache, 6, ids, replacement), targets
    )
    gradient = t.autograd.grad(tail_loss, raw)[0]
    tail_derivative = float((gradient * direction).sum())
    q_full = m.orthonormal_columns(raw)
    full_replacement = m.subspace_interchange(base, source, q_full)
    full_loss = t.nn.functional.cross_entropy(
        full_forward_intervention(model, double_cache, 6, ids, full_replacement), targets
    )
    full_gradient = t.autograd.grad(full_loss, raw)[0]
    t.testing.assert_close(gradient, full_gradient, atol=1e-9, rtol=1e-7)

    # The oracle uses A (A.T A)^-1 A.T rather than the learner's QR or its CE.
    def independent_loss(matrix):
        coordinates = t.linalg.solve(matrix.T @ matrix, (source - base).matmul(matrix).T).T
        changed = base + coordinates @ matrix.T
        with t.no_grad():
            return float(
                t.nn.functional.cross_entropy(
                    full_forward_intervention(model, double_cache, 6, ids, changed), targets
                )
            )

    eps = 1e-3
    finite = (
        independent_loss(raw.detach() + eps * direction)
        - independent_loss(raw.detach() - eps * direction)
    ) / (2 * eps)
    if not gradient.norm() > 1e-7 or abs(tail_derivative - finite) > 1e-5 + 1e-4 * abs(finite):
        raise AssertionError(f"Native gradient mismatch: {tail_derivative=}, {finite=}")
    result["gradient"] = dict(
        dtype="float64_numerical_check_only",
        norm=float(gradient.norm()),
        native_hook_max_error=float((gradient - full_gradient).abs().max()),
        analytic_directional_derivative=tail_derivative,
        independent_finite_difference=finite,
        epsilon=eps,
    )
    model.float()

    # A failure inside the native forward must not leave an intervention hook.
    block = model.transformer.h[6]
    initial_hooks = len(block._forward_pre_hooks)
    try:
        full_forward_intervention(model, cache, 6, indices, t.zeros(2, 3))
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("Malformed replacement should fail.")
    assert len(block._forward_pre_hooks) == initial_hooks
    after = weight_digest(model)
    assert before == after
    assert all(
        parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()
    )
    result.update(
        weights_sha256=before,
        weights_unchanged=True,
        model_gradients_absent=True,
        hook_cleanup_pass=True,
    )
    return result, cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ["das_methods.py", "das_native.py", "das_ioi_data.py", "run_native_development.py"]
    digests = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in names:
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    state = dict(
        status="RUNNING",
        scope="native_path_validation",
        final_opened=False,
        source_inputs=digests,
        model_id=data.MODEL_ID,
        revision=data.REVISION,
        transformers_version=transformers.__version__,
        torch_version=t.__version__,
        device="cpu",
        dtype="float32",
        training_started=False,
    )
    started = time.perf_counter()

    def save():
        (output / "development.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        t.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(
            data.MODEL_ID, revision=data.REVISION, local_files_only=True
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
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
        data.name_token_ids(tokenizer, [])
        native_source = Path(inspect.getfile(type(model.transformer)))
        state["native_implementation_sha256"] = hashlib.sha256(
            native_source.read_bytes()
        ).hexdigest()
        result, cache = validate_native(model, tokenizer)
        t.save(
            dict(
                tokens=cache.tokens,
                subject_positions=cache.subject_positions,
                hidden=cache.hidden,
                causal_mask=cache.causal_mask,
                logits=cache.logits,
            ),
            output / "actual_native_checks.pt",
        )
        state.update(status="PASS", native_checks=result)
    except BaseException as error:
        state.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        state["changed_sources"] = [
            name
            for name, digest in digests.items()
            if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest
        ]
        if state["changed_sources"]:
            state["status"] = "FAIL"
        save()
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
