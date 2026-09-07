"""Build the paired learner and solution notebooks for section 15.3."""

from __future__ import annotations

import inspect
import ast
import json
import re
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chapter15_peft_misalignment.exercises.part3_qlora_from_scratch import solutions


HERE = Path(__file__).resolve().parent


def _source(*names: str) -> str:
    return "\n\n".join(inspect.getsource(getattr(solutions, name)).rstrip() for name in names) + "\n"


def _cell(cell_type: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _md(source: str) -> dict:
    return _cell("markdown", source)


def _code(source: str) -> dict:
    return _cell("code", source)


def _solution_dropdown(source: str) -> dict:
    return _md("<details>\n<summary>Solution</summary>\n\n```python\n"
               + source.rstrip() + "\n```\n\n</details>")


def _exercise_code(stub: str, solved: str, *, is_solution: bool) -> dict:
    return _code(complete_from_answer(stub, solved) if is_solution else stub)


def complete_from_answer(source: str, answer: str) -> str:
    """Replace only top-level exercise stubs, retaining supplied helpers verbatim."""

    source, answer = dedent(source).strip() + "\n", dedent(answer).strip() + "\n"
    lines, answer_lines = source.splitlines(True), answer.splitlines(True)
    answers = {}
    for node in ast.parse(answer).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
            answers[node.name] = "".join(answer_lines[start:node.end_lineno])
    for node in reversed(ast.parse(source).body):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
            if "raise NotImplementedError" in "".join(lines[start:node.end_lineno]):
                lines[start:node.end_lineno] = [answers[node.name]]
    return "".join(lines)


def complete_learner(notebook: dict) -> dict:
    """Fill the actual learner using its own following Solution dropdowns."""

    notebook = json.loads(json.dumps(notebook))
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or "raise NotImplementedError" not in source:
            continue
        following = notebook["cells"][index + 1:]
        answer_cell = next(item for item in following if item["cell_type"] == "markdown"
                           and "<summary>Solution</summary>" in "".join(item["source"]))
        answer = "\n\n".join(re.findall(r"```python\n(.*?)```", "".join(answer_cell["source"]), re.S))
        cell["source"] = complete_from_answer(source, answer).splitlines(True)
    return notebook


def build_notebook(*, is_solution: bool) -> dict:
    cells: list[dict] = []
    cells += [
        _md(
            """
            # [15.3] QLoRA from Scratch

            **Can a small trainable adapter recover a useful rule after we quantize
            a pretrained model's frozen weights to four bits?**

            Start with a weight vector you can draw by hand. You will build its NF4
            quantizer, account for the scale metadata, and add a low-rank update.
            Then teach TinyStories a new marker-to-color rule and compare your QLoRA
            against BF16 LoRA and full finetuning. The final panels show behavior,
            memory use, and the singular values of the learned updates.

            Each method appears here as editable code, followed immediately by a
            small test. Complete the exercises or reveal the answers as you go.
            This is an optional QLoRA companion; it does not replace the core
            LoRA, DoRA, subspace, and intruder-dimension material in the roadmap.

            """
        ),
        _md(
            """
            ## Learning Objectives

            1. Implement NF4 lookup quantization, dequantization, and nested scale compression.
            2. Audit exact bitsandbytes codebook values, tensor finiteness, packing, and compute dtype.
            3. Add visible FP32 LoRA factors around a genuinely materialized nested-NF4 base.
            4. Remove name and template confounding with a complete subject-template-label cross.
            5. Evaluate marker swaps, marker removal, name-only, random-label, and same-norm controls.
            6. Distinguish two-label conditional NLL from full-vocabulary next-token NLL.
            7. Compare total resident training state and report a 95% Student-t interval for n=3.
            8. Explain which behavior and memory claims your measured controls support.
            """
        ),
        _md(
            """
            ## Cold Open: Where Do the Missing Bits Go?

            A float32 weight uses 32 bits. NF4 picks one of 16 values, so the index
            needs four bits. But weights in different blocks need different scales,
            and training an adapter also needs gradients and optimizer state.

            Predict first: will a 4-bit base make the whole training process eight
            times smaller? Will adding one large outlier damage its neighbors?
            Keep both predictions; our plots will make the answers visible.
            """
        ),
        _md("## Setup"),
        _code(
            """
            import gc
            import os
            import hashlib
            import inspect
            import json
            import math
            import statistics
            import sys
            import warnings
            from dataclasses import asdict
            from pathlib import Path
            from typing import Any, Callable, Iterable, Sequence

            import bitsandbytes as bnb
            import copy
            import matplotlib.pyplot as plt
            import numpy as np
            import torch as t
            import torch.nn.functional as F
            from IPython.display import Image, display
            from torch import nn

            ROOT = Path.cwd().resolve()
            while not (ROOT / "pyproject.toml").exists():
                if ROOT.parent == ROOT:
                    raise RuntimeError("Run this notebook from inside the ARENA repository")
                ROOT = ROOT.parent
            sys.path.insert(0, str(ROOT))

            from chapter15_peft_misalignment.exercises.part3_qlora_from_scratch import solutions as reference
            from chapter15_peft_misalignment.exercises.part3_qlora_from_scratch import tests
            from chapter15_peft_misalignment.exercises.part3_qlora_from_scratch.solutions import (
                DOUBLE_QUANT_BLOCK_SIZE,
                INTENDED_QUANTIZED_SUFFIXES,
                LABEL_TEXT,
                MARKERS,
                MODEL_ID,
                MODEL_REVISION,
                NF4_BLOCK_SIZE,
                TARGET_MODULES,
                BehaviorMetrics,
                DenseLoRALinear,
                DoubleQuantizedScales,
                LearnerFunctions,
                NF4Tensor,
                PromptExample,
                SpectrumMetrics,
                TrainTrace,
            )

            warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")
            t.manual_seed(0)
            print(f"torch={t.__version__}; bitsandbytes={bnb.__version__}; revision={MODEL_REVISION[:12]}")
            """
        ),
    ]

    nf4_solution = _source(
        "nf4_codebook",
        "expected_dynamic_codebook",
        "blockwise_absmax",
        "quantize_nf4",
        "dequantize_nf4",
        "quantization_error",
    )
    nf4_stub = """
def nf4_codebook(*, device: t.device | str = "cpu", dtype: t.dtype = t.float32) -> t.Tensor:
    raise NotImplementedError()

def expected_dynamic_codebook(*, device: t.device | str = "cpu", dtype: t.dtype = t.float32) -> t.Tensor:
    raise NotImplementedError()

def blockwise_absmax(values: t.Tensor, *, block_size: int = NF4_BLOCK_SIZE) -> t.Tensor:
    raise NotImplementedError()

def quantize_nf4(values: t.Tensor, *, block_size: int = NF4_BLOCK_SIZE) -> NF4Tensor:
    raise NotImplementedError()

def dequantize_nf4(quantized: NF4Tensor, *, dtype: t.dtype = t.float32) -> t.Tensor:
    raise NotImplementedError()

def quantization_error(reference_values: t.Tensor, reconstructed: t.Tensor) -> dict[str, float]:
    raise NotImplementedError()
"""
    cells += [
        _md(
            """
            ## 1. NF4 Is a Lookup Table

            ### Exercise: exact blockwise NF4

            Implement per-block absmax normalization, nearest-code assignment, and
            dequantization from stored codes. Also construct the signed dynamic codebook
            used by the nested scale state.

            <details><summary>Expected output</summary>

            The manual 64-value reconstruction equals bitsandbytes exactly, including
            the all-zero block. The NF4 table has 16 values; the nested dynamic table
            has 256 finite values matching the expected backend-specific map.

            </details>

            <details><summary>Help</summary>

            Pad only after flattening. Clamp zero scales before division. For the
            dynamic table, inspect the bitsandbytes constructor and the endpoints its
            blockwise runtime actually materializes.

            </details>
            """
        ),
        _exercise_code(nf4_stub, nf4_solution, is_solution=is_solution),
        _code(
            """
            tests.check_nf4_codebook(nf4_codebook)
            tests.check_blockwise_absmax(blockwise_absmax)
            tests.check_manual_nf4(quantize_nf4, dequantize_nf4)
            print("Exact NF4 lookup, padding, and zero-block checks passed.")
            """
        ),
        _solution_dropdown(nf4_solution),
    ]

    plumbing_solution = _source(
        "_tensor_storage_key",
        "_unique_tensor_storage_bytes",
        "_quant_state_tensors",
        "_require_finite_tensor",
    )
    nested_solution = _source(
        "double_quantize_scales",
        "dequantize_scales",
        "nf4_storage_accounting",
        "linear4bit_storage_accounting",
        "audit_linear4bit",
    )
    nested_stub = plumbing_solution + """

def double_quantize_scales(scales: t.Tensor, *, block_size: int = DOUBLE_QUANT_BLOCK_SIZE) -> DoubleQuantizedScales:
    raise NotImplementedError()

def dequantize_scales(quantized: DoubleQuantizedScales) -> t.Tensor:
    raise NotImplementedError()

def nf4_storage_accounting(*, numel: int, block_size: int = NF4_BLOCK_SIZE, double_quant_block_size: int = DOUBLE_QUANT_BLOCK_SIZE) -> dict[str, Any]:
    raise NotImplementedError()

def linear4bit_storage_accounting(layer: nn.Module) -> dict[str, Any]:
    raise NotImplementedError()

def audit_linear4bit(layer: nn.Module, *, expected_compute_dtype: t.dtype, expected_storage_dtype: t.dtype = t.uint8) -> dict[str, Any]:
    raise NotImplementedError()
"""
    cells += [
        _md(
            """
            ## 2. Nested Quantization and a Fail-Closed `QuantState` Audit

            ### Exercise: formulas, allocated tensors, and corrupt-state rejection

            Reconstruct mean-centered scales, separate the amortized paper formula
            from a tail-aware logical layout, then measure every unique materialized
            tensor. The audit must compare all 16 NF4 values and all 256 dynamic-map
            values, reject every non-finite nested tensor, and verify finite
            dequantization.

            For this first derivation use uniform signed-int8 scale codes. This
            arithmetic exercise differs from bitsandbytes' nonuniform dynamic
            scale map. The real model below uses and audits the latter; do not
            treat the toy scale error as an exact kernel-parity measurement.

            <details><summary>Expected output</summary>

            `paper amortized < tail-aware small-tensor formula < actual small-layer
            storage`; packed bytes equal `ceil(numel / 2)`. Changing any codebook value
            or inserting NaN/Inf into offset or nested absmax raises `RuntimeError`.

            </details>

            <details><summary>Help</summary>

            Length and dtype checks cannot detect a subtly corrupt lookup table. Use
            exact tensor equality against independently constructed expected values,
            and apply `isfinite` to every tensor reachable through `state2`.

            </details>
            """
        ),
        _exercise_code(nested_stub, plumbing_solution + "\n" + nested_solution, is_solution=is_solution),
        _code(
            """
            tests.check_double_quantization(double_quantize_scales, dequantize_scales, nf4_storage_accounting)
            tests.check_linear4bit_storage(linear4bit_storage_accounting)
            tests.check_quant_state_audit(audit_linear4bit)

            corrupt = tests._actual_nf4_linear()
            corrupt.weight.quant_state.code[3] += 0.125
            try:
                audit_linear4bit(corrupt, expected_compute_dtype=t.float32)
            except RuntimeError as error:
                assert "canonical NF4" in str(error)
                print("Corrupt codebook rejected:", error)
            else:
                raise AssertionError("A changed NF4 codebook value must be rejected")
            """
        ),
        _solution_dropdown(nested_solution),
    ]

    adapter_solution = _source(
        "ste_nf4",
        "toy_qlora_forward",
        "QLoRALinear",
        "dequantize_bnb_weight",
        "dequantized_base_max_diff",
        "_get_parent_and_leaf",
        "freeze_except_adapters",
        "inject_qlora",
    )
    adapter_stub = """
def ste_nf4(values: t.Tensor, *, block_size: int = NF4_BLOCK_SIZE) -> t.Tensor:
    raise NotImplementedError()

def toy_qlora_forward(inputs: t.Tensor, base_weight: t.Tensor, lora_a: t.Tensor, lora_b: t.Tensor, *, alpha: float, block_size: int = NF4_BLOCK_SIZE) -> t.Tensor:
    raise NotImplementedError()

class QLoRALinear(nn.Module):
    def __init__(self, base: nn.Module, *, rank: int, alpha: float) -> None:
        super().__init__()
        raise NotImplementedError()

    def delta_weight(self) -> t.Tensor:
        raise NotImplementedError()

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        raise NotImplementedError()

def dequantize_bnb_weight(layer: nn.Module) -> t.Tensor:
    raise NotImplementedError()

def dequantized_base_max_diff(layer: nn.Module, inputs: t.Tensor) -> float:
    raise NotImplementedError()

""" + _source("_get_parent_and_leaf") + """

def freeze_except_adapters(model: nn.Module) -> int:
    raise NotImplementedError()

def inject_qlora(model: nn.Module, target_modules: Sequence[str], *, rank: int, alpha: float) -> dict[str, QLoRALinear]:
    raise NotImplementedError()
"""
    cells += [
        _md(
            """
            ## 3. A Frozen Quantized Base Plus a Trainable Low-Rank Path

            ### Exercise: implement the STE oracle and real QLoRA wrapper

            The toy STE makes the gradient path explicit. The real wrapper must accept
            only a materialized nested-NF4 `Linear4bit`, preserve its function when B
            is zero, freeze the packed base, and train visible FP32 A/B factors.

            <details><summary>Expected output</summary>

            STE backward gradient is all ones. A zero-initialized adapter exactly
            matches the real bitsandbytes base. Both factors receive gradients once B
            is nonzero, while the packed base receives none.

            </details>

            <details><summary>Help</summary>

            Use `x + (quantized_x - x).detach()` for the STE. Compute the adapter in
            FP32 and cast its output to the base output dtype only at the addition.

            </details>
            """
        ),
        _exercise_code(adapter_stub, adapter_solution, is_solution=is_solution),
        _code(
            """
            tests.check_ste(ste_nf4)
            tests.check_toy_qlora_forward(toy_qlora_forward)
            tests.check_real_qlora_wrapper(QLoRALinear)
            tests.check_dequantized_base_parity(dequantized_base_max_diff)
            tests.check_inject_qlora(inject_qlora, QLoRALinear)
            print("Quantized-base parity, freezing, injection, and gradient checks passed.")
            """
        ),
        _solution_dropdown(adapter_solution),
    ]

    data_solution = _source(
        "_crossed_codebook_examples",
        "make_codebook_examples",
        "make_unseen_subject_examples",
        "make_marker_control_examples",
        "counterbalance_audit",
        "name_only_baseline",
        "randomize_training_labels",
    )
    data_stub = """
def _crossed_codebook_examples(subjects: Sequence[str], templates: Sequence[tuple[str, str]]) -> list[PromptExample]:
    raise NotImplementedError()

def make_codebook_examples() -> tuple[list[PromptExample], list[PromptExample]]:
    raise NotImplementedError()

def make_unseen_subject_examples() -> list[PromptExample]:
    raise NotImplementedError()

def make_marker_control_examples(examples: Sequence[PromptExample]) -> dict[str, list[PromptExample]]:
    raise NotImplementedError()

def counterbalance_audit(examples: Sequence[PromptExample]) -> dict[str, Any]:
    raise NotImplementedError()

def name_only_baseline(train: Sequence[PromptExample], evaluation: Sequence[PromptExample]) -> dict[str, Any]:
    raise NotImplementedError()

def randomize_training_labels(examples: Sequence[PromptExample], *, seed: int) -> list[PromptExample]:
    raise NotImplementedError()
"""
    cells += [
        _md(
            """
            ## 4. Remove the Name Shortcut Before Training

            ### Exercise: a complete crossed design and paired counterfactuals

            Cross every subject and template with both labels. Training subjects recur
            under both markers; OOD subjects are disjoint but are crossed in the same
            way. Then create a marker swap that flips the target, a marker-removal
            control that retains the paired labels, and a name-only empirical baseline.

            <details><summary>Expected output</summary>

            Train/template-transfer/OOD sizes are `60 / 40 / 40`. Every
            `(subject, template, label)` cell occurs once. Name-only accuracy is exactly
            `0.5` for known and unseen names. Swap/removal sets each contain 40 paired
            examples.

            </details>

            <details><summary>Help</summary>

            Counterbalancing is a property to compute, not prose to assert. Compare the
            observed tuple set with the full Cartesian product and report per-name and
            per-template label counts.

            </details>
            """
        ),
        _exercise_code(data_stub, data_solution, is_solution=is_solution),
        _code(
            """
            tests.check_counterbalanced_data(
                make_codebook_examples,
                make_unseen_subject_examples,
                make_marker_control_examples,
                counterbalance_audit,
                name_only_baseline,
            )
            tests.check_random_labels(make_codebook_examples, randomize_training_labels)
            train_examples, transfer_examples = make_codebook_examples()
            ood_examples = make_unseen_subject_examples()
            controls = make_marker_control_examples(transfer_examples)
            display({
                "sizes": [len(train_examples), len(transfer_examples), len(ood_examples)],
                "name_only_transfer_accuracy": name_only_baseline(train_examples, transfer_examples)["accuracy"],
                "canonical": transfer_examples[0],
                "marker_swap": controls["marker_swap"][0],
                "marker_removed": controls["marker_removed"][0],
            })
            """
        ),
        _solution_dropdown(data_solution),
    ]

    train_helpers = _source(
        "label_token_ids",
        "tokenize_prompts",
        "two_label_logits",
        "protected_replay_batch",
        "optimizer_state_bytes",
        "optimizer_state_dtypes",
    )
    train_solution = _source("train_on_codebook")
    train_stub = train_helpers + """

def train_on_codebook(model: nn.Module, tokenizer: Any, examples: Sequence[PromptExample], *, steps: int, learning_rate: float, batch_size: int, seed: int, protected_replay_weight: float = 0.1) -> TrainTrace:
    raise NotImplementedError()
"""
    cells += [
        _md(
            """
            ## 5. Train on Two Labels Without Renaming the Objective

            ### Exercise: matched optimizer loop

            The task term selects only the ` red` and ` blue` logits and applies
            cross-entropy. This is **two-label conditional NLL**, not full-vocabulary
            language-model loss. The protected replay term is a separate full-vocabulary
            next-token NLL. Return measured trainable bytes and actual AdamW state.

            <details><summary>Expected output</summary>

            One real TinyStories CPU step updates the rank-2 adapter, allocates nonzero
            AdamW state, reports 512 trainable FP32 values, and returns a finite loss.

            </details>

            <details><summary>Help</summary>

            Seed batch sampling explicitly. Build the optimizer only from
            `requires_grad` parameters, clip that same list, and count optimizer tensor
            storage after the first step creates it.

            </details>
            """
        ),
        _exercise_code(train_stub, train_helpers + "\n" + train_solution, is_solution=is_solution),
        _code(
            """
            tests.check_training_loop(train_on_codebook)
            print("Real TinyStories CPU adapter step passed with measured AdamW state.")
            """
        ),
        _solution_dropdown(train_solution),
    ]

    diagnostics_solution = _source("summarize_spectrum", "quantization_stress_test")
    diagnostics_stub = """
def summarize_spectrum(update: t.Tensor) -> SpectrumMetrics:
    raise NotImplementedError()

def quantization_stress_test(*, seed: int = 0) -> dict[str, Any]:
    raise NotImplementedError()
"""
    cells += [
        _md(
            """
            ## 6. Mechanism Diagnostics and an Outlier Control

            ### Exercise: spectra and shared-scale damage

            Report singular values per target matrix rather than concatenating away
            layer identity. Then compare clean weights with one raw value set to 12
            while varying NF4 block size.

            <details><summary>Expected output</summary>

            Rank-one and full-rank matrix oracles separate cleanly. For the outlier
            condition, non-outlier MSE at block size 128 exceeds block size 8 by more
            than 10x; the matched clean condition is lower.

            An all-zero update has effective, stable, and numerical rank zero.

            </details>

            <details><summary>Help</summary>

            Effective rank is the exponential entropy of normalized singular values.
            Handle zero total mass before computing that entropy. Low rank alone
            cannot identify a useful adaptation: compare the random-adapter curve.

            </details>
            """
        ),
        _exercise_code(diagnostics_stub, diagnostics_solution, is_solution=is_solution),
        _code(
            """
            tests.check_spectrum(summarize_spectrum)
            tests.check_outlier_stress(quantization_stress_test)
            stress = quantization_stress_test(seed=0)
            display({size: stress["outlier"][size]["non_outlier_mse"] for size in ("8", "32", "64", "128")})
            """
        ),
        _solution_dropdown(diagnostics_solution),
    ]

    model_solution = _source("intended_nf4_module_names", "audit_nf4_model", "load_nf4_model")
    model_stub = """
def intended_nf4_module_names(model: nn.Module) -> tuple[str, ...]:
    raise NotImplementedError()

def audit_nf4_model(model: nn.Module, intended_modules: Sequence[str], *, expected_compute_dtype: t.dtype = t.bfloat16, expected_storage_dtype: t.dtype = t.uint8) -> dict[str, Any]:
    raise NotImplementedError()

def load_nf4_model(*, device: t.device) -> nn.Module:
    raise NotImplementedError()
"""
    cells += [
        _md(
            """
            ## 7. Strict Pretrained NF4 Loading

            ### Exercise: audit every intended projection before returning the model

            Enumerate all attention and MLP projections in every GPT-Neo block. The
            loader is CUDA-only and must abort if any intended layer is dense,
            unmaterialized, unnested, corrupt, non-finite, wrongly packed, or configured
            with the wrong compute dtype.

            <details><summary>Expected output</summary>

            The real CPU audit accepts two materialized nested-NF4 layers and rejects an
            unmaterialized replacement. Calling the pretrained loader on CPU raises;
            there is no alternate pretrained path.

            </details>

            <details><summary>Help</summary>

            A configuration flag does not pack a weight. Check the module type and
            its materialized quantization state after loading, then traverse every
            intended projection rather than accepting the first successful layer.

            </details>
            """
        ),
        _exercise_code(model_stub, model_solution, is_solution=is_solution),
        _code(
            """
            tests.check_nf4_model_audit(audit_nf4_model)
            try:
                load_nf4_model(device=t.device("cpu"))
            except RuntimeError as error:
                assert "no substitute path" in str(error)
                print("CPU pretrained load rejected:", error)
            else:
                raise AssertionError("The pretrained NF4 path must require CUDA")
            """
        ),
        _solution_dropdown(model_solution),
    ]

    cells += [
        _md(
            """
            ## 8. Assemble the Learner Provenance Bundle

            This bundle is the dependency of both release APIs. The strict audit records
            a qualified name and executable SHA-256 fingerprint for every callback and
            rejects any callback object silently taken from `solutions.py`.
            """
        ),
        _code(
            """
            learner_functions = LearnerFunctions(
                nf4_codebook=nf4_codebook,
                expected_dynamic_codebook=expected_dynamic_codebook,
                blockwise_absmax=blockwise_absmax,
                quantize_nf4=quantize_nf4,
                dequantize_nf4=dequantize_nf4,
                quantization_error=quantization_error,
                double_quantize_scales=double_quantize_scales,
                dequantize_scales=dequantize_scales,
                ste_nf4=ste_nf4,
                nf4_storage_accounting=nf4_storage_accounting,
                linear4bit_storage_accounting=linear4bit_storage_accounting,
                audit_linear4bit=audit_linear4bit,
                qlora_linear=QLoRALinear,
                dequantize_bnb_weight=dequantize_bnb_weight,
                dequantized_base_max_diff=dequantized_base_max_diff,
                audit_nf4_model=audit_nf4_model,
                inject_qlora=inject_qlora,
                make_codebook_examples=make_codebook_examples,
                make_unseen_subject_examples=make_unseen_subject_examples,
                make_marker_control_examples=make_marker_control_examples,
                counterbalance_audit=counterbalance_audit,
                name_only_baseline=name_only_baseline,
                randomize_training_labels=randomize_training_labels,
                train_on_codebook=train_on_codebook,
                summarize_spectrum=summarize_spectrum,
                quantization_stress_test=quantization_stress_test,
                load_nf4_model=load_nf4_model,
            )
            tests.check_learner_bundle(learner_functions)
            print("Your function bundle is connected. The numerical tests check its behavior; hashes only identify code.")
            """
        ),
        _md(
            """
            ## 9. Signature Result: Quantization and Storage

            `run_cpu_release` requires the bundle. It derives `accepted` from measured
            checks; it cannot default to reference functions. The panel is regenerated
            from those same callbacks.
            """
        ),
        _code(
            """
            cpu_result = reference.run_cpu_release(
                implementation=learner_functions,
                make_figure=True,
            )
            assert cpu_result["accepted"] == all(cpu_result["checks"].values())
            display({
                "accepted": cpu_result["accepted"],
                "checks": cpu_result["checks"],
                "dataset_sizes": {
                    key: cpu_result["dataset"][key]
                    for key in ("train_examples", "template_transfer_examples", "unseen_subject_ood_examples")
                },
                "name_only": cpu_result["dataset"]["name_only_baselines"],
            })
            display(Image(filename=str(ROOT / "chapter15_peft_misalignment/instructions/assets/qlora_nf4_toy_signature.png")))
            """
        ),
        _md(
            """
            ## 10. Train the Marker Rule on TinyStories

            Each method result carries its seed. Acceptance extracts exactly three
            unique seeds from returned runs, verifies they match every method payload,
            and uses a two-sided 95% Student-t interval with `t(0.975, df=2) =
            4.3026527299`. The comparison reports:

            - canonical, marker-swap, marker-removal, name-only, random-label, and same-norm behavior;
            - paired marker-swap prediction-flip and both-sides-correct rates;
            - template-transfer and OOD accuracy plus two-label conditional NLL;
            - exact codebook/finiteness audits for every quantized method and seed;
            - model tensors, gradients, AdamW tensors, and their non-overlapping resident-state total;
            - `resident_training_state_bytes = model_tensor_bytes + gradient_bytes + optimizer_state_bytes`;
            - QLoRA-vs-BF16-LoRA and QLoRA-vs-full-BF16 state comparators;
            - per-layer spectra, protected full-vocabulary NLL, and peak CUDA allocation.

            The generic `run_gpu_test` hook is for the visible section reference. This
            notebook calls only `run_cuda_release`, whose implementation argument is
            mandatory.
            """
        ),
        _code(
            """
            def run_cuda_release(max_vram_gb: float = 24.0) -> dict[str, Any]:
                return reference.run_cuda_release(
                    implementation=learner_functions,
                    max_vram_gb=max_vram_gb,
                    steps=32,
                    batch_size=12,
                )

            RUN_CUDA = t.cuda.is_available() and os.environ.get("ARENA_QLORA_CPU_ONLY") != "1"
            if RUN_CUDA:
                cuda_result = run_cuda_release(max_vram_gb=24.0)
                assert cuda_result["accepted"] == all(cuda_result["checks"].values())
                display(Image(filename=str(ROOT / "chapter15_peft_misalignment/instructions/assets/qlora_real_signature_panel.png")))
                for method in ("qlora_r8", "bf16_lora_r8", "full_bf16", "random_labels", "same_norm_random"):
                    result = cuda_result["summary"][method]
                    print(f"{method:18s} accuracy={result['template_transfer']['accuracy']['mean']:.3f}; "
                          f"resident state={result['resident_training_state_mib']['mean']:.2f} MiB")
                print("Unmet comparison checks:", [name for name, passed in cuda_result["checks"].items() if not passed])
            else:
                print("CPU mechanics completed. The pretrained comparison requires CUDA and was not run.")
            """
        ),
        _md(
            """
            <details><summary>Expected output: measured TinyStories comparison</summary>

            The September 5 reference run uses seeds 0, 1, and 2, with 32 updates per
            trained method. QLoRA r=8 and BF16 LoRA r=8 both reach 1.000 accuracy on
            these 40 template-transfer and 40 narrow OOD examples. Random labels
            average 0.517 and random same-norm adapters 0.575 on template transfer;
            marker removal and the name-only baseline give 0.500.

            Resident training state is 38.67 MiB for QLoRA, 39.18 MiB for BF16 LoRA,
            and 60.58 MiB for full BF16. This is about 1.3% savings against BF16 LoRA,
            not an eightfold reduction. The original 0.01 dequantized-matmul error
            limit fails (maximum 0.03125), so the candidate remains unreleased even
            though its behavior and memory checks pass. Thresholds are unchanged.

            </details>

            <details><summary>Interpreting the curves</summary>

            Low rank is enforced by A and B even for random adapters; it is not
            evidence of useful learning by itself. Compare their behavior and
            protected-prompt loss with the trained adapters. The protected prompts
            are also used for replay, so improvement there measures the replay
            objective and does not establish general capability preservation.

            All seeds solving this tiny rule gives a zero-width seed interval; it
            does not imply zero uncertainty on new tasks or examples. The next
            diagnostic inspects the small memory gain and failed parity check.

            </details>
            """
        ),
        _md("## Try It Yourself"),
        _code(
            """
            PLAY_BLOCK_SIZE = 32
            PLAY_OUTLIER_VALUE = 6.0
            play_generator = t.Generator().manual_seed(7)
            play_weights = 0.2 * t.randn(256, generator=play_generator)
            play_weights[0] = PLAY_OUTLIER_VALUE
            play_reconstructed = dequantize_nf4(quantize_nf4(play_weights, block_size=PLAY_BLOCK_SIZE))
            play_clean = play_weights.clone()
            play_clean[0] = 0.0
            play_clean_rec = dequantize_nf4(quantize_nf4(play_clean, block_size=PLAY_BLOCK_SIZE))
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), layout="constrained")
            for label, original, reconstructed in (("with outlier", play_weights, play_reconstructed),
                                                    ("outlier removed", play_clean, play_clean_rec)):
                axes[0].plot((original[1:] - reconstructed[1:]).abs(), label=label, alpha=0.7)
                axes[1].plot(original[1:33], reconstructed[1:33], ".", label=label)
            axes[0].set(xlabel="Weight index (excluding outlier)", ylabel="Absolute reconstruction error")
            axes[1].plot([-0.5, 0.5], [-0.5, 0.5], "k--", label="exact reconstruction")
            axes[1].set(xlabel="Original weight", ylabel="Reconstructed weight")
            for ax in axes: ax.legend()
            plt.show()
            display({
                "block_size": PLAY_BLOCK_SIZE,
                "raw_outlier_value": PLAY_OUTLIER_VALUE,
                "non_outlier_mse": float((play_weights[1:] - play_reconstructed[1:]).square().mean().item()),
            })
            """
        ),
        _code(
            """
            _, play_transfer = make_codebook_examples()
            play_controls = make_marker_control_examples(play_transfer)
            PLAY_INDEX = 0
            for condition, example in (("canonical", play_transfer[PLAY_INDEX]),
                                        ("marker swap", play_controls["marker_swap"][PLAY_INDEX]),
                                        ("marker removed", play_controls["marker_removed"][PLAY_INDEX])):
                print(f"{condition}: {example.prompt}")
                print(f"  paired target: {LABEL_TEXT[example.label]!r}")
            print("After marker removal the same prompt occurs with both targets; name alone cannot decide.")
            """
        ),
        _md(
            """
            ### Try It Yourself: inspect memory and arithmetic precision

            The small resident-state saving needs an explanation. Count parameters
            and buffers separately, then change `PLAY_PARITY_LAYER` or the input
            scale. Compare the real BF16 kernel with a dequantized dense layer, and
            repeat in FP32 as a precision control. This diagnostic does not change
            the registered BF16 parity threshold or the training configuration.

            <details><summary>Interpreting the diagnostic</summary>

            Eight 2048-by-2048 boolean attention masks occupy 32 MiB in this GPT-Neo
            implementation. These buffers and the unquantized embedding table limit
            total savings. In the tested runtime, the FP32 comparison agrees closely
            while the BF16 comparison exceeds the original 0.01 absolute-error gate.
            This supports a precision-dependent difference; it does not prove every
            BF16 discrepancy harmless. Keep that failed gate visible.

            </details>
            """
        ),
        _code(
            """
            PLAY_PARITY_LAYER = TARGET_MODULES[0]
            PLAY_INPUT_SCALE = 1.0
            precision_diagnostic = None
            if RUN_CUDA:
                diagnostic_model = load_nf4_model(device=t.device("cuda:0"))
                parent, leaf = _get_parent_and_leaf(diagnostic_model, PLAY_PARITY_LAYER)
                diagnostic_layer = getattr(parent, leaf)
                generator = t.Generator(device="cuda").manual_seed(1)
                inputs = PLAY_INPUT_SCALE * t.randn(3, diagnostic_layer.in_features,
                                                     generator=generator, device="cuda", dtype=t.bfloat16)
                bf16_error = dequantized_base_max_diff(diagnostic_layer, inputs)
                state32 = copy.copy(diagnostic_layer.weight.quant_state)
                state32.dtype = t.float32
                weight32 = bnb.functional.dequantize_4bit(diagnostic_layer.weight.data, state32)
                diagnostic_layer.compute_dtype = t.float32
                bias32 = None if diagnostic_layer.bias is None else diagnostic_layer.bias.float()
                with t.no_grad():
                    dense32 = F.linear(inputs.float(), weight32, bias32)
                    fp32_error = float((diagnostic_layer(inputs.float()) - dense32).abs().max())
                precision_diagnostic = {
                    "bf16_max_abs_error": bf16_error, "fp32_max_abs_error": fp32_error,
                    "parameter_bytes": sum(p.numel() * p.element_size() for p in diagnostic_model.parameters()),
                    "buffer_bytes": sum(b.numel() * b.element_size() for b in diagnostic_model.buffers()),
                    "layer": PLAY_PARITY_LAYER, "input_scale": PLAY_INPUT_SCALE,
                }
                fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
                axes[0].bar(["BF16", "FP32 control"], [bf16_error, fp32_error])
                axes[0].axhline(0.01, linestyle="--", color="black", label="original BF16 gate")
                axes[0].set(yscale="log", ylabel="Maximum absolute output difference")
                axes[0].legend()
                axes[1].bar(["parameters", "buffers"],
                            [precision_diagnostic[k] / 2**20 for k in ("parameter_bytes", "buffer_bytes")])
                axes[1].set(ylabel="MiB (nested quantization metadata separate)")
                plt.show()
                print(f"BF16 error {bf16_error:.8f}; FP32 error {fp32_error:.8f}")
                del diagnostic_layer, parent, diagnostic_model, inputs, weight32, state32, dense32
                gc.collect()
                t.cuda.empty_cache()
            else:
                print("This precision and model-buffer diagnostic requires CUDA.")
            """
        ),
        _md(
            """
            ## Bonus Anomaly Hunt

            Register one directional prediction before changing rank, target layer,
            block size, nested statistics, or NF4 versus FP4. A useful anomaly is a
            case where stored model tensors shrink but total resident state or peak
            allocation does not, or where accuracy agrees while two-label conditional
            NLL, marker-removal behavior, or protected full-vocabulary NLL diverges.
            """
        ),
        _md(
            """
            ## Interpreting the Result

            Aggregate swap accuracy is redundant on a fully crossed split. The paired
            prediction-flip and both-sides-correct rates ask whether predictions follow
            the causal marker after the subject is held fixed. Marker-removal and name-only results should be at
            chance because the crossed design makes identity uninformative. The memory
            claim is only that measured resident state, `model tensors + gradients +
            optimizer tensors`, is smaller for QLoRA under explicit comparators. Peak
            allocator memory is reported separately and may not follow that ordering.
            """
        ),
        _md(
            """
            ## Limitations

            - Three fixed seeds estimate only optimizer, batching, and initialization variability; the wide t interval is not dataset uncertainty.
            - The OOD split changes names and templates but keeps the explicit two-marker rule.
            - Two-label conditional NLL says nothing about probability assigned outside ` red` and ` blue`.
            - TinyStories-1M and two layer-6 targets do not establish large-model quality or memory scaling.
            - Four protected prompts are a guardrail, not a capability benchmark.
            - AdamW is measured here; paged optimizers are not reimplemented.
            - Callback hashes identify code, and direct-dependency checks catch simple reference aliases.
              They cannot prove correctness or detect arbitrary dynamic delegation.
            - CUDA is needed for the pretrained comparison. CPU-only execution covers the mechanics,
              not the behavior or training-memory claims.
            """
        ),
        _md(
            """
            ## Reading

            - Dettmers et al. (2023), [QLoRA](https://arxiv.org/abs/2305.14314)
            - Hu et al. (2021), [LoRA](https://arxiv.org/abs/2106.09685)
            - Eldan and Li (2023), [TinyStories](https://arxiv.org/abs/2305.07759)
            """
        ),
    ]

    for index, cell in enumerate(cells):
        cell["id"] = f"qlora-15-3-{index:03d}"

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for suffix, is_solution in (("exercises", False), ("solutions", True)):
        path = HERE / f"15.3_QLoRA_from_Scratch_{suffix}.ipynb"
        notebook = build_notebook(is_solution=is_solution)
        if path.exists():
            old_code = {"".join(cell["source"]): cell for cell in json.loads(path.read_text())["cells"]
                        if cell["cell_type"] == "code"}
            for cell in notebook["cells"]:
                old = old_code.get("".join(cell["source"]))
                if cell["cell_type"] == "code" and old:
                    cell["outputs"], cell["execution_count"] = old["outputs"], old["execution_count"]
        path.write_text(json.dumps(notebook, indent=1) + "\n")
        print(path)
    learner = build_notebook(is_solution=False)
    page = HERE.parents[1] / "instructions/pages/03_[15.3]_QLoRA_from_Scratch.md"
    chunks = []
    for cell in learner["cells"]:
        source = "".join(cell["source"]).rstrip()
        chunks.append(source if cell["cell_type"] == "markdown" else f"```python\n{source}\n```")
        if cell["cell_type"] == "code":
            for asset in ("qlora_nf4_toy_signature.png", "qlora_real_signature_panel.png"):
                if asset in source and "display(Image" in source:
                    chunks.append(f"![Measured QLoRA signature result](../assets/{asset})")
    chunks.insert(1, "> **Notebooks: [exercises](../../exercises/part3_qlora_from_scratch/15.3_QLoRA_from_Scratch_exercises.ipynb) | [solutions](../../exercises/part3_qlora_from_scratch/15.3_QLoRA_from_Scratch_solutions.ipynb)**")
    page.write_text("\n\n".join(chunks) + "\n")
    print(page)


if __name__ == "__main__":
    main()
