"""Generate the GT-2 IOI/path-patching verification report.

This is a narrow evidence generator for the roadmap final-completeness gate. It
uses the original ARENA IOI dataset utilities and a pinned GPT-2 small
checkpoint, then runs a lightweight Appendix-B-style path-patching check on
published IOI heads plus random controls.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("BNB_CUDA_VERSION", "130")

import torch as t
import yaml

from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parents[1]
EXERCISES_DIR = ROOT / "chapter1_transformer_interp" / "exercises"
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "ioi_path_patching"
REPORT_PATH = EVIDENCE_DIR / "verification_report.json"
LOCK_PATH = EVIDENCE_DIR / "artifacts.lock.yml"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXERCISES_DIR) not in sys.path:
    sys.path.insert(0, str(EXERCISES_DIR))

from arena_ext.verification import (
    build_report_input_manifest,
    build_verification_report,
    write_verification_report,
)
from part41_indirect_object_identification.ioi_dataset import IOIDataset
from scripts.run_extension_verification_reports import (
    expected_metric_failures,
    gpu_evidence_summary,
    stamp_runtime_cuda_metadata,
)


SELECTED_HEADS: dict[str, tuple[int, int]] = {
    "name_mover_9_9": (9, 9),
    "name_mover_10_0": (10, 0),
    "name_mover_9_6": (9, 6),
    "negative_name_mover_10_7": (10, 7),
    "negative_name_mover_11_10": (11, 10),
    "s2_inhibition_5_5": (5, 5),
    "s2_inhibition_7_3": (7, 3),
    "s2_inhibition_7_9": (7, 9),
    "s2_inhibition_8_6": (8, 6),
    "s2_inhibition_8_10": (8, 10),
    "random_control_0_0": (0, 0),
    "random_control_1_1": (1, 1),
    "random_control_8_8": (8, 8),
}


def run_smoke_test() -> dict[str, Any]:
    """Return the static contract for the selected published-head check."""

    known_heads = {
        label: head
        for label, head in SELECTED_HEADS.items()
        if label.startswith(("name_mover", "negative_name_mover", "s2_inhibition"))
    }
    random_controls = {
        label: head for label, head in SELECTED_HEADS.items() if label.startswith("random_control")
    }
    return {
        "selected_head_count": len(SELECTED_HEADS),
        "known_published_head_count": len(known_heads),
        "random_control_count": len(random_controls),
        "uses_original_arena_ioi_dataset": True,
        "uses_appendix_b_freeze_all_heads_patch_one_sender": True,
        "full_heatmap_claimed": False,
    }


def _logit_diff(logits: t.Tensor, dataset: IOIDataset) -> t.Tensor:
    batch_index = t.arange(logits.shape[0], device=logits.device)
    end_positions = dataset.word_idx["end"].to(logits.device)
    io_token_ids = t.tensor(dataset.io_tokenIDs, device=logits.device)
    s_token_ids = t.tensor(dataset.s_tokenIDs, device=logits.device)
    io_logits = logits[batch_index, end_positions, io_token_ids]
    s_logits = logits[batch_index, end_positions, s_token_ids]
    return (io_logits - s_logits).float().mean()


def _single_name_token_id(tokenizer: Any, name: str) -> int:
    token_ids = tokenizer.encode(f" {name}", add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer returned no ids for name {name!r}.")
    return int(token_ids[0])


def _repair_special_token_name_ids(dataset: IOIDataset) -> None:
    """Keep original IOIDataset behavior stable under Transformers 5 tokenizers.

    The original ARENA helper indexes `tokenizer.encode(' ' + name)[0]`. Newer
    GPT-2 tokenizers may prepend the special token unless explicitly disabled,
    which makes IO and S token IDs both equal to 50256 and destroys the logit
    difference. The evidence report should adapt at the boundary instead of
    editing the preserved original ARENA file.
    """

    dataset.io_tokenIDs = [
        _single_name_token_id(dataset.tokenizer, prompt["IO"])
        for prompt in dataset.ioi_prompts
    ]
    dataset.s_tokenIDs = [
        _single_name_token_id(dataset.tokenizer, prompt["S"])
        for prompt in dataset.ioi_prompts
    ]


def run_gpu_test(max_vram_gb: float = 24.0, dataset_size: int = 12) -> dict[str, Any]:
    """Run the lightweight published-replication-style IOI path-patching check."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "full_path": "GPT-2 small IOI path-patching published-replication-style check.",
        }

    t.set_grad_enabled(False)
    t.manual_seed(0)
    t.cuda.reset_peak_memory_stats()
    device = "cuda"
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device=device,
        revision=GPT2_REVISION,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    model.eval()
    if hasattr(model.tokenizer, "add_bos_token"):
        model.tokenizer.add_bos_token = False

    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=dataset_size,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=1,
        device=device,
    )
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")
    _repair_special_token_name_ids(ioi_dataset)
    _repair_special_token_name_ids(abc_dataset)
    z_filter = lambda name: name.endswith("z")

    with t.inference_mode():
        clean_logits, clean_cache = model.run_with_cache(ioi_dataset.toks, names_filter=z_filter)
        corrupt_logits, corrupt_cache = model.run_with_cache(abc_dataset.toks, names_filter=z_filter)

    clean_logit_diff = _logit_diff(clean_logits, ioi_dataset).item()
    corrupt_logit_diff = _logit_diff(corrupt_logits, ioi_dataset).item()
    clean_corrupt_gap = clean_logit_diff - corrupt_logit_diff
    if clean_corrupt_gap <= 0:
        raise RuntimeError(
            "Expected IOI clean logit diff to exceed ABC corrupt logit diff, got "
            f"{clean_logit_diff=}, {corrupt_logit_diff=}."
        )

    selected_results: dict[str, dict[str, Any]] = {}
    for label, (sender_layer, sender_head) in SELECTED_HEADS.items():

        def freeze_all_z_patch_sender(
            z: t.Tensor,
            hook,
            *,
            sender_layer: int = sender_layer,
            sender_head: int = sender_head,
        ) -> t.Tensor:
            patched = clean_cache[hook.name].clone()
            if hook.layer() == sender_layer:
                patched[:, :, sender_head, :] = corrupt_cache[hook.name][:, :, sender_head, :]
            return patched

        with t.inference_mode():
            patched_logits = model.run_with_hooks(
                ioi_dataset.toks,
                fwd_hooks=[(z_filter, freeze_all_z_patch_sender)],
                return_type="logits",
            )

        patched_logit_diff = _logit_diff(patched_logits, ioi_dataset).item()
        ioi_metric = (patched_logit_diff - clean_logit_diff) / clean_corrupt_gap
        selected_results[label] = {
            "head": [sender_layer, sender_head],
            "patched_logit_diff": patched_logit_diff,
            "ioi_metric": ioi_metric,
            "damage_fraction": -ioi_metric,
            "abs_damage_fraction": abs(ioi_metric),
        }

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3

    name_mover_signed_damage = [
        result["damage_fraction"]
        for label, result in selected_results.items()
        if label.startswith("name_mover")
    ]
    name_mover_abs_damage = [
        result["abs_damage_fraction"]
        for label, result in selected_results.items()
        if label.startswith("name_mover")
    ]
    random_control_abs_damage = [
        result["abs_damage_fraction"]
        for label, result in selected_results.items()
        if label.startswith("random_control")
    ]
    negative_name_mover_signed_damage = [
        result["damage_fraction"]
        for label, result in selected_results.items()
        if label.startswith("negative_name_mover")
    ]

    name_mover_mean_damage = sum(name_mover_abs_damage) / len(name_mover_abs_damage)
    name_mover_signed_mean_damage = sum(name_mover_signed_damage) / len(
        name_mover_signed_damage
    )
    random_control_abs_mean_damage = sum(random_control_abs_damage) / len(
        random_control_abs_damage
    )
    negative_name_mover_mean_damage = sum(negative_name_mover_signed_damage) / len(
        negative_name_mover_signed_damage
    )
    known_to_random_control_ratio = name_mover_mean_damage / max(
        random_control_abs_mean_damage,
        1e-12,
    )
    within_vram_budget = peak_vram_gb <= max_vram_gb
    random_controls_passed = random_control_abs_mean_damage <= 0.02
    published_heads_beat_random_controls = (
        name_mover_mean_damage >= 0.15 and known_to_random_control_ratio >= 10.0
    )
    negative_name_mover_sign_control_passed = all(
        damage <= -0.05 for damage in negative_name_mover_signed_damage
    )
    preflight_passed = (
        clean_corrupt_gap >= 1.0
        and name_mover_mean_damage >= 0.15
        and selected_results["name_mover_9_9"]["abs_damage_fraction"] >= 0.10
        and selected_results["name_mover_9_6"]["abs_damage_fraction"] >= 0.20
        and random_controls_passed
        and known_to_random_control_ratio >= 10.0
        and negative_name_mover_sign_control_passed
        and within_vram_budget
    )

    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "preflight_passed": preflight_passed,
        "model_name": "gpt2-small",
        "hf_model_id": "gpt2",
        "hf_revision": GPT2_REVISION,
        "transformerlens_model_name": "gpt2-small",
        "bnb_cuda_override": os.environ.get("BNB_CUDA_VERSION"),
        "dataset_size": dataset_size,
        "clean_prompt_type": "mixed IOI",
        "corrupt_prompt_type": "ABC counterfactual via ABB->XYZ, BAB->XYZ",
        "clean_logit_diff": clean_logit_diff,
        "corrupt_logit_diff": corrupt_logit_diff,
        "clean_corrupt_gap": clean_corrupt_gap,
        "selected_head_count": len(SELECTED_HEADS),
        "selected_path_patch_results": selected_results,
        "name_mover_mean_damage": name_mover_mean_damage,
        "name_mover_signed_mean_damage": name_mover_signed_mean_damage,
        "random_control_abs_mean_damage": random_control_abs_mean_damage,
        "negative_name_mover_mean_damage": negative_name_mover_mean_damage,
        "known_to_random_control_ratio": known_to_random_control_ratio,
        "name_mover_9_9_damage": selected_results["name_mover_9_9"][
            "abs_damage_fraction"
        ],
        "name_mover_9_6_damage": selected_results["name_mover_9_6"][
            "abs_damage_fraction"
        ],
        "random_controls_passed": random_controls_passed,
        "published_heads_beat_random_controls": published_heads_beat_random_controls,
        "negative_name_mover_sign_control_passed": negative_name_mover_sign_control_passed,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": (
            "GPT-2 small IOI selected-head Appendix-B-style z activation "
            "path-patching check with published heads and low-effect random controls."
        ),
    }


def generate_report(max_vram_gb: float = 24.0, report_path: Path = REPORT_PATH) -> bool:
    """Run the report generator and write the universal verification report."""

    start = time.perf_counter()
    metrics: dict[str, Any] = {}
    failures: list[str] = []
    lock = yaml.safe_load(LOCK_PATH.read_text())
    try:
        metrics["notebook_contract"] = run_smoke_test()
        metrics["gpu_test"] = stamp_runtime_cuda_metadata(
            run_gpu_test(max_vram_gb=max_vram_gb)
        )
        metrics["gpu_evidence"] = gpu_evidence_summary(metrics["gpu_test"])
    except Exception as exc:  # noqa: BLE001 - report should capture the blocker.
        failures.append(repr(exc))

    failures.extend(expected_metric_failures(lock.get("expected_metrics", {}), metrics))
    elapsed = time.perf_counter() - start
    report = build_verification_report(
        lock,
        root=ROOT,
        report_inputs=build_report_input_manifest(
            ROOT,
            [
                Path(__file__),
                LOCK_PATH,
                EVIDENCE_DIR / "verification_report.schema.json",
                EXERCISES_DIR / "part41_indirect_object_identification/ioi_dataset.py",
                ROOT / "scripts/run_extension_verification_reports.py",
                ROOT / "arena_ext/verification.py",
            ],
        ),
        wall_clock_seconds=elapsed,
        metrics=metrics,
        baselines={
            "declared_controls": lock.get("controls", []),
            "expected_metrics": lock.get("expected_metrics", {}),
        },
        negative_controls={
            "declared_controls": [
                control
                for control in lock.get("controls", [])
                if "negative" in str(control).lower() or "random" in str(control).lower()
            ]
        },
        ood_tests={"declared": "not_required_for_this_lightweight_gt2_path_patch_smoke"},
        known_failures=failures,
        tests_passed=not failures,
        accepted=not failures,
    )
    write_verification_report(report, report_path)
    return report.accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-vram-gb", type=float, default=24.0)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    accepted = generate_report(max_vram_gb=args.max_vram_gb, report_path=args.report_path)
    print(f"Wrote {args.report_path}")
    print("GT2_IOI_PATH_PATCHING=PASS" if accepted else "GT2_IOI_PATH_PATCHING=FAIL")
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
