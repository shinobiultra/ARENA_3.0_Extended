"""Generate a modern verification report for the Othello-GPT exercise.

This is intentionally scoped to the existing ARENA Othello-GPT assets: the
TransformerLens checkpoint, the bundled small board-sequence fixture, and the
bundled linear probe. It does not train a new model or mock evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("BNB_CUDA_VERSION", "130")

import numpy as np
import torch as t
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = Path(__file__).resolve().parent
OTHELLO_SECTION_DIR = (
    ROOT / "chapter1_transformer_interp" / "exercises" / "part53_othellogpt"
)
EXERCISES_DIR = ROOT / "chapter1_transformer_interp" / "exercises"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXERCISES_DIR) not in sys.path:
    sys.path.append(str(EXERCISES_DIR))

from part53_othellogpt import utils  # noqa: E402
from arena_ext.verification import build_report_input_manifest  # noqa: E402
from scripts.run_extension_verification_reports import runtime_cuda_metadata  # noqa: E402


REPO_ID = "NeelNanda/Othello-GPT-Transformer-Lens"
REVISION = "905ca1a68b9f7dff77adc56af1962e5f6fcac274"
CHECKPOINT_FILE = "synthetic_model.pth"
MODEL_PRECISION = "float32"
NUM_GAMES = 64
NUM_MOVES = 59
MAX_VRAM_GB = 8.0
SAMPLE_INPUT = [20, 19, 18, 10, 2, 1, 27, 3, 41, 42]
EXPECTED_FIRST_TOP3 = [21, 33, 19]


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


def build_model(device: t.device) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
        device=device,
    )
    model = HookedTransformer(cfg)
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=CHECKPOINT_FILE,
        revision=REVISION,
    )
    state_dict = t.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def board_states_and_legal_moves(
    games_square: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    states = t.zeros((NUM_GAMES, NUM_MOVES, 8, 8), dtype=t.int64)
    legal = t.zeros((NUM_GAMES, NUM_MOVES, 8, 8), dtype=t.bool)
    for game_index in range(NUM_GAMES):
        board = utils.OthelloBoardState()
        for move_index in range(NUM_MOVES):
            board.umpire(int(games_square[game_index, move_index].item()))
            states[game_index, move_index] = t.from_numpy(board.state).long()
            legal[game_index, move_index].flatten()[board.get_valid_moves()] = True
    return states, legal


def legal_token_mask_from_board(legal_board: t.Tensor) -> t.Tensor:
    mask = t.zeros((NUM_GAMES, NUM_MOVES, 61), dtype=t.bool)
    flat_legal = legal_board.flatten(2)
    for square, token_id in utils.SQUARE_TO_ID.items():
        if square >= 0 and token_id >= 1:
            mask[:, :, token_id] = flat_legal[:, :, square]
    return mask


def relative_board_labels(states: t.Tensor) -> t.Tensor:
    """Map board states to empty/theirs/mine relative to the side to move."""

    labels = t.zeros_like(states)
    for move_index in range(NUM_MOVES):
        current_player = -1 if move_index % 2 == 0 else 1
        labels[:, move_index][states[:, move_index] == 0] = 0
        labels[:, move_index][states[:, move_index] == -current_player] = 1
        labels[:, move_index][states[:, move_index] == current_player] = 2
    return labels


def swapped_parity_labels(states: t.Tensor) -> t.Tensor:
    labels = t.zeros_like(states)
    for move_index in range(NUM_MOVES):
        current_player = 1 if move_index % 2 == 0 else -1
        labels[:, move_index][states[:, move_index] == 0] = 0
        labels[:, move_index][states[:, move_index] == -current_player] = 1
        labels[:, move_index][states[:, move_index] == current_player] = 2
    return labels


def build_relative_linear_probe(device: t.device) -> tuple[t.Tensor, tuple[int, ...]]:
    full_probe = t.load(
        OTHELLO_SECTION_DIR / "main_linear_probe.pth",
        map_location=device,
        weights_only=True,
    )
    black_to_play, white_to_play, _ = (0, 1, 2)
    empty, white, black = (0, 1, 2)
    relative_probe = t.stack(
        [
            full_probe[[black_to_play, white_to_play], ..., [empty, empty]].mean(0),
            full_probe[[black_to_play, white_to_play], ..., [white, black]].mean(0),
            full_probe[[black_to_play, white_to_play], ..., [black, white]].mean(0),
        ],
        dim=-1,
    ).to(device)
    return relative_probe, tuple(full_probe.shape)


def run_verification() -> dict[str, Any]:
    t.set_grad_enabled(False)
    cuda_available = t.cuda.is_available()
    device = t.device("cuda" if cuda_available else "cpu")
    if cuda_available:
        t.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    model = build_model(device)

    sample_input = t.tensor([SAMPLE_INPUT], device=device)
    sample_logits = model(sample_input)
    sample_logprobs = sample_logits.log_softmax(-1)
    sample_top3 = sample_logprobs[0, 0].topk(3).indices.detach().cpu().tolist()

    board_seqs_id = t.from_numpy(
        np.load(OTHELLO_SECTION_DIR / "board_seqs_id_small.npy")
    ).long()
    board_seqs_square = t.from_numpy(
        np.load(OTHELLO_SECTION_DIR / "board_seqs_square_small.npy")
    ).long()
    focus_ids = board_seqs_id[:NUM_GAMES, :NUM_MOVES].to(device)
    focus_square = board_seqs_square[:NUM_GAMES, :NUM_MOVES]
    states, legal_board = board_states_and_legal_moves(focus_square)

    logits, cache = model.run_with_cache(focus_ids)
    logits_cpu = logits.detach().cpu()
    legal_token_mask = legal_token_mask_from_board(legal_board)
    nonempty_positions = legal_token_mask.any(dim=-1)
    top1 = logits_cpu.argmax(dim=-1)
    legal_top1 = (
        legal_token_mask.gather(-1, top1.unsqueeze(-1)).squeeze(-1)
        & nonempty_positions
    )
    legal_top1_accuracy = float(legal_top1.float().mean().item())
    random_legal_top1_baseline = float(legal_token_mask.float().mean(dim=-1).mean().item())

    linear_probe, full_probe_shape = build_relative_linear_probe(device)
    residual_stream = cache["resid_post", 6]
    probe_logits = t.einsum(
        "btd,drcq->btrcq",
        residual_stream,
        linear_probe,
    ).detach().cpu()
    probe_prediction = probe_logits.argmax(dim=-1)
    labels = relative_board_labels(states)
    occupied_mask = states != 0
    board_probe_accuracy = float((probe_prediction == labels).float().mean().item())
    board_probe_occupied_accuracy = float(
        (probe_prediction[occupied_mask] == labels[occupied_mask]).float().mean().item()
    )
    board_probe_empty_accuracy = float(
        (probe_prediction[~occupied_mask] == labels[~occupied_mask]).float().mean().item()
    )

    swapped = swapped_parity_labels(states)
    swapped_parity_accuracy = float((probe_prediction == swapped).float().mean().item())

    if cuda_available:
        t.cuda.synchronize()
        peak_vram_gb = float(t.cuda.max_memory_allocated() / 1e9)
        gpu_name = t.cuda.get_device_name(0)
    else:
        peak_vram_gb = 0.0
        gpu_name = "cpu_only"

    wall_clock_seconds = time.perf_counter() - start
    gpu_test = {
        "cuda_available": cuda_available,
        "device": gpu_name,
        "full_path": (
            "Loaded the pinned Othello-GPT synthetic TransformerLens checkpoint, "
            "ran legal-move prediction and bundled linear board-state probe on "
            f"{NUM_GAMES} games x {NUM_MOVES} positions."
        ),
        "model_loaded": True,
        "model_id": REPO_ID,
        "revision": REVISION,
        "sample_shape": list(sample_logprobs.shape),
        "sample_first_top3": sample_top3,
        "sample_first_top3_matches_expected": sample_top3 == EXPECTED_FIRST_TOP3,
        "games_checked": NUM_GAMES,
        "positions_checked": int(nonempty_positions.sum().item()),
        "legal_top1_accuracy": legal_top1_accuracy,
        "legal_top1_accuracy_threshold": 0.99,
        "random_legal_top1_baseline": random_legal_top1_baseline,
        "board_probe_layer": 6,
        "board_probe_accuracy": board_probe_accuracy,
        "board_probe_occupied_accuracy": board_probe_occupied_accuracy,
        "board_probe_empty_accuracy": board_probe_empty_accuracy,
        "board_probe_accuracy_threshold": 0.95,
        "swapped_parity_probe_accuracy": swapped_parity_accuracy,
        "swapped_parity_negative_control_threshold": 0.60,
        "full_probe_shape": list(full_probe_shape),
        "relative_probe_shape": list(linear_probe.shape),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= MAX_VRAM_GB,
    }
    gpu_test.update(runtime_cuda_metadata())
    section_specific_keys = sorted(
        key
        for key in gpu_test
        if key
        not in {
            "cuda_available",
            "cuda_version",
            "device",
            "full_path",
            "gpu_total_memory_gb",
            "peak_vram_gb",
            "torch_version",
            "within_vram_budget",
        }
    )
    checks = {
        "cuda_available": cuda_available,
        "sample_first_top3_matches_expected": sample_top3 == EXPECTED_FIRST_TOP3,
        "legal_top1_accuracy_min": legal_top1_accuracy >= 0.99,
        "board_probe_accuracy_min": board_probe_accuracy >= 0.95,
        "board_probe_occupied_accuracy_min": board_probe_occupied_accuracy >= 0.95,
        "board_probe_empty_accuracy_min": board_probe_empty_accuracy >= 0.95,
        "swapped_parity_negative_control_max": swapped_parity_accuracy <= 0.60,
        "within_vram_budget": peak_vram_gb <= MAX_VRAM_GB,
    }
    accepted = all(checks.values())
    known_failures = [key for key, value in checks.items() if not value]

    return {
        "accepted": accepted,
        "baselines": {
            "declared_controls": [
                "expected_sample_legal_move_top3",
                "random_legal_move_baseline",
                "swapped_player_parity_probe_negative_control",
                "bundled_linear_probe_reference",
            ],
            "expected_metrics": {
                "sample_first_top3": EXPECTED_FIRST_TOP3,
                "legal_top1_accuracy_min": 0.99,
                "board_probe_accuracy_min": 0.95,
                "swapped_parity_probe_accuracy_max": 0.60,
                "max_allowed_gpu_gb": MAX_VRAM_GB,
            },
        },
        "claim_scope": (
            "GT-2 Othello-GPT world-model smoke replication using the pinned "
            "NeelNanda/Othello-GPT-Transformer-Lens synthetic checkpoint, the "
            "existing ARENA board-sequence fixture, and the bundled linear probe. "
            "The report verifies real CUDA model loading, legal-move prediction, "
            "and linear board-state decoding; it does not claim new SAE/circuit "
            "discovery or a fresh probe-training replication."
        ),
        "datasets": [
            {
                "id": "board_seqs_id_small.npy",
                "source": "bundled_arena_othello_fixture",
                "games_used": NUM_GAMES,
                "moves_per_game": NUM_MOVES,
            },
            {
                "id": "board_seqs_square_small.npy",
                "source": "bundled_arena_othello_fixture",
                "games_used": NUM_GAMES,
                "moves_per_game": NUM_MOVES,
            },
        ],
        "date_run": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_level": "pinned_othello_gpt_cuda_legal_move_and_board_probe_smoke",
        "git_commit": git_commit(),
        "gpu_name": gpu_name,
        "gt_tier": "GT-2",
        "known_failures": known_failures,
        "metrics": {
            "gpu_evidence": {
                "category": "cuda_section_metric" if cuda_available else "missing",
                "placeholder_only": False,
                "section_specific_metric_keys": section_specific_keys,
                "uses_cuda": cuda_available,
            },
            "gpu_test": gpu_test,
            "notebook_contract": {
                "checks": checks,
                "tests_passed": accepted,
            },
        },
        "models": [
            {
                "gated": False,
                "id": REPO_ID,
                "precision": MODEL_PRECISION,
                "revision": REVISION,
                "source": "huggingface",
                "weights_file": CHECKPOINT_FILE,
            },
            {
                "gated": False,
                "id": "main_linear_probe.pth",
                "precision": MODEL_PRECISION,
                "revision": "bundled_arena_probe",
                "source": "local_fixture",
            },
        ],
        "negative_controls": {
            "declared_controls": [
                "random_legal_move_baseline",
                "swapped_player_parity_probe_negative_control",
            ]
        },
        "notebook_id": "1_5_3_othellogpt",
        "ood_tests": {
            "declared": (
                "held-out games from the bundled small fixture; no broader OOD "
                "board distribution is claimed by this smoke report"
            )
        },
        "peak_vram_gb": round(peak_vram_gb, 6),
        "report_inputs": build_report_input_manifest(
            ROOT,
            [
                Path(__file__),
                EVIDENCE_DIR / "artifacts.lock.yml",
                EVIDENCE_DIR / "verification_report.schema.json",
                OTHELLO_SECTION_DIR / "utils.py",
                OTHELLO_SECTION_DIR / "board_seqs_id_small.npy",
                OTHELLO_SECTION_DIR / "board_seqs_square_small.npy",
                OTHELLO_SECTION_DIR / "main_linear_probe.pth",
                ROOT / "scripts/run_extension_verification_reports.py",
                ROOT / "arena_ext/verification.py",
            ],
        ),
        "safety_notes": [
            "Othello-GPT is a board-game model; no unsafe prompts or personal data are used.",
            "No raw model weights are added to the repository; the checkpoint is loaded from the pinned Hugging Face revision.",
        ],
        "tests_passed": accepted,
        "wall_clock_seconds": round(wall_clock_seconds, 6),
    }


def main() -> None:
    report = run_verification()
    output_path = EVIDENCE_DIR / "verification_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path.relative_to(ROOT)}")
    print(
        json.dumps(
            {
                "accepted": report["accepted"],
                "gpu_name": report["gpu_name"],
                "legal_top1_accuracy": report["metrics"]["gpu_test"]["legal_top1_accuracy"],
                "board_probe_accuracy": report["metrics"]["gpu_test"]["board_probe_accuracy"],
                "peak_vram_gb": report["peak_vram_gb"],
                "known_failures": report["known_failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
