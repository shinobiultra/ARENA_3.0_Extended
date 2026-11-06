"""Regenerate the three-seed CPU reference metrics and learner-facing assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch as t

import solutions


SECTION_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SECTION_DIR.parents[1] / "instructions" / "assets"
EXPECTED_DIR = SECTION_DIR / "expected_outputs"
SEEDS = (0, 1, 2)
STEPS = 1_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_world_asset(world: solutions.VideoWorldBatch, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.1), constrained_layout=True)
    for frame_index, axis in enumerate(axes):
        axis.imshow(world.videos[24, frame_index].cpu().permute(1, 2, 0))
        axis.set_title(f"t={frame_index}")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Exact toroidal motion: every next frame is a known transition")
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        metadata={"EvaluationProtocol": solutions.EVALUATION_PROTOCOL_VERSION},
    )
    plt.close(fig)


def main() -> None:
    t.set_num_threads(min(8, t.get_num_threads()))
    metrics_by_seed: dict[int, solutions.VJEPAMetrics] = {}
    seed_zero = None
    for seed in SEEDS:
        run = solutions.run_signature_experiment(device="cpu", steps=STEPS, seed=seed)
        model, history, metrics, train_world, test_world = run
        metrics_by_seed[seed] = metrics
        if seed == 0:
            seed_zero = (model, history, metrics, train_world, test_world)

    assert seed_zero is not None
    model, history, metrics, _, test_world = seed_zero
    intervals = solutions.summarize_multiseed(metrics_by_seed)
    predicted, target = solutions.predict_future_features(model, test_world.videos)
    retrieval = solutions.retrieval_details(predicted, target, test_world.clip_keys, test_world.clip_keys)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    signature_path = ASSETS_DIR / "vjepa_from_scratch_signature.png"
    world_path = ASSETS_DIR / "vjepa_from_scratch_world.png"
    save_world_asset(test_world, world_path)
    solutions.save_signature_figure(
        history,
        metrics,
        test_world,
        retrieval,
        signature_path,
        intervals,
    )

    per_seed = {str(seed): solutions.metrics_dict(values) for seed, values in metrics_by_seed.items()}
    interval_payload = {name: asdict(interval) for name, interval in intervals.items()}
    signature_fingerprint = solutions.evaluation_fingerprint(
        {"seed_metrics": per_seed["0"], "intervals": interval_payload}
    )
    payload = {
        "notebook_id": "14_3_vjepa_from_scratch",
        "protocol_version": solutions.EVALUATION_PROTOCOL_VERSION,
        "run": {
            "seeds": list(SEEDS),
            "train_videos_per_seed": 512,
            "heldout_videos_per_seed": 128,
            "steps_per_seed": STEPS,
            "device": "cpu",
        },
        "acceptance_thresholds": {
            "train_loss_ratio_max": 0.65,
            "random_label_state_accuracy_max": 0.25,
            "random_predictor_teacher_latent_cosine_max": 0.20,
            "random_predictor_retrieval_accuracy_max": 0.05,
            "exact_dynamics_teacher_latent_cosine_min": 0.999999,
            "exact_dynamics_retrieval_accuracy_min": 0.999999,
            "feature_std_min": 0.05,
            "effective_rank_min": 5.0,
        },
        "per_seed": per_seed,
        "intervals": interval_payload,
        "freshness": {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "solutions_sha256": sha256(SECTION_DIR / "solutions.py"),
            "generator_sha256": sha256(Path(__file__)),
            "world_asset_sha256": sha256(world_path),
            "signature_asset_sha256": sha256(signature_path),
            "signature_evaluation_fingerprint": signature_fingerprint,
        },
    }
    output_path = EXPECTED_DIR / "reference_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
