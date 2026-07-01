# %%
"""Reference solutions for [17.1] checkpoint archaeology."""

import sys
import tempfile
from pathlib import Path

import torch as t

chapter = "chapter17_training_dynamics"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.training_dynamics import (
    developmental_comparison_report,
    first_threshold_crossing,
    mechanism_emergence_report,
    monotonicity_violations,
    phase_transition_report,
    random_control_report,
    stable_threshold_step,
    toy_training_trajectories,
)

MAIN = __name__ == "__main__"

MODULAR_ARCHAEOLOGY_MODULUS = 13
MODULAR_ARCHAEOLOGY_EMBED_DIM = 32
MODULAR_ARCHAEOLOGY_HIDDEN_DIM = 64
MODULAR_ARCHAEOLOGY_LR = 5e-3
MODULAR_ARCHAEOLOGY_STEPS = 80
MODULAR_ARCHAEOLOGY_CHECKPOINT_STEPS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80]
MODULAR_ARCHAEOLOGY_THRESHOLD = 0.9
MODULAR_ARCHAEOLOGY_MIN_CONSECUTIVE = 2


# %%
def _report_dict(report) -> dict:
    return report.__dict__.copy()


def checkpoint_emergence_smoke_test() -> dict:
    steps, trajectories = toy_training_trajectories()
    return _report_dict(
        mechanism_emergence_report(
            steps,
            trajectories["autoregressive"],
            metric_name="induction_probe_accuracy",
            threshold=0.6,
            min_consecutive=2,
        )
    )


def phase_transition_smoke_test() -> dict:
    steps, trajectories = toy_training_trajectories()
    return _report_dict(
        phase_transition_report(
            steps,
            trajectories["autoregressive"],
            metric_name="induction_probe_accuracy",
            min_jump=0.3,
        )
    )


def random_control_smoke_test() -> dict:
    _, trajectories = toy_training_trajectories()
    return _report_dict(
        random_control_report(
            trajectories["random_control"],
            metric_name="label_shuffled_probe",
            max_allowed_value=0.2,
        )
    )


def developmental_comparison_smoke_test() -> dict:
    steps, trajectories = toy_training_trajectories()
    return _report_dict(
        developmental_comparison_report(
            steps,
            trajectories,
            threshold=0.6,
            min_consecutive=2,
        )
    )


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "checkpoint_emergence": checkpoint_emergence_smoke_test(),
        "phase_transition": phase_transition_smoke_test(),
        "random_control": random_control_smoke_test(),
        "developmental_comparison": developmental_comparison_smoke_test(),
        "live_checkpoint_archaeology": live_checkpoint_archaeology_smoke_test(device="cpu"),
    }


class _TinyModularAdditionMLP(t.nn.Module):
    """Tiny finite-table model organism for checkpoint archaeology."""

    def __init__(
        self,
        *,
        modulus: int = MODULAR_ARCHAEOLOGY_MODULUS,
        embed_dim: int = MODULAR_ARCHAEOLOGY_EMBED_DIM,
        hidden_dim: int = MODULAR_ARCHAEOLOGY_HIDDEN_DIM,
    ):
        super().__init__()
        self.modulus = modulus
        self.embed = t.nn.Embedding(modulus, embed_dim)
        self.mlp = t.nn.Sequential(
            t.nn.Linear(2 * embed_dim, hidden_dim),
            t.nn.GELU(),
            t.nn.Linear(hidden_dim, modulus),
        )

    def forward(self, input_pairs: t.Tensor) -> t.Tensor:
        embedded = self.embed(input_pairs)
        return self.mlp(embedded.flatten(start_dim=1))


def _modular_addition_table(
    *,
    device: t.device,
    modulus: int = MODULAR_ARCHAEOLOGY_MODULUS,
) -> tuple[t.Tensor, t.Tensor]:
    pairs = t.tensor(
        [[left, right] for left in range(modulus) for right in range(modulus)],
        device=device,
        dtype=t.long,
    )
    labels = (pairs[:, 0] + pairs[:, 1]) % modulus
    return pairs, labels


def _checkpoint_metrics_from_file(
    checkpoint_path: Path,
    *,
    device: t.device,
    input_pairs: t.Tensor,
    true_labels: t.Tensor,
) -> tuple[float, float]:
    model = _TinyModularAdditionMLP().to(device)
    state_dict = t.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    with t.inference_mode():
        logits = model(input_pairs)
        accuracy = logits.argmax(dim=-1).eq(true_labels).float().mean().item()
        loss = t.nn.functional.cross_entropy(logits, true_labels).item()
    return float(accuracy), float(loss)


def _train_and_save_modular_checkpoints(
    *,
    checkpoint_dir: Path,
    device: t.device,
    seed: int,
    random_labels: bool = False,
) -> dict:
    t.manual_seed(seed)
    if device.type == "cuda":
        t.cuda.manual_seed_all(seed)
    input_pairs, true_labels = _modular_addition_table(device=device)
    generator = t.Generator(device=device).manual_seed(seed + 1000)
    train_labels = true_labels
    if random_labels:
        train_labels = t.randint(
            0,
            MODULAR_ARCHAEOLOGY_MODULUS,
            true_labels.shape,
            generator=generator,
            device=device,
        )

    model = _TinyModularAdditionMLP().to(device)
    optimizer = t.optim.AdamW(
        model.parameters(),
        lr=MODULAR_ARCHAEOLOGY_LR,
        weight_decay=1e-3,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[Path] = []
    checkpoint_steps = set(MODULAR_ARCHAEOLOGY_CHECKPOINT_STEPS)
    for step in range(MODULAR_ARCHAEOLOGY_STEPS + 1):
        if step in checkpoint_steps:
            path = checkpoint_dir / f"step_{step:04d}.pt"
            t.save(model.state_dict(), path)
            checkpoint_paths.append(path)
        if step == MODULAR_ARCHAEOLOGY_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = t.nn.functional.cross_entropy(model(input_pairs), train_labels)
        loss.backward()
        optimizer.step()

    reloaded_accuracies: list[float] = []
    reloaded_losses: list[float] = []
    for path in checkpoint_paths:
        accuracy, loss = _checkpoint_metrics_from_file(
            path,
            device=device,
            input_pairs=input_pairs,
            true_labels=true_labels,
        )
        reloaded_accuracies.append(accuracy)
        reloaded_losses.append(loss)

    return {
        "steps": t.tensor(MODULAR_ARCHAEOLOGY_CHECKPOINT_STEPS, device=device),
        "accuracies": t.tensor(reloaded_accuracies, device=device),
        "losses": t.tensor(reloaded_losses, device=device),
        "checkpoint_count": len(checkpoint_paths),
        "checkpoint_total_bytes": sum(path.stat().st_size for path in checkpoint_paths),
    }


def live_checkpoint_archaeology_smoke_test(
    checkpoint_root: Path | None = None,
    *,
    device: str | t.device = "cpu",
    seed: int = 0,
) -> dict:
    """Train, save, reload, analyze, and control a tiny checkpoint run live."""

    device = t.device(device)
    if device.type == "cuda" and not t.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is not available.")

    def _run(root: Path) -> dict:
        target = _train_and_save_modular_checkpoints(
            checkpoint_dir=root / "target_labels",
            device=device,
            seed=seed,
            random_labels=False,
        )
        random_control = _train_and_save_modular_checkpoints(
            checkpoint_dir=root / "random_labels",
            device=device,
            seed=seed,
            random_labels=True,
        )
        checkpoint_files = sorted(root.glob("*/*.pt"))

        target_emergence = mechanism_emergence_report(
            target["steps"],
            target["accuracies"],
            metric_name="modular_addition_table_accuracy",
            threshold=MODULAR_ARCHAEOLOGY_THRESHOLD,
            min_consecutive=MODULAR_ARCHAEOLOGY_MIN_CONSECUTIVE,
        )
        target_phase = phase_transition_report(
            target["steps"],
            target["accuracies"],
            metric_name="modular_addition_table_accuracy",
            min_jump=0.2,
        )
        random_report = random_control_report(
            random_control["accuracies"],
            metric_name="random_label_true_table_accuracy",
            max_allowed_value=0.2,
        )
        checkpoint_count = target["checkpoint_count"] + random_control["checkpoint_count"]
        final_accuracy = float(target["accuracies"][-1].item())
        random_control_peak = float(random_control["accuracies"].max().item())
        preflight_passed = (
            target_emergence.emerged
            and final_accuracy >= 0.99
            and target_phase.phase_transition_detected
            and random_report.control_passed
            and random_control_peak <= 0.2
            and len(checkpoint_files) == checkpoint_count
            and checkpoint_count == 2 * len(MODULAR_ARCHAEOLOGY_CHECKPOINT_STEPS)
        )

        return {
            "preflight_passed": preflight_passed,
            "device": str(device),
            "model_family": "tiny_modular_addition_mlp",
            "modulus": MODULAR_ARCHAEOLOGY_MODULUS,
            "table_example_count": MODULAR_ARCHAEOLOGY_MODULUS**2,
            "complete_finite_domain_evaluated": True,
            "ood_generalization_claimed": False,
            "generalization_scope": (
                "Complete finite mod-13 addition table (169/169 input pairs); "
                "no held-out OOD extrapolation is claimed."
            ),
            "checkpoint_steps": MODULAR_ARCHAEOLOGY_CHECKPOINT_STEPS,
            "checkpoint_count": checkpoint_count,
            "checkpoint_files_written": len(checkpoint_files),
            "checkpoint_total_bytes": (
                target["checkpoint_total_bytes"] + random_control["checkpoint_total_bytes"]
            ),
            "target_accuracy_trajectory": target["accuracies"].detach().cpu().tolist(),
            "target_loss_trajectory": target["losses"].detach().cpu().tolist(),
            "random_control_accuracy_trajectory": random_control["accuracies"]
            .detach()
            .cpu()
            .tolist(),
            "first_crossing_step": target_emergence.first_crossing_step,
            "stable_from_step": target_emergence.stable_from_step,
            "final_accuracy": final_accuracy,
            "phase_transition_step": target_phase.transition_step,
            "phase_transition_jump": target_phase.jump,
            "phase_transition_detected": target_phase.phase_transition_detected,
            "random_control_peak_accuracy": random_report.peak_value,
            "random_control_passed": random_report.control_passed,
            "real_checkpoints_reloaded": True,
        }

    if checkpoint_root is not None:
        root = Path(checkpoint_root)
        root.mkdir(parents=True, exist_ok=True)
        return _run(root)

    with tempfile.TemporaryDirectory(prefix="arena17_live_checkpoint_archaeology_") as tmp:
        return _run(Path(tmp))


def run_modular_addition_checkpoint_archaeology_preflight(
    max_vram_gb: float = 24.0,
) -> dict:
    """Train a tiny model organism and analyze real saved checkpoints."""

    if not t.cuda.is_available():
        raise RuntimeError("Checkpoint archaeology GPU verification requires CUDA.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix="arena17_checkpoint_archaeology_") as tmp:
        tmp_dir = Path(tmp)
        target = _train_and_save_modular_checkpoints(
            checkpoint_dir=tmp_dir / "target_labels",
            device=device,
            seed=0,
            random_labels=False,
        )
        random_control = _train_and_save_modular_checkpoints(
            checkpoint_dir=tmp_dir / "random_labels",
            device=device,
            seed=0,
            random_labels=True,
        )

    target_emergence = mechanism_emergence_report(
        target["steps"],
        target["accuracies"],
        metric_name="modular_addition_table_accuracy",
        threshold=MODULAR_ARCHAEOLOGY_THRESHOLD,
        min_consecutive=MODULAR_ARCHAEOLOGY_MIN_CONSECUTIVE,
    )
    target_phase = phase_transition_report(
        target["steps"],
        target["accuracies"],
        metric_name="modular_addition_table_accuracy",
        min_jump=0.2,
    )
    random_report = random_control_report(
        random_control["accuracies"],
        metric_name="random_label_true_table_accuracy",
        max_allowed_value=0.2,
    )
    accuracy_gain = target["accuracies"][-1] - target["accuracies"][0]
    loss_drop = target["losses"][0] - target["losses"][-1]
    checkpoint_count = target["checkpoint_count"] + random_control["checkpoint_count"]
    checkpoint_total_bytes = (
        target["checkpoint_total_bytes"] + random_control["checkpoint_total_bytes"]
    )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    within_vram_budget = peak_vram_gb <= max_vram_gb
    preflight_passed = (
        target_emergence.emerged
        and target_emergence.stable_from_step == 30
        and target["accuracies"][-1].item() == 1.0
        and target_phase.phase_transition_detected
        and random_report.control_passed
        and random_control["accuracies"].max().item() <= 0.2
        and checkpoint_count == 2 * len(MODULAR_ARCHAEOLOGY_CHECKPOINT_STEPS)
        and within_vram_budget
    )

    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "preflight_passed": preflight_passed,
        "model_family": "tiny_modular_addition_mlp",
        "modulus": MODULAR_ARCHAEOLOGY_MODULUS,
        "table_example_count": MODULAR_ARCHAEOLOGY_MODULUS**2,
        "complete_finite_domain_evaluated": True,
        "ood_generalization_claimed": False,
        "generalization_scope": (
            "Complete finite mod-13 addition table (169/169 input pairs); "
            "no held-out OOD extrapolation is claimed."
        ),
        "checkpoint_steps": MODULAR_ARCHAEOLOGY_CHECKPOINT_STEPS,
        "checkpoint_count": checkpoint_count,
        "checkpoint_total_bytes": checkpoint_total_bytes,
        "target_accuracy_trajectory": target["accuracies"].detach().cpu().tolist(),
        "target_loss_trajectory": target["losses"].detach().cpu().tolist(),
        "random_control_accuracy_trajectory": random_control["accuracies"]
        .detach()
        .cpu()
        .tolist(),
        "accuracy_gain_from_training": float(accuracy_gain.item()),
        "loss_drop": float(loss_drop.item()),
        "first_crossing_step": target_emergence.first_crossing_step,
        "stable_from_step": target_emergence.stable_from_step,
        "peak_step": target_emergence.peak_step,
        "peak_accuracy": target_emergence.peak_value,
        "final_accuracy": float(target["accuracies"][-1].item()),
        "phase_transition_step": target_phase.transition_step,
        "phase_transition_jump": target_phase.jump,
        "phase_transition_detected": target_phase.phase_transition_detected,
        "random_control_peak_accuracy": random_report.peak_value,
        "random_control_passed": random_report.control_passed,
        "real_checkpoints_reloaded": True,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": within_vram_budget,
        "full_path": (
            "Train a tiny modular-addition model organism on CUDA, save and reload "
            "real checkpoints, detect stable mechanism emergence, and reject a "
            "random-label checkpoint control. The finite-domain generalization "
            "claim is exhaustive coverage of all 169 mod-13 input pairs, not OOD "
            "extrapolation."
        ),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    return run_modular_addition_checkpoint_archaeology_preflight(
        max_vram_gb=max_vram_gb,
    )


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
