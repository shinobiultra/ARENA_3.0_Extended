from pathlib import Path

import torch as t

LIVE_MODULAR_ARCHAEOLOGY_MODULUS = 13
LIVE_MODULAR_ARCHAEOLOGY_EMBED_DIM = 32
LIVE_MODULAR_ARCHAEOLOGY_HIDDEN_DIM = 64
LIVE_MODULAR_ARCHAEOLOGY_LR = 5e-3
LIVE_MODULAR_ARCHAEOLOGY_STEPS = 80
LIVE_MODULAR_ARCHAEOLOGY_CHECKPOINT_STEPS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80]
LIVE_MODULAR_ARCHAEOLOGY_THRESHOLD = 0.9
LIVE_MODULAR_ARCHAEOLOGY_MIN_CONSECUTIVE = 2


class TinyModularAdditionMLP(t.nn.Module):
    """Tiny finite-table model organism for the checkpoint exercise."""

    def __init__(
        self,
        *,
        modulus: int = LIVE_MODULAR_ARCHAEOLOGY_MODULUS,
        embed_dim: int = LIVE_MODULAR_ARCHAEOLOGY_EMBED_DIM,
        hidden_dim: int = LIVE_MODULAR_ARCHAEOLOGY_HIDDEN_DIM,
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


def modular_addition_table(
    *,
    device: t.device,
    modulus: int = LIVE_MODULAR_ARCHAEOLOGY_MODULUS,
) -> tuple[t.Tensor, t.Tensor]:
    pairs = t.tensor(
        [[left, right] for left in range(modulus) for right in range(modulus)],
        device=device,
        dtype=t.long,
    )
    labels = (pairs[:, 0] + pairs[:, 1]) % modulus
    return pairs, labels


def checkpoint_metrics_from_file(
    checkpoint_path: Path,
    *,
    device: t.device,
    input_pairs: t.Tensor,
    true_labels: t.Tensor,
) -> tuple[float, float]:
    model = TinyModularAdditionMLP().to(device)
    state_dict = t.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    with t.inference_mode():
        logits = model(input_pairs)
        accuracy = logits.argmax(dim=-1).eq(true_labels).float().mean().item()
        loss = t.nn.functional.cross_entropy(logits, true_labels).item()
    return float(accuracy), float(loss)


def print_report(title: str, report: dict):
    print(f"\n{title}")
    for key, value in report.items():
        print(f"  {key}: {value}")
