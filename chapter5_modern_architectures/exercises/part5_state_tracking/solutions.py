# %%


MAIN = __name__ == "__main__"

# %%

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch as t
from torch import nn
import torch.nn.functional as F

chapter = "chapter5_modern_architectures"
section = "part5_state_tracking"
root_dir = Path(__file__).resolve().parents[3]
exercises_dir = root_dir / chapter / "exercises"
assets_dir = exercises_dir / section / "media"
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part5_state_tracking.tests as tests
from part5_state_tracking.mamba_backbone import MambaConfig, TinyMambaModel

if MAIN:
    t.set_num_threads(min(4, t.get_num_threads()))
SAVE_FIGURES = True


@dataclass(frozen=True)
class StateTrackingBatch:
    tokens: t.Tensor
    states: t.Tensor
    task: str
    vocab: dict[int, str]

# %%

def generate_bracket_depth_task(
    batch: int,
    seq_len: int,
    *,
    max_depth: int = 4,
    seed: int = 0,
) -> StateTrackingBatch:
    """Generate bracket actions labelled by the bounded nonnegative stack depth."""

    generator = t.Generator().manual_seed(seed)
    tokens = t.zeros(batch, seq_len, dtype=t.long)
    states = t.zeros(batch, seq_len, dtype=t.long)
    depth = t.zeros(batch, dtype=t.long)

    for pos in range(seq_len):
        proposed_open = t.randint(0, 2, (batch,), generator=generator).bool()
        must_open = depth == 0
        must_close = depth == max_depth
        open_token = (proposed_open | must_open) & ~must_close
        depth = depth + t.where(open_token, t.ones_like(depth), -t.ones_like(depth))
        tokens[:, pos] = open_token.long()
        states[:, pos] = depth

    return StateTrackingBatch(
        tokens=tokens,
        states=states,
        task="bracket_depth",
        vocab={0: ")", 1: "("},
    )

# %%

if MAIN:
    tests.test_generate_bracket_depth_task_is_bounded_and_consistent(
        generate_bracket_depth_task
    )

if MAIN:
    cold_open = generate_bracket_depth_task(batch=1, seq_len=24, max_depth=3, seed=7)
if MAIN:
    deltas = t.where(cold_open.tokens.bool(), 1, -1)
if MAIN:
    exact = t.equal(cold_open.states, deltas.cumsum(dim=-1))
if MAIN:
    actions = "".join(cold_open.vocab[int(token)] for token in cold_open.tokens[0])
if MAIN:
    print("actions:", actions)
if MAIN:
    print("depths: ", " ".join(map(str, cold_open.states[0].tolist())))
if MAIN:
    print(
        f"exact recurrence: {exact}; bounds: "
        f"[{cold_open.states.min().item()}, {cold_open.states.max().item()}]"
    )

if MAIN:
    fig, ax = plt.subplots(figsize=(10, 2.8))
if MAIN:
    positions = t.arange(cold_open.tokens.shape[1])
if MAIN:
    ax.step(positions, cold_open.states[0], where="mid", linewidth=2.2, color="#176b87")
if MAIN:
    open_mask = cold_open.tokens[0].bool()
if MAIN:
    ax.scatter(
        positions[open_mask], cold_open.states[0, open_mask],
        color="#d1495b", edgecolor="white", linewidth=0.7, label="(", zorder=3,
    )
if MAIN:
    ax.scatter(
        positions[~open_mask], cold_open.states[0, ~open_mask],
        color="#3f51b5", edgecolor="white", linewidth=0.7, label=")", zorder=3,
    )
if MAIN:
    ax.set(title="Exact bracket-depth trajectory", xlabel="position", ylabel="depth")
if MAIN:
    ax.set_yticks(range(4))
if MAIN:
    ax.grid(axis="y", alpha=0.2)
if MAIN:
    ax.legend(title="action", frameon=False, ncol=2)
if MAIN:
    plt.show()

# %%

class TinyMambaStateClassifier(nn.Module):
    """Tiny supervised Mamba organism for bracket-depth state tracking."""

    def __init__(self, num_states: int = 4):
        super().__init__()
        config = MambaConfig(
            vocab_size=2,
            d_model=32,
            d_inner=64,
            d_state=8,
            d_conv=3,
            dt_rank=4,
            num_layers=1,
            tie_word_embeddings=False,
        )
        self.backbone = TinyMambaModel(config)
        self.head = nn.Linear(config.d_model, num_states)

    def encode(self, input_ids: t.Tensor) -> t.Tensor:
        hidden_states, _ = self.backbone(input_ids)
        return hidden_states

    def forward(
        self,
        input_ids: t.Tensor,
        *,
        return_hidden_states: bool = False,
    ) -> t.Tensor | tuple[t.Tensor, t.Tensor]:
        hidden_states = self.encode(input_ids)
        logits = self.head(hidden_states)
        if return_hidden_states:
            return logits, hidden_states
        return logits

def train_state_tracker_cpu(
    *,
    steps: int = 160,
    batch_size: int = 64,
    seq_len: int = 16,
    max_depth: int = 3,
    seed: int = 0,
) -> tuple[TinyMambaStateClassifier, list[float]]:
    """Train the section 5.3 tiny Mamba path on exact bracket-depth labels."""

    t.manual_seed(seed)
    model = TinyMambaStateClassifier(num_states=max_depth + 1).cpu()
    optimizer = t.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    losses: list[float] = []

    model.train()
    for step in range(steps):
        batch = generate_bracket_depth_task(
            batch=batch_size,
            seq_len=seq_len,
            max_depth=max_depth,
            seed=seed + step,
        )
        logits = model(batch.tokens)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            batch.states.flatten(),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    return model.eval(), losses

# %%

if MAIN:
    tests.test_mamba_state_tracker_shapes(TinyMambaStateClassifier)
if MAIN:
    tests.test_cpu_training_reduces_loss(train_state_tracker_cpu)

if MAIN:
    model, training_losses = train_state_tracker_cpu(
        steps=160,
        batch_size=64,
        seq_len=16,
        max_depth=3,
        seed=0,
    )


@t.inference_mode()
def evaluate_tracker(model, *, seq_len: int, seed: int, batch_size: int = 128):
    batch = generate_bracket_depth_task(
        batch=batch_size,
        seq_len=seq_len,
        max_depth=3,
        seed=seed,
    )
    logits = model(batch.tokens)
    predictions = logits.argmax(dim=-1)
    accuracy = predictions.eq(batch.states).float().mean().item()
    late_accuracy = predictions[:, -8:].eq(batch.states[:, -8:]).float().mean().item()
    position_accuracy = predictions.eq(batch.states).float().mean(dim=0)
    return batch, logits, accuracy, late_accuracy, position_accuracy


if MAIN:
    id_batch, id_logits, id_accuracy, id_late_accuracy, id_position_accuracy = evaluate_tracker(
        model, seq_len=16, seed=1000
    )
if MAIN:
    ood_batch, ood_logits, ood_accuracy, ood_late_accuracy, ood_position_accuracy = evaluate_tracker(
        model, seq_len=32, seed=2000
    )
if MAIN:
    majority_baseline = (
        t.bincount(ood_batch.states.flatten(), minlength=4).max().item()
        / ood_batch.states.numel()
    )
if MAIN:
    print(f"loss: {training_losses[0]:.3f} -> {training_losses[-1]:.3f}")
if MAIN:
    print(f"ID length 16 accuracy: {id_accuracy:.3f}")
if MAIN:
    print(f"OOD length 32 accuracy: {ood_accuracy:.3f}")
if MAIN:
    print(f"OOD final-8 accuracy: {ood_late_accuracy:.3f}")
if MAIN:
    print(f"majority-depth baseline: {majority_baseline:.3f}")

if MAIN:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
if MAIN:
    axes[0].plot(training_losses, color="#176b87")
if MAIN:
    axes[0].set(title="Training curve", xlabel="optimizer step", ylabel="cross-entropy")
if MAIN:
    axes[1].plot(range(16), id_position_accuracy, label="ID length 16", color="#176b87")
if MAIN:
    axes[1].plot(range(32), ood_position_accuracy, label="OOD length 32", color="#d1495b")
if MAIN:
    axes[1].axhline(majority_baseline, color="#555555", linestyle="--", label="majority")
if MAIN:
    axes[1].set(title="Accuracy by position", xlabel="position", ylabel="accuracy", ylim=(0, 1.03))
if MAIN:
    axes[1].legend(frameon=False)
if MAIN:
    fig.tight_layout()
if MAIN:
    plt.show()

# %%

@dataclass(frozen=True)
class StateProbe:
    """Standardized linear probe over flattened recurrent SSM states."""

    mean: t.Tensor
    scale: t.Tensor
    weight: t.Tensor
    bias: t.Tensor

@t.inference_mode()
def collect_recurrent_states(
    model: TinyMambaStateClassifier,
    tokens: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Run cached one-token steps and collect logits plus the first-layer SSM state."""

    cache = None
    logits_by_position: list[t.Tensor] = []
    states_by_position: list[t.Tensor] = []
    for position in range(tokens.shape[1]):
        hidden, cache = model.backbone(
            tokens[:, position : position + 1],
            states=cache,
            use_cache=True,
        )
        assert cache is not None and len(cache) == 1
        logits_by_position.append(model.head(hidden))
        states_by_position.append(cache[0].ssm_state.flatten(start_dim=1))

    return t.cat(logits_by_position, dim=1), t.stack(states_by_position, dim=1)

def fit_state_probe(
    recurrent_states: t.Tensor,
    labels: t.Tensor,
    *,
    num_classes: int | None = None,
    ridge: float = 1e-2,
    shuffle_labels: bool = False,
    seed: int = 0,
) -> StateProbe:
    """Fit a standardized closed-form ridge probe to recurrent SSM states."""

    if recurrent_states.shape[:-1] != labels.shape:
        raise ValueError("recurrent state leading dimensions must match labels")
    x = recurrent_states.flatten(0, -2).float()
    y = labels.flatten().long()
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1
    if shuffle_labels:
        generator = t.Generator(device=y.device).manual_seed(seed)
        y = y[t.randperm(y.numel(), generator=generator, device=y.device)]

    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-4)
    standardized = (x - mean) / scale
    design = t.cat([standardized, t.ones(x.shape[0], 1, device=x.device)], dim=-1)
    targets = F.one_hot(y, num_classes=num_classes).float()
    penalty = t.eye(design.shape[-1], device=x.device, dtype=x.dtype)
    penalty[-1, -1] = 0.0
    solution = t.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ targets,
    )
    return StateProbe(
        mean=mean,
        scale=scale,
        weight=solution[:-1],
        bias=solution[-1],
    )

def state_probe_logits(recurrent_states: t.Tensor, probe: StateProbe) -> t.Tensor:
    standardized = (recurrent_states.float() - probe.mean) / probe.scale
    return standardized @ probe.weight + probe.bias

def state_probe_accuracy(
    recurrent_states: t.Tensor,
    labels: t.Tensor,
    probe: StateProbe,
    mask: t.Tensor | None = None,
) -> float:
    predictions = state_probe_logits(recurrent_states, probe).argmax(dim=-1)
    if mask is not None:
        predictions = predictions[mask]
        labels = labels[mask]
    return float(predictions.eq(labels).float().mean().item())

# %%

if MAIN:
    tests.test_collect_recurrent_states_matches_full_forward(collect_recurrent_states)
if MAIN:
    tests.test_state_probe_recovers_exact_control(fit_state_probe, state_probe_accuracy)

if MAIN:
    probe_train_batch = generate_bracket_depth_task(96, 16, max_depth=3, seed=4000)
if MAIN:
    probe_id_batch = generate_bracket_depth_task(96, 16, max_depth=3, seed=4001)
if MAIN:
    probe_ood_batch = generate_bracket_depth_task(96, 32, max_depth=3, seed=4002)

if MAIN:
    _, probe_train_states = collect_recurrent_states(model, probe_train_batch.tokens)
if MAIN:
    _, probe_id_states = collect_recurrent_states(model, probe_id_batch.tokens)
if MAIN:
    _, probe_ood_states = collect_recurrent_states(model, probe_ood_batch.tokens)

if MAIN:
    state_probe = fit_state_probe(probe_train_states, probe_train_batch.states)
if MAIN:
    shuffled_probe = fit_state_probe(
        probe_train_states,
        probe_train_batch.states,
        shuffle_labels=True,
        seed=77,
    )
if MAIN:
    late_mask = t.zeros_like(probe_ood_batch.states, dtype=t.bool)
if MAIN:
    late_mask[:, 16:] = True
if MAIN:
    probe_metrics = {
        "train": state_probe_accuracy(probe_train_states, probe_train_batch.states, state_probe),
        "heldout_id": state_probe_accuracy(probe_id_states, probe_id_batch.states, state_probe),
        "ood_all": state_probe_accuracy(probe_ood_states, probe_ood_batch.states, state_probe),
        "ood_late": state_probe_accuracy(
            probe_ood_states, probe_ood_batch.states, state_probe, late_mask
        ),
        "shuffled_ood": state_probe_accuracy(
            probe_ood_states, probe_ood_batch.states, shuffled_probe
        ),
    }
if MAIN:
    for name, value in probe_metrics.items():
        print(f"{name:>13}: {value:.3f}")

# %%

@t.inference_mode()
def cache_after_position(
    model: TinyMambaStateClassifier,
    tokens: t.Tensor,
    position: int,
):
    """Return the recurrent cache after consuming tokens through `position`."""

    if not 0 <= position < tokens.shape[1]:
        raise ValueError("position is outside the sequence")
    cache = None
    for index in range(position + 1):
        _, cache = model.backbone(
            tokens[:, index : index + 1],
            states=cache,
            use_cache=True,
        )
    assert cache is not None
    return cache

def transplant_ssm_state(source_cache, donor_cache):
    """Copy only the donor SSM state, preserving the source convolutional history."""

    if len(source_cache) != 1 or len(donor_cache) != 1:
        raise ValueError("this lesson expects a one-layer tiny Mamba")
    state_type = type(source_cache[0])
    return (
        state_type(
            conv_state=source_cache[0].conv_state,
            ssm_state=donor_cache[0].ssm_state,
        ),
    )

def matched_random_state_edit(source_cache, donor_cache, *, seed: int = 0):
    """Add a random SSM-state delta with the transplant delta's exact norm."""

    source_state = source_cache[0]
    target_delta = donor_cache[0].ssm_state - source_state.ssm_state
    generator = t.Generator(device=target_delta.device).manual_seed(seed)
    random_delta = t.randn(
        target_delta.shape,
        generator=generator,
        device=target_delta.device,
        dtype=target_delta.dtype,
    )
    random_delta = random_delta / random_delta.norm().clamp_min(1e-8)
    random_delta = random_delta * target_delta.norm()
    state_type = type(source_state)
    return (
        state_type(
            conv_state=source_state.conv_state,
            ssm_state=source_state.ssm_state + random_delta,
        ),
    )

@t.inference_mode()
def continue_from_cache(
    model: TinyMambaStateClassifier,
    suffix_tokens: t.Tensor,
    cache,
) -> t.Tensor:
    """Continue cached Mamba inference and return one logit vector per suffix token."""

    outputs: list[t.Tensor] = []
    for position in range(suffix_tokens.shape[1]):
        hidden, cache = model.backbone(
            suffix_tokens[:, position : position + 1],
            states=cache,
            use_cache=True,
        )
        outputs.append(model.head(hidden))
    return t.cat(outputs, dim=1)

def make_matched_state_pair() -> tuple[t.Tensor, t.Tensor, int]:
    """Return prefixes with different depths but identical local history and suffix."""

    source = t.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])
    donor = t.tensor([[1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])
    return source, donor, 5

@t.inference_mode()
def run_state_transplant(
    model: TinyMambaStateClassifier,
    source_tokens: t.Tensor,
    donor_tokens: t.Tensor,
    edit_position: int,
    *,
    random_seed: int = 0,
) -> dict[str, t.Tensor | float]:
    """Compare an exact donor-state transplant with a matched random SSM edit."""

    conv_width = model.backbone.config.d_conv - 1
    local_start = max(0, edit_position - conv_width + 1)
    if not t.equal(
        source_tokens[:, local_start : edit_position + 1],
        donor_tokens[:, local_start : edit_position + 1],
    ):
        raise ValueError("source and donor must share the convolutional history at the edit")
    if not t.equal(source_tokens[:, edit_position + 1 :], donor_tokens[:, edit_position + 1 :]):
        raise ValueError("source and donor must share the suffix after the edit")

    source_cache = cache_after_position(model, source_tokens, edit_position)
    donor_cache = cache_after_position(model, donor_tokens, edit_position)
    suffix = source_tokens[:, edit_position + 1 :]
    patched_cache = transplant_ssm_state(source_cache, donor_cache)
    random_cache = matched_random_state_edit(
        source_cache,
        donor_cache,
        seed=random_seed,
    )

    source_logits = model(source_tokens)[:, edit_position + 1 :]
    donor_logits = model(donor_tokens)[:, edit_position + 1 :]
    patched_logits = continue_from_cache(model, suffix, patched_cache)
    random_logits = continue_from_cache(model, suffix, random_cache)
    donor_states = t.where(donor_tokens.bool(), 1, -1).cumsum(dim=-1)[:, edit_position + 1 :]

    def score(logits: t.Tensor) -> tuple[float, float]:
        probabilities = logits.softmax(dim=-1)
        predictions = probabilities.argmax(dim=-1)
        accuracy = predictions.eq(donor_states).float().mean().item()
        target_probability = probabilities.gather(-1, donor_states[..., None]).mean().item()
        return float(accuracy), float(target_probability)

    source_match, source_probability = score(source_logits)
    patched_match, patched_probability = score(patched_logits)
    random_match, random_probability = score(random_logits)
    return {
        "source_logits": source_logits,
        "donor_logits": donor_logits,
        "patched_logits": patched_logits,
        "random_logits": random_logits,
        "donor_states": donor_states,
        "source_match": source_match,
        "source_target_probability": source_probability,
        "patched_match": patched_match,
        "patched_target_probability": patched_probability,
        "random_match": random_match,
        "random_target_probability": random_probability,
    }

# %%

if MAIN:
    tests.test_state_transplant_matches_donor_dynamics(
        run_state_transplant,
        make_matched_state_pair,
    )

if MAIN:
    source_tokens, donor_tokens, edit_position = make_matched_state_pair()
if MAIN:
    transplant = run_state_transplant(
        model,
        source_tokens,
        donor_tokens,
        edit_position,
        random_seed=0,
    )
if MAIN:
    print(f"source donor-trajectory match: {transplant['source_match']:.3f}")
if MAIN:
    print(f"targeted transplant donor-trajectory match: {transplant['patched_match']:.3f}")
if MAIN:
    print(f"matched random donor-trajectory match: {transplant['random_match']:.3f}")
if MAIN:
    print(f"targeted mean donor-state probability: {transplant['patched_target_probability']:.3f}")
if MAIN:
    print(f"random mean donor-state probability: {transplant['random_target_probability']:.6f}")

# %%

if MAIN:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

if MAIN:
    ax = axes[0, 0]
if MAIN:
    positions = t.arange(cold_open.tokens.shape[1])
if MAIN:
    ax.step(positions, cold_open.states[0], where="mid", color="#176b87", linewidth=2.2)
if MAIN:
    open_mask = cold_open.tokens[0].bool()
if MAIN:
    ax.scatter(
        positions[open_mask], cold_open.states[0, open_mask],
        color="#d1495b", edgecolor="white", linewidth=0.6, label="(",
    )
if MAIN:
    ax.scatter(
        positions[~open_mask], cold_open.states[0, ~open_mask],
        color="#3f51b5", edgecolor="white", linewidth=0.6, label=")",
    )
if MAIN:
    ax.set(title="A. Exact latent trajectory", xlabel="position", ylabel="depth")
if MAIN:
    ax.set_yticks(range(4))
if MAIN:
    ax.legend(title="action", frameon=False, ncol=2)

if MAIN:
    ax = axes[0, 1]
if MAIN:
    model_names = ["ID\n16", "OOD\n32", "OOD\nlast 8", "majority\ncontrol"]
if MAIN:
    model_values = [id_accuracy, ood_accuracy, ood_late_accuracy, majority_baseline]
if MAIN:
    bars = ax.bar(model_names, model_values, color=["#176b87", "#3a86a8", "#65a9bd", "#777777"])
if MAIN:
    ax.bar_label(bars, fmt="%.3f", padding=3)
if MAIN:
    ax.set(title="B. Model generalization", ylabel="token accuracy", ylim=(0, 1.08))

if MAIN:
    ax = axes[1, 0]
if MAIN:
    probe_names = ["held-out\nID", "OOD\nall", "OOD\nlate", "shuffled\nlabels"]
if MAIN:
    probe_values = [
        probe_metrics["heldout_id"],
        probe_metrics["ood_all"],
        probe_metrics["ood_late"],
        probe_metrics["shuffled_ood"],
    ]
if MAIN:
    bars = ax.bar(probe_names, probe_values, color=["#2a9d8f", "#58aa92", "#8ab89b", "#d1495b"])
if MAIN:
    ax.bar_label(bars, fmt="%.3f", padding=3)
if MAIN:
    ax.set(title="C. SSM-state probe and control", ylabel="depth accuracy", ylim=(0, 1.08))

if MAIN:
    donor_truth = transplant["donor_states"][0]
if MAIN:
    source_predictions = transplant["source_logits"].argmax(dim=-1)[0]
if MAIN:
    patched_predictions = transplant["patched_logits"].argmax(dim=-1)[0]
if MAIN:
    random_predictions = transplant["random_logits"].argmax(dim=-1)[0]
if MAIN:
    trajectory_matrix = t.stack(
        [donor_truth, source_predictions, patched_predictions, random_predictions]
    )
if MAIN:
    ax = axes[1, 1]
if MAIN:
    image = ax.imshow(trajectory_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=3)
if MAIN:
    ax.set(
        title="D. Future depth after cache edit",
        xlabel="steps after intervention",
        yticks=range(4),
        yticklabels=["exact donor", "source", "targeted", "random"],
    )
if MAIN:
    for row in range(trajectory_matrix.shape[0]):
        for col in range(trajectory_matrix.shape[1]):
            ax.text(col, row, int(trajectory_matrix[row, col]), ha="center", va="center", color="white")
if MAIN:
    fig.colorbar(image, ax=ax, label="predicted depth", ticks=range(4), shrink=0.8)

if MAIN:
    if SAVE_FIGURES:
        fig.savefig(assets_dir / "mamba_state_tracking_signature.png", dpi=180, bbox_inches="tight")
if MAIN:
    plt.show()

# %%

# Change these values and rerun this cell.
if MAIN:
    PLAY_SEQ_LEN = 48
if MAIN:
    PLAY_SEED = 3100

if MAIN:
    play_batch, play_logits, play_accuracy, play_late, play_position_accuracy = evaluate_tracker(
        model,
        seq_len=PLAY_SEQ_LEN,
        seed=PLAY_SEED,
    )
if MAIN:
    below = (play_position_accuracy < 0.8).nonzero(as_tuple=False).flatten()
if MAIN:
    first_below = None if below.numel() == 0 else int(below[0])
if MAIN:
    print(f"length {PLAY_SEQ_LEN} accuracy: {play_accuracy:.3f}")
if MAIN:
    print(f"final-8 accuracy: {play_late:.3f}")
if MAIN:
    print(f"first position below 0.8: {first_below}")

if MAIN:
    plt.figure(figsize=(9, 2.8))
if MAIN:
    plt.plot(play_position_accuracy, color="#176b87")
if MAIN:
    plt.axhline(0.8, color="#d1495b", linestyle="--")
if MAIN:
    plt.ylim(0, 1.03)
if MAIN:
    plt.xlabel("position")
if MAIN:
    plt.ylabel("accuracy")
if MAIN:
    plt.title("Play: where does length generalization weaken?")
if MAIN:
    plt.show()

# %%

def find_confident_errors(
    tokens: t.Tensor,
    labels: t.Tensor,
    logits: t.Tensor,
    *,
    k: int = 8,
) -> list[dict[str, object]]:
    """Return the most confident wrong OOD predictions with their exact prefixes."""

    probabilities = logits.softmax(dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    wrong = predictions.ne(labels)
    candidates = wrong.nonzero(as_tuple=False)
    if candidates.numel() == 0:
        return []
    scores = confidence[wrong]
    order = scores.argsort(descending=True)[:k]
    records: list[dict[str, object]] = []
    for candidate_index in order:
        batch_index, position = candidates[int(candidate_index)]
        prefix = "".join(
            "(" if int(token) == 1 else ")"
            for token in tokens[batch_index, : position + 1]
        )
        records.append(
            {
                "sequence": int(batch_index),
                "position": int(position),
                "prefix": prefix,
                "true_depth": int(labels[batch_index, position]),
                "predicted_depth": int(predictions[batch_index, position]),
                "confidence": float(confidence[batch_index, position]),
            }
        )
    return records

# %%

if MAIN:
    tests.test_find_confident_errors_orders_real_mistakes(find_confident_errors)
if MAIN:
    anomalies = find_confident_errors(ood_batch.tokens, ood_batch.states, ood_logits, k=8)
if MAIN:
    for item in anomalies:
        print(
            f"seq={item['sequence']:>3} pos={item['position']:>2} "
            f"true={item['true_depth']} pred={item['predicted_depth']} "
            f"confidence={item['confidence']:.3f} prefix={item['prefix']}"
        )


def binned_error_rate(logits: t.Tensor, labels: t.Tensor, *, bin_width: int = 4) -> t.Tensor:
    predictions = logits.argmax(dim=-1)
    num_states = logits.shape[-1]
    num_bins = (labels.shape[1] + bin_width - 1) // bin_width
    rates = t.full((num_states, num_bins), float("nan"))
    for state in range(num_states):
        for bin_index in range(num_bins):
            start = bin_index * bin_width
            stop = min(labels.shape[1], start + bin_width)
            mask = labels[:, start:stop].eq(state)
            if mask.any():
                correct = predictions[:, start:stop].eq(labels[:, start:stop])
                rates[state, bin_index] = 1 - correct[mask].float().mean()
    return rates


if MAIN:
    error_heatmap = binned_error_rate(ood_logits, ood_batch.states)
if MAIN:
    fig, ax = plt.subplots(figsize=(9, 3.4))
if MAIN:
    image = ax.imshow(error_heatmap, aspect="auto", cmap="Reds", vmin=0, vmax=0.35)
if MAIN:
    ax.set(
        title="OOD error rate by true depth and position bin",
        xlabel="position bin (width 4)",
        ylabel="true depth",
        yticks=range(4),
        xticks=range(error_heatmap.shape[1]),
        xticklabels=[f"{4*i}-{4*i+3}" for i in range(error_heatmap.shape[1])],
    )
if MAIN:
    for row in range(error_heatmap.shape[0]):
        for col in range(error_heatmap.shape[1]):
            value = error_heatmap[row, col]
            if t.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
if MAIN:
    fig.colorbar(image, ax=ax, label="error rate")
if MAIN:
    fig.tight_layout()
if MAIN:
    if SAVE_FIGURES:
        fig.savefig(assets_dir / "mamba_state_tracking_anomaly_heatmap.png", dpi=180, bbox_inches="tight")
if MAIN:
    plt.show()

# %%

import importlib


def _release_solution_module():
    return importlib.import_module("chapter5_modern_architectures.exercises.part4_mamba_state_tracking.solutions")


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    """Run the section's pinned CUDA verification contract."""
    return _release_solution_module().run_gpu_test(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the full experiment used for the checked-in release evidence."""
    return _release_solution_module().run_full_experiment(
        max_vram_gb=max_vram_gb
    )

# %%
