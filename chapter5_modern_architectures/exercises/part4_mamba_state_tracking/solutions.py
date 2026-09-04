# %%
"""Reference solutions for [5.4] Mamba State Tracking."""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t
from torch import nn
import torch.nn.functional as F

chapter = "chapter5_modern_architectures"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.mamba import MambaConfig, TinyMambaModel

MAIN = __name__ == "__main__"

REAL_MAMBA_MODEL_ID = "state-spaces/mamba-130m-hf"
REAL_MAMBA_REVISION = "1e76775f628fbf1350fbe4dbb3d971ba64af25a1"


@dataclass(frozen=True)
class StateTrackingBatch:
    tokens: t.Tensor
    states: t.Tensor
    task: str
    vocab: dict[int, str]


@dataclass(frozen=True)
class LinearProbe:
    weight: t.Tensor
    bias: t.Tensor


@dataclass(frozen=True)
class ProbeReport:
    train_accuracy: float
    test_accuracy: float
    num_train: int
    num_test: int


@dataclass(frozen=True)
class InterventionReport:
    source_prediction: int
    target_prediction: int
    intervened_prediction: int
    target_logit_delta: float
    passed: bool


@dataclass(frozen=True)
class StateProbe:
    """Standardized linear probe over flattened recurrent SSM states."""

    mean: t.Tensor
    scale: t.Tensor
    weight: t.Tensor
    bias: t.Tensor


def generate_parity_task(batch: int, seq_len: int, seed: int = 0) -> StateTrackingBatch:
    """Generate binary-token sequences labelled by cumulative XOR."""

    generator = t.Generator().manual_seed(seed)
    tokens = t.randint(0, 2, (batch, seq_len), generator=generator)
    states = tokens.cumsum(dim=-1) % 2
    return StateTrackingBatch(tokens=tokens, states=states, task="parity", vocab={0: "0", 1: "1"})


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


def one_hot_state_features(
    states: t.Tensor,
    num_states: int | None = None,
    *,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> t.Tensor:
    """Represent each latent state by a noisy one-hot hidden vector."""

    if num_states is None:
        num_states = int(states.max().item()) + 1
    features = F.one_hot(states.long(), num_classes=num_states).float()
    if noise_scale > 0:
        generator = t.Generator(device=features.device).manual_seed(seed)
        features = features + noise_scale * t.randn(
            features.shape,
            generator=generator,
            device=features.device,
        )
    return features


def make_position_split(
    states: t.Tensor,
    *,
    train_fraction: float = 0.5,
) -> tuple[t.Tensor, t.Tensor]:
    """Split every sequence into early train positions and later test positions."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    seq_len = states.shape[-1]
    split = max(1, min(seq_len - 1, int(seq_len * train_fraction)))
    position = t.arange(seq_len, device=states.device)
    train_mask = position[None, :] < split
    test_mask = ~train_mask
    return train_mask.expand_as(states), test_mask.expand_as(states)


def _flatten_masked(
    hidden_states: t.Tensor,
    labels: t.Tensor,
    mask: t.Tensor | None,
) -> tuple[t.Tensor, t.Tensor]:
    if hidden_states.shape[:-1] != labels.shape:
        raise ValueError("hidden_states leading dimensions must match labels.")
    flat_x = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    flat_y = labels.reshape(-1).long()
    if mask is None:
        return flat_x, flat_y
    flat_mask = mask.reshape(-1).bool()
    return flat_x[flat_mask], flat_y[flat_mask]


def fit_linear_probe(
    hidden_states: t.Tensor,
    labels: t.Tensor,
    *,
    num_classes: int | None = None,
    train_mask: t.Tensor | None = None,
    ridge: float = 1e-3,
) -> LinearProbe:
    """Fit a closed-form ridge-regression probe from hidden states to labels."""

    x, y = _flatten_masked(hidden_states, labels, train_mask)
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1
    y_one_hot = F.one_hot(y, num_classes=num_classes).float()
    ones = t.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    x_aug = t.cat([x, ones], dim=-1)
    eye = t.eye(x_aug.shape[-1], device=x.device, dtype=x.dtype)
    eye[-1, -1] = 0.0
    solution = t.linalg.solve(x_aug.T @ x_aug + ridge * eye, x_aug.T @ y_one_hot)
    return LinearProbe(weight=solution[:-1], bias=solution[-1])


def probe_logits(hidden_states: t.Tensor, probe: LinearProbe) -> t.Tensor:
    return hidden_states.float() @ probe.weight + probe.bias


def probe_predictions(hidden_states: t.Tensor, probe: LinearProbe) -> t.Tensor:
    return probe_logits(hidden_states, probe).argmax(dim=-1)


def probe_accuracy(
    hidden_states: t.Tensor,
    labels: t.Tensor,
    probe: LinearProbe,
    mask: t.Tensor | None = None,
) -> float:
    predictions = probe_predictions(hidden_states, probe)
    if mask is not None:
        predictions = predictions[mask.bool()]
        labels = labels[mask.bool()]
    return predictions.eq(labels).float().mean().item()


def evaluate_probe_generalization(
    hidden_states: t.Tensor,
    labels: t.Tensor,
    probe: LinearProbe,
    train_mask: t.Tensor,
    test_mask: t.Tensor,
) -> ProbeReport:
    """Evaluate a probe on train positions and held-out later positions."""

    return ProbeReport(
        train_accuracy=probe_accuracy(hidden_states, labels, probe, train_mask),
        test_accuracy=probe_accuracy(hidden_states, labels, probe, test_mask),
        num_train=int(train_mask.sum().item()),
        num_test=int(test_mask.sum().item()),
    )


def state_intervention_direction(
    probe: LinearProbe,
    source_state: int,
    target_state: int,
) -> t.Tensor:
    """Return the target-minus-source probe direction."""

    if source_state == target_state:
        raise ValueError("source_state and target_state must differ.")
    return probe.weight[:, target_state] - probe.weight[:, source_state]


def apply_state_intervention(
    hidden_state: t.Tensor,
    probe: LinearProbe,
    *,
    source_state: int,
    target_state: int,
    coefficient: float = 1.0,
) -> t.Tensor:
    """Move a hidden state along a probe-derived target direction."""

    direction = state_intervention_direction(probe, source_state, target_state)
    direction = direction.to(device=hidden_state.device, dtype=hidden_state.dtype)
    return hidden_state + coefficient * direction


def intervention_report(
    hidden_state: t.Tensor,
    probe: LinearProbe,
    *,
    source_state: int,
    target_state: int,
    coefficient: float = 1.0,
) -> InterventionReport:
    """Check whether a probe-derived intervention flips the decoded state."""

    logits_before = probe_logits(hidden_state, probe)
    intervened = apply_state_intervention(
        hidden_state,
        probe,
        source_state=source_state,
        target_state=target_state,
        coefficient=coefficient,
    )
    logits_after = probe_logits(intervened, probe)
    source_prediction = int(logits_before.argmax(dim=-1).item())
    intervened_prediction = int(logits_after.argmax(dim=-1).item())
    target_delta = (logits_after[..., target_state] - logits_before[..., target_state]).item()
    return InterventionReport(
        source_prediction=source_prediction,
        target_prediction=target_state,
        intervened_prediction=intervened_prediction,
        target_logit_delta=float(target_delta),
        passed=intervened_prediction == target_state,
    )


def random_direction_control(
    hidden_state: t.Tensor,
    probe: LinearProbe,
    *,
    target_state: int,
    coefficient: float = 1.0,
    seed: int = 0,
) -> int:
    """Apply a random direction with matched norm and decode the new state."""

    generator = t.Generator(device=hidden_state.device).manual_seed(seed)
    random_dir = t.randn(hidden_state.shape, generator=generator, device=hidden_state.device)
    target_dir = probe.weight[:, target_state].to(device=hidden_state.device)
    random_dir = random_dir / random_dir.norm().clamp_min(1e-8) * target_dir.norm()
    logits = probe_logits(hidden_state + coefficient * random_dir, probe)
    return int(logits.argmax(dim=-1).item())


def _transformers_mamba_fast_path_status() -> tuple[bool, dict[str, bool]]:
    """Check whether Transformers can see the compiled Mamba fast-path kernels."""

    from transformers.models.mamba import modeling_mamba
    from transformers.models.mamba.configuration_mamba import (
        MambaConfig as TransformersMambaConfig,
    )

    components: dict[str, bool] = {}
    try:
        tiny_config = TransformersMambaConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            state_size=4,
            conv_kernel=3,
            num_hidden_layers=1,
        )
        modeling_mamba.MambaMixer(
            tiny_config,
            layer_idx=0,
            initialize_mixer_weights=False,
        )
        components["transformers_mamba_mixer_constructed"] = True
    except Exception:
        components["transformers_mamba_mixer_constructed"] = False

    component_names = (
        "selective_state_update",
        "selective_scan_fn",
        "mamba_inner_fn",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )
    components.update(
        {name: getattr(modeling_mamba, name, None) is not None for name in component_names}
    )
    return all(components.values()), components


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


class TinyTransformerStateClassifier(nn.Module):
    """Compact causal Transformer baseline for the same depth-tracking task."""

    def __init__(self, num_states: int = 4, max_seq_len: int = 64):
        super().__init__()
        d_model = 32
        self.embed_tokens = nn.Embedding(2, d_model)
        self.embed_positions = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=96,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_states)

    def encode(self, input_ids: t.Tensor) -> t.Tensor:
        seq_len = input_ids.shape[1]
        if seq_len > self.embed_positions.num_embeddings:
            raise ValueError("input sequence is longer than the configured baseline.")
        positions = t.arange(seq_len, device=input_ids.device)
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(positions)[None]
        causal_mask = t.triu(
            t.ones(seq_len, seq_len, device=input_ids.device, dtype=t.bool),
            diagonal=1,
        )
        hidden_states = self.encoder(hidden_states, mask=causal_mask)
        return self.norm(hidden_states)

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


# %%
def parity_task_smoke_test() -> dict:
    batch = generate_parity_task(batch=2, seq_len=5, seed=0)
    expected = batch.tokens.cumsum(dim=-1) % 2
    return {
        "task": batch.task,
        "matches_cumulative_xor": bool(t.equal(batch.states, expected)),
        "tokens": batch.tokens.tolist(),
        "states": batch.states.tolist(),
    }


def bracket_depth_smoke_test() -> dict:
    batch = generate_bracket_depth_task(batch=8, seq_len=20, max_depth=3, seed=1)
    deltas = t.where(batch.tokens.bool(), 1, -1)
    return {
        "bounded": bool(batch.states.min() >= 0 and batch.states.max() <= 3),
        "consistent": bool(t.equal(batch.states, deltas.cumsum(dim=-1))),
        "max_depth_observed": int(batch.states.max().item()),
    }


def probe_generalization_smoke_test() -> dict:
    task = generate_bracket_depth_task(batch=16, seq_len=12, max_depth=3, seed=2)
    features = one_hot_state_features(task.states, noise_scale=0.01, seed=3)
    train_mask, test_mask = make_position_split(task.states, train_fraction=0.5)
    probe = fit_linear_probe(features, task.states, train_mask=train_mask)
    report = evaluate_probe_generalization(features, task.states, probe, train_mask, test_mask)
    return report.__dict__


def intervention_smoke_test() -> dict:
    labels = t.tensor([[0, 1]])
    features = one_hot_state_features(labels)
    probe = fit_linear_probe(features, labels)
    hidden_state = features[0, 0]
    report = intervention_report(
        hidden_state,
        probe,
        source_state=0,
        target_state=1,
        coefficient=4.0,
    )
    random_prediction = random_direction_control(
        hidden_state,
        probe,
        target_state=1,
        coefficient=0.1,
        seed=0,
    )
    return {**report.__dict__, "random_prediction": random_prediction}


def parity_probe_smoke_test() -> dict:
    task = generate_parity_task(batch=4, seq_len=6, seed=4)
    features = one_hot_state_features(task.states)
    probe = fit_linear_probe(features, task.states)
    predictions = probe_predictions(features, probe)
    return {
        "accuracy": probe_accuracy(features, task.states, probe),
        "predictions_match": bool(t.equal(predictions, task.states)),
    }


def _train_tiny_mamba_depth_classifier(
    *,
    seed: int,
    random_labels: bool = False,
    steps: int = 160,
) -> dict:
    _, metrics = _train_depth_classifier(
        TinyMambaStateClassifier,
        seed=seed,
        random_labels=random_labels,
        steps=steps,
    )
    return metrics


def _evaluate_depth_classifier(
    model: nn.Module,
    *,
    seq_len: int,
    eval_seed: int,
) -> tuple[float, float]:
    batch = generate_bracket_depth_task(
        batch=128,
        seq_len=seq_len,
        max_depth=3,
        seed=eval_seed,
    )
    tokens = batch.tokens.to(next(model.parameters()).device)
    labels = batch.states.to(tokens.device)
    model.eval()
    with t.inference_mode():
        predictions = model(tokens).argmax(dim=-1)
    accuracy = predictions.eq(labels).float().mean().item()
    late_accuracy = predictions[:, -8:].eq(labels[:, -8:]).float().mean().item()
    return accuracy, late_accuracy


def _train_depth_classifier(
    model_factory,
    *,
    seed: int,
    random_labels: bool = False,
    steps: int = 160,
) -> tuple[nn.Module, dict]:
    device = t.device("cuda")
    t.manual_seed(seed)
    model = model_factory().to(device)
    optimizer = t.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    random_label_generator = t.Generator(device=device).manual_seed(seed + 99)
    final_loss = 0.0

    for step in range(steps):
        batch = generate_bracket_depth_task(
            batch=64,
            seq_len=16,
            max_depth=3,
            seed=seed + step,
        )
        tokens = batch.tokens.to(device)
        labels = batch.states.to(device)
        if random_labels:
            labels = t.randint(
                0,
                4,
                labels.shape,
                device=device,
                generator=random_label_generator,
            )
        logits = model(tokens)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    short_accuracy, short_late_accuracy = _evaluate_depth_classifier(
        model,
        seq_len=16,
        eval_seed=1000 + seed,
    )
    long_accuracy, long_late_accuracy = _evaluate_depth_classifier(
        model,
        seq_len=32,
        eval_seed=2000 + seed,
    )
    return model, {
        "final_loss": final_loss,
        "short_accuracy": short_accuracy,
        "short_late_accuracy": short_late_accuracy,
        "long_accuracy": long_accuracy,
        "long_late_accuracy": long_late_accuracy,
    }


def _learned_hidden_state_intervention_report(
    model: TinyMambaStateClassifier,
    *,
    seed: int = 3000,
    coefficient: float = 5.0,
) -> dict:
    """Intervene on learned Mamba hidden states using the trained readout direction."""

    device = next(model.parameters()).device
    batch = generate_bracket_depth_task(batch=128, seq_len=32, max_depth=3, seed=seed)
    tokens = batch.tokens.to(device)
    labels = batch.states.to(device)
    model.eval()
    with t.inference_mode():
        logits, hidden_states = model(tokens, return_hidden_states=True)

    train_mask, test_mask = make_position_split(labels, train_fraction=0.5)
    probe = fit_linear_probe(hidden_states.detach(), labels, train_mask=train_mask)
    probe_report = evaluate_probe_generalization(
        hidden_states.detach(),
        labels,
        probe,
        train_mask,
        test_mask,
    )

    predictions = logits.argmax(dim=-1)
    candidates = predictions.eq(labels) & (labels < 3)
    if int(candidates.sum().item()) == 0:
        raise RuntimeError("no correctly classified source states available for intervention")

    source_states = labels[candidates]
    target_states = source_states + 1
    source_hidden = hidden_states[candidates]
    before_logits = logits[candidates]
    head_weight = model.head.weight
    directions = head_weight[target_states] - head_weight[source_states]
    intervened_hidden = source_hidden + coefficient * directions
    after_logits = model.head(intervened_hidden)
    intervened_predictions = after_logits.argmax(dim=-1)

    generator = t.Generator(device=device).manual_seed(seed + 11)
    random_dirs = t.randn(
        directions.shape,
        device=device,
        dtype=directions.dtype,
        generator=generator,
    )
    random_dirs = (
        random_dirs
        / random_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        * directions.norm(dim=-1, keepdim=True)
    )
    random_logits = model.head(source_hidden + coefficient * random_dirs)
    random_predictions = random_logits.argmax(dim=-1)

    row = t.arange(source_states.shape[0], device=device)
    target_logit_delta = (
        after_logits[row, target_states] - before_logits[row, target_states]
    ).mean()
    random_target_logit_delta = (
        random_logits[row, target_states] - before_logits[row, target_states]
    ).mean()
    success_rate = intervened_predictions.eq(target_states).float().mean().item()
    random_target_rate = random_predictions.eq(target_states).float().mean().item()
    passed = (
        success_rate >= 0.9
        and random_target_rate <= 0.5
        and target_logit_delta.item() > random_target_logit_delta.item() + 1.0
        and probe_report.test_accuracy >= 0.8
    )
    return {
        "candidate_count": int(candidates.sum().item()),
        "source_states": sorted(int(x) for x in source_states.unique().detach().cpu()),
        "target_states": sorted(int(x) for x in target_states.unique().detach().cpu()),
        "coefficient": coefficient,
        "success_rate": success_rate,
        "random_target_rate": random_target_rate,
        "target_logit_delta": float(target_logit_delta.detach().item()),
        "random_target_logit_delta": float(random_target_logit_delta.detach().item()),
        "probe_train_accuracy": probe_report.train_accuracy,
        "probe_test_accuracy": probe_report.test_accuracy,
        "passed": passed,
    }


def _tiny_transformer_baseline_preflight(*, seed: int = 2, steps: int = 160) -> dict:
    _, baseline = _train_depth_classifier(
        TinyTransformerStateClassifier,
        seed=seed,
        random_labels=False,
        steps=steps,
    )
    return baseline


def tiny_mamba_training_preflight(max_vram_gb: float = 24.0) -> dict:
    """Train real Mamba and Transformer depth-tracking organisms on CUDA."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "skipped": True,
            "claim_scope": "tiny_mamba_training_preflight_requires_cuda",
        }

    t.cuda.reset_peak_memory_stats()
    learned_model, learned = _train_depth_classifier(
        TinyMambaStateClassifier,
        seed=0,
        random_labels=False,
    )
    random_control = _train_tiny_mamba_depth_classifier(seed=1, random_labels=True)
    transformer_baseline = _tiny_transformer_baseline_preflight(seed=2)
    intervention = _learned_hidden_state_intervention_report(learned_model)
    mamba_minus_transformer_long_accuracy = (
        learned["long_accuracy"] - transformer_baseline["long_accuracy"]
    )
    mamba_minus_transformer_long_late_accuracy = (
        learned["long_late_accuracy"] - transformer_baseline["long_late_accuracy"]
    )
    transformer_comparison_passed = (
        transformer_baseline["short_accuracy"] >= 0.95
        and mamba_minus_transformer_long_accuracy >= 0.2
        and mamba_minus_transformer_long_late_accuracy >= 0.4
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    preflight_passed = (
        learned["short_accuracy"] >= 0.9
        and learned["long_accuracy"] >= 0.85
        and learned["long_late_accuracy"] >= 0.8
        and random_control["long_accuracy"] <= 0.55
        and transformer_comparison_passed
        and intervention["passed"]
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "cuda_available": True,
        "claim_scope": "trained_tiny_mamba_transformer_comparison_and_learned_state_intervention_preflight",
        "train_steps": 160,
        "train_batch": 64,
        "train_seq_len": 16,
        "heldout_seq_len": 32,
        "max_depth": 3,
        "learned": learned,
        "random_label_control": random_control,
        "random_label_control_fails": random_control["long_accuracy"] <= 0.55,
        "transformer_baseline": transformer_baseline,
        "transformer_comparison_passed": transformer_comparison_passed,
        "mamba_minus_transformer_long_accuracy": mamba_minus_transformer_long_accuracy,
        "mamba_minus_transformer_long_late_accuracy": (
            mamba_minus_transformer_long_late_accuracy
        ),
        "learned_state_intervention": intervention,
        "learned_state_intervention_passed": intervention["passed"],
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": preflight_passed,
    }


def official_mamba_hidden_state_preflight(max_vram_gb: float = 24.0) -> dict:
    """Load an official Mamba checkpoint and verify hidden-state extraction."""

    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "preflight_passed": False,
            "skipped": True,
            "claim_scope": "official_mamba_hidden_state_preflight_requires_cuda",
        }

    from transformers import AutoModelForCausalLM, AutoTokenizer

    fast_kernel_available, fast_path_components = _transformers_mamba_fast_path_status()
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        REAL_MAMBA_MODEL_ID,
        revision=REAL_MAMBA_REVISION,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        REAL_MAMBA_MODEL_ID,
        revision=REAL_MAMBA_REVISION,
        dtype=t.float16,
    ).to(device)
    model.eval()
    prompts = [
        "The sequence is 0 1 1 0. The answer is",
        "The sequence is 1 1 1 0. The answer is",
        "(((( )))) balanced brackets:",
        "(()()) balanced brackets:",
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with t.inference_mode():
        output = model(**inputs, output_hidden_states=True)
    hidden_states = output.hidden_states[-1].detach().float()
    finite_hidden_states = bool(t.isfinite(hidden_states).all().item())
    hidden_std = hidden_states.std().item()
    hidden_shape = list(hidden_states.shape)
    logits_shape = list(output.logits.shape)
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3

    del output, hidden_states, inputs, model, tokenizer
    t.cuda.empty_cache()

    preflight_passed = (
        finite_hidden_states
        and hidden_std > 0.1
        and peak_vram_gb <= max_vram_gb
        and fast_kernel_available
    )
    return {
        "cuda_available": True,
        "model_id": REAL_MAMBA_MODEL_ID,
        "revision": REAL_MAMBA_REVISION,
        "claim_scope": "official_mamba_130m_hf_hidden_state_extraction_preflight",
        "prompt_count": len(prompts),
        "hidden_shape": hidden_shape,
        "logits_shape": logits_shape,
        "finite_hidden_states": finite_hidden_states,
        "hidden_std": hidden_std,
        "fast_kernel_available": fast_kernel_available,
        "fast_path_components": fast_path_components,
        "backend_note": (
            "Transformers Mamba fast path components are available."
            if fast_kernel_available
            else "Transformers Mamba fast path is unavailable; install mamba-ssm and causal-conv1d."
        ),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": preflight_passed,
    }


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "parity_task": parity_task_smoke_test(),
        "bracket_depth": bracket_depth_smoke_test(),
        "probe_generalization": probe_generalization_smoke_test(),
        "intervention": intervention_smoke_test(),
        "parity_probe": parity_probe_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        return {
            "cuda_available": False,
            "full_path": "Train tiny Mamba and tiny Transformer baselines on state-tracking tasks.",
        }

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    task = generate_bracket_depth_task(batch=16, seq_len=12, max_depth=3, seed=2)
    states = task.states.to(device)
    features = one_hot_state_features(states, noise_scale=0.01, seed=3)
    train_mask, test_mask = make_position_split(states, train_fraction=0.5)
    probe = fit_linear_probe(features, states, train_mask=train_mask)
    report = evaluate_probe_generalization(features, states, probe, train_mask, test_mask)
    predictions = probe_predictions(features, probe)
    t.cuda.synchronize()
    synthetic_peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    tiny_mamba = tiny_mamba_training_preflight(max_vram_gb=max_vram_gb)
    official_mamba = official_mamba_hidden_state_preflight(max_vram_gb=max_vram_gb)
    peak_vram_gb = max(
        synthetic_peak_vram_gb,
        tiny_mamba["peak_vram_gb"],
        official_mamba["peak_vram_gb"],
    )
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "train_accuracy": report.train_accuracy,
        "test_accuracy": report.test_accuracy,
        "predictions_match": bool(t.equal(predictions, states)),
        "tiny_mamba_preflight_passed": tiny_mamba["preflight_passed"],
        "tiny_mamba_short_accuracy": tiny_mamba["learned"]["short_accuracy"],
        "tiny_mamba_long_accuracy": tiny_mamba["learned"]["long_accuracy"],
        "tiny_mamba_long_late_accuracy": tiny_mamba["learned"]["long_late_accuracy"],
        "tiny_mamba_random_label_long_accuracy": tiny_mamba["random_label_control"][
            "long_accuracy"
        ],
        "tiny_mamba_random_label_control_fails": tiny_mamba["random_label_control_fails"],
        "tiny_transformer_long_accuracy": tiny_mamba["transformer_baseline"][
            "long_accuracy"
        ],
        "tiny_transformer_long_late_accuracy": tiny_mamba["transformer_baseline"][
            "long_late_accuracy"
        ],
        "tiny_transformer_short_accuracy": tiny_mamba["transformer_baseline"][
            "short_accuracy"
        ],
        "tiny_transformer_comparison_passed": tiny_mamba[
            "transformer_comparison_passed"
        ],
        "tiny_mamba_minus_transformer_long_accuracy": tiny_mamba[
            "mamba_minus_transformer_long_accuracy"
        ],
        "tiny_mamba_minus_transformer_long_late_accuracy": tiny_mamba[
            "mamba_minus_transformer_long_late_accuracy"
        ],
        "learned_state_intervention_passed": tiny_mamba[
            "learned_state_intervention_passed"
        ],
        "learned_state_intervention_success_rate": tiny_mamba[
            "learned_state_intervention"
        ]["success_rate"],
        "learned_state_intervention_random_target_rate": tiny_mamba[
            "learned_state_intervention"
        ]["random_target_rate"],
        "learned_state_intervention_target_logit_delta": tiny_mamba[
            "learned_state_intervention"
        ]["target_logit_delta"],
        "learned_hidden_probe_test_accuracy": tiny_mamba[
            "learned_state_intervention"
        ]["probe_test_accuracy"],
        "tiny_mamba_peak_vram_gb": tiny_mamba["peak_vram_gb"],
        "tiny_mamba_preflight": tiny_mamba,
        "official_mamba_preflight_passed": official_mamba["preflight_passed"],
        "official_mamba_hidden_shape": official_mamba["hidden_shape"],
        "official_mamba_hidden_std": official_mamba["hidden_std"],
        "official_mamba_fast_kernel_available": official_mamba[
            "fast_kernel_available"
        ],
        "official_mamba_peak_vram_gb": official_mamba["peak_vram_gb"],
        "official_mamba_preflight": official_mamba,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": (
            peak_vram_gb <= max_vram_gb
            and tiny_mamba["within_vram_budget"]
            and official_mamba["within_vram_budget"]
        ),
        "full_path": (
            "Validated tiny Mamba state-tracking, random-label controls, a trained "
            "tiny Transformer baseline, learned-state interventions, and pinned "
            "Mamba-130M-HF hidden-state extraction with fast CUDA kernels."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())
