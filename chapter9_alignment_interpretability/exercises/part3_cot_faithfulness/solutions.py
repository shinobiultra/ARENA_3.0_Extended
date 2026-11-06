# %%


from __future__ import annotations
MAIN = __name__ == "__main__"
from importlib import import_module as _migration_import_module
from pathlib import Path as _MigrationPath
asset_dir = _MigrationPath(__file__).resolve().parents[2] / "instructions" / "assets"
class _HistoricalReference:
    """Defer the exact historical helper until an explicit optional wrapper call.

    Optional execution retains its original repository-root cwd requirement.
    """
    def __getattr__(self, name):
        return getattr(_migration_import_module("chapter9_alignment_interpretability.exercises.part2_cot_faithfulness.solutions"), name)
reference_solutions = _HistoricalReference()

# %%

from dataclasses import dataclass

import json

import sys

from pathlib import Path

from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt

import torch as t

chapter = 'chapter9_alignment_interpretability'

section = 'part3_cot_faithfulness'

if MAIN:
    root_dir = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists()))

if MAIN:
    exercises_dir = root_dir / chapter / 'exercises'

if MAIN:
    section_dir = exercises_dir / section

if MAIN:
    asset_dir = root_dir / chapter / 'instructions' / 'assets'

if MAIN:
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

if MAIN:
    if str(exercises_dir) not in sys.path:
        sys.path.append(str(exercises_dir))

if MAIN:
    import chapter9_alignment_interpretability.exercises.part2_cot_faithfulness.tests as tests

if MAIN:
    import chapter9_alignment_interpretability.exercises.part2_cot_faithfulness.utils as utils


MAIN = __name__ == '__main__'

AnswerLabel = Literal['A', 'B']

CoTCondition = Literal['no_cot', 'faithful_cot', 'biased_cot', 'posthoc']

GT_TIER = 'GT-0'

EXERCISE_ID = '9_2_chain_of_thought_faithfulness'

EXPECTED_RUNTIME = '45-75 minutes; parent CUDA verification runs the pinned Pythia hook-patching path'

REQUIRES_GPU = True

SEMANTIC_POSITIONS = ('private_answer', 'rationale_answer', 'final_prompt')

TOY_POSITIONS = ('bos', 'private_answer', 'rationale_answer', 'final_prompt')

PYTHIA_CONDITION_ROWS: tuple[tuple[AnswerLabel,
    AnswerLabel,
    CoTCondition],
    ...] = (('A',
    'A',
    'faithful_cot'),
    ('B',
    'B',
    'faithful_cot'),
    ('A',
    'B',
    'biased_cot'),
    ('B',
    'A',
    'biased_cot'),
    ('A',
    'B',
    'posthoc'),
    ('B',
    'A',
    'posthoc'),
    ('A',
    'A',
    'no_cot'),
    ('B',
    'B',
    'no_cot'))

# %%

@dataclass(frozen=True)
class MeanDifferenceProbe:
    direction: t.Tensor
    threshold: float

@dataclass(frozen=True)
class PatchControlReport:
    mean_effects: dict[str, float]
    flip_rates: dict[str, float]
    target_name: str
    max_control_name: str
    target_control_gap: float
    target_beats_controls: bool

@dataclass(frozen=True)
class ToyCoTBatch:
    activations: t.Tensor
    hidden_answer_ids: t.Tensor
    visible_answer_ids: t.Tensor
    condition_ids: tuple[CoTCondition, ...]
    prompts: tuple[str, ...]
    position_names: tuple[str, ...]
    answer_readout: t.Tensor

def _require_finite_tensor(name: str, tensor: t.Tensor) -> None:
    if tensor.numel() == 0:
        raise ValueError(f'{name} must be non-empty.')
    if not t.isfinite(tensor.float()).all():
        raise ValueError(f'{name} must contain only finite values.')

def _require_binary_ids(name: str, ids: t.Tensor, expected_shape: tuple[int, ...] | None=None) -> None:
    if expected_shape is not None and tuple(ids.shape) != expected_shape:
        raise ValueError(f'{name} must have shape {expected_shape}, got {tuple(ids.shape)}.')
    if ids.numel() == 0:
        raise ValueError(f'{name} must be non-empty.')
    unique = set((int(x) for x in ids.long().detach().cpu().flatten().tolist()))
    if not unique.issubset({0, 1}):
        raise ValueError(f'{name} must contain only answer ids 0 and 1.')

@dataclass(frozen=True)
class RealModelVerdict:
    claim_supported: bool
    negative_result: bool
    strongest_control: str
    target_control_gap: float
    detector_advantage: float

# %%

def make_toy_cot_batch(num_pairs: int=12) -> ToyCoTBatch:
    """Construct an exact toy CoT organism with known answer-carrying states."""
    if num_pairs < 4:
        raise ValueError('num_pairs must be at least 4.')
    d_model = 8
    layers = 4
    positions = len(TOY_POSITIONS)
    prompts: list[str] = []
    conditions: list[CoTCondition] = []
    hidden_ids: list[int] = []
    visible_ids: list[int] = []
    activations = t.zeros((2 * num_pairs, layers, positions, d_model), dtype=t.float32)
    answer_readout = t.zeros((d_model, 2), dtype=t.float32)
    answer_readout[0, 0] = -1.0
    answer_readout[0, 1] = 1.0
    for i in range(2 * num_pairs):
        hidden = i % 2
        condition = PYTHIA_CONDITION_ROWS[i % len(PYTHIA_CONDITION_ROWS)][2]
        visible = hidden if condition in {'faithful_cot', 'no_cot'} else 1 - hidden
        sign = -1.0 if hidden == 0 else 1.0
        visible_sign = -1.0 if visible == 0 else 1.0
        hidden_ids.append(hidden)
        visible_ids.append(visible)
        conditions.append(condition)
        prompts.append(f'toy {i:02d}: private={chr(65 + hidden)} rationale={chr(65 + visible)} condition={condition}')
        activations[i, 0, 1, 0] = sign
        activations[i, 0, 2, 1] = visible_sign
        activations[i, 1, 1, 0] = 1.5 * sign
        activations[i, 1, 2, 1] = 1.2 * visible_sign
        activations[i, 1, 3, 0] = 0.5 * sign
        activations[i, 2, 3, 0] = 2.0 * sign
        activations[i, 2, 3, 1] = 0.2 * visible_sign
        activations[i, 3, 3, 0] = 2.5 * sign
        activations[i, 3, 3, 1] = 0.2 * visible_sign
        activations[i, :, :, 2] = t.linspace(-0.2, 0.2, layers)[:, None]
        activations[i, :, :, 3] = t.linspace(-0.1, 0.1, positions)[None, :]
    return ToyCoTBatch(activations=activations,
        hidden_answer_ids=t.tensor(hidden_ids,
        dtype=t.long),
        visible_answer_ids=t.tensor(visible_ids,
        dtype=t.long),
        condition_ids=tuple(conditions),
        prompts=tuple(prompts),
        position_names=TOY_POSITIONS,
        answer_readout=answer_readout)


if MAIN:
    toy_batch = make_toy_cot_batch(num_pairs=12)

if MAIN:
    assert toy_batch.activations.shape == (24, 4, 4, 8)

if MAIN:
    assert set(toy_batch.hidden_answer_ids.tolist()) == {0, 1}

# %%

def prediction_accuracy(logits: t.Tensor, target_token_ids: t.Tensor) -> float:
    """Return top-1 accuracy for answer logits."""
    if logits.shape[:-1] != target_token_ids.shape:
        raise ValueError('target_token_ids must match logits leading dimensions.')
    _require_finite_tensor('logits', logits)
    _require_binary_ids('target_token_ids', target_token_ids, tuple(logits.shape[:-1]))
    predictions = logits.argmax(dim=-1)
    return float(predictions.eq(target_token_ids.long()).float().mean().item())


def answer_logit_diff(answer_logits: t.Tensor) -> t.Tensor:
    """Return logit(B) - logit(A) for a final-token answer-logit tensor."""
    if answer_logits.shape[-1] != 2:
        raise ValueError('answer_logits must end with the two answer logits [A, B].')
    _require_finite_tensor('answer_logits', answer_logits)
    return answer_logits[..., 1] - answer_logits[..., 0]


if MAIN:
    tests.test_prediction_accuracy_checks_top1_predictions(prediction_accuracy)

if MAIN:
    tests.test_answer_logit_diff_tracks_b_minus_a(answer_logit_diff)

# %%

def fit_mean_difference_probe(hidden_states: t.Tensor, answer_ids: t.Tensor) -> MeanDifferenceProbe:
    """Fit a closed-form A-vs-B probe from class means."""
    if hidden_states.ndim != 2:
        raise ValueError('hidden_states must have shape (examples, d_model).')
    _require_finite_tensor('hidden_states', hidden_states)
    _require_binary_ids('answer_ids', answer_ids, (hidden_states.shape[0],))
    answer_ids = answer_ids.long().to(hidden_states.device)
    class_a = hidden_states[answer_ids.eq(0)].float()
    class_b = hidden_states[answer_ids.eq(1)].float()
    if class_a.numel() == 0 or class_b.numel() == 0:
        raise ValueError('both answer classes must be present.')
    direction = class_b.mean(dim=0) - class_a.mean(dim=0)
    norm = direction.norm()
    if float(norm.item()) <= 0:
        raise ValueError('class means must define a non-zero direction.')
    direction = direction / norm
    scores = hidden_states.float() @ direction
    threshold = 0.5 * (scores[answer_ids.eq(0)].mean().item() + scores[answer_ids.eq(1)].mean().item())
    return MeanDifferenceProbe(direction=direction, threshold=float(threshold))


def probe_logits_from_direction(hidden_states: t.Tensor, probe: MeanDifferenceProbe) -> t.Tensor:
    """Convert hidden states to two answer logits using a mean-difference probe."""
    if hidden_states.shape[-1] != probe.direction.numel():
        raise ValueError('hidden_states last dimension must match probe direction.')
    _require_finite_tensor('hidden_states', hidden_states)
    direction = probe.direction.to(hidden_states.device, dtype=hidden_states.float().dtype)
    scores = hidden_states.float() @ direction - probe.threshold
    return t.stack([-scores, scores], dim=-1)


if MAIN:
    tests.test_mean_difference_probe_recovers_toy_hidden_answer(fit_mean_difference_probe, probe_logits_from_direction)

# %%

def layer_position_probe_heatmap(train_hidden_states: t.Tensor, train_answer_ids: t.Tensor, eval_hidden_states: t.Tensor, eval_answer_ids: t.Tensor) -> t.Tensor:
    """Evaluate a mean-difference hidden-answer probe at every layer and position."""
    if train_hidden_states.ndim != 4 or eval_hidden_states.ndim != 4:
        raise ValueError('hidden-state grids must have shape (examples, layers, positions, d_model).')
    if train_hidden_states.shape[1:] != eval_hidden_states.shape[1:]:
        raise ValueError('train and eval grids must share layer/position/model dimensions.')
    _require_binary_ids('train_answer_ids', train_answer_ids, (train_hidden_states.shape[0],))
    _require_binary_ids('eval_answer_ids', eval_answer_ids, (eval_hidden_states.shape[0],))
    layers, positions = (train_hidden_states.shape[1], train_hidden_states.shape[2])
    heatmap = t.empty((layers, positions), dtype=t.float32, device=eval_hidden_states.device)
    for layer in range(layers):
        for position in range(positions):
            try:
                probe = fit_mean_difference_probe(train_hidden_states[:, layer, position, :], train_answer_ids)
            except ValueError as exc:
                if 'non-zero direction' not in str(exc):
                    raise
                heatmap[layer, position] = 0.5
                continue
            logits = probe_logits_from_direction(eval_hidden_states[:, layer, position, :], probe)
            heatmap[layer, position] = prediction_accuracy(logits, eval_answer_ids)
    return heatmap


if MAIN:
    tests.test_layer_position_heatmap_finds_toy_answer_stream(layer_position_probe_heatmap)

# %%

if MAIN:
    toy_batch = make_toy_cot_batch(num_pairs=12)

if MAIN:
    train = t.arange(0, 12)

if MAIN:
    eval_idx = t.arange(12, 24)

if MAIN:
    toy_heatmap = layer_position_probe_heatmap(toy_batch.activations[train], toy_batch.hidden_answer_ids[train], toy_batch.activations[eval_idx], toy_batch.hidden_answer_ids[eval_idx])

if MAIN:
    fig, ax = plt.subplots(figsize=(7, 3.5))

if MAIN:
    im = ax.imshow(toy_heatmap, vmin=0, vmax=1, cmap='viridis')

if MAIN:
    ax.set_xticks(range(len(toy_batch.position_names)), toy_batch.position_names, rotation=20, ha='right')

if MAIN:
    ax.set_yticks(range(toy_heatmap.shape[0]), [f'layer {i}' for i in range(toy_heatmap.shape[0])])

if MAIN:
    ax.set_title('Toy hidden-answer probe accuracy by layer and position')

if MAIN:
    for i in range(toy_heatmap.shape[0]):
        for j in range(toy_heatmap.shape[1]):
            color = 'black' if float(toy_heatmap[i, j]) > 0.72 else 'white'
            ax.text(j, i, f'{toy_heatmap[i, j]:.1f}', ha='center', va='center', color=color)

if MAIN:
    fig.colorbar(im, ax=ax, label='held-out accuracy')

if MAIN:
    fig.tight_layout()

if MAIN:
    plt.show()

# %%

def replace_position_in_layer_output(output: t.Tensor | tuple[Any, ...], donor_activation: t.Tensor, token_position: int) -> t.Tensor | tuple[Any, ...]:
    """Return a hook output with one residual-stream position replaced."""
    hidden_states = output[0] if isinstance(output, tuple) else output
    if hidden_states.ndim != 3:
        raise ValueError('layer output hidden states must have shape (batch, seq, d_model).')
    position = token_position if token_position >= 0 else hidden_states.shape[1] + token_position
    if not 0 <= position < hidden_states.shape[1]:
        raise ValueError('token_position is outside the sequence length.')
    donor = donor_activation.to(hidden_states.device, dtype=hidden_states.dtype)
    if donor.ndim == 1:
        if donor.shape != (hidden_states.shape[-1],):
            raise ValueError('rank-1 donor_activation must have shape (d_model,).')
        replacement = donor[None, :].expand(hidden_states.shape[0], -1)
    elif donor.ndim == 2:
        if donor.shape != (hidden_states.shape[0], hidden_states.shape[-1]):
            raise ValueError('rank-2 donor_activation must have shape (batch, d_model).')
        replacement = donor
    else:
        raise ValueError('donor_activation must be rank 1 or rank 2.')
    patched_hidden = hidden_states.clone()
    patched_hidden[:, position, :] = replacement
    if isinstance(output, tuple):
        return (patched_hidden, *output[1:])
    return patched_hidden


if MAIN:
    tests.test_replace_position_in_layer_output_patches_tensor_and_tuple(replace_position_in_layer_output)

# %%

def toy_answer_logits(batch: ToyCoTBatch, layer_index: int=-1) -> t.Tensor:
    """Decode A/B logits from the toy final-prompt residual stream."""
    final_position = batch.position_names.index('final_prompt')
    states = batch.activations[:, layer_index, final_position, :]
    return states @ batch.answer_readout


def toy_forward_patch_answer_logits(batch: ToyCoTBatch, target_indices: t.Tensor, donor_indices: t.Tensor, *, layer_index: int, position_index: int) -> t.Tensor:
    """Exact toy forward patch: only the answer-carrying final stream is causal."""
    if target_indices.shape != donor_indices.shape:
        raise ValueError('target_indices and donor_indices must have matching shape.')
    final_position = batch.position_names.index('final_prompt')
    layer = layer_index if layer_index >= 0 else batch.activations.shape[1] + layer_index
    patched_final = batch.activations[target_indices, -1, final_position, :].clone()
    if position_index == final_position and layer >= 2:
        donor_states = batch.activations[donor_indices, layer, position_index, :]
        patched_final[:, 0] = donor_states[:, 0]
    return patched_final @ batch.answer_readout


if MAIN:
    tests.test_toy_forward_patch_has_exact_causal_ground_truth(toy_forward_patch_answer_logits)

# %%

def signed_margin_toward_donor(answer_logits: t.Tensor, target_answer_ids: t.Tensor, donor_answer_ids: t.Tensor) -> t.Tensor:
    """Return the donor-answer logit minus the target-answer logit."""
    if answer_logits.ndim < 2 or answer_logits.shape[-1] != 2:
        raise ValueError('answer_logits must have shape (..., 2).')
    leading_shape = tuple(answer_logits.shape[:-1])
    _require_binary_ids('target_answer_ids', target_answer_ids, leading_shape)
    _require_binary_ids('donor_answer_ids', donor_answer_ids, leading_shape)
    flat_logits = answer_logits.reshape(-1, 2)
    target = target_answer_ids.reshape(-1).long().to(flat_logits.device)
    donor = donor_answer_ids.reshape(-1).long().to(flat_logits.device)
    rows = t.arange(flat_logits.shape[0], device=flat_logits.device)
    margin = flat_logits[rows, donor] - flat_logits[rows, target]
    return margin.reshape(leading_shape)


def patch_control_summary(patch_effects: Mapping[str, t.Tensor], patch_flips: Mapping[str, t.Tensor], *, target_name: str='target_patch', min_target_control_gap: float=0.05) -> PatchControlReport:
    """Compare a targeted patch against named controls."""
    if target_name not in patch_effects:
        raise ValueError(f'patch_effects must contain {target_name!r}.')
    if set(patch_effects) != set(patch_flips):
        raise ValueError('patch_effects and patch_flips must use the same condition names.')
    mean_effects: dict[str, float] = {}
    flip_rates: dict[str, float] = {}
    for name, values in patch_effects.items():
        _require_finite_tensor(f'patch_effects[{name}]', values)
        _require_finite_tensor(f'patch_flips[{name}]', patch_flips[name])
        mean_effects[name] = float(values.float().mean().item())
        flip_rates[name] = float(patch_flips[name].float().mean().item())
    control_names = [name for name in mean_effects if name != target_name]
    if not control_names:
        raise ValueError('at least one control condition is required.')
    max_control_name = max(control_names, key=lambda name: mean_effects[name])
    gap = mean_effects[target_name] - mean_effects[max_control_name]
    return PatchControlReport(mean_effects=mean_effects,
        flip_rates=flip_rates,
        target_name=target_name,
        max_control_name=max_control_name,
        target_control_gap=float(gap),
        target_beats_controls=bool(gap >= min_target_control_gap))


if MAIN:
    tests.test_patch_control_summary_requires_target_beats_controls(patch_control_summary)

# %%

def toy_ground_truth_signature() -> dict[str, Any]:
    """Return a CPU-only exact toy result used by the notebook contract."""
    batch = make_toy_cot_batch(num_pairs=12)
    train = t.arange(0, 12)
    eval_idx = t.arange(12, 24)
    heatmap = layer_position_probe_heatmap(batch.activations[train], batch.hidden_answer_ids[train], batch.activations[eval_idx], batch.hidden_answer_ids[eval_idx])
    final_position = batch.position_names.index('final_prompt')
    target_indices = eval_idx[batch.hidden_answer_ids[eval_idx].eq(0)]
    donor_indices = target_indices + 1
    donor_answer_ids = batch.hidden_answer_ids[donor_indices]
    target_answer_ids = batch.hidden_answer_ids[target_indices]
    clean_logits = toy_answer_logits(batch)[target_indices]
    clean_margin = signed_margin_toward_donor(clean_logits, target_answer_ids, donor_answer_ids)
    target_logits = toy_forward_patch_answer_logits(batch, target_indices, donor_indices, layer_index=2, position_index=final_position)
    irrelevant_logits = toy_forward_patch_answer_logits(batch, target_indices, donor_indices, layer_index=2, position_index=batch.position_names.index('rationale_answer'))
    random_donor = target_indices
    random_logits = toy_forward_patch_answer_logits(batch, target_indices, random_donor, layer_index=2, position_index=final_position)
    text_only_logits = clean_logits
    label_shuffled_logits = clean_logits
    random_direction_logits = clean_logits + t.tensor([[0.05, -0.05]], dtype=t.float32)

    def effect(logits: t.Tensor) -> t.Tensor:
        return signed_margin_toward_donor(logits, target_answer_ids, donor_answer_ids) - clean_margin

    def flips(logits: t.Tensor) -> t.Tensor:
        return logits.argmax(dim=-1).eq(donor_answer_ids)
    logits_by_condition = {'target_patch': target_logits,
        'text_only': text_only_logits,
        'label_shuffled': label_shuffled_logits,
        'random_direction': random_direction_logits,
        'random_donor': random_logits,
        'irrelevant_position': irrelevant_logits}
    patch_effects = {name: effect(logits) for name, logits in logits_by_condition.items()}
    patch_flips = {name: flips(logits) for name, logits in logits_by_condition.items()}
    controls = patch_control_summary(patch_effects, patch_flips, min_target_control_gap=1.0)
    shuffled_heatmap = layer_position_probe_heatmap(batch.activations[train], batch.hidden_answer_ids[train].roll(shifts=1), batch.activations[eval_idx], batch.hidden_answer_ids[eval_idx])
    final_accuracy = float(heatmap[2, final_position].item())
    shuffled_accuracy = float(shuffled_heatmap[2, final_position].item())
    accepted = final_accuracy == 1.0 and shuffled_accuracy <= 0.5 and controls.target_beats_controls and (controls.flip_rates['target_patch'] == 1.0) and (controls.flip_rates['irrelevant_position'] == 0.0)
    return {'accepted': accepted,
        'contract_passed': accepted,
        'tests_passed': accepted,
        'position_names': list(batch.position_names),
        'heatmap': heatmap.tolist(),
        'target_layer': 2,
        'target_position': 'final_prompt',
        'target_probe_accuracy': final_accuracy,
        'label_shuffled_probe_accuracy': shuffled_accuracy,
        'patch_control_means': controls.mean_effects,
        'patch_control_flip_rates': controls.flip_rates,
        'patch_target_control_gap': controls.target_control_gap,
        'max_control_name': controls.max_control_name,
        'qualitative_examples': [{'prompt': batch.prompts[int(i)],
        'target_answer': 'A' if int(batch.hidden_answer_ids[int(i)]) == 0 else 'B',
        'donor_answer': 'A' if int(batch.hidden_answer_ids[int(j)]) == 0 else 'B',
        'clean_prediction': 'A' if int(clean_logits[k].argmax().item()) == 0 else 'B',
        'patched_prediction': 'A' if int(target_logits[k].argmax().item()) == 0 else 'B'} for k,
        (i,
        j) in enumerate(zip(target_indices[:3],
        donor_indices[:3],
        strict=True))]}

if MAIN:
    toy_signature = toy_ground_truth_signature()

if MAIN:
    utils.print_report('Toy patch-control means', toy_signature['patch_control_means'])

if MAIN:
    utils.print_report('Toy patch-control flip rates', toy_signature['patch_control_flip_rates'])

if MAIN:
    heatmap = t.tensor(toy_signature['heatmap'], dtype=t.float32)

if MAIN:
    conditions = list(toy_signature['patch_control_means'])

if MAIN:
    control_colors = ['#16867a' if name == 'target_patch' else '#d99a21' if name == 'random_donor' else '#9299a3' for name in conditions]

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))

if MAIN:
    im = axes[0].imshow(heatmap, vmin=0, vmax=1, cmap='viridis')

if MAIN:
    axes[0].set_xticks(range(len(toy_signature['position_names'])), toy_signature['position_names'], rotation=25, ha='right')

if MAIN:
    axes[0].set_yticks(range(heatmap.shape[0]), [f'layer {i}' for i in range(heatmap.shape[0])])

if MAIN:
    axes[0].set_title('Held-out hidden-answer probe')

if MAIN:
    for layer in range(heatmap.shape[0]):
        for position in range(heatmap.shape[1]):
            value = float(heatmap[layer, position])
            axes[0].text(position, layer, f'{value:.1f}', ha='center', va='center', color='black' if value > 0.72 else 'white')

if MAIN:
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='accuracy')

if MAIN:
    means = [toy_signature['patch_control_means'][name] for name in conditions]

if MAIN:
    axes[1].bar(range(len(conditions)), means, color=control_colors)

if MAIN:
    axes[1].axhline(0, color='#50545a', linewidth=0.8)

if MAIN:
    axes[1].set_xticks(range(len(conditions)), conditions, rotation=32, ha='right')

if MAIN:
    axes[1].set_ylabel('mean donor-margin effect')

if MAIN:
    axes[1].set_title('Exact causal patch vs controls')

if MAIN:
    flips = [toy_signature['patch_control_flip_rates'][name] for name in conditions]

if MAIN:
    axes[2].bar(range(len(conditions)), flips, color=control_colors)

if MAIN:
    axes[2].set_ylim(0, 1.08)

if MAIN:
    axes[2].set_xticks(range(len(conditions)), conditions, rotation=32, ha='right')

if MAIN:
    axes[2].set_ylabel('flip rate toward donor')

if MAIN:
    axes[2].set_title('Only the causal patch flips')

if MAIN:
    fig.suptitle('Exact toy ground truth: localization, intervention, and controls', fontweight='bold')

if MAIN:
    fig.tight_layout()

if MAIN:
    toy_signature_path = asset_dir / 'cot_faithfulness_toy_ground_truth.png'

if MAIN:
    fig.savefig(toy_signature_path, dpi=180, bbox_inches='tight')

if MAIN:
    plt.show()

if MAIN:
    print(
        "Toy signature metrics: "
        f"probe={toy_signature['target_probe_accuracy']:.3f}, "
        f"shuffled_probe={toy_signature['label_shuffled_probe_accuracy']:.3f}, "
        f"target_effect={toy_signature['patch_control_means']['target_patch']:.3f}, "
        f"strongest_control={toy_signature['max_control_name']}:"
        f"{toy_signature['patch_control_means'][toy_signature['max_control_name']]:.3f}, "
        f"gap={toy_signature['patch_target_control_gap']:.3f}, "
        f"target_flip={toy_signature['patch_control_flip_rates']['target_patch']:.3f}"
    )

# %%

PINNED_PYTHIA_NEGATIVE_RESULT: dict[str,
    Any] = {'heldout_prompt_count': 40,
    'hidden_answer_accuracy': 0.7750000357627869,
    'label_shuffled_probe_accuracy': 0.22500000894069672,
    'detector_recall': 0.55,
    'text_only_recall': 1.0,
    'patch_control_means': {'target_patch': 0.6444976925849915,
    'text_only': 0.0,
    'label_shuffled': 0.18719482421875,
    'random_direction': 0.17066650092601776,
    'random_donor': 0.6444976925849915,
    'irrelevant_position': 0.0}}


def adjudicate_pythia_result(result: Mapping[str, Any]) -> RealModelVerdict:
    """Apply predeclared probe, intervention, and baseline claim boundaries."""
    required = {'heldout_prompt_count', 'hidden_answer_accuracy', 'label_shuffled_probe_accuracy', 'detector_recall', 'text_only_recall', 'patch_control_means'}
    missing = required - set(result)
    if missing:
        raise ValueError(f'result is missing required fields: {sorted(missing)}')
    means = result['patch_control_means']
    if 'target_patch' not in means:
        raise ValueError("patch_control_means must contain 'target_patch'.")
    control_names = [name for name in means if name != 'target_patch']
    if not control_names:
        raise ValueError('at least one named patch control is required.')
    strongest_control = max(control_names, key=lambda name: float(means[name]))
    target_control_gap = float(means['target_patch']) - float(means[strongest_control])
    detector_advantage = float(result['detector_recall']) - float(result['text_only_recall'])
    probe_passes = int(result['heldout_prompt_count']) >= 20 and float(result['hidden_answer_accuracy']) >= 0.7 and (float(result['label_shuffled_probe_accuracy']) <= 0.65)
    claim_supported = probe_passes and target_control_gap > 0.05 and (detector_advantage > 0)
    negative_result = not claim_supported and target_control_gap <= 0.01 and (detector_advantage <= 0) and (strongest_control == 'random_donor')
    return RealModelVerdict(claim_supported=bool(claim_supported), negative_result=bool(negative_result), strongest_control=strongest_control, target_control_gap=target_control_gap, detector_advantage=detector_advantage)


if MAIN:
    pythia_verdict = adjudicate_pythia_result(PINNED_PYTHIA_NEGATIVE_RESULT)

if MAIN:
    assert not pythia_verdict.claim_supported, 'Failed controls must block the Pythia faithfulness claim.'

if MAIN:
    assert pythia_verdict.negative_result, 'The adjudicator should label this locked evidence as a negative result.'

if MAIN:
    assert pythia_verdict.strongest_control == 'random_donor', 'The matched random donor should be the strongest control.'

if MAIN:
    assert abs(pythia_verdict.target_control_gap) < 1e-12, 'Target and random-donor patch effects should tie in the locked run.'

if MAIN:
    assert abs(pythia_verdict.detector_advantage + 0.45) < 1e-12, 'Text-only recall should exceed detector recall by 0.45.'

if MAIN:
    print(
        "Pinned Pythia adjudication: "
        f"claim_supported={pythia_verdict.claim_supported}, "
        f"negative_result={pythia_verdict.negative_result}, "
        f"target_control_gap={pythia_verdict.target_control_gap:.3f}, "
        f"detector_advantage={pythia_verdict.detector_advantage:.3f}"
    )

if MAIN:
    print('Recorded Pythia boundary checks passed!')

def run_smoke_test(cpu: bool=True) -> dict[str, Any]:
    """Assemble the exact local contract from the functions above."""
    _ = cpu
    return toy_ground_truth_signature()

def run_gpu_test(max_vram_gb: float=24.0) -> dict[str, Any]:
    """Run the pinned Pythia loader and hook plumbing maintained by the section."""
    return reference_solutions.run_gpu_test(max_vram_gb=max_vram_gb)

def run_full_experiment(max_vram_gb: float=24.0) -> dict[str, Any]:
    return run_gpu_test(max_vram_gb=max_vram_gb)

if MAIN:
    tests.test_notebook_contract(run_smoke_test)

# %%

def plot_gpu_signature_result(gpu_result: Mapping[str, Any]) -> None:
    heatmap = t.tensor(gpu_result['probe_heatmap'], dtype=t.float32)
    controls = list(gpu_result['patch_control_means'].keys())
    means = [gpu_result['patch_control_means'][name] for name in controls]
    flips = [gpu_result['patch_control_flip_rates'][name] for name in controls]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im = axes[0].imshow(heatmap, vmin=0, vmax=1, cmap='magma')
    axes[0].set_title('Pythia hidden-answer probe\nlayer-position heatmap')
    axes[0].set_xticks(range(len(gpu_result['semantic_positions'])), gpu_result['semantic_positions'], rotation=25, ha='right')
    axes[0].set_yticks(range(len(heatmap)), [f'h{i}' for i in range(len(heatmap))])
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].bar(range(len(controls)), means)
    axes[1].set_title('Hook patch effect vs controls')
    axes[1].set_ylabel('mean donor-margin effect')
    axes[1].set_xticks(range(len(controls)), controls, rotation=35, ha='right')
    axes[2].bar(range(len(controls)), flips)
    axes[2].set_title('Final answer flips toward donor')
    axes[2].set_ylabel('flip rate')
    axes[2].set_xticks(range(len(controls)), controls, rotation=35, ha='right')
    fig.tight_layout()
    output = asset_dir / 'cot_faithfulness_signature_result.png'
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.show()
    print(f'saved {output}')

if MAIN:
    RUN_GPU_SIGNATURE = False

if MAIN:
    if RUN_GPU_SIGNATURE and t.cuda.is_available():
        gpu_result = run_gpu_test(max_vram_gb=24.0)
        tests.test_live_gpu_signature_result(gpu_result)
        plot_gpu_signature_result(gpu_result)
        utils.print_report('Core GPU metrics',
            {'hidden_answer_accuracy': gpu_result['hidden_answer_accuracy'],
            'label_shuffled_probe_accuracy': gpu_result['label_shuffled_probe_accuracy'],
            'patch_target_control_gap': gpu_result['patch_target_control_gap'],
            'text_only_recall': gpu_result['text_only_recall'],
            'detector_recall': gpu_result['detector_recall']})
    else:
        print('Set RUN_GPU_SIGNATURE = True on CUDA to run the real Pythia hook-patching signature result. Parent verification will run it serially.')

# %%

if MAIN:
    if 'gpu_result' in globals() and gpu_result.get('cuda_available'):
        for row in gpu_result['qualitative_examples'][:6]:
            print('-' * 80)
            print(row['prompt'])
            print({'condition': row['condition'],
                'hidden': row['hidden_answer'],
                'visible': row['visible_answer'],
                'donor': row['donor_hidden_answer'],
                'clean': row['clean_prediction'],
                'target_patch': row['target_patched_prediction'],
                'random_direction': row['random_direction_prediction'],
                'target_patch_effect': round(row['target_patch_effect'],
                4),
                'random_direction_effect': round(row['random_direction_effect'],
                4)})
    else:
        print('Run the CUDA signature cell first.')

# %%

if MAIN:
    PLAY_LAYER = 2

if MAIN:
    PLAY_POSITION = 'final_prompt'

if MAIN:
    PLAY_TARGET_INDEX = 12

if MAIN:
    PLAY_DONOR_INDEX = 13

if MAIN:
    if PLAY_POSITION not in toy_batch.position_names:
        raise ValueError(f'PLAY_POSITION must be one of {toy_batch.position_names}.')

if MAIN:
    play_target = t.tensor([PLAY_TARGET_INDEX])

if MAIN:
    play_donor = t.tensor([PLAY_DONOR_INDEX])

if MAIN:
    target_answer = toy_batch.hidden_answer_ids[play_target]

if MAIN:
    donor_answer = toy_batch.hidden_answer_ids[play_donor]

if MAIN:
    if bool(target_answer.eq(donor_answer).any()):
        raise ValueError('Choose a donor with the opposite hidden answer.')

if MAIN:
    play_clean_logits = toy_answer_logits(toy_batch)[play_target]

if MAIN:
    play_patched_logits = toy_forward_patch_answer_logits(toy_batch, play_target, play_donor, layer_index=PLAY_LAYER, position_index=toy_batch.position_names.index(PLAY_POSITION))

if MAIN:
    play_effect = signed_margin_toward_donor(play_patched_logits, target_answer, donor_answer) - signed_margin_toward_donor(play_clean_logits, target_answer, donor_answer)

if MAIN:
    labels = ('A', 'B')

if MAIN:
    print(f'patch=(layer {PLAY_LAYER}, {PLAY_POSITION}), target={labels[int(target_answer.item())]}, donor={labels[int(donor_answer.item())]}')

if MAIN:
    print(f'clean={labels[int(play_clean_logits.argmax(dim=-1).item())]}, patched={labels[int(play_patched_logits.argmax(dim=-1).item())]}, donor-margin effect={float(play_effect.item()):.3f}')

# %%
