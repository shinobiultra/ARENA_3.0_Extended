# %%


MAIN = __name__ == "__main__"
from pathlib import Path as _MigrationPath
import sys as _migration_sys
_MIGRATION_ROOT = _MigrationPath(__file__).resolve().parents[3]
if str(_MIGRATION_ROOT) not in _migration_sys.path:
    _migration_sys.path.insert(0, str(_MIGRATION_ROOT))

# %%

from pathlib import Path as dm_Path

from types import SimpleNamespace as dm_SimpleNamespace

import hashlib as dm_hashlib

import json as dm_json

import math as dm_math

import os as dm_os

import sys as dm_sys

import tempfile as dm_tempfile

import numpy as dm_np

import torch as dm_torch

from torch import Tensor as dm_Tensor

import matplotlib.pyplot as dm_plt

from IPython import get_ipython as dm_get_ipython

if MAIN:
    if dm_get_ipython() is not None:
        dm_get_ipython().run_line_magic('matplotlib', 'inline')

if MAIN:
    dm_ROOT = next((p for p in [dm_Path.cwd(), *dm_Path.cwd().parents] if (p / 'chapter16_shapley_attribution_baselines').exists()))

if MAIN:
    if str(dm_ROOT) not in dm_sys.path:
        dm_sys.path.insert(0, str(dm_ROOT))

from chapter16_shapley_attribution_baselines.exercises.part10_datamodels import datamodel_methods_tests as dm_tests

from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.datamodel_data import load_data, data_manifest

if MAIN:
    dm_torch.set_num_threads(4)

if MAIN:
    DTYPE = dm_torch.float64

if MAIN:
    PART = dm_ROOT / 'chapter16_shapley_attribution_baselines/exercises/part10_datamodels'

if MAIN:
    dm_OUTPUT = dm_Path(dm_os.environ.get('ARENA_DATAMODELS_OUTPUT_DIRECTORY', '/tmp/arena16_10_datamodels'))

if MAIN:
    dm_OUTPUT.mkdir(parents=True, exist_ok=True)

if MAIN:
    dm_plt.rcParams.update({'figure.figsize': (10, 4), 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

def dm_show_figure(figure, name):
    figure.tight_layout()
    figure.savefig(dm_OUTPUT / f'{name}.png', dpi=150, bbox_inches='tight')
    dm_plt.show()

if MAIN:
    data = load_data()

if MAIN:
    assert data_manifest(data) == dm_json.loads((PART / 'data_manifest.json').read_text())

if MAIN:
    dm_train, query = (data['train'], data['query'])

# %%

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(2, 3, figsize=(9, 6))

if MAIN:
    for dm_row, dm_label in enumerate((0, 1)):
        ids = dm_torch.nonzero(dm_train['y'] == dm_label).flatten()[:2]
        q = dm_torch.nonzero(query['y'] == dm_label).flatten()[0]
        for dm_ax, dm_split, dm_i, role in zip(dm_axes[dm_row], [dm_train, dm_train, query], [*ids, q], ['Training', 'Training', 'Query']):
            dm_ax.imshow(dm_split['images'][dm_i], cmap='gray_r', vmin=0, vmax=1)
            dm_ax.set_title(f"{role}: digit {dm_split['digit'][dm_i].item()}")
            dm_ax.set_xlabel(dm_split['ids'][dm_i], fontsize=9)
            dm_ax.set_xticks([])
            dm_ax.set_yticks([])

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_images')

# %%

def sample_subsets(n: int, size: int, count: int, seed: int) -> dm_Tensor:
    """Uniform fixed-size membership masks, one independent permutation per row."""
    if not 0 < size < n or count < 1:
        raise ValueError('Use a nonempty proper subset and a positive sample count.')
    generator = dm_torch.Generator(device='cpu').manual_seed(seed)
    masks = dm_torch.zeros(count, n, dtype=dm_torch.bool)
    for row in range(count):
        included = dm_torch.randperm(n, generator=generator)[:size]
        masks[row, included] = True
    return masks

if MAIN:
    dm_tests.test_sample_subsets(sample_subsets)

# %%

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(1, 3, figsize=(12, 4), sharey=True)

if MAIN:
    for dm_ax, fraction in zip(dm_axes, (0.25, 0.5, 0.75)):
        masks = sample_subsets(32, int(32 * fraction), 20, 1610)
        dm_ax.imshow(masks, cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        dm_ax.set_title(f'α = {fraction}: {int(32 * fraction)} of 32 images')
        dm_ax.set_xlabel('Training-image index')

if MAIN:
    dm_axes[0].set_ylabel('Independently sampled model')

if MAIN:
    dm_axes[0].set_yticks([0, 5, 10, 15, 19])

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_membership')

# %%

def correct_class_margins(logits: dm_Tensor, labels: dm_Tensor) -> dm_Tensor:
    """True-class logit minus the largest other logit; preserve leading axes."""
    if logits.ndim < 2 or logits.shape[-1] < 2 or labels.shape != (logits.shape[-2],):
        raise ValueError('Expected logits [..., queries, classes] and labels [queries].')
    if labels.dtype != dm_torch.int64 or (labels < 0).any() or (labels >= logits.shape[-1]).any():
        raise ValueError('Labels must index valid classes.')
    index = labels.expand(*logits.shape[:-1]).unsqueeze(-1)
    correct = logits.gather(-1, index).squeeze(-1)
    incorrect = logits.scatter(-1, index, -dm_torch.inf).amax(-1)
    return correct - incorrect

if MAIN:
    dm_tests.test_correct_class_margins(correct_class_margins)

# %%

def soft_threshold(values: dm_Tensor, threshold: dm_Tensor | float) -> dm_Tensor:
    """Proximal map for a nonnegative elementwise L1 penalty."""
    threshold = dm_torch.as_tensor(threshold, dtype=values.dtype, device=values.device)
    if not dm_torch.isfinite(threshold).all() or (threshold < 0).any():
        raise ValueError('Thresholds must be finite and nonnegative.')
    return values.sign() * (values.abs() - threshold).clamp_min(0)

if MAIN:
    dm_tests.test_soft_threshold(soft_threshold)

# %%

if MAIN:
    dm_values = dm_torch.linspace(-3, 3, 301, dtype=DTYPE)

if MAIN:
    dm_fig, dm_ax = dm_plt.subplots(figsize=(7, 4))

if MAIN:
    dm_ax.plot(dm_values, dm_values, ':', color='0.5', label='No penalty')

if MAIN:
    for threshold in (0.5, 1.0):
        dm_ax.plot(dm_values, soft_threshold(dm_values, threshold), label=f'Threshold {threshold}')

if MAIN:
    dm_ax.set(xlabel='Before proximal step', ylabel='After proximal step')

if MAIN:
    dm_ax.axhline(0, color='0.8', linewidth=0.8)

if MAIN:
    dm_ax.axvline(0, color='0.8', linewidth=0.8)

if MAIN:
    dm_ax.legend()

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_soft_threshold')

# %%

def lasso_objective(x: dm_Tensor, y: dm_Tensor, weights: dm_Tensor, bias: dm_Tensor, penalty: dm_Tensor) -> dm_Tensor:
    """One objective per target: mean squared error / 2 plus lambda * L1."""
    residual = x @ weights + bias - y
    return residual.square().mean(0) / 2 + penalty * weights.abs().sum(0)


def lasso_kkt(gram: dm_Tensor, cross: dm_Tensor, weights: dm_Tensor, penalty: dm_Tensor) -> dm_Tensor:
    """Maximum stationarity violation per target for centered Lasso."""
    gradient = gram @ weights - cross
    active = (gradient + penalty * weights.sign()).abs()
    inactive = (gradient.abs() - penalty).clamp_min(0)
    return dm_torch.where(weights != 0, active, inactive).amax(0)

if MAIN:
    dm_tests.test_lasso_objective(lasso_objective)

if MAIN:
    dm_tests.test_lasso_kkt(lasso_kkt)

# %%

def fit_lasso(x: dm_Tensor, y: dm_Tensor, penalty: dm_Tensor, initial: dm_Tensor | None=None, tolerance: float=1e-08, max_steps: int=20000) -> dict[str, dm_Tensor | int]:
    """Accelerated proximal gradient with target-wise adaptive restart.

    Center using training means only. The Gram matrix is shared by all targets;
    each target still has its own coefficients, penalty and acceleration state.
    A checked KKT residual, not an iteration count, certifies convergence.
    """
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or (len(x) < 1):
        raise ValueError('Expected x [models, examples] and y [models, targets].')
    if penalty.shape != (y.shape[1],) or (penalty < 0).any() or (not dm_torch.isfinite(penalty).all()):
        raise ValueError('Provide one finite nonnegative penalty per target.')
    if not dm_torch.isfinite(x).all() or not dm_torch.isfinite(y).all():
        raise ValueError('Regression inputs must be finite.')
    if not dm_math.isfinite(tolerance) or tolerance <= 0 or max_steps < 1:
        raise ValueError('Use a positive finite tolerance and positive step limit.')
    x_mean, y_mean = (x.mean(0), y.mean(0))
    xc, yc = (x - x_mean, y - y_mean)
    gram, cross = (xc.T @ xc / len(x), xc.T @ yc / len(x))
    lipschitz = dm_torch.linalg.eigvalsh(gram)[-1]
    shape = (x.shape[1], y.shape[1])
    weights = x.new_zeros(shape) if initial is None else initial.clone()
    if weights.shape != shape or not dm_torch.isfinite(weights).all():
        raise ValueError('Initial coefficients have the wrong shape or nonfinite entries.')
    if lipschitz <= dm_torch.finfo(x.dtype).eps:
        weights.zero_()
        residual = lasso_kkt(gram, cross, weights, penalty)
        return dict(weights=weights, bias=y_mean, kkt=residual, steps=0, history=dm_torch.cat([x.new_zeros(1), residual.max().reshape(1)])[None])
    extrapolated = weights.clone()
    momentum_time = x.new_ones(y.shape[1])
    history = []
    for step in range(1, max_steps + 1):
        gradient = gram @ extrapolated - cross
        updated = soft_threshold(extrapolated - gradient / lipschitz, penalty / lipschitz)
        next_time = (1 + dm_torch.sqrt(1 + 4 * momentum_time.square())) / 2
        restart = ((extrapolated - updated) * (updated - weights)).sum(0) > 0
        acceleration = dm_torch.where(restart, 0, (momentum_time - 1) / next_time)
        extrapolated = updated + acceleration * (updated - weights)
        momentum_time = dm_torch.where(restart, 1, next_time)
        weights = updated
        if step == 1 or step % 10 == 0 or step == max_steps:
            residual = lasso_kkt(gram, cross, weights, penalty)
            history.append(dm_torch.stack([x.new_tensor(step), residual.max()]))
            if (residual <= tolerance).all():
                return dict(weights=weights, bias=y_mean - x_mean @ weights, kkt=residual, steps=step, history=dm_torch.stack(history))
    raise RuntimeError(f'Lasso did not converge: maximum KKT residual {residual.max().item():.3g}.')

if MAIN:
    dm_tests.test_fit_lasso(fit_lasso)

# %%

def fit_datamodel_path(x_train: dm_Tensor, y_train: dm_Tensor, x_val: dm_Tensor, y_val: dm_Tensor, fractions: dm_Tensor) -> dict[str, dm_Tensor]:
    """Fit every path point on train rows, then select per target on validation."""
    if fractions.ndim != 1 or len(fractions) < 1 or (not dm_torch.isfinite(fractions).all()):
        raise ValueError('Use a nonempty finite path of penalty fractions.')
    if (fractions <= 0).any() or (fractions > 1).any() or (fractions[1:] > fractions[:-1]).any():
        raise ValueError('Penalty fractions must descend within (0, 1].')
    if x_val.shape[1:] != x_train.shape[1:] or y_val.shape != (len(x_val), y_train.shape[1]):
        raise ValueError('Validation masks and targets must align with the training axes.')
    xc, yc = (x_train - x_train.mean(0), y_train - y_train.mean(0))
    lambda_max = (xc.T @ yc / len(x_train)).abs().amax(0)
    weights, biases, residuals, steps, validation, penalties = ([], [], [], [], [], [])
    initial = None
    for fraction in fractions:
        penalty = fraction * lambda_max
        fit = fit_lasso(x_train, y_train, penalty, initial=initial)
        initial = fit['weights']
        weights.append(fit['weights'])
        biases.append(fit['bias'])
        residuals.append(fit['kkt'])
        steps.append(fit['steps'])
        penalties.append(penalty)
        prediction = x_val @ fit['weights'] + fit['bias']
        validation.append((prediction - y_val).square().mean(0))
    weights, biases = (dm_torch.stack(weights), dm_torch.stack(biases))
    validation = dm_torch.stack(validation)
    selected = validation.argmin(0)
    targets = dm_torch.arange(y_train.shape[1], device=x_train.device)
    return dict(weights=weights, bias=biases, penalty=dm_torch.stack(penalties), kkt=dm_torch.stack(residuals), steps=dm_torch.tensor(steps), validation_mse=validation, selected=selected, selected_weights=weights[selected, :, targets].T, selected_bias=biases[selected, targets])

if MAIN:
    dm_tests.test_fit_datamodel_path(fit_datamodel_path)

# %%

if MAIN:
    INTERACTION = 0.0

if MAIN:
    SUBSET_SIZE = 12

if MAIN:
    fixture_masks = [sample_subsets(24, SUBSET_SIZE, count, seed).double() for count, seed in ((384, 31), (128, 32), (256, 33))]

if MAIN:
    fixture_weights = dm_torch.zeros(24, 1, dtype=DTYPE)

if MAIN:
    fixture_weights[[0, 4, 11], 0] = dm_torch.tensor([1.5, -1.0, 0.7], dtype=DTYPE)

def fixture_response(x):
    interaction = (x[:, 1] - x[:, 2]) * (x[:, 7] - x[:, 8])
    return 0.3 + x @ fixture_weights + INTERACTION * interaction[:, None]

if MAIN:
    fixture_y = [fixture_response(x) for x in fixture_masks]

if MAIN:
    fixture_path = fit_datamodel_path(*[fixture_masks[0], fixture_y[0], fixture_masks[1], fixture_y[1]], dm_torch.logspace(0, -4, 16, dtype=DTYPE))

if MAIN:
    fixture_pred = fixture_masks[2] @ fixture_path['selected_weights'] + fixture_path['selected_bias']

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(1, 2, figsize=(11, 4))

if MAIN:
    dm_axes[0].scatter(fixture_pred[:, 0], fixture_y[2][:, 0], s=12, alpha=0.45, label='Datamodel')

if MAIN:
    dm_axes[0].scatter(dm_torch.full_like(fixture_y[2][:, 0], fixture_y[0].mean()), fixture_y[2][:, 0], s=10, alpha=0.15, color='0.5', label='Mean only')

if MAIN:
    bounds = [float(fixture_y[2].min()), float(fixture_y[2].max())]

if MAIN:
    dm_axes[0].plot(bounds, bounds, 'k--', linewidth=1)

if MAIN:
    dm_axes[0].set(xlabel='Predicted output', ylabel='Exact held-out output', title=f'Interaction = {INTERACTION}')

if MAIN:
    dm_axes[0].legend()

if MAIN:
    dm_axes[1].plot(fixture_path['penalty'][:, 0], fixture_path['validation_mse'][:, 0], 'o-')

if MAIN:
    dm_chosen = fixture_path['selected'][0]

if MAIN:
    dm_axes[1].scatter(fixture_path['penalty'][dm_chosen, 0], fixture_path['validation_mse'][dm_chosen, 0], color='crimson', zorder=3, label='Validation choice')

if MAIN:
    dm_axes[1].axhline(float((fixture_y[1] - fixture_y[0].mean()).square().mean()), color='0.5', linestyle=':', label='Mean only')

if MAIN:
    dm_axes[1].set(xscale='log', yscale='log', xlabel='L1 penalty', ylabel='Validation MSE')

if MAIN:
    dm_axes[1].legend()

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_fixture_play')

# %%

if MAIN:
    gauge_masks = sample_subsets(24, 12, 48, 615).double()

if MAIN:
    gauge_w = fixture_path['selected_weights'][:, 0]

if MAIN:
    gauge_b = fixture_path['selected_bias'][0]

if MAIN:
    offset = 0.2

if MAIN:
    original = gauge_masks @ gauge_w + gauge_b

if MAIN:
    shifted = gauge_masks @ (gauge_w + offset) + gauge_b - 12 * offset

if MAIN:
    dm_torch.testing.assert_close(original, shifted)

if MAIN:
    dm_sizes = [4, 8, 12, 16, 20]

if MAIN:
    means, shifted_means = ([], [])

if MAIN:
    for size in dm_sizes:
        masks = sample_subsets(24, size, 48, 615).double()
        means.append(float((masks @ gauge_w + gauge_b).mean()))
        shifted_means.append(float((masks @ (gauge_w + offset) + gauge_b - 12 * offset).mean()))

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(1, 2, figsize=(11, 4))

if MAIN:
    dm_axes[0].scatter(original, shifted, alpha=0.6)

if MAIN:
    dm_axes[0].plot([float(original.min()), float(original.max())], [float(original.min()), float(original.max())], 'k--')

if MAIN:
    dm_axes[0].set(xlabel='Original prediction', ylabel='Offset prediction', title='Same subset size: identical')

if MAIN:
    dm_axes[1].plot(dm_sizes, means, 'o-', label='Original coefficients')

if MAIN:
    dm_axes[1].plot(dm_sizes, shifted_means, 's--', label='Same-size equivalent coefficients')

if MAIN:
    dm_axes[1].axvline(12, color='0.7', linestyle=':')

if MAIN:
    dm_axes[1].set(xlabel='Included images', ylabel='Mean predicted output', title='Changed size: different predictions')

if MAIN:
    dm_axes[1].legend(fontsize=9)

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_gauge')

# %%

def centered_weight_similarity(weights: dm_Tensor) -> dm_Tensor:
    """Cosine similarity of target datamodels, invariant to target-wise offsets."""
    centered = weights - weights.mean(0, keepdim=True)
    norms = centered.norm(dim=0)
    normalized = centered / norms.clamp_min(dm_torch.finfo(weights.dtype).eps)
    return normalized.T @ normalized

if MAIN:
    dm_tests.test_centered_weight_similarity(centered_weight_similarity)

# %%

def select_supportive(scores: dm_Tensor, budget: int) -> dm_Tensor:
    """Indices with largest scores, breaking ties by the original source order."""
    if scores.ndim != 1 or not dm_torch.isfinite(scores).all() or (not 0 <= budget < len(scores)):
        raise ValueError('Provide finite scores and a budget leaving at least one example.')
    return dm_torch.argsort(scores, descending=True, stable=True)[:budget].sort().values


def matched_random_removals(labels: dm_Tensor, selected: dm_Tensor, count: int, seed: int) -> dm_Tensor:
    """Random index sets matching the selected set's count in every class."""
    if labels.ndim != 1 or labels.dtype != dm_torch.int64 or selected.dtype != dm_torch.int64:
        raise ValueError('Labels and selected indices must be integer vectors.')
    if selected.ndim != 1 or len(selected.unique()) != len(selected) or count < 1:
        raise ValueError('Use distinct selected indices and a positive control count.')
    if (selected < 0).any() or (selected >= len(labels)).any():
        raise ValueError('Selected indices are outside the training pool.')
    generator = dm_torch.Generator(device='cpu').manual_seed(seed)
    sets = []
    for _ in range(count):
        pieces = []
        for label in labels.unique(sorted=True):
            candidates = dm_torch.nonzero(labels == label).flatten()
            number = int((labels[selected] == label).sum())
            permutation = dm_torch.randperm(len(candidates), generator=generator)
            pieces.append(candidates[permutation[:number]])
        sets.append(dm_torch.cat(pieces).sort().values)
    return dm_torch.stack(sets)

if MAIN:
    dm_tests.test_select_supportive(select_supportive)

if MAIN:
    dm_tests.test_matched_random_removals(matched_random_removals)

# %%

from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_methods import fit_weighted_logistic

if MAIN:
    FRACTIONS = (0.25, 0.5, 0.75)

if MAIN:
    BUDGETS = (2, 8, 16, 32)

# %%

def make_designs(methods) -> dict:
    """Build and check the prescribed mask splits without fitting any classifier."""
    designs = {}
    for a, fraction in enumerate(FRACTIONS):
        sizes = (('train', 1024), ('validation', 128), ('test', 256))
        design = {name: methods.sample_subsets(128, int(fraction * 128), count, 161000 + 100 * a + split) for split, (name, count) in enumerate(sizes)}
        together = dm_torch.cat(list(design.values()))
        if together.unique(dim=0).shape[0] != len(together):
            raise ValueError('A duplicate subset crossed or appeared within the prescribed splits.')
        designs[fraction] = design
    return designs

def refit_masks(train: dict, masks: dm_Tensor) -> dict[str, dm_Tensor]:
    """Batched independent classifier fits using the optimizer taught in 16.9."""
    if masks.ndim != 2 or masks.shape[1] != len(train['x']) or masks.dtype != dm_torch.bool:
        raise ValueError('Provide boolean training membership masks.')
    if (masks.sum(1) == 0).any():
        raise ValueError('A model cannot be trained on an empty subset.')
    weights, gradients, objectives = ([], [], [])
    for batch in masks.split(128):
        fit = fit_weighted_logistic(train['x'], train['y'], batch.double(), l2=0.02)
        weights.append(fit['theta'])
        gradients.append(fit['gradient_norms'][-1])
        objectives.append(fit['objective'])
    return dict(theta=dm_torch.cat(weights), gradient_norm=dm_torch.cat(gradients), objective=dm_torch.cat(objectives))

def model_margins(fit: dict, query: dict, methods) -> dm_Tensor:
    logit = fit['theta'] @ query['x'].T
    two_class_logits = dm_torch.stack([dm_torch.zeros_like(logit), logit], dim=-1)
    return methods.correct_class_margins(two_class_logits, query['y'])

def least_squares_datamodel(x: dm_Tensor, y: dm_Tensor) -> dict[str, dm_Tensor]:
    """Unpenalized minimum-norm baseline; center on training rows only."""
    xc, yc = (x - x.mean(0), y - y.mean(0))
    weights = dm_torch.linalg.lstsq(xc, yc, driver='gelsd').solution
    return dict(weights=weights, bias=y.mean(0) - x.mean(0) @ weights)

# %%

def fit_response_models(x: dm_Tensor, y: dm_Tensor, xv: dm_Tensor, yv: dm_Tensor, methods) -> dict:
    """Lasso and controls see only their fit and validation subset responses."""
    penalties = dm_torch.logspace(0, -4, 16, dtype=x.dtype)
    result = {'lasso': methods.fit_datamodel_path(x, y, xv, yv, penalties)}
    permutation = dm_torch.randperm(len(y), generator=dm_torch.Generator().manual_seed(161440))
    result['permuted'] = methods.fit_datamodel_path(x, y[permutation], xv, yv, penalties)
    result['permutation'] = permutation
    result['least_squares'] = least_squares_datamodel(x, y)
    result['intercept'] = y.mean(0)
    return result

def build_deletion_masks(train: dict, query: dict, responses: dict, methods) -> dict:
    """Freeze all selected and matched control sets before seeing refit outcomes."""
    query_ids = dm_torch.cat([dm_torch.nonzero(query['y'] == label).flatten()[:12] for label in (0, 1)]).sort().values
    pixel = dm_torch.nn.functional.normalize(train['x'][:, :64], dim=1)
    target_pixel = dm_torch.nn.functional.normalize(query['x'][:, :64], dim=1)
    pixel_scores = pixel @ target_pixel.T * dm_torch.where(train['y'][:, None] == query['y'][None], 1.0, -1.0)
    all_scores = [responses[a]['lasso']['selected_weights'] for a in FRACTIONS] + [pixel_scores]
    masks, target, budget_values, method_values, replicate, class_counts = ([], [], [], [], [], [])
    for method_index, scores in enumerate(all_scores):
        for position, q in enumerate(query_ids.tolist()):
            for budget in BUDGETS:
                selected = methods.select_supportive(scores[:, q], budget)
                seed = 162000 + 10000 * method_index + 100 * position + budget
                controls = methods.matched_random_removals(train['y'], selected, 16, seed)
                for rep, removed in enumerate(dm_torch.cat([selected[None], controls])):
                    mask = dm_torch.ones(len(train['y']), dtype=dm_torch.bool)
                    mask[removed] = False
                    masks.append(mask)
                    target.append(q)
                    budget_values.append(budget)
                    method_values.append(method_index)
                    replicate.append(rep - 1)
                    class_counts.append(dm_torch.bincount(train['y'][removed], minlength=2))
    return dict(masks=dm_torch.stack(masks), target=dm_torch.tensor(target), budget=dm_torch.tensor(budget_values), method=dm_torch.tensor(method_values), replicate=dm_torch.tensor(replicate), removed_class_counts=dm_torch.stack(class_counts), query_indices=query_ids, pixel_scores=pixel_scores)

def prepare_study(train: dict, query: dict, designs: dict, methods, directory: dm_Path) -> dict:
    """Fit/validate datamodels and construct deletions; never fit test subsets."""
    directory.mkdir(parents=True, exist_ok=False)
    subset_fits, outputs, responses, learning = ({}, {}, {}, {})
    for fraction in FRACTIONS:
        fit = {name: refit_masks(train, designs[fraction][name]) for name in ('train', 'validation')}
        margins = {name: model_margins(value, query, methods) for name, value in fit.items()}
        x, xv = [designs[fraction][name].double() for name in ('train', 'validation')]
        y, yv = (margins['train'], margins['validation'])
        response = fit_response_models(x, y, xv, yv, methods)
        if fraction == 0.5:
            for count in (64, 128, 256, 512):
                learning[count] = fit_response_models(x[:count], y[:count], xv, yv, methods)
        subset_fits[fraction], outputs[fraction], responses[fraction] = (fit, margins, response)
    deletions = build_deletion_masks(train, query, responses, methods)
    result = dict(phase='FIT_VALIDATION_AND_DELETION_SELECTION_COMPLETE', designs=designs, subset_fits=subset_fits, margins=outputs, responses=responses, learning=learning, deletions=deletions, training_ids=train['ids'], query_ids=query['ids'], test_subset_responses_seen=False, counterfactual_responses_seen=False)
    dm_torch.save(result, directory / 'prepared.pt')
    return result

# %%

if MAIN:
    method_names = ['sample_subsets', 'correct_class_margins', 'fit_datamodel_path', 'select_supportive', 'matched_random_removals']

if MAIN:
    dm_methods = dm_SimpleNamespace(**{name: globals()[name] for name in method_names})

if MAIN:
    designs = make_designs(dm_methods)

if MAIN:
    NATIVE = dm_Path(dm_tempfile.mkdtemp(prefix='native-', dir=dm_OUTPUT)) / 'study'

if MAIN:
    dm_prepared = prepare_study(dm_train, query, designs, dm_methods, NATIVE)

if MAIN:
    dm_boundary = {'phase': dm_prepared['phase'], 'prepared_sha256': dm_hashlib.sha256((NATIVE / 'prepared.pt').read_bytes()).hexdigest(), 'test_subset_responses_seen': False, 'counterfactual_responses_seen': False}

if MAIN:
    (dm_OUTPUT / 'selection_boundary.json').write_text(dm_json.dumps(dm_boundary, indent=2) + '\n')

if MAIN:
    print('3,456 classifier fits complete. All paths and deletion masks fixed; test responses still unseen.')

# %%

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(1, 3, figsize=(14, 4))

if MAIN:
    path_fractions = dm_torch.logspace(0, -4, 16, dtype=DTYPE)

if MAIN:
    for fraction, color in zip(FRACTIONS, ('C0', 'C1', 'C2')):
        response = dm_prepared['responses'][fraction]
        path = response['lasso']
        yv = dm_prepared['margins'][fraction]['validation']
        baseline = (yv - response['intercept']).square().mean(0)
        ratio = path['validation_mse'] / baseline
        dm_axes[0].plot(path_fractions, ratio.median(1).values, color=color, label=f'α={fraction}')
        dm_axes[0].fill_between(path_fractions, ratio.quantile(0.1, dim=1), ratio.quantile(0.9, dim=1), color=color, alpha=0.15)
        dm_axes[1].plot(path_fractions, (path['weights'] != 0).sum(1).double().median(1).values, color=color)
        dm_axes[2].plot(range(16), dm_torch.bincount(path['selected'], minlength=16), 'o-', color=color)

if MAIN:
    dm_axes[0].axhline(1, color='0.5', linestyle=':', label='Mean-only baseline')

if MAIN:
    dm_axes[0].set(xscale='log', yscale='log', xlabel='λ / λ_max', ylabel='Validation MSE / baseline MSE')

if MAIN:
    dm_axes[0].legend(fontsize=8)

if MAIN:
    dm_axes[1].set(xscale='log', xlabel='λ / λ_max', ylabel='Median nonzero coefficients', ylim=(-3, 132))

if MAIN:
    dm_axes[2].set(xlabel='Selected path index (15 = weakest penalty)', ylabel='Number of queries', xticks=[0, 5, 10, 15])

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_validation_path')

# %%

def response_predictions(response: dict, masks: dm_Tensor) -> dict[str, dm_Tensor]:
    x = masks.double()
    predictions = {}
    for name in ('lasso', 'permuted'):
        model = response[name]
        predictions[name] = x @ model['selected_weights'] + model['selected_bias']
    model = response['least_squares']
    predictions['least_squares'] = x @ model['weights'] + model['bias']
    predictions['intercept'] = response['intercept'].expand(len(x), -1)
    return predictions

def dm_evaluate_study(prepared: dict, train: dict, query: dict, methods, directory: dm_Path) -> dict:
    """Retrain held-out subsets and frozen counterfactuals, with no model selection."""
    if prepared.get('phase') != 'FIT_VALIDATION_AND_DELETION_SELECTION_COMPLETE':
        raise ValueError('Complete all datamodel fitting and deletion selection before evaluation.')
    if prepared['training_ids'] != train['ids'] or prepared['query_ids'] != query['ids']:
        raise ValueError('Data row identity/order changed between preparation and evaluation.')
    if prepared['test_subset_responses_seen'] or prepared['counterfactual_responses_seen']:
        raise ValueError('The prepared experiment claims it already inspected evaluation outcomes.')
    tests, cross_fraction, learning_predictions = ({}, {}, {})
    for fraction in FRACTIONS:
        masks = prepared['designs'][fraction]['test']
        fit = refit_masks(train, masks)
        margin = model_margins(fit, query, methods)
        predictions = response_predictions(prepared['responses'][fraction], masks)
        tests[fraction] = dict(fit=fit, margin=margin, predictions=predictions)
        cross_fraction[fraction] = {source: response_predictions(prepared['responses'][source], masks)['lasso'] for source in FRACTIONS}
        if fraction == 0.5:
            for count, response in prepared['learning'].items():
                learning_predictions[count] = response_predictions(response, masks)
    base_fit = refit_masks(train, dm_torch.ones(1, len(train['y']), dtype=dm_torch.bool))
    base_margin = model_margins(base_fit, query, methods)[0]
    deletion = prepared['deletions']
    refits = refit_masks(train, deletion['masks'])
    all_margin = model_margins(refits, query, methods)
    rows = dm_torch.arange(len(all_margin))
    query_index = deletion['target']
    target_margin = all_margin[rows, query_index]
    prediction = []
    for fraction in FRACTIONS:
        weights = prepared['responses'][fraction]['lasso']['selected_weights']
        full_changes = -(~deletion['masks']).double() @ weights
        prediction.append(full_changes[rows, query_index])
    result = dict(phase='FROZEN_EVALUATION_COMPLETE', test=tests, cross_fraction=cross_fraction, learning_predictions=learning_predictions, base_fit=base_fit, base_margin=base_margin, counterfactual=dict(fit=refits, all_margin=all_margin, target_margin=target_margin, actual_change=target_margin - base_margin[query_index], predicted_change=dm_torch.stack(prediction, dim=1)), selection_used_test_responses=False)
    dm_torch.save(result, directory / 'evaluated.pt')
    return result

# %%

if MAIN:
    dm_evaluated = dm_evaluate_study(dm_prepared, dm_train, query, dm_methods, NATIVE)

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(1, 3, figsize=(13, 4))

if MAIN:
    for dm_ax, fraction in zip(dm_axes, FRACTIONS):
        result = dm_evaluated['test'][fraction]
        observed = result['margin']
        center = observed.mean(0)
        predicted = result['predictions']['lasso']
        xp, yp = ((predicted - center).flatten().numpy(), (observed - center).flatten().numpy())
        dm_ax.hexbin(xp, yp, gridsize=35, mincnt=1, cmap='Blues')
        control = (result['predictions']['permuted'] - center).flatten().numpy()
        dm_ax.scatter(control[::32], yp[::32], color='darkorange', alpha=0.2, s=7, label='Permuted responses')
        limits = [min(xp.min(), yp.min()), max(xp.max(), yp.max())]
        dm_ax.plot(limits, limits, 'k--', linewidth=1)
        dm_ax.set(xlabel='Predicted margin − query mean', ylabel='Retrained margin − query mean', title=f'α = {fraction}')

if MAIN:
    dm_axes[0].legend(fontsize=8)

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_heldout_predictions')

# %%

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(3, 1, figsize=(12, 8), sharex=True)

if MAIN:
    styles = {'lasso': ('C0', '-'), 'least_squares': ('C2', '--'), 'permuted': ('C1', '-'), 'intercept': ('0.5', ':')}

if MAIN:
    for dm_ax, fraction in zip(dm_axes, FRACTIONS):
        result = dm_evaluated['test'][fraction]
        observed = result['margin']
        for dm_name, prediction in result['predictions'].items():
            mse = (prediction - observed).square().mean(0)
            r2 = 1 - mse / observed.var(0, unbiased=False)
            color, linestyle = styles[dm_name]
            dm_ax.plot(r2, label=dm_name.replace('_', ' '), color=color, linestyle=linestyle, linewidth=1.4)
        dm_ax.set(ylabel='Per-query R²', title=f'α = {fraction}', ylim=(-0.09, 1.03))
        dm_ax.axhline(0, color='0.8', linewidth=0.7)

if MAIN:
    dm_axes[0].legend(ncol=4, fontsize=9)

if MAIN:
    dm_axes[-1].set_xlabel('Query index in fixed source order')

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_query_r2')

# %%

if MAIN:
    dm_counts = [64, 128, 256, 512, 1024]

if MAIN:
    observed = dm_evaluated['test'][0.5]['margin']

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(1, 3, figsize=(15, 4))

if MAIN:
    for dm_name, (color, linestyle) in styles.items():
        errors = []
        for count in dm_counts:
            predictions = dm_evaluated['test'][0.5]['predictions'] if count == 1024 else dm_evaluated['learning_predictions'][count]
            errors.append(float((predictions[dm_name] - observed).square().mean()))
        dm_axes[0].plot(dm_counts, errors, 'o', linestyle=linestyle, color=color, label=dm_name.replace('_', ' '))

if MAIN:
    densities = []

if MAIN:
    for count in dm_counts:
        response = dm_prepared['responses'][0.5] if count == 1024 else dm_prepared['learning'][count]
        densities.append((response['lasso']['selected_weights'] != 0).sum(0).double())

if MAIN:
    densities = dm_torch.stack(densities)

if MAIN:
    dm_axes[1].plot(dm_counts, densities.median(1).values, 'o-', label='Median query')

if MAIN:
    dm_axes[1].fill_between(dm_counts, densities.min(1).values, densities.max(1).values, alpha=0.15, label='All-query range')

if MAIN:
    dm_axes[0].set(xscale='log', yscale='log', xlabel='Subset models used for regression', ylabel='Held-out margin MSE')

if MAIN:
    dm_axes[1].set(xscale='log', xlabel='Subset models used for regression', ylabel='Selected nonzero coefficients', ylim=(-3, 132))

if MAIN:
    condition_numbers = []

if MAIN:
    design = dm_prepared['designs'][0.5]['train'].double()

if MAIN:
    for count in dm_counts:
        singular_values = dm_torch.linalg.svdvals(design[:count] - design[:count].mean(0))
        nonzero = singular_values[singular_values > 1e-10]
        condition_numbers.append(float(nonzero.max() / nonzero.min()))

if MAIN:
    dm_axes[2].plot(dm_counts, condition_numbers, 'o-', color='C3')

if MAIN:
    dm_axes[2].set(xscale='log', yscale='log', xlabel='Subset models used for regression', ylabel='Nonzero singular-value condition number')

if MAIN:
    for dm_ax in dm_axes:
        dm_ax.set_xticks(dm_counts, [str(c) for c in dm_counts])

if MAIN:
    for dm_ax in dm_axes[:2]:
        dm_ax.legend(fontsize=8)

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_sample_complexity')

# %%

if MAIN:
    cross_mse = dm_torch.tensor([[float((dm_evaluated['cross_fraction'][target][source] - dm_evaluated['test'][target]['margin']).square().mean()) for target in FRACTIONS] for source in FRACTIONS])

if MAIN:
    dm_fig, dm_ax = dm_plt.subplots(figsize=(7, 5))

from matplotlib.colors import LogNorm

if MAIN:
    heat = dm_ax.imshow(cross_mse, cmap='magma_r', norm=LogNorm(vmin=float(cross_mse.min()), vmax=float(cross_mse.max())))

if MAIN:
    for dm_row in range(3):
        for dm_col in range(3):
            dm_ax.text(dm_col, dm_row, f'{cross_mse[dm_row, dm_col]:.4f}', ha='center', va='center', color='white' if cross_mse[dm_row, dm_col] > 0.01 else 'black')

if MAIN:
    dm_ax.set(xticks=range(3), xticklabels=FRACTIONS, yticks=range(3), yticklabels=FRACTIONS, xlabel='Test subset fraction', ylabel='Fitted subset fraction', title='Actual held-out margin MSE')

if MAIN:
    dm_fig.colorbar(heat, ax=dm_ax, label='MSE (log color scale)')

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_cross_fraction')

# %%

if MAIN:
    dm_gallery = dm_torch.cat([dm_torch.nonzero(query['y'] == label).flatten()[:2] for label in (0, 1)]).sort().values

if MAIN:
    weights = dm_prepared['responses'][0.5]['lasso']['selected_weights']

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(4, 7, figsize=(14, 9))

if MAIN:
    for dm_row, q in enumerate(dm_gallery.tolist()):
        w = weights[:, q]
        high = dm_torch.argsort(w, descending=True, stable=True)[:2]
        low = dm_torch.argsort(w, stable=True)[:2]
        close = dm_torch.argsort(dm_prepared['deletions']['pixel_scores'][:, q], descending=True, stable=True)[:2]
        dm_axes[dm_row, 0].imshow(query['images'][q], cmap='gray_r', vmin=0, vmax=1)
        dm_axes[dm_row, 0].set_title(f"Query: {query['digit'][q].item()}", fontsize=10)
        dm_axes[dm_row, 0].set_xlabel(query['ids'][q], fontsize=7)
        for dm_col, (dm_index, role) in enumerate(zip([*high, *low, *close], ['Support', 'Support', 'Oppose', 'Oppose', 'Pixel', 'Pixel']), start=1):
            dm_axes[dm_row, dm_col].imshow(dm_train['images'][dm_index], cmap='gray_r', vmin=0, vmax=1)
            dm_axes[dm_row, dm_col].set_title(f"{role}: {dm_train['digit'][dm_index].item()}\nw = {w[dm_index]:+.3f}", fontsize=9)
            dm_axes[dm_row, dm_col].set_xlabel(dm_train['ids'][dm_index], fontsize=7)

if MAIN:
    for dm_ax in dm_axes.flat:
        dm_ax.set_xticks([])
        dm_ax.set_yticks([])

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_coefficient_gallery')

# %%

if MAIN:
    ordered = dm_torch.argsort(query['y'], stable=True)

if MAIN:
    weights = dm_prepared['responses'][0.5]['lasso']['selected_weights']

if MAIN:
    shuffled_weights = weights.clone()

if MAIN:
    generator = dm_torch.Generator().manual_seed(161771)

if MAIN:
    for q in range(weights.shape[1]):
        shuffled_weights[:, q] = weights[dm_torch.randperm(len(weights), generator=generator), q]

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(1, 2, figsize=(10, 4.5))

if MAIN:
    for dm_ax, current, dm_title in zip(dm_axes, [weights, shuffled_weights], ['Datamodel directions', 'Independent coordinate shuffle']):
        similarity = centered_weight_similarity(current)[ordered][:, ordered]
        heat = dm_ax.imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1)
        dm_ax.axvline(31.5, color='0.3', linewidth=0.8)
        dm_ax.axhline(31.5, color='0.3', linewidth=0.8)
        dm_ax.set(title=dm_title, xlabel='Queries: 32 threes, then 32 eights', ylabel='Query')
        dm_fig.colorbar(heat, ax=dm_ax, fraction=0.046, pad=0.04, label='Centered cosine')

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_similarity')

# %%

if MAIN:
    deletions = dm_prepared['deletions']

if MAIN:
    counterfactual = dm_evaluated['counterfactual']

if MAIN:
    method_labels = ['Datamodel α=.25', 'Datamodel α=.5', 'Datamodel α=.75', 'Pixel similarity']

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

if MAIN:
    for method_index, (dm_ax, dm_label) in enumerate(zip(dm_axes.flat, method_labels)):
        actual, prediction, low, high, average = ([0.0], [0.0], [0.0], [0.0], [0.0])
        for budget in BUDGETS:
            scope = (deletions['method'] == method_index) & (deletions['budget'] == budget)
            dm_chosen = scope & (deletions['replicate'] == -1)
            actual.append(float(counterfactual['actual_change'][dm_chosen].mean()))
            prediction.append(float(counterfactual['predicted_change'][dm_chosen, method_index if method_index < 3 else 1].mean()))
            controls = dm_torch.stack([counterfactual['actual_change'][scope & (deletions['replicate'] == rep)].mean() for rep in range(16)])
            low.append(float(controls.min()))
            high.append(float(controls.max()))
            average.append(float(controls.mean()))
        dm_budgets = [0, *BUDGETS]
        dm_ax.plot(dm_budgets, actual, 'o-', label='Actual selected deletion')
        dm_ax.plot(dm_budgets, prediction, 's--', label='Datamodel prediction')
        dm_ax.plot(dm_budgets, average, color='0.5', label='Matched random mean')
        dm_ax.fill_between(dm_budgets, low, high, color='0.5', alpha=0.2, label='16 random replicate means')
        dm_ax.set(title=dm_label, xlabel='Images removed', ylabel='Mean correct-margin change')

if MAIN:
    dm_axes[0, 0].legend(fontsize=8)

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_deletion_curves')

# %%

if MAIN:
    dm_fig, dm_axes = dm_plt.subplots(1, 2, figsize=(13, 4.5))

if MAIN:
    for method_index, dm_label in enumerate(method_labels):
        dm_chosen = (deletions['method'] == method_index) & (deletions['budget'] == 32) & (deletions['replicate'] == -1)
        targets = deletions['target'][dm_chosen]
        dm_axes[0].plot(range(len(targets)), counterfactual['target_margin'][dm_chosen], 'o-', markersize=3, label=dm_label)
        flips = []
        for budget in BUDGETS:
            dm_selected = (deletions['method'] == method_index) & (deletions['budget'] == budget) & (deletions['replicate'] == -1)
            flips.append(int((counterfactual['target_margin'][dm_selected] < 0).sum()))
        dm_axes[1].plot([0, *BUDGETS], [0, *flips], 'o-', label=dm_label)

if MAIN:
    dm_axes[0].plot(range(24), dm_evaluated['base_margin'][deletions['query_indices']], 'k:', label='Full pool')

if MAIN:
    dm_axes[0].axhline(0, color='crimson', linewidth=1)

if MAIN:
    dm_axes[0].set(xlabel='Fixed intervention-query position', ylabel='Actual retrained margin', title='Remove 32 images')

if MAIN:
    random_low, random_high = ([0], [0])

if MAIN:
    for budget in BUDGETS:
        dm_counts = []
        for method_index in range(4):
            for dm_replicate in range(16):
                dm_selected = (deletions['method'] == method_index) & (deletions['budget'] == budget) & (deletions['replicate'] == dm_replicate)
                dm_counts.append(int((counterfactual['target_margin'][dm_selected] < 0).sum()))
        random_low.append(min(dm_counts))
        random_high.append(max(dm_counts))

if MAIN:
    dm_axes[1].fill_between([0, *BUDGETS], random_low, random_high, color='0.5', alpha=0.25, label='Matched random: all method/seed counts')

if MAIN:
    dm_axes[1].legend(fontsize=7, loc='upper left')

if MAIN:
    dm_axes[1].set(xlabel='Images removed', ylabel='Correct → incorrect queries (of 24)', yticks=range(0, 6))

if MAIN:
    dm_axes[0].legend(fontsize=7, ncol=2)

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_individual_flips')

# %%

def deletion_play(prepared: dict, train: dict, query: dict, methods, query_index: int=0, fraction: float=0.5, budget: int=8) -> dict:
    """An editable intervention: base, selected deletion and sixteen random refits."""
    if not 0 <= query_index < len(query['y']) or fraction not in FRACTIONS:
        raise ValueError('Choose an available query and fitted subsampling fraction.')
    weights = prepared['responses'][fraction]['lasso']['selected_weights'][:, query_index]
    selected = methods.select_supportive(weights, budget)
    controls = methods.matched_random_removals(train['y'], selected, 16, 163001)
    removals = dm_torch.cat([selected[None], controls])
    masks = dm_torch.ones(18, len(train['y']), dtype=dm_torch.bool)
    for i, removed in enumerate(removals, start=1):
        masks[i, removed] = False
    fit = refit_masks(train, masks)
    margin = model_margins(fit, query, methods)[:, query_index]
    prediction = -(~masks).double() @ weights
    return dict(query_index=query_index, fraction=fraction, budget=budget, removals=removals, masks=masks, fit=fit, margin=margin, actual_change=margin - margin[0], predicted_change=prediction)

# %%

if MAIN:
    QUERY_INDEX = 0

if MAIN:
    SUBSET_FRACTION = 0.5

if MAIN:
    REMOVAL_BUDGET = 8

if MAIN:
    dm_play = deletion_play(dm_prepared, dm_train, query, dm_methods, QUERY_INDEX, SUBSET_FRACTION, REMOVAL_BUDGET)

if MAIN:
    dm_torch.save(dm_play, NATIVE / 'play.pt')

if MAIN:
    dm_fig = dm_plt.figure(figsize=(12, 6.5))

if MAIN:
    grid = dm_fig.add_gridspec(2, 2, height_ratios=[1, 1.25])

if MAIN:
    ax_query = dm_fig.add_subplot(grid[0, 0])

if MAIN:
    ax_query.imshow(query['images'][QUERY_INDEX], cmap='gray_r', vmin=0, vmax=1)

if MAIN:
    ax_query.set_title(f"Query {QUERY_INDEX}: digit {query['digit'][QUERY_INDEX].item()} | base margin {dm_play['margin'][0]:.3f}")

if MAIN:
    ax_query.axis('off')

if MAIN:
    ax_changes = dm_fig.add_subplot(grid[0, 1])

if MAIN:
    ax_changes.scatter(dm_play['actual_change'][2:], dm_np.arange(16), color='0.5', s=20, label='Matched random refits')

if MAIN:
    ax_changes.axvline(float(dm_play['actual_change'][1]), label='Selected deletion: actual', color='C0')

if MAIN:
    ax_changes.axvline(float(dm_play['predicted_change'][1]), label='Selected deletion: prediction', color='C1', linestyle='--')

if MAIN:
    ax_changes.axvline(0, color='0.8', linewidth=0.7)

if MAIN:
    ax_changes.set(xlabel='Correct-margin change', ylabel='Random replicate', yticks=[0, 5, 10, 15])

if MAIN:
    ax_changes.legend(fontsize=7)

if MAIN:
    removed = dm_play['removals'][0]

if MAIN:
    columns = min(8, max(1, len(removed)))

if MAIN:
    dm_rows = dm_math.ceil(max(1, len(removed)) / columns)

if MAIN:
    image_grid = grid[1, :].subgridspec(dm_rows, columns)

if MAIN:
    for dm_position in range(dm_rows * columns):
        dm_ax = dm_fig.add_subplot(image_grid[dm_position // columns, dm_position % columns])
        if dm_position < len(removed):
            dm_i = removed[dm_position]
            dm_ax.imshow(dm_train['images'][dm_i], cmap='gray_r', vmin=0, vmax=1)
            dm_ax.set_title(f"{dm_train['digit'][dm_i].item()} | {dm_train['ids'][dm_i].split(':')[-1]}", fontsize=8)
        dm_ax.axis('off')

if MAIN:
    dm_show_figure(dm_fig, 'datamodel_real_play')

# %%

from chapter16_shapley_attribution_baselines.exercises.part10_datamodels.run_native import reference_checks as dm_reference_checks

if MAIN:
    references = dm_reference_checks(data, dm_prepared, dm_evaluated)

if MAIN:
    gradient_values = [fit['gradient_norm'].max() for values in dm_prepared['subset_fits'].values() for fit in values.values()]

if MAIN:
    gradient_values += [value['fit']['gradient_norm'].max() for value in dm_evaluated['test'].values()]

if MAIN:
    gradient_values += [dm_evaluated['base_fit']['gradient_norm'].max(), counterfactual['fit']['gradient_norm'].max()]

if MAIN:
    dm_summary = {'native_directory': str(NATIVE), 'classifier_refits': 10753, 'separate_refits': 6, 'independent_scipy_optima': 10, 'independent_lasso_fits': 12, 'play_refits': 18, 'train_images': 128, 'query_images': 64, 'heldout_subsets_per_fraction': 256, 'selection_used_test_responses': False, 'max_classifier_gradient': float(dm_torch.stack(gradient_values).max()), 'base_query_correct': int((dm_evaluated['base_margin'] > 0).sum())}

if MAIN:
    (dm_OUTPUT / 'native_summary.json').write_text(dm_json.dumps(dm_summary, indent=2) + '\n')

if MAIN:
    (dm_OUTPUT / 'reference_checks.json').write_text(dm_json.dumps(references, indent=2) + '\n')

if MAIN:
    print('Independent classifier and Lasso checks passed. All native fits and figures are saved.')

# %%

from pathlib import Path as cart_Path

from types import SimpleNamespace as cart_SimpleNamespace

import hashlib as cart_hashlib

import json as cart_json

import math as cart_math

import os as cart_os

import sys as cart_sys

import tempfile as cart_tempfile

import numpy as cart_np

import torch as cart_torch

from torch import Tensor as cart_Tensor

import torch.nn.functional as F

import matplotlib.pyplot as cart_plt

from IPython import get_ipython as cart_get_ipython

if MAIN:
    if cart_get_ipython() is not None:
        cart_get_ipython().run_line_magic('matplotlib', 'inline')

if MAIN:
    cart_ROOT = next((p for p in [cart_Path.cwd(), *cart_Path.cwd().parents] if (p / 'chapter16_shapley_attribution_baselines').exists()))

if MAIN:
    if str(cart_ROOT) not in cart_sys.path:
        cart_sys.path.insert(0, str(cart_ROOT))

from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography import cartography_methods_tests as cart_tests

from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography.cartography_data import load_training, load_evaluation

from chapter16_shapley_attribution_baselines.exercises.part11_dataset_cartography.run_native import validate_freeze, reference_checks as cart_reference_checks, summarize

if MAIN:
    cart_torch.set_num_threads(4)

if MAIN:
    cart_torch.use_deterministic_algorithms(True)

if MAIN:
    cart_OUTPUT = cart_Path(cart_os.environ.get('ARENA_CARTOGRAPHY_OUTPUT_DIRECTORY', '/tmp/arena16_11_cartography'))

if MAIN:
    cart_OUTPUT.mkdir(parents=True, exist_ok=True)

if MAIN:
    cart_plt.rcParams.update({'figure.figsize': (10, 4), 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

def cart_show_figure(figure, name):
    figure.savefig(cart_OUTPUT / f'{name}.png', dpi=145, bbox_inches='tight')
    cart_plt.show()
    cart_plt.close(figure)

if MAIN:
    validate_freeze()

if MAIN:
    cart_train = load_training()

if MAIN:
    print(f"Training pool: {len(cart_train['ids'])} images; 300 per digit; 300 known injected label errors.")

# %%

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(2, 5, figsize=(11, 4.6))

if MAIN:
    for cart_position, cart_ax in enumerate(cart_axes.flat):
        indices = cart_torch.where((cart_train['y'] == cart_position) & cart_train['changed'])[0]
        cart_index = int(indices[0])
        cart_ax.imshow(cart_train['x'][cart_index].reshape(8, 8), cmap='gray_r', vmin=0, vmax=1)
        cart_ax.set_title(f"Original {cart_position} → assigned {int(cart_train['noisy_y'][cart_index])}", fontsize=10)
        cart_ax.set_xlabel(cart_train['ids'][cart_index], fontsize=8)
        cart_ax.set_xticks([])
        cart_ax.set_yticks([])

if MAIN:
    cart_fig.suptitle('The same pixels, with a controlled label change', y=1.03)

if MAIN:
    cart_fig.tight_layout()

if MAIN:
    cart_show_figure(cart_fig, 'cartography_images')

# %%

def align_epoch(logits: cart_Tensor, visited_ids: cart_Tensor) -> cart_Tensor:
    """Restore B × N × C logits to canonical integer IDs 0 through N-1."""
    if logits.ndim != 3 or visited_ids.shape != logits.shape[:2]:
        raise ValueError('Expected one visited ID per model and example.')
    expected = cart_torch.arange(logits.shape[1], device=visited_ids.device)
    if not cart_torch.equal(visited_ids.sort(dim=1).values, expected.expand_as(visited_ids)):
        raise ValueError('Every epoch must visit each ID exactly once; check missing/duplicate IDs.')
    return logits.gather(1, visited_ids.argsort(dim=1)[..., None].expand_as(logits))

if MAIN:
    cart_tests.test_align_epoch(align_epoch)

# %%

def gold_probabilities(logits: cart_Tensor, labels: cart_Tensor) -> cart_Tensor:
    """Gather the assigned label, preserving any leading epoch dimension."""
    probabilities = logits.softmax(dim=-1)
    indices = labels.expand(logits.shape[:-1]).unsqueeze(-1)
    return probabilities.gather(-1, indices).squeeze(-1)

if MAIN:
    cart_tests.test_gold_probabilities(gold_probabilities)


def training_statistics(logits: cart_Tensor, labels: cart_Tensor) -> dict[str, cart_Tensor]:
    """Reduce E × B × N × C over epochs only; variability uses population SD."""
    if logits.ndim != 4 or logits.shape[0] == 0 or labels.shape != logits.shape[1:3]:
        raise ValueError('Use nonempty epoch/model/example/class logits and model/example labels.')
    probabilities = gold_probabilities(logits, labels)
    return dict(confidence=probabilities.mean(dim=0), variability=probabilities.std(dim=0, correction=0), correctness=(logits.argmax(dim=-1) == labels).float().mean(dim=0))

if MAIN:
    cart_tests.test_training_statistics(training_statistics)

# %%

if MAIN:
    history = cart_torch.tensor([0.05, 0.08, 0.16, 0.32, 0.55, 0.73, 0.87, 0.95])

if MAIN:
    assert ((history > 0) & (history < 1)).all(), 'Keep fixture probabilities strictly between zero and one.'

if MAIN:
    curves = {'Your history': history, 'Reversed': history.flip(0), 'Constant mean': history.mean().expand_as(history)}

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(1, 2, figsize=(10, 4))

if MAIN:
    for cart_name, cart_values in curves.items():
        logits = cart_torch.stack([cart_values.log(), (1 - cart_values).log()], dim=-1)[:, None, None]
        point = training_statistics(logits, cart_torch.zeros(1, 1, dtype=cart_torch.long))
        cart_axes[0].plot(range(1, len(cart_values) + 1), cart_values, 'o--' if cart_name == 'Constant mean' else 'o-', label=cart_name)
        cart_axes[1].scatter(point['variability'], point['confidence'], s=110 if cart_name == 'Your history' else 40, label=cart_name, alpha=0.8)

if MAIN:
    mu = cart_torch.linspace(0, 1, 401)

if MAIN:
    cart_axes[1].plot(cart_torch.sqrt(mu * (1 - mu)), mu, color='gray', label='Feasible boundary')

if MAIN:
    cart_axes[0].set(xlabel='Epoch', ylabel='Assigned-label probability', ylim=(-0.03, 1.03))

if MAIN:
    cart_axes[1].set(xlabel='Variability (population SD)', ylabel='Confidence', xlim=(-0.02, 0.52), ylim=(-0.03, 1.03))

if MAIN:
    for cart_ax in cart_axes:
        cart_ax.legend(fontsize=9)

if MAIN:
    cart_fig.tight_layout()

if MAIN:
    cart_show_figure(cart_fig, 'cartography_trajectory_play')

# %%

def initialize_parameters(features, classes, width, generators):
    """Small architecture setup; independent generators also supply batch orders."""
    draws = []
    for generator in generators:
        if width:
            item = dict(w1=cart_torch.randn(features, width, generator=generator) * cart_math.sqrt(2 / features), b1=cart_torch.zeros(1, width), w2=cart_torch.randn(width, classes, generator=generator) / cart_math.sqrt(width), b2=cart_torch.zeros(1, classes))
        else:
            item = dict(w=cart_torch.randn(features, classes, generator=generator) / cart_math.sqrt(features), b=cart_torch.zeros(1, classes))
        draws.append(item)
    return {name: cart_torch.stack([item[name] for item in draws]).requires_grad_() for name in draws[0]}

# %%

def batched_logits(parameters: dict[str, cart_Tensor], x: cart_Tensor) -> cart_Tensor:
    """Apply independent linear or one-hidden-layer networks along model axis B."""
    if 'w1' in parameters:
        hidden = (x @ parameters['w1'] + parameters['b1']).relu()
        return hidden @ parameters['w2'] + parameters['b2']
    return x @ parameters['w'] + parameters['b']

if MAIN:
    cart_tests.test_batched_logits(batched_logits)

# %%

def train_ensemble(x: cart_Tensor, labels: cart_Tensor, seeds: list[int], width: int=64, epochs: int=30, batch_size: int=128, lr: float=0.05, momentum: float=0.9, weight_decay: float=0.0001, classes: int=10, record: bool=True) -> dict:
    """Fresh SGD runs, with pre-update visits and epoch-end snapshots aligned by ID."""
    if x.ndim != 3 or labels.shape != x.shape[:2] or len(seeds) != x.shape[0]:
        raise ValueError('Provide B × N × D inputs, B × N labels and B run seeds.')
    if min(epochs, batch_size, x.shape[1]) < 1:
        raise ValueError('Epoch, minibatch and dataset sizes must be positive.')
    generators = [cart_torch.Generator().manual_seed(seed) for seed in seeds]
    parameters = initialize_parameters(x.shape[-1], classes, width, generators)
    velocity = {name: cart_torch.zeros_like(value) for name, value in parameters.items()}
    model_ids = cart_torch.arange(len(seeds))[:, None]
    online_trace, end_trace, losses = ([], [], [])
    steps = 0
    for _ in range(epochs):
        order = cart_torch.stack([cart_torch.randperm(x.shape[1], generator=g) for g in generators])
        visits = []
        for start in range(0, x.shape[1], batch_size):
            ids = order[:, start:start + batch_size]
            batch_x, batch_y = (x[model_ids, ids], labels[model_ids, ids])
            logits = batched_logits(parameters, batch_x)
            if record:
                visits.append(logits.detach().clone())
            individual_losses = F.cross_entropy(logits.flatten(0, 1), batch_y.flatten(), reduction='none').reshape_as(batch_y).mean(dim=1)
            gradients = cart_torch.autograd.grad(individual_losses.sum(), tuple(parameters.values()))
            with cart_torch.no_grad():
                for (name, value), gradient in zip(parameters.items(), gradients):
                    gradient = gradient + weight_decay * value
                    velocity[name].mul_(momentum).add_(gradient)
                    value.add_(velocity[name], alpha=-lr)
            steps += 1
        with cart_torch.no_grad():
            end = batched_logits(parameters, x)
            losses.append(F.cross_entropy(end.flatten(0, 1), labels.flatten(), reduction='none').reshape_as(labels).mean(dim=1))
            if record:
                online_trace.append(align_epoch(cart_torch.cat(visits, dim=1), order))
                end_trace.append(end.clone())
    return dict(parameters={name: value.detach().clone() for name, value in parameters.items()}, online_logits=cart_torch.stack(online_trace) if record else cart_torch.empty(0), end_logits=cart_torch.stack(end_trace) if record else cart_torch.empty(0), loss=cart_torch.stack(losses), steps=steps, seeds=list(seeds), width=width, n=x.shape[1])

if MAIN:
    cart_tests.test_train_ensemble(train_ensemble)

# %%

def select_indices(score: cart_Tensor, k: int, largest: bool=True) -> cart_Tensor:
    """Rank examples; ties retain canonical source order."""
    if score.ndim != 1 or not 0 < k <= score.numel() or (not cart_torch.isfinite(score).all()):
        raise ValueError('Use a finite score vector and a positive retained size at most N.')
    return cart_torch.argsort(score, descending=largest, stable=True)[:k]

if MAIN:
    cart_tests.test_select_indices(select_indices)

# %%

def retrieval_curve(score: cart_Tensor, changed: cart_Tensor, budgets: cart_Tensor) -> dict[str, cart_Tensor]:
    """Low-confidence retrieval of known injected errors, with stable ranking ties."""
    if changed.shape != score.shape or not changed.any() or changed.dtype != cart_torch.bool:
        raise ValueError('Provide an aligned boolean error mask containing at least one error.')
    if budgets.ndim != 1 or not ((budgets > 0) & (budgets <= score.numel())).all():
        raise ValueError('Retrieval budgets must lie between one and the dataset size.')
    order = select_indices(score, len(score), largest=False)
    found = changed[order].cumsum(dim=0)[budgets - 1]
    return dict(found=found, precision=found / budgets, recall=found / changed.sum())

if MAIN:
    cart_tests.test_retrieval_curve(retrieval_curve)

# %%

def selection_overlap(left: cart_Tensor, right: cart_Tensor) -> cart_Tensor:
    """Jaccard overlap of two selected index sets, not agreement of their ordering."""
    if left.numel() == 0 or right.numel() == 0:
        raise ValueError('Compare nonempty selected sets.')
    intersection = cart_torch.isin(left.unique(), right.unique()).sum()
    union = cart_torch.cat([left, right]).unique().numel()
    return intersection.float() / union

if MAIN:
    cart_tests.test_selection_overlap(selection_overlap)

# %%

if MAIN:
    cart_methods = cart_SimpleNamespace(train_ensemble=train_ensemble, training_statistics=training_statistics, select_indices=select_indices, retrieval_curve=retrieval_curve)

def map_training_data(train, methods, directory):
    seeds = list(range(161100, 161105))
    mapping = dict(ids=train['ids'], runs={}, statistics={}, windows={}, diagnosis={})
    for name, label_name, width in (('clean', 'y', 64), ('noisy', 'noisy_y', 64), ('shuffled', 'shuffled_y', 64), ('linear', 'y', 0)):
        labels = train[label_name].expand(5, -1)
        run = methods.train_ensemble(train['x'].expand(5, -1, -1), labels, seeds, width=width)
        mapping['runs'][name] = run
        mapping['statistics'][name] = {mode: methods.training_statistics(run[f'{mode}_logits'], labels) for mode in ('online', 'end')}
    clean = mapping['runs']['clean']['end_logits']
    labels = train['y'].expand(5, -1)
    mapping['windows'] = {'early': methods.training_statistics(clean[:10], labels), 'late': methods.training_statistics(clean[-10:], labels), 'repeated_final': methods.training_statistics(clean[-1:].expand_as(clean), labels)}
    budgets = cart_torch.tensor([30, 100, 300, 600, 1000])
    for replicate in range(5):
        noisy_score = mapping['statistics']['noisy']['end']['confidence'][replicate]
        clean_score = mapping['statistics']['clean']['end']['confidence'][replicate]
        generator = cart_torch.Generator().manual_seed(166100 + replicate)
        random_score = cart_torch.rand(len(train['y']), generator=generator)
        mapping['diagnosis'][str(replicate)] = {name: methods.retrieval_curve(score, train['changed'], budgets) for name, score in (('noisy', noisy_score), ('clean', clean_score), ('random', random_score))}
    mapping['diagnosis_budgets'] = budgets
    mapping['mapping_runs'] = 20
    directory.mkdir(parents=True, exist_ok=False)
    cart_torch.save(mapping, directory / 'mapping.pt')
    return mapping

if MAIN:
    native_parent = cart_Path(cart_tempfile.mkdtemp(prefix='native-', dir=cart_OUTPUT))

if MAIN:
    native_directory = native_parent / 'study'

if MAIN:
    mapping = map_training_data(cart_train, cart_methods, native_directory)

if MAIN:
    print('20 fresh training runs recorded; both logging conventions retain every example.')

# %%

def draw_map(ax, stats, title, replicate=0):
    points = ax.scatter(stats['variability'][replicate], stats['confidence'][replicate], c=stats['correctness'][replicate], s=8, cmap='viridis', vmin=0, vmax=1, alpha=0.65, rasterized=True)
    mean = cart_torch.linspace(0, 1, 301)
    ax.plot(cart_torch.sqrt(mean * (1 - mean)), mean, color='gray', lw=0.8, ls='--')
    ax.set(title=title, xlabel='Variability', ylabel='Confidence', xlim=(-0.01, 0.51), ylim=(-0.02, 1.02))
    return points

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

if MAIN:
    for cart_ax, cart_name, cart_title in zip(cart_axes.flat, ('clean', 'noisy', 'shuffled', 'linear'), ('Clean labels · MLP', '10% injected errors · MLP', 'Permuted labels · MLP', 'Clean labels · linear')):
        points = draw_map(cart_ax, mapping['statistics'][cart_name]['end'], cart_title)

if MAIN:
    cart_fig.colorbar(points, ax=cart_axes, label='Correctness fraction', shrink=0.7)

if MAIN:
    cart_show_figure(cart_fig, 'cartography_maps')

# %%

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(1, 2, figsize=(11, 4))

if MAIN:
    for cart_name, cart_label in (('clean', 'Clean MLP'), ('noisy', 'Noisy MLP'), ('shuffled', 'Permuted MLP'), ('linear', 'Clean linear')):
        loss = mapping['runs'][cart_name]['loss']
        cart_axes[0].plot(range(1, 31), loss.mean(1), label=cart_label)
        cart_axes[0].fill_between(range(1, 31), loss.min(1).values, loss.max(1).values, alpha=0.12)
        correct = (mapping['runs'][cart_name]['end_logits'].argmax(-1) == cart_train[{'clean': 'y', 'noisy': 'noisy_y', 'shuffled': 'shuffled_y', 'linear': 'y'}[cart_name]]).float().mean(-1)
        cart_axes[1].plot(range(1, 31), correct.mean(1), label=cart_label)

if MAIN:
    cart_axes[0].axhline(cart_math.log(10), color='gray', ls='--', label='Uniform probability')

if MAIN:
    cart_axes[1].axhline(0.1, color='gray', ls='--', label='Uniform guess expectation')

if MAIN:
    cart_axes[0].set(xlabel='Epoch', ylabel='Training cross entropy', ylim=(0, 2.5))

if MAIN:
    cart_axes[1].set(xlabel='Epoch', ylabel='Assigned-label training accuracy', ylim=(0, 1.03))

if MAIN:
    cart_axes[0].legend(fontsize=9)

if MAIN:
    cart_axes[1].legend(fontsize=9)

if MAIN:
    cart_fig.tight_layout()

if MAIN:
    cart_show_figure(cart_fig, 'cartography_training')

# %%

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(1, 2, figsize=(10, 4))

if MAIN:
    for cart_ax, metric in zip(cart_axes, ('confidence', 'variability')):
        online = mapping['statistics']['clean']['online'][metric][0]
        end = mapping['statistics']['clean']['end'][metric][0]
        upper = 1 if metric == 'confidence' else 0.5
        cart_ax.scatter(online, end, s=6, alpha=0.3)
        cart_ax.plot([0, upper], [0, upper], 'k--', lw=1, label='Identity')
        correlation = float(cart_torch.corrcoef(cart_torch.stack([online, end]))[0, 1])
        cart_ax.set(xlabel=f'Visited minibatch {metric}', ylabel=f'Epoch-end {metric}', title=f'Pearson r = {correlation:.3f}', xlim=(0, upper), ylim=(0, upper))
        cart_ax.legend()

if MAIN:
    cart_fig.tight_layout()

if MAIN:
    cart_show_figure(cart_fig, 'cartography_logging')

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)

if MAIN:
    for cart_ax, stats, cart_title in zip(cart_axes, [mapping['windows']['early'], mapping['statistics']['clean']['end'], mapping['windows']['late'], mapping['windows']['repeated_final']], ['Epochs 1–10', 'Epochs 1–30', 'Epochs 21–30', 'Final checkpoint repeated']):
        points = draw_map(cart_ax, stats, cart_title)

if MAIN:
    cart_fig.colorbar(points, ax=cart_axes, label='Correctness', shrink=0.8)

if MAIN:
    cart_show_figure(cart_fig, 'cartography_windows')

# %%

if MAIN:
    stats = mapping['statistics']['clean']['end']

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)

if MAIN:
    for cart_ax, metric in zip(cart_axes[:2], ('confidence', 'variability')):
        matrix = cart_torch.corrcoef(stats[metric])
        image = cart_ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
        cart_ax.set(title=f'{metric.title()} correlation', xlabel='Mapping seed index', ylabel='Mapping seed index')
        cart_ax.set_xticks(range(5))
        cart_ax.set_yticks(range(5))
        for cart_i in range(5):
            for j in range(5):
                cart_ax.text(j, cart_i, f'{matrix[cart_i, j]:.2f}', ha='center', va='center', fontsize=8)

if MAIN:
    cart_selected = [select_indices(row, 1000) for row in stats['variability']]

if MAIN:
    jaccard = cart_torch.stack([cart_torch.stack([selection_overlap(a, b) for b in cart_selected]) for a in cart_selected])

if MAIN:
    cart_axes[2].imshow(jaccard, vmin=0, vmax=1, cmap='viridis')

if MAIN:
    cart_axes[2].set(title='Top-1000 variability Jaccard', xlabel='Mapping seed index', ylabel='Mapping seed index')

if MAIN:
    cart_axes[2].set_xticks(range(5))

if MAIN:
    cart_axes[2].set_yticks(range(5))

if MAIN:
    for cart_i in range(5):
        for j in range(5):
            cart_axes[2].text(j, cart_i, f'{jaccard[cart_i, j]:.2f}', ha='center', va='center', fontsize=8)

if MAIN:
    cart_fig.colorbar(image, ax=cart_axes, shrink=0.8)

if MAIN:
    cart_show_figure(cart_fig, 'cartography_seed_stability')

# %%

if MAIN:
    alternatives = {'Visited minibatch': mapping['statistics']['clean']['online'], 'Epochs 1–10': mapping['windows']['early'], 'Epochs 21–30': mapping['windows']['late'], 'Linear learner': mapping['statistics']['linear']['end'], 'Repeated final': mapping['windows']['repeated_final'], 'Identical measurement': stats}

if MAIN:
    comparison = []

if MAIN:
    for cart_name, other in alternatives.items():
        cart_rows = []
        for cart_replicate in range(5):
            cart_values = []
            for metric in ('confidence', 'variability'):
                left, right = (stats[metric][cart_replicate], other[metric][cart_replicate])
                cart_values.append(float(cart_torch.corrcoef(cart_torch.stack([left, right]))[0, 1]) if right.std() > 0 else float('nan'))
            for metric, largest in (('confidence', True), ('confidence', False), ('variability', True)):
                left = select_indices(stats[metric][cart_replicate], 1000, largest=largest)
                right = select_indices(other[metric][cart_replicate], 1000, largest=largest)
                cart_values.append(float(selection_overlap(left, right)))
            cart_rows.append(cart_values)
        comparison.append(cart_torch.tensor(cart_rows).mean(0))

if MAIN:
    comparison = cart_torch.stack(comparison)

if MAIN:
    cart_fig, cart_ax = cart_plt.subplots(figsize=(10, 5), constrained_layout=True)

if MAIN:
    colormap = cart_plt.colormaps['viridis'].copy()

if MAIN:
    colormap.set_bad('lightgray')

if MAIN:
    image = cart_ax.imshow(comparison, vmin=0, vmax=1, cmap=colormap, aspect='auto')

if MAIN:
    cart_ax.set_yticks(range(len(alternatives)), labels=list(alternatives))

if MAIN:
    cart_ax.set_xticks(range(5), labels=['Confidence\ncorrelation', 'Variability\ncorrelation', 'Easy\nJaccard', 'Hard\nJaccard', 'Ambiguous\nJaccard'])

if MAIN:
    for cart_i in range(len(alternatives)):
        for j in range(5):
            cart_label = f'{comparison[cart_i, j]:.2f}' if cart_torch.isfinite(comparison[cart_i, j]) else 'undefined'
            cart_ax.text(j, cart_i, cart_label, ha='center', va='center', fontsize=9)

if MAIN:
    cart_ax.set_title('Agreement with full-window epoch-end MLP map · five paired seeds')

if MAIN:
    cart_fig.colorbar(image, ax=cart_ax, label='Correlation or Jaccard')

if MAIN:
    cart_show_figure(cart_fig, 'cartography_measurement_stability')

# %%

if MAIN:
    cart_gallery = {'Easy': select_indices(stats['confidence'][0], 4), 'Hard': select_indices(stats['confidence'][0], 4, largest=False), 'Ambiguous': select_indices(stats['variability'][0], 4), 'Random control': cart_torch.randperm(3000, generator=cart_torch.Generator().manual_seed(166200))[:4]}

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(4, 4, figsize=(10, 9))

if MAIN:
    for cart_row, (region, indices) in enumerate(cart_gallery.items()):
        for cart_ax, cart_index in zip(cart_axes[cart_row], indices):
            cart_ax.imshow(cart_train['x'][cart_index].reshape(8, 8), cmap='gray_r', vmin=0, vmax=1)
            cart_ax.set_title(f"Label {int(cart_train['y'][cart_index])} · μ={stats['confidence'][0, cart_index]:.2f}\nσ={stats['variability'][0, cart_index]:.2f}", fontsize=9)
            cart_ax.set_xlabel(cart_train['ids'][cart_index], fontsize=8)
            cart_ax.set_xticks([])
            cart_ax.set_yticks([])
        cart_axes[cart_row, 0].set_ylabel(region, fontsize=11)

if MAIN:
    cart_fig.tight_layout()

if MAIN:
    cart_show_figure(cart_fig, 'cartography_region_gallery')

if MAIN:
    probabilities = gold_probabilities(mapping['runs']['clean']['end_logits'], cart_train['y'].expand(5, -1))

if MAIN:
    top_probability = mapping['runs']['clean']['end_logits'].softmax(-1).max(-1).values

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)

if MAIN:
    for cart_ax, (region, indices) in zip(cart_axes.flat, cart_gallery.items()):
        for cart_index in indices:
            line, = cart_ax.plot(range(1, 31), probabilities[:, 0, cart_index], label=cart_train['ids'][cart_index])
            cart_ax.plot(range(1, 31), top_probability[:, 0, cart_index], ls=':', color=line.get_color(), alpha=0.6)
        cart_ax.set(title=region, xlabel='Epoch', ylabel='Probability', ylim=(-0.02, 1.02))
        cart_ax.legend(fontsize=8, loc='lower right')

if MAIN:
    cart_fig.suptitle('Solid: assigned label · dotted: most probable class', y=1.01)

if MAIN:
    cart_fig.tight_layout()

if MAIN:
    cart_show_figure(cart_fig, 'cartography_region_trajectories')

# %%

if MAIN:
    cart_budgets = mapping['diagnosis_budgets']

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(1, 2, figsize=(11, 4))

if MAIN:
    for method, cart_label in (('noisy', 'Noisy-run low confidence'), ('clean', 'Clean-run low confidence'), ('random', 'Random inspection')):
        for cart_ax, metric in zip(cart_axes, ('precision', 'recall')):
            cart_values = cart_torch.stack([mapping['diagnosis'][str(i)][method][metric] for i in range(5)])
            cart_ax.plot(cart_budgets, cart_values.mean(0), 'o-', label=cart_label)
            cart_ax.fill_between(cart_budgets, cart_values.min(0).values, cart_values.max(0).values, alpha=0.15)
            cart_ax.set(xlabel='Images inspected', ylabel=metric.title(), ylim=(0, 1.03))

if MAIN:
    cart_axes[0].axhline(0.1, color='gray', ls='--', label='Injected-error prevalence')

if MAIN:
    cart_axes[1].plot(cart_budgets, cart_budgets / 3000, 'k--', label='Random expected recall')

if MAIN:
    for cart_ax in cart_axes:
        cart_ax.legend(fontsize=8)

if MAIN:
    cart_fig.tight_layout()

if MAIN:
    cart_show_figure(cart_fig, 'cartography_error_retrieval')

if MAIN:
    indices = select_indices(mapping['statistics']['noisy']['end']['confidence'][0], 4, largest=False)

if MAIN:
    random = cart_torch.randperm(3000, generator=cart_torch.Generator().manual_seed(166201))[:4]

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(2, 4, figsize=(11, 5))

if MAIN:
    for cart_row, (cart_title, cart_chosen) in enumerate((('Lowest noisy confidence', indices), ('Random control', random))):
        for cart_ax, cart_index in zip(cart_axes[cart_row], cart_chosen):
            cart_ax.imshow(cart_train['x'][cart_index].reshape(8, 8), cmap='gray_r', vmin=0, vmax=1)
            cart_ax.set_title(f"Original {int(cart_train['y'][cart_index])} → assigned {int(cart_train['noisy_y'][cart_index])}\nInjected error: {bool(cart_train['changed'][cart_index])}", fontsize=9)
            cart_ax.set_xlabel(cart_train['ids'][cart_index], fontsize=8)
            cart_ax.set_xticks([])
            cart_ax.set_yticks([])
        cart_axes[cart_row, 0].set_ylabel(cart_title, fontsize=10)

if MAIN:
    cart_fig.tight_layout()

if MAIN:
    cart_show_figure(cart_fig, 'cartography_noise_gallery')

# %%

def matched_random_indices(labels: cart_Tensor, selected: cart_Tensor, seed: int) -> cart_Tensor:
    """Uniform sampling within each observed class, matching selected label counts."""
    if selected.ndim != 1 or selected.unique().numel() != selected.numel():
        raise ValueError('Selected training indices must be a vector without duplicates.')
    generator = cart_torch.Generator().manual_seed(seed)
    result = []
    for label in labels.unique(sorted=True):
        candidates = cart_torch.where(labels == label)[0]
        count = int((labels[selected] == label).sum())
        order = cart_torch.randperm(len(candidates), generator=generator)
        result.append(candidates[order[:count]])
    return cart_torch.cat(result).sort().values

if MAIN:
    cart_tests.test_matched_random_indices(matched_random_indices)

# %%

def evaluate_models(parameters: dict[str, cart_Tensor], x: cart_Tensor, labels: cart_Tensor) -> dict[str, cart_Tensor]:
    """One fixed holdout, evaluated for every model; no selection or fitting."""
    models = len(next(iter(parameters.values())))
    with cart_torch.no_grad():
        logits = batched_logits(parameters, x.expand(models, -1, -1))
        repeated_labels = labels.expand(models, -1)
        correct = logits.argmax(dim=-1) == repeated_labels
        losses = F.cross_entropy(logits.flatten(0, 1), repeated_labels.flatten(), reduction='none').reshape_as(repeated_labels)
        class_accuracy = cart_torch.stack([correct[:, labels == label].float().mean(dim=1) for label in range(logits.shape[-1])], dim=-1)
    return dict(logits=logits, correct=correct, loss_per_example=losses, accuracy=correct.float().mean(dim=1), cross_entropy=losses.mean(dim=1), class_accuracy=class_accuracy)

if MAIN:
    cart_tests.test_evaluate_models(evaluate_models)

# %%

if MAIN:
    cart_methods.matched_random_indices = matched_random_indices

if MAIN:
    cart_methods.evaluate_models = evaluate_models

def make_selections(train, mapping, methods):
    stats = mapping['statistics']['clean']['end']
    selections = {'full': cart_torch.arange(len(train['y'])).expand(5, -1).clone()}
    for size_index, size in enumerate((300, 1000, 2000)):
        for region_index, region in enumerate(('easy', 'hard', 'ambiguous')):
            score = stats['variability' if region == 'ambiguous' else 'confidence'][0]
            selected = methods.select_indices(score, size, largest=region != 'hard').sort().values
            name = f'{region}_{size}'
            selections[name] = selected.expand(5, -1).clone()
            selections[f'{name}_matched'] = cart_torch.stack([methods.matched_random_indices(train['y'], selected, 161300 + 1000 * region_index + 10 * size_index + replicate) for replicate in range(5)])
        selections[f'uniform_{size}'] = cart_torch.stack([cart_torch.randperm(len(train['y']), generator=cart_torch.Generator().manual_seed(165000 + 10 * size_index + replicate))[:size].sort().values for replicate in range(5)])
    return selections

def retrain_selections(train, mapping, methods, directory):
    selections = make_selections(train, mapping, methods)
    cart_torch.save(selections, directory / 'selections_before_refits.pt')
    fits = {}
    seeds = list(range(161200, 161205))
    for name, indices in selections.items():
        fits[name] = methods.train_ensemble(train['x'][indices], train['y'][indices], seeds, record=False)
    for source in ('ambiguous_1000', 'ambiguous_1000_matched'):
        indices = selections[source]
        name = f'{source}_equal_updates'
        fits[name] = methods.train_ensemble(train['x'][indices], train['y'][indices], seeds, epochs=90, record=False)
        cart_torch.testing.assert_close(fits[name]['loss'][:30], fits[source]['loss'], rtol=0, atol=0)
        selections[name] = indices.clone()
    result = dict(phase='ALL_SELECTIONS_AND_FRESH_TRAINING_COMPLETE', ids=train['ids'], selections=selections, fits=fits, heldout_predictions_computed=False, refits=sum((len(run['seeds']) for run in fits.values())))
    if result['refits'] != 120:
        raise ValueError('The complete selection study requires 120 fresh refits.')
    cart_torch.save(result, directory / 'prepared.pt')
    return result

if MAIN:
    cart_prepared = retrain_selections(cart_train, mapping, cart_methods, native_directory)

if MAIN:
    cart_boundary = {'phase': cart_prepared['phase'], 'heldout_predictions_computed': False, 'mapping_sha256': cart_hashlib.sha256((native_directory / 'mapping.pt').read_bytes()).hexdigest(), 'prepared_sha256': cart_hashlib.sha256((native_directory / 'prepared.pt').read_bytes()).hexdigest()}

if MAIN:
    (cart_OUTPUT / 'before_evaluation.json').write_text(cart_json.dumps(cart_boundary, indent=2) + '\n')

if MAIN:
    print('All selections and 120 fresh refits saved. Holdout predictions have not been computed.')

# %%

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)

if MAIN:
    for cart_ax, region in zip(cart_axes, ('easy', 'hard', 'ambiguous')):
        cart_counts = cart_torch.stack([cart_torch.bincount(cart_train['y'][cart_prepared['selections'][f'{region}_{n}'][0]], minlength=10) for n in (300, 1000, 2000)])
        image = cart_ax.imshow(cart_counts, vmin=0, vmax=300, cmap='Blues', aspect='auto')
        cart_ax.set(title=region.title(), xlabel='Observed digit class', ylabel='Retained images')
        cart_ax.set_xticks(range(10))
        cart_ax.set_yticks(range(3), labels=[300, 1000, 2000])
        for cart_i in range(3):
            for j in range(10):
                cart_ax.text(j, cart_i, int(cart_counts[cart_i, j]), ha='center', va='center', fontsize=7, color='white' if cart_counts[cart_i, j] > 170 else 'black')

if MAIN:
    cart_fig.colorbar(image, ax=cart_axes, label='Examples in class', shrink=0.8)

if MAIN:
    cart_show_figure(cart_fig, 'cartography_class_counts')

# %%

def cart_evaluate_study(mapping, prepared, evaluation, methods, directory):
    if prepared['phase'] != 'ALL_SELECTIONS_AND_FRESH_TRAINING_COMPLETE' or prepared['heldout_predictions_computed']:
        raise ValueError('Save every selection and fresh training run before evaluating holdouts.')
    output = dict(holdout_ids={name: split['ids'] for name, split in evaluation.items()}, mapping={}, refits={})
    for category, runs in (('mapping', mapping['runs']), ('refits', prepared['fits'])):
        for name, run in runs.items():
            output[category][name] = {split_name: methods.evaluate_models(run['parameters'], split['x'], split['y']) for split_name, split in evaluation.items()}
    output['evaluated_models'] = 140
    output['evaluation_used_for_selection'] = False
    cart_torch.save(output, directory / 'evaluated.pt')
    return output

if MAIN:
    evaluation = load_evaluation()

if MAIN:
    cart_evaluated = cart_evaluate_study(mapping, cart_prepared, evaluation, cart_methods, native_directory)

if MAIN:
    print('Evaluated all 140 models on 823 same-source and 1797 separate-writer images.')

# %%

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)

if MAIN:
    conditions = ('clean', 'noisy', 'shuffled', 'linear')

if MAIN:
    for cart_ax, cart_split in zip(cart_axes, ('same_source', 'separate_writers')):
        for cart_index, cart_name in enumerate(conditions):
            cart_values = cart_evaluated['mapping'][cart_name][cart_split]['accuracy']
            cart_ax.bar(cart_index, cart_values.mean(), color=f'C{cart_index}', alpha=0.55)
            cart_ax.scatter(cart_index + cart_np.linspace(-0.1, 0.1, 5), cart_values, color='black', s=18)
            cart_ax.text(cart_index, float(cart_values.max()) + 0.025, f'{cart_values.mean():.1%}', ha='center', fontsize=9)
        cart_ax.axhline(0.1, color='gray', ls='--', label='Uniform guess expectation')
        cart_ax.set_xticks(range(4), labels=['Clean MLP', 'Noisy MLP', 'Permuted MLP', 'Clean linear'], fontsize=9)
        cart_ax.set(title=cart_split.replace('_', ' ').title(), ylabel='Original-label held-out accuracy', ylim=(0, 1.05))
        cart_ax.legend(fontsize=8, loc='center right')

if MAIN:
    cart_show_figure(cart_fig, 'cartography_mapping_generalization')

# %%

if MAIN:
    cart_sizes = (300, 1000, 2000)

if MAIN:
    for metric, ylabel, filename in (('accuracy', 'Held-out accuracy', 'cartography_selection_accuracy'), ('cross_entropy', 'Held-out cross entropy', 'cartography_selection_loss')):
        cart_fig, cart_axes = cart_plt.subplots(2, 3, figsize=(13, 7), sharey=True, constrained_layout=True)
        for cart_row, cart_split in enumerate(('same_source', 'separate_writers')):
            for cart_col, region in enumerate(('easy', 'hard', 'ambiguous')):
                cart_ax = cart_axes[cart_row, cart_col]
                for kind, cart_label, style in ((region, 'Selected region', 'o-'), (region + '_matched', 'Class-matched random', 's--'), ('uniform', 'Uniform random', '^:')):
                    keys = [f'{region}_{n}_matched' if kind.endswith('_matched') else f'{kind}_{n}' for n in cart_sizes]
                    cart_values = cart_torch.stack([cart_evaluated['refits'][key][cart_split][metric] for key in keys])
                    line, = cart_ax.plot(cart_sizes, cart_values.mean(1), style, label=cart_label)
                    cart_ax.fill_between(cart_sizes, cart_values.min(1).values, cart_values.max(1).values, alpha=0.13, color=line.get_color())
                full = cart_evaluated['refits']['full'][cart_split][metric]
                cart_ax.axhline(float(full.mean()), color='black', ls='--', lw=1, label='Full pool')
                cart_ax.axhspan(float(full.min()), float(full.max()), color='gray', alpha=0.12)
                cart_ax.set(title=f"{region.title()} · {cart_split.replace('_', ' ')}", xlabel='Retained training images', ylabel=ylabel)
                cart_ax.set_xticks(cart_sizes)
                if metric == 'accuracy':
                    cart_ax.set_ylim(0.45, 1.01)
                else:
                    cart_ax.set_ylim(bottom=0)
                if cart_row == 0 and cart_col == 0:
                    cart_ax.legend(fontsize=8)
        cart_show_figure(cart_fig, filename)

# %%

if MAIN:
    keys = ['ambiguous_1000', 'ambiguous_1000_matched', 'ambiguous_1000_equal_updates', 'ambiguous_1000_matched_equal_updates', 'full']

if MAIN:
    labels = ['Ambiguous\n240 steps', 'Matched random\n240 steps', 'Ambiguous\n720 steps', 'Matched random\n720 steps', 'Full pool\n720 steps']

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

if MAIN:
    for cart_ax, metric, ylabel in zip(cart_axes, ('accuracy', 'cross_entropy'), ('Separate-writer accuracy', 'Separate-writer cross entropy')):
        cart_values = cart_torch.stack([cart_evaluated['refits'][key]['separate_writers'][metric] for key in keys])
        cart_ax.bar(range(5), cart_values.mean(1), color=['C0', 'C1', 'C0', 'C1', 'gray'], alpha=0.65)
        for seed in range(5):
            cart_ax.scatter(cart_np.arange(5) + (seed - 2) * 0.04, cart_values[:, seed], color='black', s=15)
        cart_ax.set_xticks(range(5), labels=labels, fontsize=8)
        cart_ax.set_ylabel(ylabel)
        if metric == 'accuracy':
            cart_ax.set_ylim(0, 1.02)
        else:
            cart_ax.set_ylim(bottom=0)

if MAIN:
    cart_show_figure(cart_fig, 'cartography_equal_updates')

# %%

if MAIN:
    keys = ['full', 'easy_1000', 'easy_1000_matched', 'hard_1000', 'hard_1000_matched', 'ambiguous_1000', 'ambiguous_1000_matched', 'uniform_1000']

if MAIN:
    cart_values = cart_torch.stack([cart_evaluated['refits'][key]['separate_writers']['class_accuracy'].mean(0) for key in keys])

if MAIN:
    cart_fig, cart_ax = cart_plt.subplots(figsize=(11, 5.5), constrained_layout=True)

if MAIN:
    image = cart_ax.imshow(cart_values, cmap='viridis', vmin=0, vmax=1, aspect='auto')

if MAIN:
    cart_ax.set_xticks(range(10))

if MAIN:
    cart_ax.set_yticks(range(len(keys)), labels=[k.replace('_1000', '').replace('_', ' ') for k in keys])

if MAIN:
    cart_ax.set(xlabel='Digit class', title='Separate-writer accuracy by class · five-run mean')

if MAIN:
    for cart_i in range(len(keys)):
        for j in range(10):
            cart_ax.text(j, cart_i, f'{cart_values[cart_i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if cart_values[cart_i, j] < 0.6 else 'black')

if MAIN:
    cart_fig.colorbar(image, ax=cart_ax, label='Accuracy')

if MAIN:
    cart_show_figure(cart_fig, 'cartography_class_accuracy')

if MAIN:
    cart_fig, cart_ax = cart_plt.subplots(figsize=(9, 4), constrained_layout=True)

if MAIN:
    full_correct = cart_evaluated['refits']['full']['separate_writers']['correct']

if MAIN:
    for j, cart_name in enumerate(('easy', 'hard', 'ambiguous', 'uniform')):
        correct = cart_evaluated['refits'][f'{cart_name}_1000']['separate_writers']['correct']
        gains = (correct & ~full_correct).sum(1)
        losses = (~correct & full_correct).sum(1)
        cart_ax.scatter(cart_torch.full((5,), j - 0.1), gains, color='C0', alpha=0.7, label='Gained versus full' if j == 0 else None)
        cart_ax.scatter(cart_torch.full((5,), j + 0.1), -losses, color='C3', alpha=0.7, label='Lost versus full' if j == 0 else None)

if MAIN:
    cart_ax.axhline(0, color='gray', lw=1)

if MAIN:
    cart_ax.set_xticks(range(4), labels=['Easy', 'Hard', 'Ambiguous', 'Uniform random'])

if MAIN:
    cart_ax.set(ylabel='Held-out examples gained (+) or lost (−)', title='One-third selections · each point is a paired seed')

if MAIN:
    cart_ax.legend()

if MAIN:
    cart_show_figure(cart_fig, 'cartography_paired_changes')

# %%

def retraining_play(train, mapping, evaluation, methods, region='ambiguous', size=1000, seed=166000, epochs=30):
    if region not in ('easy', 'hard', 'ambiguous'):
        raise ValueError('Choose easy, hard or ambiguous.')
    stats = mapping['statistics']['clean']['end']
    score = stats['variability' if region == 'ambiguous' else 'confidence'][0]
    selected = methods.select_indices(score, size, largest=region != 'hard').sort().values
    random = methods.matched_random_indices(train['y'], selected, seed + 1)
    indices = cart_torch.stack([selected, random])
    run = methods.train_ensemble(train['x'][indices], train['y'][indices], [seed, seed], epochs=epochs, record=False)
    return dict(region=region, indices=indices, run=run, evaluation={name: methods.evaluate_models(run['parameters'], split['x'], split['y']) for name, split in evaluation.items()}, exploratory_holdout_reuse=True)

if MAIN:
    region = 'ambiguous'

if MAIN:
    retained = 1000

if MAIN:
    seed = 166000

if MAIN:
    epochs = 30

if MAIN:
    cart_play = retraining_play(cart_train, mapping, evaluation, cart_methods, region, retained, seed, epochs)

if MAIN:
    cart_torch.save(cart_play, native_directory / 'play.pt')

if MAIN:
    cart_fig, cart_axes = cart_plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

if MAIN:
    for cart_index, cart_label in enumerate(('Selected', 'Class-matched random')):
        cart_axes[0].plot(range(1, epochs + 1), cart_play['run']['loss'][:, cart_index], label=cart_label)
        cart_counts = cart_torch.bincount(cart_train['y'][cart_play['indices'][cart_index]], minlength=10)
        cart_axes[1].plot(range(10), cart_counts, 'o-' if cart_index == 0 else 'x--', label=cart_label)
        results = [float(cart_play['evaluation'][name]['accuracy'][cart_index]) for name in ('same_source', 'separate_writers')]
        cart_axes[2].bar(cart_np.arange(2) + (cart_index - 0.5) * 0.32, results, width=0.32, label=cart_label)

if MAIN:
    cart_axes[0].set(xlabel='Epoch', ylabel='Training cross entropy')

if MAIN:
    cart_axes[1].set(xlabel='Digit class', ylabel='Retained images')

if MAIN:
    cart_axes[1].set_xticks(range(10))

if MAIN:
    cart_axes[2].set_xticks(range(2), labels=['Same source', 'Separate writers'])

if MAIN:
    cart_axes[2].set(ylabel='Held-out accuracy', ylim=(0, 1.02))

if MAIN:
    for cart_ax in cart_axes:
        cart_ax.legend(fontsize=8)

if MAIN:
    cart_show_figure(cart_fig, 'cartography_real_play')

# %%

if MAIN:
    checks = cart_reference_checks(cart_train, mapping, cart_methods)

if MAIN:
    (cart_OUTPUT / 'reference_checks.json').write_text(cart_json.dumps(checks, indent=2) + '\n')

if MAIN:
    cart_summary = summarize(mapping, cart_prepared, cart_evaluated, cart_play, native_directory)

if MAIN:
    (cart_OUTPUT / 'native_summary.json').write_text(cart_json.dumps(cart_summary, indent=2) + '\n')

if MAIN:
    print('Separate full training and independent NumPy coordinate checks passed.')

if MAIN:
    print(f"Frozen experiment: {cart_summary['main_training_runs']} real training runs, {cart_summary['main_optimizer_updates']} optimizer updates, plus two separate checks and two play fits.")

# %%
