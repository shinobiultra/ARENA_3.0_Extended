# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

chapter = 'chapter15_peft_misalignment'
section = 'part4_lora_phase_transitions'
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).is_dir())
exercises_dir = root_dir / chapter / 'exercises'
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part4_lora_phase_transitions.tests as tests
import part4_lora_phase_transitions.utils as utils
MAIN = __name__ == '__main__'
np.set_printoptions(precision=4, suppress=True)

# %%

def lora_update(A, B, alpha):
    r = A.shape[0]
    return (alpha / r) * (B @ A)

if MAIN:
    tests.test_lora_update(lora_update)

# %%

def planted_update(time):
    return np.diag([time, 2.4 * max(time - 0.5, 0.0), 0.0, 0.0])

if MAIN:
    tests.test_planted_update(planted_update)

# %%

def rank_projection(update, rank):
    U, singular, Vh = np.linalg.svd(update, full_matrices=False)
    return (U[:, :rank] * singular[:rank]) @ Vh[:rank]

if MAIN:
    tests.test_rank_projection(rank_projection)

# %%

if MAIN:
    times = np.linspace(0., 1., 9)
    updates = np.stack([planted_update(time) for time in times])
    print('time  narrow coefficient  proxy coefficient  rank')
    for time, update in zip(times, updates):
        print(f'{time:4.3f}       {update[0,0]:5.3f}             {update[1,1]:5.3f}       {np.linalg.matrix_rank(update)}')

# %%

def spectral_energy(update):
    squared = np.linalg.svd(update, compute_uv=False) ** 2
    return squared / squared.sum() if squared.sum() > 0 else np.zeros_like(squared)

if MAIN:
    tests.test_spectral_energy(spectral_energy)

# %%

def matched_random_update(update, rank, seed):
    if np.linalg.norm(update) == 0:
        return np.zeros_like(update)
    rng = np.random.default_rng(seed)
    left = np.linalg.qr(rng.normal(size=(update.shape[0], rank)))[0]
    right = np.linalg.qr(rng.normal(size=(update.shape[1], rank)))[0]
    random = left @ right.T
    return random * (np.linalg.norm(update) / np.linalg.norm(random))

if MAIN:
    tests.test_matched_random_update(matched_random_update)

# %%

def proxy_margins(inputs, weight, direction, thresholds):
    return (inputs @ weight.T) @ direction - thresholds

if MAIN:
    tests.test_proxy_margins(proxy_margins)

# %%

def flip_rate(base_margins, adapted_margins):
    return np.mean((base_margins > 0) != (adapted_margins > 0))

if MAIN:
    tests.test_flip_rate(flip_rate)

# %%

def first_crossing(times, values, threshold):
    indices = np.flatnonzero(np.asarray(values) >= threshold)
    return float(times[indices[0]]) if len(indices) else None

if MAIN:
    tests.test_first_crossing(first_crossing)

# %%

if MAIN:
    base = np.zeros((4, 4))
    proxy_inputs = np.tile(np.eye(4)[1], (24, 1))
    thresholds = np.linspace(.1, 1.1, 24)
    proxy_direction = np.eye(4)[1]
    base_margins = proxy_margins(proxy_inputs, base, proxy_direction, thresholds)
    flips = np.array([flip_rate(base_margins, proxy_margins(proxy_inputs, base + update, proxy_direction, thresholds)) for update in updates])
    print('Proxy flips:', flips)
    print('First sampled 50% crossing:', first_crossing(times, flips, .5))
    print('Final spectral energy:', spectral_energy(updates[-1]))

# %%

def top_output_alignment(update, direction):
    if np.linalg.norm(update) == 0:
        return 0.0
    U, _, _ = np.linalg.svd(update, full_matrices=False)
    return float(abs(U[:, 0] @ (direction / np.linalg.norm(direction))))

if MAIN:
    tests.test_top_output_alignment(top_output_alignment)

# %%

def remove_output_direction(update, direction):
    unit = direction / np.linalg.norm(direction)
    return update - np.outer(unit, unit @ update)

if MAIN:
    tests.test_remove_output_direction(remove_output_direction)

# %%

def remove_dominant_mode(update):
    U, singular, Vh = np.linalg.svd(update, full_matrices=False)
    return update - singular[0] * np.outer(U[:, 0], Vh[0])

if MAIN:
    tests.test_remove_dominant_mode(remove_dominant_mode)

# %%

def task_rmse(inputs, weight, direction, targets):
    predictions = (inputs @ weight.T) @ direction
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))

if MAIN:
    tests.test_task_rmse(task_rmse)

# %%

if MAIN:
    narrow_inputs = np.tile(np.eye(4)[0], (8, 1))
    narrow_targets = np.ones(8)
    narrow_direction = np.eye(4)[0]
    conditions = {
        'adapter': list(updates),
        'targeted removal': [remove_output_direction(d, proxy_direction) for d in updates],
        'dominant removal': [remove_dominant_mode(d) for d in updates],
        'rank-1 truncation': [rank_projection(d, 1) for d in updates],
        'same-rank/norm random': [matched_random_update(d, int(np.linalg.matrix_rank(d)), 17) for d in updates],
        'zero adapter': [np.zeros_like(base) for _ in times],
    }
    curves = {name: np.array([flip_rate(base_margins, proxy_margins(proxy_inputs, base + d, proxy_direction, thresholds)) for d in ds]) for name, ds in conditions.items()}
    errors = {name: np.array([task_rmse(narrow_inputs, base + d, narrow_direction, narrow_targets) for d in ds]) for name, ds in conditions.items()}
    spectra = np.stack([np.linalg.svd(d, compute_uv=False) for d in updates])
    alignments = np.array([top_output_alignment(d, proxy_direction) for d in updates])
    print('50% proxy crossing:', first_crossing(times, curves['adapter'], .5))
    print('Top-mode proxy alignment:', alignments)
    for name in conditions:
        print(f'{name:25s} final flips={curves[name][-1]:.4f}; task RMSE={errors[name][-1]:.4f}')
    utils.signature(times, spectra, curves, errors, alignments)
    plt.show()

# %%

if MAIN:
    threshold_shift = 0.3  # editable context change; 0 recovers the worked example
    changed_thresholds = thresholds + threshold_shift
    changed_base = proxy_margins(proxy_inputs, base, proxy_direction, changed_thresholds)
    changed_flips = np.array([flip_rate(changed_base, proxy_margins(proxy_inputs, base + d, proxy_direction, changed_thresholds)) for d in updates])
    print('Changed 50% crossing:', first_crossing(times, changed_flips, .5))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, curves['adapter'], 'o-', label='original contexts')
    ax.plot(times, changed_flips, 's-', label=f'threshold shift {threshold_shift:+.1f}')
    ax.plot(times, curves['zero adapter'], ':', label='zero adapter')
    ax.axhline(.5, color='grey', linestyle='--', label='50% detector')
    ax.set(xlabel='Planted checkpoint time', ylabel='Proxy label flip fraction', title='A detector can move with unchanged weights')
    ax.legend()
    plt.show()

# %%
