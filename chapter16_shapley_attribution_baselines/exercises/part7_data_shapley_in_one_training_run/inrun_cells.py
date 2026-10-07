"""Visible demonstration cells for the extended 16.7 notebook."""

CELLS = {}
CELLS['setup'] = r'''
# Retain the exact-retraining foundation before the digit study reuses data names.
foundation_evidence = dict(
    coalition_values=[{'coalition': sorted(k), 'utility': coalition_values[k]}
                      for k in sorted(coalition_values, key=lambda k: sum(1 << i for i in k))],
    exact=exact, one_run=one_run, sampled_256=sampled_256, ascending=ascending,
    descending=descending, budget_sweep=budget_sweep, leave_one_out=leave_one_out,
    influence=influence, shuffled_labels=shuffled_labels, shuffled_exact=shuffled_exact,
    harmful_removal=harmful_removal, matched_removal=matched_removal,
    play_exact=try_exact, play_one_run=try_one_run, play_sampled=try_sampled,
)
import os
import time
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from chapter16_shapley_attribution_baselines.exercises.part7_data_shapley_in_one_training_run import inrun_tests
from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_data import load_study_data, data_manifest

torch.set_num_threads(1)
INRUN_OUTPUT = Path(os.environ.get('ARENA_INRUN_OUTPUT', '/tmp/arena-inrun-play'))
INRUN_OUTPUT.mkdir(parents=True, exist_ok=True)

def show_inrun(figure, name):
    figure.savefig(INRUN_OUTPUT / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.show()

# Initialization is supplied; all forward, gradient, attribution and update code follows below.
'''

CELLS['ghost_picture'] = r'''
demo_params = init_mlp(3, 2, 41)
demo_x = torch.tensor([[1.,0.,1.], [1.,0.,1.], [-1.,1.,0.], [0.,2.,1.]], dtype=torch.float64)
demo_y = torch.tensor([1.,1.,0.,1.], dtype=torch.float64)
demo_factors = mlp_factors(demo_params, demo_x, demo_y)
direct = per_example_gradients(demo_factors)
ghost = ghost_dot_products(demo_factors, demo_factors)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
limit = float(ghost.abs().max())
im = axes[0].imshow(ghost, cmap='RdBu_r', vmin=-limit, vmax=limit)
axes[0].set(title='Gradient similarities, including biases', xlabel='Example', ylabel='Example')
fig.colorbar(im, ax=axes[0], label='Gradient dot product')
error = (ghost - direct @ direct.T).abs()
axes[1].imshow(error, cmap='Greys', vmin=0, vmax=1e-12)
axes[1].set(title=f'Ghost minus explicit: max {error.max():.1e}', xlabel='Example', ylabel='Example')
show_inrun(fig, '167_05_ghost_products')
print('The duplicate examples have identical rows and columns; opposite labels can give negative alignment.')
'''

CELLS['local_play'] = r'''
# Try changing this list. Large steps can break both Taylor approximations.
TRY_STEP_SIZES = (0.005, 0.02, 0.1, 0.5, 1.0)
flat = torch.cat([p.flatten() for p in demo_params])
def local_validation_loss(vector):
    hidden = torch.tanh(demo_x @ vector[:6].reshape(3, 2) + vector[6:8])
    logits = hidden @ vector[8:10] + vector[10]
    return F.binary_cross_entropy_with_logits(logits, demo_y)
validation_gradient = torch.func.grad(local_validation_loss)(flat)
validation_hessian = torch.func.hessian(local_validation_loss)(flat)
local_results = []
for eta in TRY_STEP_SIZES:
    values = local_coalition_values(demo_params, demo_x, demo_y, demo_x, demo_y, eta)
    exact_local = batch_shapley(values)
    first = first_order_scores(ghost, eta, len(demo_x))
    second = second_order_scores(direct, validation_gradient, validation_hessian, eta, len(demo_x))
    local_results.append(dict(eta=eta, exact=exact_local, first=first, second=second, values=values))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for key, label in [('first', 'First order'), ('second', 'Second order')]:
    axes[0].loglog(TRY_STEP_SIZES, [float((r[key]-r['exact']).abs().max()) for r in local_results], 'o-', label=label)
    axes[1].semilogx(TRY_STEP_SIZES, [float(r[key].sum()-r['values'][-1]) for r in local_results], 'o-', label=label)
axes[0].set(xlabel='SGD learning rate', ylabel='Maximum local credit error', title='Exact local Shapley is the oracle')
axes[1].axhline(0, color='black', lw=1, label='Exact allocation')
axes[1].set(xlabel='SGD learning rate', ylabel='Allocated minus actual improvement', title='Approximate credits need not balance')
for ax in axes: ax.legend()
show_inrun(fig, '167_06_local_taylor')
'''

CELLS['digits'] = r'''
digits = load_study_data()
train_x = digits['train']['x'][:, :64]
train_y = digits['train']['corrupted_y'].double()
ranking_x = digits['ranking']['x'][:, :64]
ranking_y = digits['ranking']['y'].double()
flipped = digits['train']['flipped']
STUDY_SEEDS = (16701, 16702, 16703)
N_EPOCHS, BATCH_SIZE, LEARNING_RATE = 4, 4, 0.1
ranking_shuffle = torch.randperm(len(ranking_y), generator=torch.Generator().manual_seed(16730))
digit_runs, deletion_choices = {}, {}
for seed_index, seed in enumerate(STUDY_SEEDS):
    initial = init_mlp(64, 16, seed)
    schedule = make_schedule(len(train_x), BATCH_SIZE, N_EPOCHS, seed)
    started = time.perf_counter()
    run = run_accountant(initial, train_x, train_y, ranking_x, ranking_y, schedule, LEARNING_RATE, exact=(seed_index == 0))
    run['wall_seconds'] = time.perf_counter() - started
    shuffled = run_accountant(initial, train_x, train_y, ranking_x, ranking_y[ranking_shuffle], schedule, LEARNING_RATE)
    for real_param, shuffled_param in zip(run['params'], shuffled['params']):
        torch.testing.assert_close(real_param, shuffled_param, atol=0, rtol=0)
    order = run['scores'].argsort(stable=True)
    choices = dict(lowest=order[:12], highest=order[-12:], known_corruption=flipped.nonzero().flatten(),
                   shuffled=shuffled['scores'].argsort(stable=True)[:12])
    # Random deletions match the lowest-score set's observed-label counts.
    for replicate in range(8):
        generator = torch.Generator().manual_seed(16750 + 100 * seed_index + replicate)
        chosen = []
        for label in (0., 1.):
            candidates = (train_y == label).nonzero().flatten()
            count = int((train_y[choices['lowest']] == label).sum())
            chosen.extend(candidates[torch.randperm(len(candidates), generator=generator)[:count]].tolist())
        choices[f'random_{replicate}'] = torch.tensor(sorted(chosen))
    digit_runs[seed] = dict(run=run, shuffled=shuffled, initial=initial, schedule=schedule)
    deletion_choices[seed] = choices

# No final test score is used above. Only now compare the fixed interventions.
final_x, final_y = digits['final']['x'][:, :64], digits['final']['y'].double()
deletion_results = {}
for seed in STUDY_SEEDS:
    record = digit_runs[seed]
    full_loss = float(F.binary_cross_entropy_with_logits(mlp_forward(record['run']['params'], final_x)[0], final_y))
    results = {'full_loss': full_loss, 'conditions': {}}
    for name, removed in deletion_choices[seed].items():
        retrained = run_accountant(record['initial'], train_x, train_y, ranking_x, ranking_y,
                                  record['schedule'], LEARNING_RATE, removed=tuple(removed.tolist()))
        loss = float(F.binary_cross_entropy_with_logits(mlp_forward(retrained['params'], final_x)[0], final_y))
        results['conditions'][name] = dict(removed=removed, final_loss=loss, improvement=full_loss-loss, run=retrained)
    deletion_results[seed] = results

primary = digit_runs[STUDY_SEEDS[0]]['run']
for checkpoint in (digit_runs[STUDY_SEEDS[0]]['initial'], primary['params']):
    inrun_tests.check_native_gradient_identity(checkpoint, train_x, train_y, mlp_factors, ghost_dot_products)
torch.testing.assert_close(primary['exact_allocations'].sum(1), primary['losses'][:-1]-primary['losses'][1:], atol=1e-12, rtol=1e-10)
print(f'{len(train_x)} training images; {len(ranking_x)} ranking images; {len(final_x)} untouched test images.')
print(f'{len(primary["allocations"])} SGD steps per run; all three seeds and all deletion controls retained.')
print(f'Exact total credit {primary["exact_scores"].sum():.6f}; measured ranking-loss improvement {primary["losses"][0]-primary["losses"][-1]:.6f}.')
'''

CELLS['trajectory_plot'] = r'''
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
actual = primary['losses'][0] - primary['losses'][1:]
axes[0].plot(actual, color='black', label='Actual ranking-loss improvement')
axes[0].plot(primary['exact_allocations'].sum(1).cumsum(0), '--', label='Exact local Shapley sum')
axes[0].plot(primary['allocations'].sum(1).cumsum(0), label='First-order sum')
axes[0].set(xlabel='SGD step', ylabel='Cumulative loss improvement', title='An exact accountant, an approximate shortcut')
axes[0].legend()
axes[1].scatter(primary['exact_scores'][~flipped], primary['scores'][~flipped], s=22, label='Unchanged labels')
axes[1].scatter(primary['exact_scores'][flipped], primary['scores'][flipped], s=38, marker='x', color='#bb4433', label='Known label flips')
lo = float(torch.minimum(primary['scores'].min(), primary['exact_scores'].min()))
hi = float(torch.maximum(primary['scores'].max(), primary['exact_scores'].max()))
axes[1].plot([lo, hi], [lo, hi], 'k--', lw=1, label='Exact agreement')
axes[1].set(xlabel='Exact trajectory credit', ylabel='First-order trajectory credit', title='Every training image is shown')
axes[1].legend()
show_inrun(fig, '167_07_training_accounting')

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
limit = float(primary['allocations'].abs().max())
im = axes[0].imshow(primary['allocations'].T, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
axes[0].set(xlabel='SGD step', ylabel='Training row', title='When did each image contribute?')
fig.colorbar(im, ax=axes[0], label='First-order loss-improvement credit')
for seed in STUDY_SEEDS:
    r = digit_runs[seed]
    axes[1].scatter(r['run']['scores'], r['shuffled']['scores'], s=10, alpha=.45, label=f'Seed {seed}')
axes[1].axhline(0, color='gray', lw=.7); axes[1].axvline(0, color='gray', lw=.7)
axes[1].set(xlabel='Credit for true ranking labels', ylabel='Credit for shuffled ranking labels', title='Same training trajectory, different question')
axes[1].legend()
show_inrun(fig, '167_08_credit_over_time')
'''

CELLS['deletion_plot'] = r'''
order = primary['scores'].argsort(stable=True)
shown = torch.cat([order[:6], order[-6:]])
fig, axes = plt.subplots(2, 6, figsize=(12, 4.6), constrained_layout=True)
for ax, index in zip(axes.flat, shown.tolist()):
    ax.imshow(digits['train']['images'][index], cmap='gray_r', vmin=0, vmax=1)
    ax.set_title(f'row {index}: {primary["scores"][index]:+.3f}\nlabel {int(train_y[index])}; digit {int(digits["train"]["digit"][index])}', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('#bb4433' if flipped[index] else '#cccccc')
fig.suptitle('Lowest and highest trajectory credits; red border marks a known label flip')
show_inrun(fig, '167_09_digit_examples')

names = ('lowest', 'highest', 'shuffled', 'known_corruption')
labels = ('Lowest credit', 'Highest credit', 'Shuffled target', 'Known flips', 'Matched random')
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
for i, name in enumerate(names):
    values = [deletion_results[seed]['conditions'][name]['improvement'] for seed in STUDY_SEEDS]
    ax.scatter([i] * len(values), values, s=32, zorder=3)
    ax.plot([i-.16, i+.16], [np.mean(values)]*2, color='black', lw=2)
random_gains = [r['improvement'] for row in deletion_results.values() for name, r in row['conditions'].items() if name.startswith('random_')]
ax.scatter(np.linspace(3.83, 4.17, len(random_gains)), random_gains, s=18, alpha=.65, color='#777777')
ax.plot([3.84, 4.16], [np.mean(random_gains)]*2, color='black', lw=2)
ax.axhline(0, color='black', lw=1)
ax.set(xticks=range(5), xticklabels=labels, ylabel='Untouched test loss: full minus retrained',
       title='Delete twelve images, retrain the entire fixed schedule')
show_inrun(fig, '167_10_deletion_controls')
print('Positive points mean deletion helped. These are measured interventions, not guarantees from the attribution sign.')
'''

CELLS['timing'] = r'''
# A small CPU microbenchmark, not the paper's foundation-model throughput study.
timing_rows = []
timing_params = init_mlp(64, 16, 16701)
timing_train = mlp_factors(timing_params, train_x[:32], train_y[:32])
timing_val = mlp_factors(timing_params, ranking_x, ranking_y)
for mode in ('explicit', 'ghost'):
    for repetition in range(7):
        started = time.perf_counter()
        for _ in range(100):
            if mode == 'explicit':
                observed = per_example_gradients(timing_train) @ per_example_gradients(timing_val).T
            else:
                observed = ghost_dot_products(timing_train, timing_val)
        timing_rows.append(dict(mode=mode, repetition=repetition, seconds=(time.perf_counter()-started)/100))
torch.testing.assert_close(ghost_dot_products(timing_train, timing_val),
                           per_example_gradients(timing_train) @ per_example_gradients(timing_val).T,
                           atol=1e-12, rtol=1e-10)
training_times = []
for repetition in range(3):
    initial = digit_runs[STUDY_SEEDS[0]]['initial']
    schedule = digit_runs[STUDY_SEEDS[0]]['schedule']
    started = time.perf_counter()
    plain_params = tuple(p.clone() for p in initial)
    for batch in schedule:
        plain_params = sgd_step(plain_params, mlp_factors(plain_params, train_x[batch], train_y[batch]), LEARNING_RATE, len(batch))
    training_times.append(dict(mode='SGD only', repetition=repetition, seconds=time.perf_counter()-started, params=plain_params))
    started = time.perf_counter()
    accounted = run_accountant(initial, train_x, train_y, ranking_x, ranking_y, schedule, LEARNING_RATE)
    training_times.append(dict(mode='SGD + accounting', repetition=repetition, seconds=time.perf_counter()-started, run=accounted))
    for plain, attributed in zip(plain_params, accounted['params']):
        torch.testing.assert_close(plain, attributed, atol=0, rtol=0)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
ax = axes[0]
for i, mode in enumerate(('explicit', 'ghost')):
    values = [r['seconds']*1000 for r in timing_rows if r['mode'] == mode]
    ax.scatter(np.linspace(i-.1, i+.1, len(values)), values, label=mode)
    ax.plot([i-.16,i+.16], [np.median(values)]*2, color='black', lw=2)
ax.set(xticks=[0,1], xticklabels=['Explicit parameter gradients', 'Ghost factors'],
       ylabel='Milliseconds per similarity matrix', title='Measure the cost on this machine')
for i, mode in enumerate(('SGD only', 'SGD + accounting')):
    values = [r['seconds'] for r in training_times if r['mode'] == mode]
    axes[1].scatter(np.linspace(i-.1, i+.1, len(values)), values)
    axes[1].plot([i-.16,i+.16], [np.median(values)]*2, color='black', lw=2)
axes[1].set(xticks=[0,1], xticklabels=['SGD only', 'SGD + accounting'], ylabel='Seconds for 128 steps',
            title='Actual full training overhead')
show_inrun(fig, '167_11_measured_cost')
'''

CELLS['native_play'] = r'''
# Try It Yourself: change the removed row, then explain the whole retrained path.
TRY_REMOVED_ROW = 0
record = digit_runs[STUDY_SEEDS[0]]
play_run = run_accountant(record['initial'], train_x, train_y, ranking_x, ranking_y,
                         record['schedule'], LEARNING_RATE, removed=(TRY_REMOVED_ROW,))
fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
ax.plot(record['run']['losses'], label='Original training')
ax.plot(play_run['losses'], label=f'Retrain without row {TRY_REMOVED_ROW}')
ax.set(xlabel='SGD step', ylabel='Ranking-set cross entropy', title='Deletion changes later gradients too')
ax.legend()
show_inrun(fig, '167_12_retraining_play')
print(f'Fixed-trajectory first-order credit: {record["run"]["scores"][TRY_REMOVED_ROW]:+.6f}')
print(f'Actual ranking-loss increase after retraining: {play_run["losses"][-1]-record["run"]["losses"][-1]:+.6f}')
'''
