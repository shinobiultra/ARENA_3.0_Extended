# %%


from pathlib import Path
import sys
import itertools
import numpy as np
import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
chapter = "chapter14_jepa_world_models"
section = "part3_action_conditioned_planning"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part3_action_conditioned_planning.tests as tests
MAIN = __name__ == "__main__"
device = t.device("cpu")
np.set_printoptions(precision=4, suppress=True)

# %%

def world_step(state, action, gain=1.0):
    state = np.asarray(state, dtype=float)
    action = np.asarray(action, dtype=float)
    x, v, c = np.moveaxis(state, -1, 0)
    return np.stack([x + v + 0.5 * gain * action, v + gain * action, c + np.zeros_like(action)], axis=-1)


if MAIN:
    tests.test_world_step(world_step)

# %%

def transition_features(states, actions):
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    return np.concatenate([states, actions[..., None], np.ones_like(actions[..., None])], axis=-1)


if MAIN:
    tests.test_transition_features(transition_features)

# %%

def fit_dynamics(features, next_states):
    return np.linalg.lstsq(features, next_states, rcond=None)[0]


if MAIN:
    tests.test_fit_dynamics(fit_dynamics)

# %%

def predict_step(states, actions, weights):
    return transition_features(states, actions) @ weights


if MAIN:
    tests.test_predict_step(predict_step)

# %%

if MAIN:
    states = np.array(list(itertools.product([-2., 0., 2.], [-1., 0., 1.], [-1., 1.])))
    train_states = np.repeat(states, 3, axis=0)
    train_actions = np.tile([-1., 0., 1.], len(states))
    targets = world_step(train_states, train_actions)
    W = fit_dynamics(transition_features(train_states, train_actions), targets)
    shuffled = np.random.default_rng(31).permutation(train_actions)
    W_shuffled = fit_dynamics(transition_features(train_states, shuffled), targets)
    print("Fitted action row:", W[3])
    print("Shuffled action row:", W_shuffled[3])

# %%

def imagine(initial, actions, weights):
    actions = np.asarray(actions, dtype=float)
    state = np.broadcast_to(initial, (actions.shape[0], 3)).copy()
    trajectory = [state]
    for i in range(actions.shape[1]):
        state = predict_step(state, actions[:, i], weights)
        trajectory.append(state)
    return np.stack(trajectory, axis=1)


if MAIN:
    tests.test_imagine(imagine)

# %%

def goal_cost(trajectories, actions, goal, nuisance_weight=0.0):
    error = trajectories[:, -1] - np.asarray(goal)
    return error[:, 0]**2 + 2 * error[:, 1]**2 + nuisance_weight * error[:, 2]**2 + 0.02 * np.sum(actions**2, axis=1)


if MAIN:
    tests.test_goal_cost(goal_cost)

# %%

def select_elites(actions, costs, count):
    if not 1 <= count <= len(actions):
        raise ValueError("elite count must be between one and population size")
    order = np.argsort(costs, kind="stable")[:count]
    return actions[order]


if MAIN:
    tests.test_select_elites(select_elites)

# %%

def cem_plan(initial, goal, weights, horizon=6, seed=7, population=256, rounds=7):
    rng = np.random.default_rng(seed)
    mean, std = np.zeros(horizon), np.ones(horizon)
    best, best_cost = mean.copy(), np.inf
    history = []
    for _ in range(rounds):
        candidates = np.clip(rng.normal(mean, std, size=(population, horizon)), -1., 1.)
        candidates[0] = mean
        costs = goal_cost(imagine(initial, candidates, weights), candidates, goal)
        j = int(np.argmin(costs))
        if costs[j] < best_cost:
            best, best_cost = candidates[j].copy(), float(costs[j])
        elite = select_elites(candidates, costs, max(2, population // 8))
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), 0.08)
        history.append(best_cost)
    return best, np.asarray(history)


if MAIN:
    tests.test_cem_plan(cem_plan)

# %%

if MAIN:
    initial, goal = np.array([0., 0., -1.]), np.array([3., 0., 1.])
    discrete = np.array(list(itertools.product([-1., 0., 1.], repeat=4)))
    costs4 = goal_cost(imagine(initial, discrete, W), discrete, goal)
    oracle4 = discrete[np.argmin(costs4)]
    plan, history = cem_plan(initial, goal, W)
    bad_plan, bad_history = cem_plan(initial, goal, W_shuffled)
    zero_plan = np.zeros(6)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for label, acts in [("CEM, aligned actions", plan), ("CEM, shuffled actions", bad_plan), ("zero-action baseline", zero_plan)]:
        real = [initial]
        for a in acts:
            real.append(world_step(real[-1], a))
        real = np.array(real)
        axes[0].plot(real[:, 0], real[:, 1], "o-", label=label)
        axes[1].plot(np.arange(7), real[:, 0], "o-", label=label)
        print(label, "terminal [x,v]", real[-1,:2].round(4), "true cost", round(float(goal_cost(real[None], acts[None], goal)[0]),4))
    axes[0].plot([goal[0]], [goal[1]], "k*", ms=13, label="goal")
    axes[0].set(xlabel="position", ylabel="velocity", title="Actual trajectories of selected actions")
    axes[1].axhline(3., color="k", ls=":", label="goal position")
    axes[1].set(xlabel="step", ylabel="position", title="Control compared with zero actions")
    axes[2].plot(history, label="aligned model best imagined cost")
    axes[2].plot(bad_history, label="shuffled model best imagined cost")
    axes[2].set(xlabel="CEM round", ylabel="cost", title="Search uses its own world model")
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); plt.show()
    print("81-sequence horizon-four oracle:", oracle4, "cost", round(float(costs4.min()),4))

# %%

def run_mpc(initial, goal, weights, decisions=6, gain=1.0, seed=19):
    states, actions = [np.asarray(initial, dtype=float)], []
    for i in range(decisions):
        plan, _ = cem_plan(states[-1], goal, weights, seed=seed+i)
        actions.append(plan[0])
        states.append(world_step(states[-1], plan[0], gain=gain))
    return np.asarray(states), np.asarray(actions)


if MAIN:
    tests.test_run_mpc(run_mpc)

# %%

def physical_error(trajectory, goal):
    return np.sqrt(np.sum((trajectory[..., :2] - np.asarray(goal)[:2])**2, axis=-1))


if MAIN:
    tests.test_physical_error(physical_error)

# %%

if MAIN:
    open_states = [initial]
    for a in plan: open_states.append(world_step(open_states[-1], a, gain=.7))
    open_states = np.array(open_states)
    mpc_states, mpc_actions = run_mpc(initial, goal, W, gain=.7)
    zero_states = np.repeat(initial[None], 7, axis=0)
    fig, ax = plt.subplots(figsize=(7,3.5))
    for name, tr in [("open loop, wrong gain", open_states), ("MPC, fresh six-step horizon", mpc_states), ("zero-action baseline", zero_states)]:
        ax.plot(physical_error(tr, goal), "o-", label=name)
        print(name, "final physical distance", round(float(physical_error(tr, goal)[-1]),4))
    ax.set(xlabel="executed step", ylabel="physical distance", title="Feedback under actuator mismatch")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); plt.show()

# %%

def controllability(weights, horizon=3):
    A, B = weights[:3].T, weights[3]
    columns = []
    for _ in range(horizon):
        columns.append(B)
        B = A @ B
    return np.stack(columns, axis=1)


if MAIN:
    tests.test_controllability(controllability)

# %%

def patch_latent(recipient, donor, coordinates):
    out = np.array(recipient, dtype=float, copy=True)
    out[..., coordinates] = np.asarray(donor)[..., coordinates]
    return out


if MAIN:
    tests.test_patch_latent(patch_latent)

# %%

if MAIN:
    C = controllability(W)
    physical_patch = patch_latent(initial, goal, [0,1])
    colour_patch = patch_latent(initial, goal, [2])
    variants = [initial, colour_patch, physical_patch]
    phys = [physical_error(v,goal) for v in variants]
    full = [float(goal_cost(np.array(v)[None,None,:],np.zeros((1,0)),goal,nuisance_weight=1)[0]) for v in variants]
    fig, axes = plt.subplots(1,2,figsize=(10,3.5))
    im=axes[0].imshow(C,aspect="auto",cmap="coolwarm");fig.colorbar(im,ax=axes[0],label="action influence")
    axes[0].set(xticks=range(3),xticklabels=["B","AB","A²B"],yticks=range(3),yticklabels=["position","velocity","colour"],title="Reachable physical subspace")
    xx=np.arange(3);axes[1].bar(xx-.18,phys,.36,label="physical distance");axes[1].bar(xx+.18,full,.36,label="bad full-latent squared cost")
    axes[1].set(xticks=xx,xticklabels=["unpatched","colour-only","physical-only"],ylabel="metric value",title="Intervention versus metric shortcut");axes[1].legend(fontsize=8)
    fig.tight_layout();plt.show()
    print("controllability rank:",np.linalg.matrix_rank(C))
    print("physical distances:",np.array(phys))
    print("full-latent costs:",np.array(full))

# %%

if MAIN:
    play_initial, play_goal = np.array([0.,1.,-1.]), np.array([-2.,0.,1.])
    play_plan, _ = cem_plan(play_initial, play_goal, W)
    play_state, _ = run_mpc(play_initial, play_goal, W, gain=.7)
    print("planned first action:",round(float(play_plan[0]),4))
    print("feedback final physical distance:",round(float(physical_error(play_state,play_goal)[-1]),4))

# %%
