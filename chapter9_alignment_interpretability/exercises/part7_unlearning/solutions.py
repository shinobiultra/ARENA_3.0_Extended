# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

chapter = "chapter9_alignment_interpretability"
section = "part7_unlearning"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part7_unlearning.tests as tests
import part7_unlearning.utils as utils
MAIN = __name__ == "__main__"
np.set_printoptions(precision=4, suppress=True)

# %%

def log_probability(logits, labels):
    sign = 2 * labels - 1
    return -np.logaddexp(0., -sign * logits)


if MAIN:
    tests.test_log_probability(log_probability)

# %%

def memory_gradient(memory, labels):
    sign = (2 * labels - 1)[:, None]
    return -sign * np.exp(-np.logaddexp(0., sign * memory)) / len(labels)


if MAIN:
    tests.test_memory_gradient(memory_gradient)

# %%

def train_memory(labels, steps=32, lr=1.):
    memory = np.zeros((len(labels), 2), dtype=np.float64)
    for _ in range(steps):
        memory -= lr * memory_gradient(memory, labels)
    return memory


if MAIN:
    tests.test_train_memory(train_memory)

# %%

def masked_mean(values, mask):
    if not np.any(mask):
        raise ValueError("The declared cohort is empty")
    return np.mean(values[mask])


if MAIN:
    tests.test_masked_mean(masked_mean)

# %%

if MAIN:
    entities = ["Aro", "Bex", "Ceri", "Dax"]
    labels = np.array([1, 0, 1, 0])
    forget = np.array([True, True, False, False])
    retain = ~forget
    original_memory = train_memory(labels)
    original_head = np.array([1., 0.])
    reference_logp = log_probability(original_memory @ original_head, labels)[forget].copy()
    for entity, label, hidden, is_forget in zip(entities, labels, original_memory, forget):
        print(f"{entity:4s} color={['teal','amber'][label]:5s} cohort={'forget' if is_forget else 'retain':6s} h={hidden}")
    print("Original target probability:", np.exp(reference_logp).mean())

# %%

def ga_head_step(memory, labels, forget, weights, lr, retain_weight=0.):
    sign = 2 * labels - 1
    logits = memory @ weights
    derivative = sign * np.exp(-np.logaddexp(0., sign * logits))
    gradient = memory[forget].T @ derivative[forget] / np.sum(forget)
    gradient -= retain_weight * memory[~forget].T @ derivative[~forget] / np.sum(~forget)
    return weights - lr * gradient


if MAIN:
    tests.test_ga_head_step(ga_head_step)

# %%

def npo_loss(logp, reference_logp, beta):
    return (2. / beta) * np.logaddexp(0., beta * (logp - reference_logp)).mean()


if MAIN:
    tests.test_npo_loss(npo_loss)

# %%

def npo_head_step(memory, labels, forget, weights, reference_logp, beta, lr, retain_weight=0.):
    sign = 2 * labels - 1
    logits = memory @ weights
    logp = log_probability(logits, labels)
    derivative = sign * np.exp(-np.logaddexp(0., sign * logits))
    coefficient = 2. * np.exp(-np.logaddexp(0., -beta * (logp[forget] - reference_logp)))
    gradient = memory[forget].T @ (coefficient * derivative[forget]) / np.sum(forget)
    gradient -= retain_weight * memory[~forget].T @ derivative[~forget] / np.sum(~forget)
    return weights - lr * gradient


if MAIN:
    tests.test_npo_head_step(npo_head_step)

# %%

if MAIN:
    ga_head, npo_head = original_head.copy(), original_head.copy()
    ga_trace, npo_trace = [], []
    for step in range(9):
        ga_trace.append(np.exp(log_probability(original_memory @ ga_head, labels)[forget]).mean())
        npo_trace.append(np.exp(log_probability(original_memory @ npo_head, labels)[forget]).mean())
        if step < 8:
            ga_head = ga_head_step(original_memory, labels, forget, ga_head, .3, 0.)
            npo_head = npo_head_step(original_memory, labels, forget, npo_head, reference_logp, .5, .3, 0.)
    print("Step     GA target-p    NPO target-p")
    for step, a, b in zip(range(9), ga_trace, npo_trace):
        print(f"{step:4d}     {a:.4f}         {b:.4f}")
    print("Fixed budget final heads:", ga_head, npo_head)

# %%

def rmu_objective(memory, reference, forget, target, retain_weight):
    forget_error = np.sum((memory - target)**2, axis=1)
    retain_error = np.sum((memory - reference)**2, axis=1)
    return masked_mean(forget_error, forget) + retain_weight * masked_mean(retain_error, ~forget)


if MAIN:
    tests.test_rmu_objective(rmu_objective)

# %%

def rmu_memory_step(memory, reference, forget, target, lr, retain_weight):
    gradient = np.zeros_like(memory)
    gradient[forget] = 2. * (memory[forget] - target) / np.sum(forget)
    gradient[~forget] = 2. * retain_weight * (memory[~forget] - reference[~forget]) / np.sum(~forget)
    return memory - lr * gradient


if MAIN:
    tests.test_rmu_memory_step(rmu_memory_step)

# %%

def sae_encode(memory, encoder):
    return np.maximum(memory @ encoder, 0.)


if MAIN:
    tests.test_sae_encode(sae_encode)

# %%

def ablate_features(features, decoder, selected):
    edited = features.copy()
    edited[:, selected] = 0.
    return edited @ decoder


if MAIN:
    tests.test_ablate_features(ablate_features)

# %%

if MAIN:
    target = np.array([1., 2.]) / np.sqrt(5.)
    rmu_memory = rmu_memory_step(original_memory, original_memory, forget, target, 1., 1.)
    encoder = np.array([[1., -1., 0., 0.], [0., 0., 1., -1.]])
    decoder = encoder.T
    features = sae_encode(original_memory, encoder)
    reconstructed = ablate_features(features, decoder, [])
    primary_ablated = ablate_features(features, decoder, [0, 1])
    backup_ablated = ablate_features(features, decoder, [2, 3])
    print("Exact dictionary reconstruction max error:", np.max(np.abs(reconstructed-original_memory)))
    print("RMU-style forget rows:", rmu_memory[forget])
    print("RMU-style retain max change:", np.max(np.abs(rmu_memory[retain]-original_memory[retain])))
    print("Primary-ablated backup signs:", np.sign(primary_ablated[:, 1]))

# %%

def retrieval_accuracy(logits, labels, cohort):
    return masked_mean((logits > 0).astype(int) == labels, cohort)


if MAIN:
    tests.test_retrieval_accuracy(retrieval_accuracy)

# %%

def probe_memory(memory, readout):
    return memory @ readout


if MAIN:
    tests.test_probe_memory(probe_memory)

# %%

def relearn_head(memory, labels, forget, weights, steps=3, lr=2.):
    result = weights.copy()
    for _ in range(steps):
        sign = 2 * labels[forget] - 1
        logits = memory[forget] @ result
        derivative = -sign * np.exp(-np.logaddexp(0., sign * logits))
        result -= lr * (memory[forget].T @ derivative) / np.sum(forget)
    return result


if MAIN:
    tests.test_relearn_head(relearn_head)

# %%

def relearn_memory(memory, labels, forget, steps=3, lr=2.):
    result = memory.copy()
    for _ in range(steps):
        result[forget] -= lr * memory_gradient(result[forget], labels[forget])
    return result


if MAIN:
    tests.test_relearn_memory(relearn_memory)

# %%

if MAIN:
    methods = ["original", "output mask", "head-only GA", "head-only NPO", "direct-memory RMU",
               "primary features zero", "backup features zero"]
    memories = [original_memory, original_memory, original_memory, original_memory, rmu_memory,
                primary_ablated, backup_ablated]
    heads = [original_head, original_head, ga_head, npo_head, original_head, original_head, original_head]
    matrix, probability_rows = [], []
    for name, hidden, head in zip(methods, memories, heads):
        logits = hidden @ head
        if name == "output mask":
            logits = logits.copy()
            logits[forget] = 0.
        backup_logits = probe_memory(hidden, np.array([0., 1.]))
        new_head = relearn_head(hidden, labels, forget, head, steps=3, lr=2.)
        new_memory = relearn_memory(hidden, labels, forget, steps=3, lr=2.)
        matrix.append([retrieval_accuracy(logits, labels, forget),
                       retrieval_accuracy(logits, labels, retain),
                       retrieval_accuracy(backup_logits, labels, forget),
                       retrieval_accuracy(hidden @ new_head, labels, forget),
                       retrieval_accuracy(new_memory @ original_head, labels, forget)])
        probability_rows.append([masked_mean(np.exp(log_probability(logits, labels)), forget),
                                 masked_mean(np.exp(log_probability(logits, labels)), retain)])
    matrix = np.array(matrix)
    probability_rows = np.array(probability_rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    image = utils.metric_heatmap(axes[0], matrix, methods,
                               ["forget acc", "retain acc", "fixed backup", "head relearn", "memory relearn"])
    fig.colorbar(image, ax=axes[0], label="Accuracy on the declared cohort")
    for j, label in enumerate(["forget target probability", "retain target probability"]):
        axes[1].scatter(probability_rows[:, j], np.arange(len(methods)), label=label, s=60)
    axes[1].axvline(.5, color="black", ls="--", label="binary chance / uniform probability")
    axes[1].set(yticks=np.arange(len(methods)), yticklabels=methods,
                xlabel="Mean correct-color probability", ylabel="Restricted CPU method",
                title="Likelihood also needs a retain control", xlim=(0, 1))
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    plt.show()
    print("method                    forget retain backup head-attack memory-attack")
    for name, row in zip(methods, matrix):
        print(f"{name:25s} " + " ".join(f"{value:.2f}" for value in row))

# %%

if MAIN:
    play_memory = original_memory.copy()
    play_memory[:, 1] = 0.0  # changed redundancy condition
    play_features = sae_encode(play_memory, encoder)
    play_hidden = ablate_features(play_features, decoder, [0, 1])
    print("Changed condition normal forget accuracy:", retrieval_accuracy(play_hidden @ original_head, labels, forget))
    print("Changed condition backup forget accuracy:", retrieval_accuracy(probe_memory(play_hidden, np.array([0., 1.])), labels, forget))
    print("Unmodified primary path before ablation:", retrieval_accuracy(play_memory @ original_head, labels, forget))

# %%
