# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

chapter = "chapter9_alignment_interpretability"
section = "part2_truthfulness_hallucination"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part2_truthfulness_hallucination.tests as tests
import part2_truthfulness_hallucination.utils as utils
MAIN = __name__ == "__main__"
np.set_printoptions(precision=4, suppress=True)

# %%

if MAIN:
    statements = [("3 is smaller than 7", 1), ("7 is smaller than 3", 0),
                  ("2 is even", 1), ("3 is even", 0)]
    for statement, label in statements:
        print(f"{statement:25s} truth={label}")
    train_rows, iid_rows, transfer_rows, labels, known = [], [], [], [], []
    for sign in (-1., 1.):
        for nuisance, other_nuisance in zip((-1.5, -.5, .5, 1.5), (-1.25, -.25, .25, 1.25)):
            for familiarity in (-1., 1.):
                train_rows.append([sign+nuisance, nuisance, sign, familiarity])
                iid_rows.append([sign+other_nuisance, other_nuisance, sign, familiarity])
                transfer_rows.append([sign+other_nuisance, other_nuisance, -sign, familiarity])
                labels.append(int(sign > 0))
                known.append(familiarity > 0)
    X_train, X_iid, X_transfer = map(lambda rows: np.array(rows, dtype=np.float64),
                                   (train_rows, iid_rows, transfer_rows))
    y = np.array(labels)
    known = np.array(known)
    print("rows per role:", len(y), "label counts:", np.bincount(y))

# %%

def center_from_train(train, evaluated):
    center = np.mean(train, axis=0)
    return center, evaluated - center


if MAIN:
    tests.test_center_from_train(center_from_train)

# %%

def mass_mean(x, labels):
    return x[labels == 1].mean(axis=0) - x[labels == 0].mean(axis=0)


if MAIN:
    tests.test_mass_mean(mass_mean)

# %%

def class_covariance(x, labels):
    residual = x.copy()
    for label in (0, 1):
        mask = labels == label
        residual[mask] -= x[mask].mean(axis=0)
    return residual.T @ residual / len(x)


if MAIN:
    tests.test_class_covariance(class_covariance)

# %%

def iid_direction(direction, covariance, regularizer):
    return np.linalg.solve(covariance + regularizer * np.eye(len(direction)), direction)


if MAIN:
    tests.test_iid_direction(iid_direction)

# %%

if MAIN:
    center, train = center_from_train(X_train, X_train)
    _, iid = center_from_train(X_train, X_iid)
    _, transfer = center_from_train(X_train, X_transfer)
    mm = mass_mean(train, y)
    covariance = class_covariance(train, y)
    iid_mm = iid_direction(mm, covariance, .1)
    print("TRAIN center:", center)
    print("mass-mean direction:", mm)
    print("within-class diagonal:", np.diag(covariance))

# %%

def logistic_gradient(x, labels, weights, penalty):
    z = x @ weights
    probabilities = np.exp(-np.logaddexp(0., -z))
    return x.T @ (probabilities - labels) / len(x) + penalty * weights


if MAIN:
    tests.test_logistic_gradient(logistic_gradient)

# %%

def fit_logistic(x, labels, steps=200, lr=.2, penalty=.1):
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(steps):
        weights -= lr * logistic_gradient(x, labels, weights, penalty)
    return weights


if MAIN:
    tests.test_fit_logistic(fit_logistic)

# %%

def balanced_accuracy(predicted, labels):
    return .5 * (np.mean(predicted[labels == 1] == 1) + np.mean(predicted[labels == 0] == 0))


if MAIN:
    tests.test_balanced_accuracy(balanced_accuracy)

# %%

def permute_labels(labels, permutation):
    if not np.array_equal(np.sort(permutation), np.arange(len(labels))):
        raise ValueError("Expected each label index exactly once")
    return labels[permutation]


if MAIN:
    tests.test_permute_labels(permute_labels)

# %%

def transfer_matrix(probes, datasets):
    return np.array([[balanced_accuracy((x @ w > 0).astype(int), labels)
                      for x, labels in datasets] for w in probes])


if MAIN:
    tests.test_transfer_matrix(transfer_matrix)

# %%

if MAIN:
    lr_weights = fit_logistic(train, y)
    permutation = np.array([j for i in range(8) for j in (i, i+8)])
    shuffled_y = permute_labels(y, permutation)
    shuffled_mm = mass_mean(train, shuffled_y)
    tag = np.array([0., 0., 1., 0.])
    probe_names = ["mass mean", "IID covariance", "logistic", "spare tag", "shuffled labels"]
    probes = np.stack([mm, iid_mm, lr_weights, tag, shuffled_mm])
    accuracy = transfer_matrix(probes, [(iid, y), (transfer, y)])
    print("Probe                 IID    tag-reversed TRANSFER")
    for name, row in zip(probe_names, accuracy):
        print(f"{name:20s} {row[0]:.3f}  {row[1]:.3f}")

# %%

def steer(x, direction, alpha):
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Direction must have finite positive norm")
    return x + alpha * direction / norm


if MAIN:
    tests.test_steer(steer)

# %%

def answer_probability(margins):
    return np.exp(-np.logaddexp(0., -margins))


if MAIN:
    tests.test_answer_probability(answer_probability)

# %%

if MAIN:
    strengths = np.linspace(-3., 3., 13)
    false_rows = transfer[y == 0]
    nuisance_direction = np.array([1., 1., 0., 0.])
    directions = [mm, lr_weights, tag, shuffled_mm, nuisance_direction]
    direction_names = ["mass mean", "logistic", "spare tag", "shuffled labels", "nuisance"]
    intervention_curves = []
    for direction in directions:
        curve = []
        for alpha in strengths:
            changed = steer(false_rows, direction, alpha)
            curve.append(answer_probability(changed[:, 0] - changed[:, 1] - .8).mean())
        intervention_curves.append(curve)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    image = utils.labelled_heatmap(axes[0], accuracy, probe_names, ["IID", "tag reversed"],
                                  "Decoding transfers differently across domains")
    fig.colorbar(image, ax=axes[0], label="Balanced accuracy")
    for name, curve in zip(direction_names, intervention_curves):
        axes[1].plot(strengths, curve, marker="o", label=name)
    axes[1].axhline(answer_probability(np.array([-1.8]))[0], color="black", ls="--", label="no intervention")
    axes[1].set(xlabel="Matched-norm activation shift α", ylabel="P(TRUE) on false statements",
                title="A perfect tag probe has zero answer effect", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    plt.show()
    print("Tag-probe IID accuracy:", accuracy[3, 0])
    print("Tag-direction probability range:", np.ptp(intervention_curves[2]))

# %%

def selective_rates(known, emitted, correct):
    coverage = np.mean(emitted)
    selective = np.mean(correct[emitted]) if np.any(emitted) else np.nan
    unknown_assertion = np.mean(emitted[~known])
    known_refusal = np.mean(~emitted[known])
    return np.array([coverage, selective, unknown_assertion, known_refusal])


if MAIN:
    tests.test_selective_rates(selective_rates)

# %%

def head_shift(z, directions, scales, selected, alpha):
    result = z.copy()
    for head in selected:
        result[:, head] += alpha * scales[head] * directions[head] / np.linalg.norm(directions[head])
    return result


if MAIN:
    tests.test_head_shift(head_shift)

# %%

if MAIN:
    print("Recognition shift   coverage  selective-acc  unknown-assert  known-refusal")
    correct = known.copy()  # Declared: this organism recalls only known entity answers.
    for alpha in (-2., 0., 2.):
        emitted = answer_probability(transfer[:, 3] + alpha) > .5
        rates = selective_rates(known, emitted, correct)
        print(f"{alpha:17.1f}   " + "  ".join(f"{rate:.3f}" for rate in rates))

# %%

if MAIN:
    readout_sign = 1.0
    tag_weight = 0.0
    play_alphas = np.array([-2., 0., 2.])
    for name, direction in (("mass mean", mm), ("spare tag", tag)):
        probabilities = []
        for alpha in play_alphas:
            changed = steer(transfer[y == 0], direction, alpha)
            margin = readout_sign * (changed[:, 0] - changed[:, 1]) + tag_weight * changed[:, 2] - .8
            probabilities.append(answer_probability(margin).mean())
        print(name, np.round(probabilities, 4))

# %%
