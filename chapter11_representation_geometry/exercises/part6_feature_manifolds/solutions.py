# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

chapter = "chapter11_representation_geometry"
section = "part6_feature_manifolds"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part6_feature_manifolds.tests as tests
import part6_feature_manifolds.utils as utils
MAIN = __name__ == "__main__"
np.set_printoptions(precision=4, suppress=True)

# %%

def cyclic_coordinates(indices, period):
    angles = 2. * np.pi * np.asarray(indices) / period
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


if MAIN:
    tests.test_cyclic_coordinates(cyclic_coordinates)

# %%

def embed_manifold(coordinates, basis, nuisance):
    return coordinates @ basis + nuisance


if MAIN:
    tests.test_embed_manifold(embed_manifold)

# %%

def pca_plane(hidden, rank=2):
    center = hidden.mean(axis=0)
    _, _, right = np.linalg.svd(hidden - center, full_matrices=False)
    return center, right[:rank]


if MAIN:
    tests.test_pca_plane(pca_plane)

# %%

def fit_fourier_plane(hidden, indices, period):
    design = np.column_stack([np.ones(len(indices)), cyclic_coordinates(indices, period)])
    coefficients = np.linalg.lstsq(design, hidden, rcond=None)[0]
    return coefficients[0], coefficients[1:]


if MAIN:
    tests.test_fit_fourier_plane(fit_fourier_plane)

# %%

def subspace_overlap(first, second):
    return np.sum((first @ second.T)**2) / len(first)


if MAIN:
    tests.test_subspace_overlap(subspace_overlap)

# %%

if MAIN:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_ids = np.repeat(np.arange(7), 2)
    context = np.tile(np.array([-.3, .3]), 7)
    basis = np.array([[1., 0., 1., 0., 0., 0.], [0., 1., 0., 1., 0., 0.]]) / np.sqrt(2.)
    nuisance = np.zeros((14, 6)); nuisance[:, 4] = context
    train_hidden = embed_manifold(cyclic_coordinates(day_ids, 7), basis, nuisance)
    evaluated_nuisance = nuisance.copy(); evaluated_nuisance[:, 4] *= 3.
    evaluated_hidden = embed_manifold(cyclic_coordinates(day_ids, 7), basis, evaluated_nuisance)
    pca_center, pca_basis = pca_plane(train_hidden)
    fourier_center, fitted_basis = fit_fourier_plane(train_hidden, day_ids, 7)
    np.testing.assert_allclose(fitted_basis @ fitted_basis.T, np.eye(2), atol=1e-12)
    print("PCA / true plane overlap:", subspace_overlap(pca_basis, basis))
    print("Fourier / true plane overlap:", subspace_overlap(fitted_basis, basis))
    _, single_context_basis = pca_plane(train_hidden[::2])
    print("Across-context fitted plane overlap:", subspace_overlap(pca_basis, single_context_basis))
    print("Monday, Tuesday, Thursday coordinates:")
    print(cyclic_coordinates(np.array([0, 1, 3]), 7))
    month_ids = np.arange(12)
    month_hidden = embed_manifold(cyclic_coordinates(month_ids, 12), basis, np.zeros((12, 6)))
    print("Month coordinates have shape:", cyclic_coordinates(month_ids, 12).shape)

# %%

def pairwise_distances(hidden):
    differences = hidden[:, None, :] - hidden[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=-1))


if MAIN:
    tests.test_pairwise_distances(pairwise_distances)

# %%

def distance_stress(original, projected):
    mask = np.triu(np.ones_like(original, dtype=bool), k=1)
    return np.sqrt(np.sum((original[mask] - projected[mask])**2) / np.sum(original[mask]**2))


if MAIN:
    tests.test_distance_stress(distance_stress)

# %%

def cyclic_neighbor_score(distances, indices, period):
    scores = []
    for row, label in enumerate(indices):
        order = np.argsort(distances[row], kind="stable")
        nearest = [index for index in order if index != row][:2]
        wanted = {(int(label)-1) % period, (int(label)+1) % period}
        scores.extend(int(indices[index]) in wanted for index in nearest)
    return np.mean(scores)


if MAIN:
    tests.test_cyclic_neighbor_score(cyclic_neighbor_score)

# %%

def energy_fraction(hidden, basis):
    centered = hidden - hidden.mean(axis=0)
    return np.sum((centered @ basis.T)**2) / np.sum(centered**2)


if MAIN:
    tests.test_energy_fraction(energy_fraction)

# %%

if MAIN:
    calendar_centroids = cyclic_coordinates(np.arange(7), 7) @ basis
    calendar_view = calendar_centroids @ fitted_basis.T
    simplex = np.eye(7)
    chosen_projection = cyclic_coordinates(np.arange(7), 7) * np.sqrt(2./7.)
    # Columns are orthonormal, but they were deliberately chosen using day labels.
    simplex_view = simplex @ chosen_projection
    full_circle_d = pairwise_distances(calendar_centroids)
    full_simplex_d = pairwise_distances(simplex)
    print("Calendar distance stress:", distance_stress(full_circle_d, pairwise_distances(calendar_view)))
    print("Simplex distance stress:", distance_stress(full_simplex_d, pairwise_distances(simplex_view)))
    print("Cyclic neighbor scores, calendar / simplex:",
          cyclic_neighbor_score(full_circle_d, np.arange(7), 7),
          cyclic_neighbor_score(full_simplex_d, np.arange(7), 7))
    print("Simplex rank-two captured energy:", energy_fraction(simplex, chosen_projection.T))
    reversed_labels = np.array([0, 2, 4, 6, 1, 3, 5])
    print("Calendar shuffled-label neighbor score:", cyclic_neighbor_score(full_circle_d, reversed_labels, 7))

# %%

def rotate_in_plane(hidden, basis, delta):
    coordinates = hidden @ basis.T
    rotation = np.array([[np.cos(delta), np.sin(delta)], [-np.sin(delta), np.cos(delta)]])
    return hidden - coordinates @ basis + (coordinates @ rotation) @ basis


if MAIN:
    tests.test_rotate_in_plane(rotate_in_plane)

# %%

def decode_cycle(hidden, decoder_basis, period):
    coordinates = hidden @ decoder_basis.T
    prototypes = cyclic_coordinates(np.arange(period), period)
    return np.argmax(coordinates @ prototypes.T, axis=-1)


if MAIN:
    tests.test_decode_cycle(decode_cycle)

# %%

def chord_shift(hidden, basis, period, source_index, target_index):
    points = cyclic_coordinates(np.array([source_index, target_index]), period)
    return hidden + (points[1] - points[0]) @ basis


if MAIN:
    tests.test_chord_shift(chord_shift)

# %%

def matched_wrong_direction(hidden, changed, control_direction):
    displacements = np.linalg.norm(changed - hidden, axis=-1, keepdims=True)
    return hidden + displacements * control_direction / np.linalg.norm(control_direction)


if MAIN:
    tests.test_matched_wrong_direction(matched_wrong_direction)

# %%

if MAIN:
    # Use the frozen new contexts, not the training states or refitted evaluation axes.
    calendar_hidden = evaluated_hidden.copy()
    rotation_predictions = []
    rotation_accuracy, wrong_accuracy, chord_accuracy = [], [], []
    control_direction = np.array([0., 0., 0., 0., 0., 1.])
    for days_added in range(7):
        changed = rotate_in_plane(calendar_hidden, fitted_basis, 2*np.pi*days_added/7)
        wrong = matched_wrong_direction(calendar_hidden, changed, control_direction)
        chord = chord_shift(calendar_hidden, fitted_basis, 7, 1, (1+days_added)%7)
        expected = (day_ids + days_added) % 7
        predictions = decode_cycle(changed, basis, 7)
        rotation_predictions.append(predictions[::2])  # one token per weekday for the display
        rotation_accuracy.append(np.mean(predictions == expected))
        wrong_accuracy.append(np.mean(decode_cycle(wrong, basis, 7) == expected))
        chord_accuracy.append(np.mean(decode_cycle(chord, basis, 7) == expected))
    rotation_predictions = np.array(rotation_predictions)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    utils.label_points(axes[0, 0], calendar_view, days, "Known cyclic states: full distances are cyclic")
    utils.label_points(axes[0, 1], simplex_view, days, "Projection control: full distances are all equal")
    utils.token_heatmap(axes[1, 0], rotation_predictions, days)
    for name, scores in (("rotate calendar plane", rotation_accuracy),
                         ("matched-size orthogonal control", wrong_accuracy),
                         ("single Tuesday chord", chord_accuracy)):
        axes[1, 1].plot(range(7), scores, marker="o", label=name)
    axes[1, 1].axhline(1/7, color="gray", ls="--", label="uniform token chance")
    axes[1, 1].set(xlabel="Requested days added", ylabel="Accuracy: seven weekdays × two new contexts",
                   title="A one-prompt chord is not global cyclic addition", ylim=(-.05, 1.05))
    axes[1, 1].legend(fontsize=8)
    plt.show()
    print("Tuesday +2 =", days[rotation_predictions[2, 1]])
    print("+2 accuracy, rotation / chord / wrong-plane:", rotation_accuracy[2], chord_accuracy[2], wrong_accuracy[2])
    month_changed = rotate_in_plane(month_hidden, fitted_basis, 2*np.pi*2/12)
    month_predictions = decode_cycle(month_changed, basis, 12)
    print("12-month +2 accuracy:", np.mean(month_predictions == (month_ids+2)%12))

# %%

def fit_affine_probe(hidden, targets):
    design = np.column_stack([hidden, np.ones(len(hidden))])
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    return coefficients[:-1], coefficients[-1]


if MAIN:
    tests.test_fit_affine_probe(fit_affine_probe)

# %%

def probe_mse(predicted, targets):
    return np.mean((predicted - targets)**2)


if MAIN:
    tests.test_probe_mse(probe_mse)

# %%

def residual_readout(hidden, coordinates, weights, bias):
    return hidden - (coordinates @ weights + bias)


if MAIN:
    tests.test_residual_readout(residual_readout)

# %%

def interaction_feature(coordinates):
    return (coordinates[:, 0] * coordinates[:, 2])[:, None]


if MAIN:
    tests.test_interaction_feature(interaction_feature)

# %%

if MAIN:
    # Four fictional map corners and years1900/2000, each in two irrelevant contexts.
    train_coordinates, world_hidden = [], []
    for x in (-1., 1.):
        for y_coord in (-1., 1.):
            for time in (-1., 1.):
                for style in (-.2, .2):
                    train_coordinates.append([x, y_coord, time])
                    world_hidden.append([x+y_coord, x-y_coord, time, x*time, style, 0.])
    train_coordinates = np.array(train_coordinates)
    world_hidden = np.array(world_hidden)
    evaluated_coordinates = np.array([[-.75, .5, -.5], [.25, -.5, .5], [.75, .25, .75]])
    evaluated_world = np.array([[x+y_coord, x-y_coord, time, x*time, .7, 0.]
                                for x, y_coord, time in evaluated_coordinates])
    probe_w, probe_b = fit_affine_probe(world_hidden, train_coordinates)
    predicted_coordinates = evaluated_world @ probe_w + probe_b
    # Fixed shuffled-target control: complement only the x target in TRAIN.
    shuffled_targets = train_coordinates.copy(); shuffled_targets[:, 0] *= -1
    shuffled_w, shuffled_b = fit_affine_probe(world_hidden, shuffled_targets)
    print("Frozen coordinate probe MSE:", probe_mse(predicted_coordinates, evaluated_coordinates))
    print("Shuffled-x target control MSE:", probe_mse(evaluated_world @ shuffled_w + shuffled_b, evaluated_coordinates))
    hidden_w, hidden_b = fit_affine_probe(train_coordinates, world_hidden)
    train_residual = residual_readout(world_hidden, train_coordinates, hidden_w, hidden_b)
    evaluated_residual = residual_readout(evaluated_world, evaluated_coordinates, hidden_w, hidden_b)
    interaction_w, interaction_b = fit_affine_probe(interaction_feature(train_coordinates), train_residual[:, 3:4])
    predicted_residual = interaction_feature(evaluated_coordinates) @ interaction_w + interaction_b
    print("x      y      year     actual residual   interaction forecast   affine-only forecast")
    for point, actual, predicted in zip(evaluated_coordinates, evaluated_residual[:, 3], predicted_residual[:, 0]):
        print(f"{point[0]: .2f} {point[1]: .2f} {1950+50*point[2]:7.1f}       {actual: .4f}              {predicted: .4f}               0.0000")

# %%

if MAIN:
    second_harmonic_weight = 0.0  # Try3.0 after recording a forecast.
    first_codes = cyclic_coordinates(np.arange(7), 7)
    second_codes = cyclic_coordinates(2*np.arange(7), 7)
    shifted_primary = rotate_in_plane(first_codes, np.eye(2), 4*np.pi/7)
    # The readout now consults a second, unedited circular code in omitted dimensions.
    play_logits = shifted_primary @ first_codes.T + second_harmonic_weight * second_codes @ second_codes.T
    play_tokens = np.argmax(play_logits, axis=-1)
    print("New-readout +2 accuracy:", np.mean(play_tokens == (np.arange(7)+2)%7))
    print("Input -> output:", list(zip(days, [days[index] for index in play_tokens])))

# %%
