"""Independent geometry, intervention and regression oracles on small CPU arrays."""
import numpy as np


def done(name):
    print(f'All tests in {name} passed!')


def test_cyclic_coordinates(fn):
    np.testing.assert_allclose(fn(np.arange(4), 4), [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], atol=1e-14)
    np.testing.assert_allclose(fn(np.array([0, 7]), 7), [[1., 0.], [1., 0.]], atol=1e-14)
    done('test_cyclic_coordinates')


def test_embed_manifold(fn):
    coordinates = np.array([[1., 2.], [-1., 3.]])
    basis = np.array([[1., 0., 0.], [0., 1., 0.]])
    nuisance = np.array([[0., 0., 4.], [0., 0., -2.]])
    np.testing.assert_allclose(fn(coordinates, basis, nuisance), [[1., 2., 4.], [-1., 3., -2.]])
    done('test_embed_manifold')


def test_pca_plane(fn):
    h = np.array([[2., 0., 7.], [-2., 0., 7.], [0., 1., 7.], [0., -1., 7.]])
    center, basis = fn(h, 2)
    np.testing.assert_allclose(center, [0., 0., 7.])
    np.testing.assert_allclose(basis.T @ basis, np.diag([1., 1., 0.]), atol=1e-12)
    done('test_pca_plane')


def test_fit_fourier_plane(fn):
    labels = np.arange(4)
    h = np.array([[3., 2., 0.], [2., 4., 0.], [1., 2., 0.], [2., 0., 0.]])
    center, basis = fn(h, labels, 4)
    np.testing.assert_allclose(center, [2., 2., 0.], atol=1e-12)
    np.testing.assert_allclose(basis, [[1., 0., 0.], [0., 2., 0.]], atol=1e-12)
    done('test_fit_fourier_plane')


def test_subspace_overlap(fn):
    a = np.eye(4)[:2]; b = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.]])
    assert abs(fn(a, b)-1.) < 1e-12
    assert abs(fn(a, np.eye(4)[2:])) < 1e-12
    done('test_subspace_overlap')


def test_pairwise_distances(fn):
    h = np.array([[0., 0.], [3., 4.], [3., 0.]])
    np.testing.assert_allclose(fn(h), [[0., 5., 3.], [5., 0., 4.], [3., 4., 0.]])
    done('test_pairwise_distances')


def test_distance_stress(fn):
    distance = np.array([[0., 2.], [2., 0.]])
    assert abs(fn(distance, distance)) < 1e-12
    assert abs(fn(distance, distance/2)-.5) < 1e-12
    done('test_distance_stress')


def test_cyclic_neighbor_score(fn):
    # Clockwise adjacent vertices of a square are nearest; diagonals are farther.
    distance = np.array([[0., 1., 2., 1.], [1., 0., 1., 2.], [2., 1., 0., 1.], [1., 2., 1., 0.]])
    assert fn(distance, np.arange(4), 4) == 1.
    assert fn(distance, np.array([0, 2, 1, 3]), 4) == .5
    done('test_cyclic_neighbor_score')


def test_energy_fraction(fn):
    h = np.array([[2., 1.], [-2., -1.]])
    assert abs(fn(h, np.array([[1., 0.]]))-.8) < 1e-12
    done('test_energy_fraction')


def test_rotate_in_plane(fn):
    basis = np.array([[1., 0., 0.], [0., 1., 0.]])
    h = np.array([[1., 0., 7.], [0., 1., -3.]])
    before = h.copy()
    np.testing.assert_allclose(fn(h, basis, np.pi/2), [[0., 1., 7.], [-1., 0., -3.]], atol=1e-12)
    np.testing.assert_array_equal(h, before)
    done('test_rotate_in_plane')


def test_decode_cycle(fn):
    basis = np.array([[1., 0., 0.], [0., 1., 0.]])
    np.testing.assert_array_equal(fn(np.array([[0., 1., 9.], [-1., 0., -8.]]), basis, 4), [1, 2])
    done('test_decode_cycle')


def test_chord_shift(fn):
    basis = np.eye(2)
    h = np.array([[1., 0.], [0., 1.]])
    np.testing.assert_allclose(fn(h, basis, 4, 0, 1), [[0., 1.], [-1., 2.]], atol=1e-12)
    done('test_chord_shift')


def test_matched_wrong_direction(fn):
    h = np.array([[1., 0., 0.], [0., 2., 0.]])
    changed = np.array([[0., 1., 0.], [2., 0., 0.]])
    control = fn(h, changed, np.array([0., 0., 3.]))
    np.testing.assert_allclose(np.linalg.norm(control-h, axis=1), np.linalg.norm(changed-h, axis=1))
    np.testing.assert_allclose(control[:, :2], h[:, :2])
    done('test_matched_wrong_direction')


def test_fit_affine_probe(fn):
    x = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    target = x @ np.array([[2., -1.], [3., 4.]]) + [5., -2.]
    weights, bias = fn(x, target)
    np.testing.assert_allclose(weights, [[2., -1.], [3., 4.]], atol=1e-12)
    np.testing.assert_allclose(bias, [5., -2.], atol=1e-12)
    done('test_fit_affine_probe')


def test_probe_mse(fn):
    assert fn(np.array([[1., 2.], [3., 4.]]), np.array([[0., 2.], [1., 4.]])) == 1.25
    done('test_probe_mse')


def test_residual_readout(fn):
    h = np.array([[5., 2.], [4., 7.]])
    coordinates = np.array([[1.], [2.]])
    np.testing.assert_allclose(fn(h, coordinates, np.array([[2., 1.]]), np.array([1., 0.])), [[2., 1.], [-1., 5.]])
    done('test_residual_readout')


def test_interaction_feature(fn):
    np.testing.assert_allclose(fn(np.array([[2., 100., 3.], [-2., 100., 4.]])), [[6.], [-8.]])
    done('test_interaction_feature')
