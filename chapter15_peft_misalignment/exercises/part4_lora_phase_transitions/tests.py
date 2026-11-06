"""Independent small analytic checks for the study-edition matrix trajectory."""
import numpy as np


def _ok(name):
    print(f"All tests in `{name}` passed!")


def test_lora_update(fn):
    A = np.array([[1., 2., -1.], [0., 3., 1.]])
    B = np.array([[1., 2.], [-1., 0.]])
    expected = np.array([[.5, 4., .5], [-.5, -1., .5]])
    np.testing.assert_allclose(fn(A, B, 1.), expected)
    np.testing.assert_array_equal(fn(A, np.zeros_like(B), 1.), np.zeros((2, 3)))
    _ok('test_lora_update')


def test_planted_update(fn):
    np.testing.assert_array_equal(fn(0.), np.zeros((4, 4)))
    np.testing.assert_allclose(fn(.75), np.diag([.75, .6, 0., 0.]))
    np.testing.assert_allclose(fn(1.), np.diag([1., 1.2, 0., 0.]))
    _ok('test_planted_update')


def test_rank_projection(fn):
    D = np.diag([1., 4., 2.])
    np.testing.assert_allclose(fn(D, 1), np.diag([0., 4., 0.]))
    np.testing.assert_allclose(fn(D, 0), np.zeros_like(D))
    np.testing.assert_allclose(np.linalg.norm(D-fn(D, 2))**2, 1.)
    _ok('test_rank_projection')


def test_spectral_energy(fn):
    np.testing.assert_allclose(fn(np.diag([3., 4., 0.])), [16/25, 9/25, 0.])
    np.testing.assert_array_equal(fn(np.zeros((3, 2))), [0., 0.])
    _ok('test_spectral_energy')


def test_matched_random_update(fn):
    D = np.diag([3., 4., 0., 0.])
    R = fn(D, 2, 17)
    np.testing.assert_allclose(np.linalg.norm(R), 5.)
    assert np.linalg.matrix_rank(R) == 2, 'Preserve the requested random rank.'
    np.testing.assert_array_equal(R, fn(D, 2, 17))
    assert not np.allclose(R, D), 'A norm control must change the direction.'
    np.testing.assert_array_equal(fn(np.zeros((4, 4)), 2, 17), np.zeros((4, 4)))
    np.testing.assert_array_equal(fn(np.zeros((4, 4)), 0, 17), np.zeros((4, 4)))
    rank_one_target = np.diag([.5, 0., 0., 0.])
    rank_one_random = fn(rank_one_target, 1, 17)
    assert np.linalg.matrix_rank(rank_one_random) == 1
    np.testing.assert_allclose(np.linalg.norm(rank_one_random), .5)
    _ok('test_matched_random_update')


def test_proxy_margins(fn):
    X = np.array([[1., 2.], [-1., 0.], [2., 3.]])
    W = np.array([[1., 0.], [2., -1.]])
    # y = 2*x0 - x1 for output direction [0,1].
    np.testing.assert_allclose(fn(X, W, np.array([0., 1.]), np.array([.2, .4, .6])), [-.2, -2.4, .4])
    _ok('test_proxy_margins')


def test_flip_rate(fn):
    # Zero is assigned to the non-positive class, by the lesson's frozen rule.
    np.testing.assert_allclose(fn(np.array([-1., 0., 2., 3.]), np.array([0., 1., -1., 4.])), .5)
    np.testing.assert_allclose(fn(np.array([-1., 1.]), np.array([-2., 2.])), 0.)
    np.testing.assert_allclose(fn(np.array([0.]), np.array([1.])), 1.)
    _ok('test_flip_rate')


def test_first_crossing(fn):
    assert fn(np.array([0., .5, 1.]), np.array([.1, .5, .8]), .5) == .5
    assert fn(np.array([0., .5, 1.]), np.array([.1, .2, .3]), .5) is None
    assert fn(np.array([0., .5]), np.array([.8, .1]), .5) == 0.
    _ok('test_first_crossing')


def test_top_output_alignment(fn):
    D = np.diag([1., 3., 0.])
    np.testing.assert_allclose(fn(D, np.array([0., -2., 0.])), 1.)
    np.testing.assert_allclose(fn(D, np.array([1., 0., 0.])), 0.)
    np.testing.assert_allclose(fn(np.zeros((3, 3)), np.ones(3)), 0.)
    rectangular = np.outer(np.array([0., 2., 0.]), np.array([1., 0.]))
    np.testing.assert_allclose(fn(rectangular, np.array([0., 1., 0.])), 1.)
    _ok('test_top_output_alignment')


def test_remove_output_direction(fn):
    D = np.array([[1., 2.], [3., 4.], [5., 6.]])
    result = fn(D, np.array([0., 2., 0.]))
    np.testing.assert_allclose(result, [[1., 2.], [0., 0.], [5., 6.]])
    np.testing.assert_array_equal(D[1], [3., 4.])
    np.testing.assert_allclose(fn(result, np.array([0., 1., 0.])), result)
    _ok('test_remove_output_direction')


def test_remove_dominant_mode(fn):
    D = np.diag([1., 3., 2.])
    np.testing.assert_allclose(fn(D), np.diag([1., 0., 2.]))
    np.testing.assert_allclose(fn(np.zeros((2, 2))), np.zeros((2, 2)))
    _ok('test_remove_dominant_mode')


def test_task_rmse(fn):
    X = np.eye(2)
    W = np.diag([2., 3.])
    np.testing.assert_allclose(fn(X, W, np.array([1., 0.]), np.array([0., 2.])), 2.)
    np.testing.assert_allclose(fn(X, W, np.array([0., 1.]), np.array([0., 3.])), 0.)
    _ok('test_task_rmse')
