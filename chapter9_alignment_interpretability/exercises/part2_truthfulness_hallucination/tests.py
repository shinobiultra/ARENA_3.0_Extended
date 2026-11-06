"""Independent tiny CPU oracles; no solutions import or model access."""
import numpy as np


def done(name):
    print(f'All tests in {name} passed!')


def test_center_from_train(fn):
    center, result = fn(np.array([[1., 2.], [3., 6.]]), np.array([[9., 10.]]))
    np.testing.assert_allclose(center, [2., 4.])
    np.testing.assert_allclose(result, [[7., 6.]])
    done('test_center_from_train')


def test_mass_mean(fn):
    x = np.array([[1., 3.], [3., 1.], [-1., 2.], [-3., 0.]])
    np.testing.assert_allclose(fn(x, np.array([1, 1, 0, 0])), [4., 1.])
    np.testing.assert_allclose(fn(x[::-1], np.array([0, 0, 1, 1])), [4., 1.])
    done('test_mass_mean')


def test_class_covariance(fn):
    x = np.array([[1., 3.], [3., 1.], [-1., 2.], [-3., 0.]])
    # Residuals [(-1,1),(1,-1),(1,1),(-1,-1)] give identity with ddof=0.
    np.testing.assert_allclose(fn(x, np.array([1, 1, 0, 0])), np.eye(2))
    done('test_class_covariance')


def test_iid_direction(fn):
    covariance = np.diag([4., 1.])
    np.testing.assert_allclose(fn(np.array([2., 3.]), covariance, 0.), [.5, 3.])
    np.testing.assert_allclose(fn(np.array([2., 3.]), covariance, 1.), [.4, 1.5])
    done('test_iid_direction')


def test_logistic_gradient(fn):
    x = np.array([[1., 2.], [-2., 1.], [3., -1.]])
    y = np.array([1., 0., 1.])
    w = np.array([.2, -.3])
    def loss(v):
        z = x @ v
        return np.mean(np.logaddexp(0., z) - y * z) + .07 * (v @ v) / 2
    eps = 1e-6
    expected = [(loss(w + eps * np.eye(2)[i]) - loss(w - eps * np.eye(2)[i])) / (2*eps) for i in range(2)]
    np.testing.assert_allclose(fn(x, y, w, .07), expected, atol=1e-9)
    done('test_logistic_gradient')


def test_fit_logistic(fn):
    x = np.array([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    w = fn(x, np.array([0, 0, 1, 1]), steps=100, lr=.2, penalty=.1)
    assert w[0] > 1 and abs(w[1]) < 1e-12
    assert np.all((x @ w > 0) == [False, False, True, True])
    done('test_fit_logistic')


def test_balanced_accuracy(fn):
    assert fn(np.array([1, 0, 0, 0]), np.array([1, 1, 0, 0])) == .75
    assert fn(np.array([0, 0, 0, 0]), np.array([1, 0, 0, 0])) == .5
    done('test_balanced_accuracy')


def test_permute_labels(fn):
    y = np.array([1, 0, 1, 0])
    np.testing.assert_array_equal(fn(y, np.array([1, 3, 0, 2])), [0, 0, 1, 1])
    try:
        fn(y, np.array([0, 0, 1, 2]))
    except ValueError:
        pass
    else:
        raise AssertionError('A repeated index is not a label permutation.')
    done('test_permute_labels')


def test_steer(fn):
    x = np.array([[1., 2.], [3., 4.]])
    before = x.copy()
    np.testing.assert_allclose(fn(x, np.array([3., 4.]), 2.), x + [1.2, 1.6])
    np.testing.assert_array_equal(x, before)
    try:
        fn(x, np.zeros(2), 1.)
    except ValueError:
        pass
    else:
        raise AssertionError('A zero direction must fail, not silently normalize.')
    done('test_steer')


def test_answer_probability(fn):
    np.testing.assert_allclose(fn(np.array([0., np.log(3), -np.log(3)])), [.5, .75, .25])
    assert np.all(np.isfinite(fn(np.array([-1000., 1000.]))))
    done('test_answer_probability')


def test_selective_rates(fn):
    actual = fn(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool), np.array([1, 0, 0, 0], bool))
    np.testing.assert_allclose(actual, [.5, .5, .5, .5])
    all_refuse = fn(np.array([1, 0], bool), np.zeros(2, bool), np.zeros(2, bool))
    assert np.isnan(all_refuse[1]), 'No emitted answers means undefined selective accuracy.'
    done('test_selective_rates')


def test_head_shift(fn):
    z = np.zeros((2, 3, 2))
    directions = np.array([[1., 0.], [0., 2.], [3., 4.]])
    actual = fn(z, directions, np.array([1., 2., 5.]), [0, 2], .5)
    expected = np.array([[[.5, 0.], [0., 0.], [1.5, 2.]]]*2)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(z, np.zeros_like(z))
    done('test_head_shift')


def test_transfer_matrix(fn):
    probes = np.array([[1., 0.], [0., 1.]])
    datasets = [(np.array([[1., -1.], [-1., 1.]]), np.array([1, 0])),
                (np.array([[-1., 1.], [1., -1.]]), np.array([1, 0]))]
    np.testing.assert_allclose(fn(probes, datasets), [[1., 0.], [0., 1.]])
    done('test_transfer_matrix')
