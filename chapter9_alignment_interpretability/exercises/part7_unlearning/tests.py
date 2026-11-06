"""Small independent loss, gradient and memory oracles; imports no reference solution."""
import numpy as np


def done(name):
    print(f'All tests in {name} passed!')


def finite_gradient(loss, x):
    result = np.zeros_like(x, dtype=float)
    for index in np.ndindex(x.shape):
        delta = np.zeros_like(x, dtype=float)
        delta[index] = 1e-6
        result[index] = (loss(x+delta)-loss(x-delta))/(2e-6)
    return result


def test_log_probability(fn):
    np.testing.assert_allclose(fn(np.array([0., np.log(3), -np.log(3)]), np.array([0, 1, 0])), np.log([.5, .75, .75]))
    assert np.all(np.isfinite(fn(np.array([-1000., 1000.]), np.array([1, 0]))))
    done('test_log_probability')


def test_memory_gradient(fn):
    h = np.array([[.2, -.3], [-.1, .7]])
    y = np.array([1, 0]); sign = 2*y-1
    expected = finite_gradient(lambda v: np.logaddexp(0., -sign[:, None]*v).sum()/len(y), h)
    np.testing.assert_allclose(fn(h, y), expected, atol=1e-9)
    done('test_memory_gradient')


def test_train_memory(fn):
    result = fn(np.array([1, 0]), steps=1, lr=1.)
    np.testing.assert_allclose(result, [[.25, .25], [-.25, -.25]])
    result = fn(np.array([1, 0]), steps=20, lr=1.)
    assert np.all(result[0] > 1) and np.all(result[1] < -1)
    done('test_train_memory')


def test_masked_mean(fn):
    assert fn(np.array([2., 100., 4.]), np.array([1, 0, 1], bool)) == 3.
    try:
        fn(np.ones(2), np.zeros(2, bool))
    except ValueError:
        pass
    else:
        raise AssertionError('An empty cohort has no mean.')
    done('test_masked_mean')


def test_ga_head_step(fn):
    h = np.array([[1., .2], [-.5, 1.], [.3, -.7], [-1., .5]])
    y = np.array([1, 0, 1, 0]); sign = 2*y-1
    f = np.array([1, 1, 0, 0], bool); w = np.array([.4, -.2])
    def loss(v):
        target_logp = -np.logaddexp(0., -sign*(h@v))
        return target_logp[f].mean() - .3*target_logp[~f].mean()
    expected = w - .1*finite_gradient(loss, w)
    np.testing.assert_allclose(fn(h, y, f, w, .1, .3), expected, atol=1e-9)
    done('test_ga_head_step')


def test_npo_loss(fn):
    assert abs(fn(np.array([-.4, -.8]), np.array([-.4, -.8]), .5) - 4*np.log(2)) < 1e-12
    assert fn(np.array([-4.]), np.array([-.4]), .5) < fn(np.array([-.4]), np.array([-.4]), .5)
    done('test_npo_loss')


def test_npo_head_step(fn):
    h = np.array([[1., .2], [-.5, 1.], [.3, -.7], [-1., .5]])
    y = np.array([1, 0, 1, 0]); sign = 2*y-1
    f = np.array([1, 1, 0, 0], bool); w = np.array([.4, -.2]); ref = np.array([-.2, -.5])
    def loss(v):
        lp = -np.logaddexp(0., -sign*(h@v))
        return (2/.7)*np.logaddexp(0., .7*(lp[f]-ref)).mean() - .3*lp[~f].mean()
    np.testing.assert_allclose(fn(h, y, f, w, ref, .7, .1, .3), w-.1*finite_gradient(loss, w), atol=1e-9)
    done('test_npo_head_step')


def test_rmu_objective(fn):
    current = np.array([[1., 2.], [4., 5.]])
    reference = np.array([[9., 9.], [3., 4.]])
    assert fn(current, reference, np.array([1, 0], bool), np.array([0., 0.]), .5) == 6.
    done('test_rmu_objective')


def test_rmu_memory_step(fn):
    h = np.array([[1., 2.], [3., 4.], [5., 6.]])
    reference = h.copy(); reference[2] = [4., 5.]
    f = np.array([1, 1, 0], bool); target = np.array([.6, .8])
    expected = np.array([[.6, .8], [.6, .8], [4., 5.]])
    np.testing.assert_allclose(fn(h, reference, f, target, 1., .5), expected)
    done('test_rmu_memory_step')


def test_sae_encode(fn):
    h = np.array([[2., -3.], [-1., 4.]])
    encoder = np.array([[1., -1., 0., 0.], [0., 0., 1., -1.]])
    np.testing.assert_allclose(fn(h, encoder), [[2., 0., 0., 3.], [0., 1., 4., 0.]])
    done('test_sae_encode')


def test_ablate_features(fn):
    features = np.array([[2., 0., 0., 3.]])
    decoder = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    before = features.copy()
    np.testing.assert_allclose(fn(features, decoder, [0, 1]), [[0., -3.]])
    np.testing.assert_array_equal(features, before)
    np.testing.assert_allclose(fn(features, decoder, []), [[2., -3.]])
    done('test_ablate_features')


def test_retrieval_accuracy(fn):
    assert fn(np.array([0., 1., -1., 0.]), np.array([1, 1, 0, 0]), np.ones(4, bool)) == .75
    assert fn(np.array([-1., 1., 9., -9.]), np.array([0, 1, 0, 1]), np.array([1, 1, 0, 0], bool)) == 1.
    done('test_retrieval_accuracy')


def test_probe_memory(fn):
    h = np.array([[3., -2.], [4., 5.]])
    np.testing.assert_allclose(fn(h, np.array([0., 1.])), [-2., 5.])
    done('test_probe_memory')


def test_relearn_head(fn):
    h = np.array([[1., 1.], [-1., -1.], [20., 20.]])
    y = np.array([1, 0, 0]); f = np.array([1, 1, 0], bool)
    np.testing.assert_allclose(fn(h, y, f, np.zeros(2), 1, 1.), [.5, .5])
    done('test_relearn_head')


def test_relearn_memory(fn):
    h = np.zeros((3, 2)); y = np.array([1, 0, 1]); f = np.array([1, 1, 0], bool)
    np.testing.assert_allclose(fn(h, y, f, 1, 1.), [[.25, .25], [-.25, -.25], [0., 0.]])
    done('test_relearn_memory')
