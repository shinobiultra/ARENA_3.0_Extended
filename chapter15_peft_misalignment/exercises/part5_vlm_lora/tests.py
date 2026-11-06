"""Independent deterministic tests for rendered-image, question-conditioned VLM LoRA."""
import numpy as np


def _ok(name):
    print(f"All tests in `{name}` passed!")


def test_render_scene(fn):
    image = fn(0, 1, .3)
    assert image.shape == (8, 8, 3)
    np.testing.assert_allclose(image[1, 1], [1., 0., 0.])
    np.testing.assert_allclose(image[1, 6], [0., 0., 1.])
    np.testing.assert_allclose(image[6, 1], [0., .3, 0.])
    assert image.min() >= 0 and image.max() <= 1
    _ok('test_render_scene')


def test_patchify(fn):
    image = np.arange(8*8*3).reshape(1, 8, 8, 3)
    patches = fn(image)
    assert patches.shape == (1, 4, 4, 4, 3)
    expected = np.stack([image[:, :4, :4], image[:, :4, 4:], image[:, 4:, :4], image[:, 4:, 4:]], axis=1)
    np.testing.assert_array_equal(patches, expected)
    _ok('test_patchify')


def test_vision_features(fn):
    patches = np.zeros((2, 4, 4, 4, 3))
    patches[0, 0, :, :, 0] = 1
    patches[1, 2, :, :, 2] = .5
    plain, shifted = fn(patches, False), fn(patches, True)
    assert plain.shape == (2, 4, 3)
    np.testing.assert_array_equal(plain[0, 0], [1, 0, 0])
    np.testing.assert_array_equal(shifted[0, 0], [0, 0, 1])
    np.testing.assert_array_equal(shifted[1, 2], [.5, 0, 0])
    # Vary every spatial position: a first-pixel sampler must fail this oracle.
    varied = np.arange(2 * 4 * 4 * 4 * 3, dtype=np.float64).reshape(2, 4, 4, 4, 3) / 384.
    expected = np.empty((2, 4, 3))
    for batch in range(2):
        for patch in range(4):
            for channel in range(3):
                expected[batch, patch, channel] = sum(
                    float(varied[batch, patch, row, column, channel])
                    for row in range(4) for column in range(4)
                ) / 16.
    np.testing.assert_allclose(fn(varied, False), expected)
    np.testing.assert_allclose(fn(varied, True), expected[..., [2, 1, 0]])
    _ok('test_vision_features')


def test_lora_update(fn):
    A = np.array([[1., -2., 3.]])
    B = np.array([[2.], [-1.]])
    np.testing.assert_allclose(fn(A, B, .5), [[1., -2., 3.], [-.5, 1., -1.5]])
    A2 = np.eye(2)
    np.testing.assert_allclose(fn(A2, np.eye(2), 1.), np.eye(2)/2)
    _ok('test_lora_update')


def test_project_tokens(fn):
    features = np.array([[[1., 2., 3.], [0., 0., 1.]]])
    P = np.array([[1., 0., 0.], [0., 1., -1.]])
    D = np.array([[0., 0., 1.], [0., 0., 0.]])
    np.testing.assert_allclose(fn(features, P, D), [[[4., -1.], [1., -1.]]])
    np.testing.assert_allclose(fn(features, P, np.zeros_like(P)), [[[1., -1.], [0., -1.]]])
    _ok('test_project_tokens')


def test_query_attention(fn):
    queries = np.eye(2)
    keys = np.array([[1., 0.], [0., 1.], [0., 0.]])
    result = fn(queries, keys, np.log(2.))
    np.testing.assert_allclose(result, [[.5, .25, .25], [.25, .5, .25]])
    np.testing.assert_allclose(result.sum(axis=1), np.ones(2))
    huge = fn(queries, keys, 1000.)
    assert np.isfinite(huge).all(), 'Subtract the per-row maximum before exp.'
    np.testing.assert_allclose(huge[:, :2], np.eye(2))
    _ok('test_query_attention')


def test_decode_answers(fn):
    tokens = np.array([[[1., 0.], [0., 2.]], [[2., 0.], [0., 3.]]])
    attention = np.array([[.25, .75], [1., 0.]])
    text = np.array([[.5, 0.], [0., 1.]])
    language = np.array([[1., 2.], [0., -1.]])
    unembed = np.array([[1., 0.], [0., 1.], [1., -1.]])
    # Independent per-example loop with literal weighted token means.
    expected = []
    for b in range(2):
        h = text[b].copy()
        for p in range(2):
            h += attention[b, p] * tokens[b, p]
        expected.append(unembed @ (language @ h))
    np.testing.assert_allclose(fn(tokens, attention, text, language, unembed), expected)
    _ok('test_decode_answers')


def test_accuracy(fn):
    logits = np.array([[2., 1.], [1., 2.], [0., 0.], [3., 0.]])
    np.testing.assert_allclose(fn(logits, np.array([0, 1, 1, 0])), .75)
    _ok('test_accuracy')


def test_answer_margin(fn):
    logits = np.array([[4., 1.], [2., 5.], [-1., 3.]])
    np.testing.assert_allclose(fn(logits, np.array([0, 1, 0])), [3., 3., -4.])
    _ok('test_answer_margin')


def test_replace_tokens(fn):
    receiver = np.zeros((2, 4, 3))
    donor = np.arange(24.).reshape(2, 4, 3)
    result = fn(receiver, donor, np.array([0, 1]))
    np.testing.assert_array_equal(result[0, 0], donor[0, 0])
    np.testing.assert_array_equal(result[1, 1], donor[1, 1])
    assert np.count_nonzero(result[0, 1:]) == 0
    assert np.count_nonzero(result[1, [0, 2, 3]]) == 0
    np.testing.assert_array_equal(receiver, np.zeros_like(receiver))
    _ok('test_replace_tokens')


def test_margin_recovery(fn):
    np.testing.assert_allclose(fn(np.array([3., 4.]), np.array([-1., 0.]), np.array([1., 4.])), [.5, 1.])
    result = fn(np.array([1.]), np.array([1.]), np.array([2.]))
    assert np.isnan(result[0]), 'Zero clean/corrupt gap is undefined, not successful recovery.'
    _ok('test_margin_recovery')


def test_random_rank_one(fn):
    D = np.zeros((4, 3)); D[0, 0] = 2.
    result = fn(D, 31)
    assert result.shape == D.shape
    assert np.linalg.matrix_rank(result) == 1
    np.testing.assert_allclose(np.linalg.norm(result), 2.)
    np.testing.assert_array_equal(result, fn(D, 31))
    np.testing.assert_array_equal(fn(np.zeros_like(D), 31), np.zeros_like(D))
    _ok('test_random_rank_one')


def test_adapter_parameters(fn):
    assert fn(3, 4, 1) == 7
    assert fn(4, 4, 1) == 8
    assert fn(3, 4, 2) == 14
    _ok('test_adapter_parameters')


def test_retained_energy(fn):
    np.testing.assert_allclose(fn(np.diag([3., 4., 0.]), 1), .64)
    np.testing.assert_allclose(fn(np.diag([3., 4., 0.]), 2), 1.)
    np.testing.assert_allclose(fn(np.zeros((4, 3)), 1), 0.)
    _ok('test_retained_energy')
