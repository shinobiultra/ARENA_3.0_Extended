"""Load only hash-checked components of the pinned MIT-licensed author code."""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import types
import urllib.request

import torch as t
from chapter11_representation_geometry.exercises.part4_concept_erasure import erasure_methods as reference

REVISION = '9f51753821316a1edacf78b52b464ab26d40e60a'
HASHES = {
    'LICENSE': '956bd01010229da10d4a0e8c9ab848bdf55050789174dba82b2c1250559180a0',
    'concept_erasure/caching.py': '4bdf3865c0aebd70cf241642e4f4f068625687f82e81c7f2f1906bfc76a9992a',
    'concept_erasure/leace.py': 'e4145ae6a1ae6671b3eb68fe1df7dbbbe3ea06eb077dfac8bf169ed4c908b463',
    'concept_erasure/shrinkage.py': '05f3269524e4f612691605325407e72a56a1305678d1b9cee56b1177d3335988',
}


def load_author():
    directory = Path.home() / '.cache/arena_extended/erasure/author_source'
    for name, digest in HASHES.items():
        path = directory / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://raw.githubusercontent.com/EleutherAI/concept-erasure/{REVISION}/{name}'
            path.write_bytes(urllib.request.urlopen(url).read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned author file differs: ' + name)
    package = types.ModuleType('arena_leace_author')
    package.__path__ = [str(directory / 'concept_erasure')]
    sys.modules[package.__name__] = package
    return importlib.import_module('arena_leace_author.leace').LeaceEraser


def compare_reference(methods=reference):
    AuthorEraser = load_author()
    generator = t.Generator().manual_seed(613)
    outcomes = []
    for name, width, latent in [('full_rank', 7, 7), ('singular', 11, 5), ('translated', 6, 6)]:
        labels = t.arange(90) % 3
        z = t.nn.functional.one_hot(labels, 3).double()
        x = (t.randn(90, latent, generator=generator, dtype=t.float64)
             + z @ t.randn(3, latent, generator=generator, dtype=t.float64))
        x = x @ t.randn(latent, width, generator=generator, dtype=t.float64)
        if name == 'translated':
            x += t.randn(width, generator=generator, dtype=t.float64) * 5
        author = AuthorEraser.fit(x, z, shrinkage=False, constrain_cov_trace=False, svd_tol=1e-10)
        own = methods.fit_leace(x, z)
        actual = methods.apply_eraser(x, own)
        expected = author(x)
        t.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
        own_matrix = t.eye(width, dtype=x.dtype) - own['left'] @ own['right']
        t.testing.assert_close(own_matrix, author.P, atol=1e-8, rtol=1e-8)
        outcomes.append(dict(case=name, rows=len(x), width=width, latent_rank=latent,
                             max_output_error=float((actual - expected).abs().max()),
                             max_matrix_error=float((own_matrix - author.P).abs().max())))
    print('Pinned author parity passed for full-rank, singular and translated data.')
    return dict(revision=REVISION, files_sha256=HASHES, author_settings=dict(shrinkage=False,
                constrain_cov_trace=False, svd_tol=1e-10), cases=outcomes)


if __name__ == '__main__':
    t.set_num_threads(4)
    print(json.dumps(compare_reference(), indent=2))
