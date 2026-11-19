import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    "How are concepts laid out in a model's activation space, and how do you tell real structure from an artefact "
    'of your projection? This chapter covers the classic geometric tools (PCA, SVD, UMAP/t-SNE, CCA, Procrustes) '
    'together with the controls they need, then moves to causal methods: distributed alignment search for finding '
    'the subspace that implements a variable, concept erasure, unsupervised truth directions with CCS, and features'
    ' that live on circles and manifolds rather than single directions.'
)

render_chapter_home(__file__, INTRO)
