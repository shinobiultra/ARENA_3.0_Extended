import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = r"""## TLDR (if you're in a hurry!)

**Prerequisites:** [1.3.1] Linear Probes.

- **11.1 PCA, SVD and Geometry Controls** first: it sets up the controls the rest of the chapter relies on.
- Then: 11.2 (comparing representations across models), 11.3 (causal abstraction and DAS, the most important section for circuit work), 11.4 (concept erasure), 11.5 (CCS) and 11.6 (feature manifolds) in any order.

## Description of material

How are concepts laid out in a model's activation space, and how do you tell real structure from an artefact of your projection? This chapter covers the classic geometric tools (PCA, SVD, UMAP/t-SNE, CCA, Procrustes) together with the controls they need, then moves to causal methods: distributed alignment search for finding the subspace that implements a variable, concept erasure, unsupervised truth directions with CCS, and features that live on circles and manifolds rather than single directions."""

render_chapter_home(__file__, INTRO)
