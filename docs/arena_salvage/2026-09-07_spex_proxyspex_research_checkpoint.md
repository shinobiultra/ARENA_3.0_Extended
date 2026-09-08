# SPEX / ProxySPEX research checkpoint — September 7

The user asked to finish work already underway and stop within about 90 minutes.
This records the source research that had started when that instruction arrived.
No new notebook, native model study, estimator experiment, dependency installation,
scientific freeze or requirement promotion was performed for this topic.

The selected requirement is `shapley.spex_proxyspex` in
`docs/roadmap_requirement_registry.yml`. It remains **UNIMPLEMENTED**. The scope is
sparse interaction recovery, query complexity and an actual LLM comparison for
both SPEX and ProxySPEX. The completed pairwise sampling lesson 16.17 is a useful
prerequisite but does not fulfill this requirement.

## Sources already acquired

Cache root: `/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/references/`.
`sources.json` binds the downloaded paper bytes, author repository revision,
license, package metadata and every Python source file. Both PDFs have searchable
`pdftotext -layout` companions. The repository checkout is clean.

| Source | Local file / revision |
| --- | --- |
| [SPEX, ICML 2025](https://proceedings.mlr.press/v267/kang25a.html) | `spex_2025.pdf`; SHA256 `eb00ec4987ede00bfd3b7d105d9e22b2af3118ab0e1342e540ba7c224ff003d4` |
| [ProxySPEX, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/68dced1638d71d4d21598f79ae91262e-Abstract-Conference.html) | `proxyspex_2025.pdf`; SHA256 `777a115c72c383c9cde41f70ef4edae0d212350b3325ce877dea63ebcff392d8` |
| [Author implementation](https://github.com/basics-lab/spectral-explain) | `spectral-explain/`; commit `61400b5388b8ecbf18fe18c4d821ffe3e8a5f9c8`; declared package version `0.0.4` |

SPEX designs binary linear subsamples and shifts so that a small transform mixes
many original coefficients into bins. Shift signs encode a candidate coefficient's
feature support. BCH decoding and iterative message passing recover candidates
without enumerating every interaction. A faithful lesson must expose this recovery
mechanism, its query accounting, collisions and failure conditions; calling a sparse
transform utility is insufficient. Read Section 5 and Appendix A of the paper.

ProxySPEX trains boosted-tree proxies on masked model responses, extracts their
Fourier representation, selects important coefficients and optionally refines them
using regression chosen by cross-validation. Read Section 3 and Appendix A.2.
The learner must distinguish proxy prediction error from Fourier extraction error
and truncation/refinement error. A decision-tree fallback is not the paper's
boosted-tree experiment. Preserve explicit distinctions between Fourier coefficients
and converted Shapley interaction indices.

## Observed environment facts

The main repository environment has `shapiq==1.5.2`. Its public `SPEX` name is an
optional-dependency placeholder: `approximator/__init__.py` catches an import error
from `sparse_transform` and substitutes a constructor that raises `ImportError`.
An importable name therefore does not establish an executable SPEX reference.

`ProxySPEX` is present in `shapiq/approximator/proxy/proxyspex.py`. Its default
backend is LightGBM and its documented fallback is a bare `DecisionTreeRegressor`.
LightGBM, XGBoost, CatBoost, galois, bchlib and spectralexplain were absent in the
checked environment. The author package declares `sparse-transform` and `lightgbm`
dependencies. No package was installed during this research checkpoint.

Do not change the shared `uv.lock` casually: completed learner receipts bind it.
Use an isolated, pinned reference environment to establish actual author-package
execution if needed, then explicitly design how the future learner will obtain its
own required dependencies. Do not silently accept an optional-backend fallback.

## Concrete route after the next explicit resume

1. Read the original ARENA notebook pair/page and completed 16.17 pair/page before
   authoring. Write the normal learner plan, including one concrete question,
   meaningful student functions, controls, visible results and editable plays.
2. Establish actual reference execution and record dependency/source identities.
   Resolve the BCH/transform and boosted-tree backends without altering completed
   lesson inputs. Check the authors' masking and Fourier sign conventions.
3. Develop exact analytic tests first: transform/inverse, subsampling/shift identity,
   support decoding, bin collisions, sparse recovery and tree-spectrum reconstruction.
   Include additive, parity, dense-spectrum and misspecified-degree controls.
4. Choose and pin an actual language-model task, feature grouping, masking operator,
   scalar output and independent explanation/audit examples. No such choices have
   been made or evaluated yet. CPU work must respect the user's unrelated GPU jobs.
5. Before native outcomes, fix query budgets, seeds, held-out mask evaluation,
   baselines, sparse/dense controls, proxy selection, resource limits and evidence
   retention. Do not substitute an all-coalition table for a scalable query claim.
6. Execute the whole learner using its own adjacent answers; reproduce from source;
   inspect actual figures, equations, dropdowns and plays; publish and run the
   registered report only when those checks pass. Keep the joint requirement open
   until both selected methods and the LLM comparison are taught.

This is a research checkpoint, not a study design approval or a partly completed
SPEX implementation. The separate 16.16 independent publication-review gate is
unchanged. Authorized subagents remain unavailable on account limits; no model
switch, reset credit or fabricated independent review was used.
