# [6.3] Transcoders and Attribution Graphs

The learner-facing result is an exact colored-shape ReLU MLP model organism.
Students implement its transcoder replacement, reconstruction decomposition,
signed feature-edge attribution, graph extraction, and causal validation in the
notebook. The known `red square` graph has three active feature nodes and ten
signed edges, including the inhibitory encoder-bias paths.

The signature result compares the recovered graph with a fixed same-size random
graph, shuffled feature identities, shuffled edge targets, and a decoder-norm
reconstruction-only baseline. The paired solution notebook contains the full
implementation in visible cells and runs entirely on CPU.

`solutions.py` and `tests.py` provide the reference implementation and semantic
tests. `_build_salvage_notebooks.py` regenerates both notebooks and the instruction
page from that source. The two PNGs in `instructions/assets` are deterministic
renders of the exact graph and intervention result.

The pinned TransformerLens `gelu-1l` CUDA preflight remains supporting evidence
in `verification_report.json`; it is not the lesson or the signature result.
The parent verification workflow owns rerunning that path.
