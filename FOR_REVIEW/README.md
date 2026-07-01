# FOR_REVIEW Bundle

Open `FOR_REVIEW.pdf` first. The LaTeX source is `FOR_REVIEW.tex`, and the
compact machine-readable evidence table is `tables/diffusiongemma_readiness.csv`.

This packet records the current DiffusionGemma review status for the 24GB RTX
5090 laptop target. The full Google BF16 checkpoint is marked `deferred`
because the weight set alone is larger than local VRAM. The quantized NVIDIA
NVFP4 checkpoint is the accepted 24GB path: its proof artifact records real
generation in an isolated vLLM 0.24.0 environment, while the main uv environment
remains torch 2.12.1+cu132 without vLLM.
