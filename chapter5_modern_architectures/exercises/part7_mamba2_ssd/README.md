# [5.7] Mamba-2 and Structured State-Space Duality

This section asks one question: when do the recurrent, full semiseparable-matrix,
and chunked SSD forms compute the same causal sequence transform, and does that
recurrent state causally matter on delayed recall?

## Learner path

Open `5.7_Mamba-2_and_SSD_exercises.ipynb`. The exercises build:

1. stable segment sums;
2. the semiseparable attention-dual matrix;
3. literal recurrent SSD;
4. the quadratic dual form;
5. chunked SSD with boundary states;
6. grouped multi-head parameters and a gated residual Mamba-2 block;
7. convolution and SSM caches for one-token decoding;
8. a trained delayed-recall sequence model and causal controls;
9. measured runtime and working-set scaling;
10. exact gradient parity, a tiny stacked cached LM, and parameter-matched
    causal-attention and Mamba-1 blocks;
11. strict mapping of all 219 tensors in the pinned Mamba-2 130M release, with
    25 hidden-state, logits, and top-token comparisons on two prompts.

The visible signature result is generated in the notebook. It is not read from
`verification_report.json`.

## Verification

The final CUDA cells require a fresh kernel with `TRITON_F32_DEFAULT=ieee`.
The notebook setup selects it before importing any Mamba kernels. Float32
tensor storage alone does not select IEEE dot products in Triton.

```bash
uv run python chapter5_modern_architectures/exercises/part7_mamba2_ssd/build_notebooks.py
uv run pytest -q chapter5_modern_architectures/exercises/part7_mamba2_ssd/tests.py
TRITON_F32_DEFAULT=ieee uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  chapter5_modern_architectures/exercises/part7_mamba2_ssd/5.7_Mamba-2_and_SSD_solutions.ipynb
```

The strict CUDA hooks are:

```python
run_gpu_test(max_vram_gb=24.0)
run_full_experiment(max_vram_gb=24.0)
```

They require CUDA, the installed official Mamba kernels, and the pinned
`state-spaces/mamba2-130m` checkpoint. They raise when those requirements are
absent; there is no CPU fallback or skipped-success path.

## Claim boundary

The section proves exact SSD operator and gradient equivalence, parity with the
official repository's minimal SSD Listing 1, full-versus-step cache parity for
both a block and a stacked LM, and a causal recurrent-state result on a
generated task. It also maps the complete released 130M checkpoint into an
independent learner implementation and verifies all 25 hidden states, logits,
and top tokens on CPU. The release hook additionally checks the official fused
operator and repeats full-model parity on CUDA.

The readable implementation does not reproduce the paper's optimized-kernel
speedup. The CUDA result establishes correctness under the recorded precision,
not the paper's training-throughput claim.

Edit `lesson.md` for prose and experiment cells, and `solutions.py` for the
canonical implementations. The builder expands those definitions into both
notebooks and the instruction page, with complete signatures in student stubs.
Tests parse every solution dropdown, check generated-source parity, and check
that copying the answers supplies every helper required by later cells.
