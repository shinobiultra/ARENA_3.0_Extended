# Project Python maintenance — 2026-09-07 UTC

The user requested Git and package updates and confirmed **project Python packages only**.
The existing course work was committed and pushed as `ff6d06e323b54c32c6b9df2f9f3bc369ecd48e19` before updating dependencies.
That commit preserves the exact lockfile used by the saved scientific results. This is a
checkpoint of unfinished course development, not a complete-course release.

`uv lock --upgrade` refreshed all resolvable dependencies within the existing constraints;
`BNB_CUDA_VERSION=130 uv sync --locked --inexact` installed them while retaining separately
installed local tools. 110 installed packages changed version and 9 were added. The
cross-platform lock contains 111 version changes, 10 additions and 7 removals. Protobuf
moved from 7.35.1 to 6.33.6 as required by the selected dependency graph; this is not 110
upgrades. Explicit pins, including NumPy 1.26.4, Torch 2.12.1+cu132, Captum 0.9.0, SHAP
0.49.1 and shapiq 1.5.2, remain unchanged. No system package or Python interpreter update
was performed. uv was 0.12.10 and the active interpreter was Python 3.14.2.

The hand-maintained `requirements.txt` had an obsolete Transformers <5 constraint and
an unpinned bitsandbytes entry. Both now match their `pyproject.toml` constraints.

## Validation and limits

- `uv lock --check`: pass. `uv pip check`: all 349 installed packages compatible.
- `uv sync --locked --inexact --dry-run`: no environment changes needed.
- 28 package imports: pass. Tiny GPT-2 with LoRA forward/backward, Diffusers UNet and
  scheduler, Matplotlib rendering, Plotly HTML, notebook validation and live Jupyter
  kernel execution: pass. CPU only; no model downloads or GPU work.
- Existing section 16.17 method/reference/study tests: **31 passed**.
- Root CPU suite: **460 passed, 9 failed**. Six failures existed in the course checkpoint:
  archive hygiene, missing legacy smoke fixtures, incomplete provenance headings,
  missing 16.17 ladder registration, and source formatting. The three additional failures
  are evidence freshness checks caused by changing `uv.lock`, affecting 16.17's course
  surface, report input hashes and learner contract.
- A detached checkout of the checkpoint, using the updated interpreter environment,
  reproduced those six existing gate failures. It also exposed an ignored diffusion
  fixture manifest absent from Git. After copying that existing fixture set, all three
  source-freshness checks pass there (`baseline_freshness_tests.log`). This comparison
  isolates file/evidence regressions; it is not a scientific rerun under the old packages.
- Original ARENA preservation: pass before the course checkpoint (1948 paths) and after
  maintenance. No original notebook, page or asset was changed.

The archived native runs remain evidence for their recorded environment. They have **not**
been relabeled, rehashed or regenerated to pretend they used the new dependencies.
The next course continuation must perform the required fresh execution before treating
those receipts as current. Full scientific notebook reproduction, GPU compatibility and
paid external APIs were not tested during this package maintenance.

## Git fixture correction

The broad `*data` ignore rule hid the existing diffusion-text fixture directory. A narrow
exception now includes its manifest and five gzip payloads, about 1.3 MB total. Every
compressed and decompressed hash matches the original manifest. These are the previously
prepared TinyStories lesson fixtures, with source/license provenance, not regenerated
training data. No large model weights or external cache directories were added.

## Reproduce the prior environment

Use a separate checkout and environment; keep current development intact:

```sh
git worktree add --detach /tmp/arena-before-package-update ff6d06e323b54c32c6b9df2f9f3bc369ecd48e19
cd /tmp/arena-before-package-update
BNB_CUDA_VERSION=130 uv sync --locked
```

The checkpoint predates the diffusion fixture packaging correction. Apply only that
fixture correction from the maintenance commit if running its diffusion evidence gate;
keep the checkpoint's `pyproject.toml` and `uv.lock`. Existing external model/data caches
are still required for full course reproductions.

Per-package versions, smoke receipts and test logs are in this directory. Trailing
whitespace was stripped from the copied console logs; the local originals are intact. The complete
local package inventory and pre-update lock backup are at `/home/bitwise/.cache/arena_extended/maintenance/2026-09-07T121732Z`.
The lock-and-sync procedure follows [uv's official documentation](https://docs.astral.sh/uv/concepts/projects/sync/).
