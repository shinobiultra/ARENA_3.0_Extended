# Original ARENA Preservation Contract

The extension is append-only with respect to the original ARENA course surface.
Original chapter exercises, instruction pages, and assets should not be edited
unless a new extension section is being added under an explicit extension path.

The preservation audit allows the following global compatibility files because
they are needed to register and verify the extension without changing original
lesson content.

| Path | Allowed reason |
| --- | --- |
| `.github/workflows/extension-quality.yml` | Real CI gates for extension audits and GPU reports. |
| `.gitignore` | Cache, model-weight, and generated-artifact hygiene for the extension. |
| `.python-version` | uv-managed Python version pin for the CUDA 13 environment. |
| `Extension-Roadmap.md` | User-authored extension specification. |
| `install.sh` | Original installer redirected to the pinned original requirements split. |
| `README.md` | Entrypoint documentation for the extended course. |
| `infrastructure/core/config.yaml` | Append-only registration of extension chapters and sections. |
| `pyproject.toml` | Project metadata, pytest import mode, and CI marker registration. |
| `requirements-ci-cpu.txt` | Minimal hosted-CI dependencies for audit tests. |
| `requirements-legacy-rl.txt` | Isolated legacy RL dependency stack kept out of the CUDA 13 env. |
| `requirements-original.txt` | Exact upstream requirements snapshot for the original installer. |
| `requirements.txt` | Default uv CUDA 13 dependency stack. |
| `uv.lock` | uv resolution metadata for the managed environment. |

Pinned original base commit:

```text
f9f034bdb5b8748f44e8b4533b5c5bea68dc8bc0
```

Environment split:

| Environment | Purpose | Dependency file |
| --- | --- | --- |
| Original ARENA installer | Preserve the upstream Python 3.11 install target used by `install.sh`. | `requirements-original.txt` |
| Extension default uv env | Python 3.14 + PyTorch CUDA 13.2 frontier-model labs. | `requirements.txt` |
| Legacy RL add-on | Optional Chapter 2 JAX/Brax/EnvPool stack that conflicts with the CUDA 13 env. | `requirements-legacy-rl.txt` |
| Hosted audit CI | Metadata/course-surface audits without model downloads. | `requirements-ci-cpu.txt` |

The legacy Chapter 2 RL dependency split is a compatibility boundary, not a
content rewrite: original RL source stays in place, the original installer uses
`requirements-original.txt`, and the older JAX/Brax/EnvPool stack is available
separately for users who want that path outside the default Python 3.14 + CUDA
13 environment.

Release-preservation check:

```bash
uv run python scripts/audit_original_arena_preservation.py
```

That audit compares original chapter files and original config entries against
the pinned upstream commit, while permitting only documented extension paths and
the compatibility files listed above.
