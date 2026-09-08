# CI Gates

The extension uses real repository gates, split by cost and hardware. These
commands are wrappers around the same scripts used locally; they do not add
mock tests or placeholder smoke checks.

## Hosted CPU

Runs on every push and pull request:

```bash
.venv/bin/python scripts/ci_quality_gate.py --mode hosted-cpu
```

This lane checks whitespace, selected metadata-only pytest tests, original
ARENA preservation, course surface, ARENA-style depth, hard-exercise ladders,
report evidence contracts, artifact hygiene, and static GPU-report evidence.
It intentionally does not download model weights or regenerate CUDA reports.

## Self-Hosted GPU

Runs manually on a self-hosted runner labelled `self-hosted`, `linux`, `x64`,
and `gpu`:

```bash
.venv/bin/python scripts/ci_quality_gate.py --mode gpu
```

For a single changed section:

```bash
.venv/bin/python scripts/ci_quality_gate.py --mode gpu --section 8.5
```

This lane requires a real NVIDIA CUDA device, verifies PyTorch CUDA availability
with `nvidia-smi`, regenerates extension verification reports through
`scripts/run_extension_verification_reports.py`, and then reruns the strict GPU
and report-evidence audits.

## Release

The release lane runs the full GPU gate and then the final roadmap/strict
completion audits:

```bash
.venv/bin/python scripts/ci_quality_gate.py --mode release
```

It is expected to fail if a documented gated artifact is missing, if a CUDA
report cannot be regenerated, or if the final roadmap/strict-completion audits
find any remaining blocker. Do not bypass those failures with public stand-ins
or placeholder reports.
