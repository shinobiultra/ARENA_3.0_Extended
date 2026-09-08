# GPU Verification Policy

GPU utilization is not itself evidence. A correct verification run can leave the
GPU mostly idle when the contract is a small exact-value check. Conversely,
keeping the GPU busy with synthetic work does not prove an interpretability
claim.

The course uses these categories in `verification_report.json`:

| category | meaning |
| --- | --- |
| `cuda_section_metric` | The notebook ran a CUDA path and returned a section-specific metric. |
| `cuda_environment_or_budget` | CUDA was checked for environment or memory-budget evidence, but no section-specific metric was produced. |
| `cpu_or_budget_metric` | The report has section-specific metrics, but not from a CUDA execution path. |
| `placeholder_only` | The `run_gpu_test` hook exists but only records that a smoke test exists. |
| `missing` | No usable GPU result was recorded. |

For a final "fully GPU tested" claim, roadmap sections should reach
`cuda_section_metric` unless the artifact lock explicitly states that the
exercise is CPU-only exact arithmetic. Real-model notebooks need more than this:
exact model revisions, dataset revisions, seeds, dtype, measured peak VRAM,
runtime, baseline metrics, negative controls, OOD checks where applicable, and
saved outputs.

Run:

```bash
BNB_CUDA_VERSION=130 uv run python scripts/audit_gpu_verification_reports.py
```

Strict mode fails until every report has section-specific CUDA evidence:

```bash
BNB_CUDA_VERSION=130 uv run python scripts/audit_gpu_verification_reports.py --require-cuda-section-metrics
```
