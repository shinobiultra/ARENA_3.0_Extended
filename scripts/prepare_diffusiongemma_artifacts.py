"""Prepare exact DiffusionGemma artifacts for the 5.5 real-model path.

The roadmap target is quantized local DiffusionGemma inference on a 24GB GPU.
This script therefore defaults to the NVIDIA NVFP4 checkpoint, while still
allowing maintainers to inspect or download the larger Google BF16 checkpoint.
It never substitutes stand-ins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_ext.gated_artifacts import (  # noqa: E402
    DIFFUSIONGEMMA_26B_A4B_IT,
    NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4,
    HFGatedArtifactSpec,
    download_required_artifact,
    hf_model_artifact_access_report,
)


ARTIFACTS: dict[str, HFGatedArtifactSpec] = {
    "bf16": DIFFUSIONGEMMA_26B_A4B_IT,
    "nvfp4": NVIDIA_DIFFUSIONGEMMA_26B_A4B_IT_NVFP4,
}


def _selected_specs(name: str) -> list[HFGatedArtifactSpec]:
    if name == "all":
        return [ARTIFACTS["bf16"], ARTIFACTS["nvfp4"]]
    return [ARTIFACTS[name]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        choices=("nvfp4", "bf16", "all"),
        default="nvfp4",
        help=(
            "Artifact to inspect/download. Defaults to the 24GB-local NVFP4 "
            "runtime path; BF16 is larger and is mainly useful for metadata or "
            "CPU/offload experiments."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the selected exact pinned artifact(s) after HF auth/access is configured.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel Hugging Face download workers. Defaults to 1 because "
            "the two large NVFP4 shards have shown more stable resumable transfer with "
            "single-worker Xet downloads on this machine."
        ),
    )
    parser.add_argument(
        "--require-local",
        action="store_true",
        help="Exit nonzero unless every selected artifact is locally complete.",
    )
    parser.add_argument(
        "--require-downloadable",
        action="store_true",
        help="Exit nonzero unless every selected artifact is authenticated and remotely downloadable.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report instead of a compact text report.",
    )
    args = parser.parse_args()

    reports = []
    for spec in _selected_specs(args.artifact):
        report = {}
        if args.download:
            report["download_path"] = download_required_artifact(
                spec,
                max_workers=args.max_workers,
            )
        report.update(hf_model_artifact_access_report(spec))
        reports.append(report)

    if args.json:
        print(json.dumps({"artifacts": reports}, indent=2, sort_keys=True))
    else:
        for report in reports:
            print(
                f"{report['repo_id']} | local={report['local_ready_for_direct_loading']} | "
                f"remote={report['remote_download_ready']} | auth={report['authenticated']} | "
                f"cache={report['cache_dir']}"
            )
            if report.get("download_path"):
                print(f"  download_path={report['download_path']}")
            if not report["local_ready_for_direct_loading"]:
                print(f"  missing_local={report['missing_local_patterns']}")
            if not report["remote_download_ready"]:
                print(f"  missing_remote={report['missing_remote_patterns']}")
                if report["auth_error_type"]:
                    print(f"  auth_error_type={report['auth_error_type']}")
                if report["access_error_type"]:
                    print(f"  access_error_type={report['access_error_type']}")

    if args.require_local and not all(
        report["local_ready_for_direct_loading"] for report in reports
    ):
        raise SystemExit(1)
    if args.require_downloadable and not all(report["remote_download_ready"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
