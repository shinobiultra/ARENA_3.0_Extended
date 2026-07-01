"""Prepare required gated Gemma-family artifacts.

This script never substitutes public stand-ins. It either verifies/downloads the
exact required Google artifacts or exits with the concrete access blocker.
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
    REQUIRED_GEMMA_FAMILY_ARTIFACTS,
    download_required_artifact,
    hf_model_artifact_access_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download exact gated artifacts after authenticated HF access is configured.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero unless every required gated artifact is locally or remotely ready.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report instead of a compact text report.",
    )
    args = parser.parse_args()

    reports = []
    if args.download:
        for spec in REQUIRED_GEMMA_FAMILY_ARTIFACTS:
            download_path = download_required_artifact(spec)
            report = hf_model_artifact_access_report(spec)
            report["download_path"] = download_path
            reports.append(report)
    else:
        reports = [
            hf_model_artifact_access_report(spec)
            for spec in REQUIRED_GEMMA_FAMILY_ARTIFACTS
        ]

    if args.json:
        print(json.dumps({"artifacts": reports}, indent=2, sort_keys=True))
    else:
        for report in reports:
            print(
                f"{report['repo_id']} | ready={report['ready_for_direct_loading']} | "
                f"local={report['local_ready_for_direct_loading']} | "
                f"remote={report['remote_download_ready']} | "
                f"auth={report['authenticated']}"
            )
            if not report["ready_for_direct_loading"]:
                print(f"  missing_local={report['missing_local_patterns']}")
                print(f"  missing_remote={report['missing_remote_patterns']}")
                if report["auth_error_type"]:
                    print(f"  auth_error_type={report['auth_error_type']}")
                if report["access_error_type"]:
                    print(f"  access_error_type={report['access_error_type']}")

    if args.require_ready and not all(report["ready_for_direct_loading"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
