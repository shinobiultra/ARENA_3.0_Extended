"""Generate the docs/evidence IOI path-patching verification report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_gt2_ioi_path_patching_report import REPORT_PATH, generate_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-vram-gb", type=float, default=24.0)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    accepted = generate_report(max_vram_gb=args.max_vram_gb, report_path=args.report_path)
    print(f"Wrote {args.report_path.relative_to(ROOT)}")
    print("GT2_IOI_PATH_PATCHING=PASS" if accepted else "GT2_IOI_PATH_PATCHING=FAIL")
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
