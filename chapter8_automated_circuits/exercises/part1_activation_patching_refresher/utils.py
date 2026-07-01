"""Helpers for [8.1] Activation Patching Refresher."""

from __future__ import annotations

from collections.abc import Mapping


def print_report(title: str, report: Mapping[str, object]) -> None:
    print(title)
    width = max((len(str(key)) for key in report), default=0)
    for key, value in report.items():
        print(f"  {key:<{width}} : {value}")
