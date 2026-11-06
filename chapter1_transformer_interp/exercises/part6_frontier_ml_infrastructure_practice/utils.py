"""Small display helpers for [1.6] Local Frontier ML Infrastructure."""

from __future__ import annotations

from collections.abc import Mapping


def print_dict_table(title: str, values: Mapping[str, object]) -> None:
    """Print a simple two-column table without adding notebook dependencies."""

    print(title)
    width = max((len(str(key)) for key in values), default=0)
    for key, value in values.items():
        print(f"  {key:<{width}} : {value}")
