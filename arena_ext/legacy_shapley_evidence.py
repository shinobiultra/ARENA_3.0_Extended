"""Strict numeric comparison for historical Data Shapley execution receipts.

The retired 16.7 ``inrun_report.compare/array`` contract survives here independently
of any retired lesson or report assembly. This module validates numeric equality;
it does not certify notebook execution, source hashes, split provenance, scientific
acceptance, or a current 16.9 publication receipt. Callers explicitly enumerate the
only positive, finite wall-clock leaves that may differ between two recordings.
All other values, tensor dtypes/shapes, dictionary order, and container types must
match exactly. No tolerance or blanket '*seconds' exclusion is applied.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Set
from typing import Any

import numpy as np

Path = tuple[str | int, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def array(value: Mapping[str, Any]) -> np.ndarray:
    """Decode a historical recorded tensor, rejecting malformed or nonfinite data."""
    _require(isinstance(value, dict) and value.get('tensor') is True, 'Expected a recorded tensor')
    _require(set(value) == {'tensor', 'dtype', 'shape', 'values'}, 'Invalid tensor schema')
    _require(isinstance(value['dtype'], str) and value['dtype'].startswith('torch.'), 'Invalid tensor dtype')
    shape = value['shape']
    _require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape), 'Invalid tensor shape')
    try:
        result = np.asarray(value['values'])
        finite = np.isfinite(result).all()
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid tensor values') from error
    _require(list(result.shape) == shape, 'Tensor shape differs from recorded shape')
    _require(bool(finite), 'Nonfinite tensor')
    return result


def compare(left: Any, right: Any, *, timing_paths: Set[Path]) -> tuple[dict[str, int | float], list[Path]]:
    """Compare recordings exactly, excluding only the complete explicit timing inventory."""
    paths = frozenset(timing_paths)
    counts = dict(tensors=0, tensor_elements=0, scalars=0, excluded_timing_fields=0, max_absolute_error=0.0)
    excluded: list[Path] = []

    def visit(a: Any, b: Any, path: Path = ()) -> None:
        if path in paths:
            _require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in (a, b)),
                     f'Invalid timing at {path}')
            excluded.append(path)
            counts['excluded_timing_fields'] += 1
            return
        _require(type(a) is type(b), f'Type changed at {path}')
        if isinstance(a, dict):
            _require(list(a) == list(b), f'Keys/order changed at {path}')
            if a.get('tensor') is True:
                x, y = array(a), array(b)
                _require(a['dtype'] == b['dtype'] and x.shape == y.shape and np.array_equal(x, y),
                         f'Tensor changed at {path}')
                counts['tensors'] += 1
                counts['tensor_elements'] += int(x.size)
            else:
                for key in a:
                    visit(a[key], b[key], path + (key,))
        elif isinstance(a, list):
            _require(len(a) == len(b), f'Length changed at {path}')
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, path + (index,))
        else:
            _require(not isinstance(a, float) or math.isfinite(a), f'Nonfinite scalar at {path}')
            _require(a == b, f'Scalar changed at {path}')
            counts['scalars'] += 1

    visit(left, right)
    _require(frozenset(excluded) == paths, 'Unexpected timing inventory')
    return counts, excluded
