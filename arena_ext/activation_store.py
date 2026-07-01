"""Disk-backed activation store for frontier-extension notebooks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch as t
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class ActivationRecord:
    """Metadata for one activation shard saved to disk."""

    name: str
    path: str
    shape: tuple[int, ...]
    dtype: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DiskActivationStore:
    """Append-only activation store using safetensors shards plus JSONL metadata."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._counter = self._next_counter()

    def _next_counter(self) -> int:
        if not self.index_path.exists():
            return 0
        count = 0
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def append(
        self,
        name: str,
        activation: t.Tensor,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ActivationRecord:
        """Save one activation tensor and append a metadata row."""

        if "/" in name or "\\" in name:
            raise ValueError("Activation names must not contain path separators.")

        tensor = activation.detach().cpu().contiguous()
        filename = f"{self._counter:06d}_{name}.safetensors"
        path = self.root / filename
        save_file({"activation": tensor}, path)

        record = ActivationRecord(
            name=name,
            path=filename,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            metadata=metadata or {},
        )
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__, sort_keys=True) + "\n")

        self._counter += 1
        return record

    def records(self) -> Iterator[ActivationRecord]:
        """Yield records in append order."""

        if not self.index_path.exists():
            return
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                data["shape"] = tuple(data["shape"])
                yield ActivationRecord(**data)

    def load(self, record: ActivationRecord | int) -> t.Tensor:
        """Load an activation by record or integer index."""

        if isinstance(record, int):
            records = list(self.records())
            try:
                record = records[record]
            except IndexError as exc:
                raise IndexError(f"Activation record index {record} is out of range.") from exc

        tensors = load_file(self.root / record.path)
        return tensors["activation"]

    def summary(self) -> dict[str, Any]:
        """Return a compact summary useful for notebook verification footers."""

        records = list(self.records())
        total_values = sum(math.prod(record.shape) for record in records)
        return {
            "root": str(self.root),
            "num_records": len(records),
            "total_values": total_values,
            "names": [record.name for record in records],
        }
