"""Finalize synthetic/historical receipts without a process or model runner.

Extracted from part18_spex/run_native.py; see PROVENANCE.json.
"""

import json
from pathlib import Path

def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def _file_inventory(directory):
    """Logical file bytes, including sparse files; never follow an output symlink."""
    directory = Path(directory)
    records = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Output must not contain a symlink: {path}")
        if path.is_file():
            records[path.relative_to(directory).as_posix()] = path.stat().st_size
    return records

def _finalize_receipt(output, record, cap):
    """Include the receipt itself and converge its self-reported byte inventory."""
    target = Path(output) / "execution.json"
    for _ in range(8):
        _write_json(target, record)
        files = _file_inventory(output)
        size = sum(files.values())
        stable = record.get("output_files") == files and record.get("new_output_bytes") == size
        record["output_files"], record["new_output_bytes"] = files, size
        if size > cap:
            if "output_bytes" not in record["resource_violations"]:
                record["resource_violations"].append("output_bytes")
                stable = False
            record["status"] = "FAIL"
        if stable:
            return
    _write_json(target, record)
