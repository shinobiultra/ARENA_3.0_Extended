#!/usr/bin/env python
"""Record a finished section in syllabus.yaml: `mark_written.py 10.1 "One-line description."`"""

import json
import sys
from pathlib import Path

import yaml

path = Path(__file__).resolve().parent / "syllabus.yaml"
number, description = sys.argv[1], sys.argv[2]
lines = path.read_text().splitlines()
start = next(i for i, line in enumerate(lines) if line.strip() == f'- number: "{number}"')
indent = " " * (lines[start].index("-") + 2)
end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("- number:") or (lines[i] and not lines[i].startswith(indent))), len(lines))
block = [l for l in lines[start:end] if not l.strip().startswith(("status:", "description:"))]
block += [f"{indent}status: written", f"{indent}description: {json.dumps(description)}"]
lines[start:end] = block
path.write_text("\n".join(lines) + "\n")
yaml.safe_load(path.read_text())
print(f"marked {number} as written")
