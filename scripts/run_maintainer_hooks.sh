#!/bin/bash
# Run the maintainer hooks (.pre-commit-config.yaml) the way CI does: on the files changed since a base.
#
#   scripts/run_maintainer_hooks.sh [<base ref>]   # default: where this branch left origin/integration/extension
#
# Needs the project's environment (`uv sync --locked`) so ty can resolve imports.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${1:-$(git merge-base origin/integration/extension HEAD)}
exec uvx prek@0.5.2 run --show-diff-on-failure --from-ref "$BASE" --to-ref HEAD
