#!/bin/bash
# Execute an extension section on the Julia worker job, and bring the executed notebooks back.
#
#   infrastructure/extension/cluster/remote_build.sh 8.4
#
# 1. syncs source files (masters, tests.py, utils.py, infrastructure, small data) to the cluster checkout
# 2. syncs Julia's environment with the laptop's uv.lock; stops here if that fails (env_sync.py)
# 3. prefetches the section's Hugging Face models on the login node (compute nodes may be offline)
# 4. queues `build.py <section> --execute` for the running worker job and waits (remote_request.py);
#    if the worker job disappears first, the request is abandoned and this fails at once
# 5. only after a successful build, pulls back the notebooks, solutions.py, media/ and Streamlit page
# Exit status is 0 if the build succeeded. Requires the university VPN and a running worker job.
set -euo pipefail
SECTION=$1
source "$(dirname "$0")/julia.sh"

read -r CHAPTER EXDIR PAGE < <("$ROOT/.venv/bin/python" - "$SECTION" "$ROOT" <<'PY'
import sys, yaml
from pathlib import Path
section, root = sys.argv[1], Path(sys.argv[2])
config = yaml.safe_load((root / "infrastructure/core/config.yaml").read_text())
m = config["conversion_map"][section]
print(config["chapter_names"][int(section.split(".")[0])], m["exercise_dir"], m["streamlit_page"])
PY
)

sync_sources
sync_env
prefetch "$SECTION"

stamp "queueing $SECTION for the worker"
status=0
julia ".venv/bin/python $CLUSTER/remote_request.py run $SECTION --queue $REMOTE/queue" || status=$?
if [ "$status" -ne 0 ]; then
    stamp "remote build of $SECTION did not succeed (status $status); nothing was pulled back"
    exit "$status"
fi

stamp "pulling results"
rsync -az -e "$SSH" --include="*.ipynb" --include="solutions.py" --include="media/***" --exclude="*" \
    "$JULIA_HOST:$JULIA_BASE/repo/$CHAPTER/exercises/$EXDIR/" "$ROOT/$CHAPTER/exercises/$EXDIR/"
rsync -az -e "$SSH" --include="$PAGE.md" --include="$PAGE.py" --exclude="*" \
    "$JULIA_HOST:$JULIA_BASE/repo/$CHAPTER/instructions/pages/" "$ROOT/$CHAPTER/instructions/pages/" || true
