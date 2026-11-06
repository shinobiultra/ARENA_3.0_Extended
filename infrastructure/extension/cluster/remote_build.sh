#!/bin/bash
# Execute an extension section on the Julia worker job, and bring the executed notebooks back.
#
#   infrastructure/extension/cluster/remote_build.sh 8.4
#
# 1. syncs source files (masters, tests.py, utils.py, infrastructure, small data) to the cluster checkout
# 2. prefetches the section's Hugging Face models on the login node (compute nodes may be offline)
# 3. queues `build.py <section> --execute` for the worker (worker.py) and waits for it
# 4. pulls back the generated notebooks, solutions.py, media/ and Streamlit page, and prints the log tail
# Exit status is 0 if the build succeeded. Requires the university VPN.
set -euo pipefail
SECTION=$1
HOST=fik46cm@julia2.hpc.uni-wuerzburg.de
BASE=projects/arena-extended
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
FILTER="$ROOT/infrastructure/extension/cluster/rsync_filter.txt"
SSH="/usr/bin/ssh -o BatchMode=yes -o ControlMaster=auto -o ControlPath=/tmp/arena-julia-%r@%h:%p -o ControlPersist=15m"

read -r CHAPTER EXDIR PAGE < <("$ROOT/.venv/bin/python" - "$SECTION" "$ROOT" <<'EOF'
import sys, yaml
from pathlib import Path
section, root = sys.argv[1], Path(sys.argv[2])
config = yaml.safe_load((root / "infrastructure/core/config.yaml").read_text())
m = config["conversion_map"][section]
print(config["chapter_names"][int(section.split(".")[0])], m["exercise_dir"], m["streamlit_page"])
EOF
)
ID="${SECTION}-$(date +%s)-$$"

echo "[$(date +%T)] syncing sources"
rsync -az -e "$SSH" --prune-empty-dirs --max-size=2m --filter="merge $FILTER" "$ROOT/" "$HOST:$BASE/repo/"

echo "[$(date +%T)] prefetching models"
$SSH "$HOST" "cd $BASE/repo && HF_HOME=\$HOME/$BASE/hf_cache MPLCONFIGDIR=/tmp/mpl-\$USER .venv/bin/python infrastructure/extension/cluster/prefetch.py $SECTION 2>&1 | grep -v Warning | tail -20"

echo "[$(date +%T)] queued $ID; waiting for the worker"
$SSH "$HOST" "echo $SECTION > $BASE/queue/$ID.req; squeue --me --name=arena-worker -h -o '  worker job %i: %T %M' ; \
  while [ ! -e $BASE/queue/$ID.done ] && [ ! -e $BASE/queue/$ID.failed ]; do sleep 10; done"

STATUS=$($SSH "$HOST" "[ -e $BASE/queue/$ID.done ] && echo done || echo failed")
echo "[$(date +%T)] build $STATUS; pulling results"
rsync -az -e "$SSH" --include="*.ipynb" --include="solutions.py" --include="media/***" --exclude="*" \
    "$HOST:$BASE/repo/$CHAPTER/exercises/$EXDIR/" "$ROOT/$CHAPTER/exercises/$EXDIR/"
rsync -az -e "$SSH" --include="$PAGE.md" --include="$PAGE.py" --exclude="*" \
    "$HOST:$BASE/repo/$CHAPTER/instructions/pages/" "$ROOT/$CHAPTER/instructions/pages/" || true
$SSH "$HOST" "grep -v 'W1008\|Warning\|warn(' $BASE/queue/$ID.log | tail -25"
[ "$STATUS" = done ]
