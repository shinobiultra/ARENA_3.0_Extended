# Settings and steps shared by the scripts that reach the Julia cluster. Source it; don't run it.
#
# Everything on Julia lives under ~/projects/arena-extended: repo (the shared checkout), queue
# (worker requests), runs (job output), hf_cache, uv_cache, tools/uv and env-synced-lock.sha256.
# Requires the university VPN. ARENA_JULIA_HOST, ARENA_JULIA_BASE and ARENA_JULIA_SSH override
# the defaults (the tests use them to stand in a local fake for ssh).

JULIA_HOST=${ARENA_JULIA_HOST:-fik46cm@julia2.hpc.uni-wuerzburg.de}
JULIA_BASE=${ARENA_JULIA_BASE:-projects/arena-extended}  # relative to the remote home
SSH=${ARENA_JULIA_SSH:-/usr/bin/ssh -o BatchMode=yes -o ControlMaster=auto -o ControlPath=/tmp/arena-julia-%r@%h:%p -o ControlPersist=15m}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CLUSTER=infrastructure/extension/cluster
REMOTE="\$HOME/$JULIA_BASE"

# Run a shell command on the login node, from the shared checkout.
julia() {
    $SSH "$JULIA_HOST" "cd $JULIA_BASE/repo && $*"
}

stamp() {
    echo "[$(date +%T)] $*"
}

# Copy the source files (masters, tests.py, utils.py, infrastructure, uv.lock, small data).
sync_sources() {
    stamp "syncing sources"
    rsync -az -e "$SSH" --prune-empty-dirs --max-size=2m \
        --filter="merge $ROOT/$CLUSTER/rsync_filter.txt" "$ROOT/" "$JULIA_HOST:$JULIA_BASE/repo/"
}

# Make Julia's environment match the laptop's uv.lock (env_sync.py); fails if it can't.
sync_env() {
    local lock uv_version
    lock=$(sha256sum "$ROOT/uv.lock" | cut -d' ' -f1)
    uv_version=$(uv --version | cut -d' ' -f2)
    stamp "checking Julia's environment against uv.lock ${lock:0:12}"
    julia "UV_CACHE_DIR=$REMOTE/uv_cache $REMOTE/tools/uv run --no-project --python \$(cat .python-version) \
        $CLUSTER/env_sync.py --repo $REMOTE/repo --state $REMOTE/env-synced-lock.sha256 \
        --queue $REMOTE/queue --expected $lock --uv $REMOTE/tools/uv --uv-version $uv_version"
}

# Download the sections' Hugging Face models on the login node (compute nodes may be offline).
prefetch() {
    stamp "prefetching models"
    julia "HF_HOME=$REMOTE/hf_cache MPLCONFIGDIR=/tmp/mpl-\$USER .venv/bin/python $CLUSTER/prefetch.py $* \
        2>&1 | grep -v Warning | tail -20"
}
