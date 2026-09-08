#!/usr/bin/env bash
set -euo pipefail

# Default ARENA_3.0_Extended setup path.
# The original upstream/legacy dependencies are kept in requirements-original.txt
# and requirements-legacy-rl.txt; this installer prepares the extension's uv
# managed Python 3.14 + PyTorch CUDA 13.2 environment.

PLATFORM="runpod"
CLONE_LLM_CONTEXT=true
PYTHON_VERSION="3.14"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --no-llm-context)
            CLONE_LLM_CONTEXT=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Setup: platform=$PLATFORM, clone_llm_context=$CLONE_LLM_CONTEXT ==="

echo "=== Installing system packages ==="
if [[ "$PLATFORM" == "runpod" ]]; then
    apt update && apt install -y git curl
elif [[ "$PLATFORM" == "vastai" ]]; then
    sudo apt update && sudo apt install -y git curl
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if $CLONE_LLM_CONTEXT && [[ ! -d arena-llm-context ]]; then
    echo "=== Cloning arena-llm-context ==="
    git clone -b main https://github.com/callummcdougall/arena-llm-context.git
fi

echo "=== Creating uv environment: Python $PYTHON_VERSION ==="
uv venv --python "$PYTHON_VERSION"

echo "=== Syncing locked ARENA extension dependencies ==="
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-130}"
uv sync --locked

echo "=== Verifying CUDA PyTorch stack ==="
uv run python - <<'PY'
import torch
import torchvision

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "=== Done. Use: source .venv/bin/activate ==="
