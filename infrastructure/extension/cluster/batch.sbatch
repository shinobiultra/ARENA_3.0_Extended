#!/bin/bash
# One Julia (Würzburg) job that executes a batch of extension sections, one per GPU.
#
#   sbatch --gres=gpu:L40s:<N> --time=<HH:MM:SS> batch.sbatch <section> [<section> ...]
#
# Layout on the cluster (everything in one place):
#   ~/projects/arena-extended/repo       source files synced from the local checkout
#   ~/projects/arena-extended/hf_cache   HF_HOME (models prefetched on the login node)
#   ~/projects/arena-extended/runs/<job> logs and summary.tsv
#SBATCH --job-name=arena-ext
#SBATCH --account=computervision
#SBATCH --partition=gpu_computervision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/home/fik46cm/projects/arena-extended/runs/slurm-%j.out

set -euo pipefail
BASE=$HOME/projects/arena-extended
REPO=$BASE/repo
OUT=$BASE/runs/$SLURM_JOB_ID
mkdir -p "$OUT"

export HF_HOME=$BASE/hf_cache
export TORCH_HOME=$BASE/hf_cache/torch
export XDG_CACHE_HOME=$BASE/cache
export MPLCONFIGDIR=$BASE/cache/matplotlib
export TRITON_CACHE_DIR=$BASE/cache/triton
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
if [ -f "$HOME/hf_token.sh" ]; then
    source "$HOME/hf_token.sh"
    export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
fi
if ! curl -s --max-time 5 -o /dev/null https://huggingface.co; then
    echo "No internet on $(hostname): running HF offline from the prefetched cache"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi

cd "$REPO"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count()); torch.ones(1, device='cuda')"

.venv/bin/python infrastructure/extension/cluster/run_batch.py "$@" --out "$OUT"
