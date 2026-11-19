#!/bin/bash
# The Julia (Würzburg) worker job: executes extension section builds on request (see worker.py).
#
#   sbatch --qos=normal --time=<about 2x the batch's expected total> infrastructure/extension/cluster/worker.sbatch
#   touch ~/projects/arena-extended/queue/STOP                     # to end it early, once running builds finish
#
# Run sbatch from ~/projects/arena-extended/repo. remote_build.sh queues builds for it. Each lesson
# gets 30 minutes; the worker exits after 30 minutes with an empty queue, releasing its GPUs.
# Keep --qos=normal: the account's default QOS (computervision) preempts other users' jobs.
#SBATCH --job-name=arena-worker
#SBATCH --account=computervision
#SBATCH --partition=gpu_computervision
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/fik46cm/projects/arena-extended/runs/worker-%j.out

set -euo pipefail
BASE=$HOME/projects/arena-extended
cd "$BASE/repo"

export HF_HOME=$BASE/hf_cache
export TORCH_HOME=$BASE/hf_cache/torch
export XDG_CACHE_HOME=$BASE/cache
export MPLCONFIGDIR=$BASE/cache/matplotlib
export TRITON_CACHE_DIR=$BASE/cache/triton
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
if [ -f "$HOME/hf_token.sh" ]; then
    source "$HOME/hf_token.sh"
    export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
fi
if curl -s --max-time 5 -o /dev/null https://huggingface.co; then
    echo "internet: yes"
else
    echo "internet: no; running HF offline from the prefetched cache"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count()); torch.ones(1, device='cuda')"
.venv/bin/python infrastructure/extension/cluster/worker.py --queue "$BASE/queue"
