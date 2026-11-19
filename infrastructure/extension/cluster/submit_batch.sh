#!/bin/bash
# Submit one finite Julia batch job (batch.sbatch) for some extension sections, one per GPU.
#
#   infrastructure/extension/cluster/submit_batch.sh --gpus 2 --time 04:00:00 8.4 9.1 12.1
#
# Syncs the sources and Julia's environment with the laptop's uv.lock first, and submits nothing
# if either fails. Then prefetches the sections' models on the login node and runs sbatch at
# --qos=normal (the account's default QOS preempts other users' jobs). Each submission counts
# toward a batch's two Julia submissions; set --time to about twice the batch's expected total.
set -euo pipefail
GPUS=2
TIME=
while [ $# -gt 0 ]; do
    case $1 in
        --gpus) GPUS=$2; shift 2 ;;
        --time) TIME=$2; shift 2 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
if [ -z "$TIME" ] || [ $# -eq 0 ]; then
    echo "usage: $0 [--gpus N] --time HH:MM:SS <section> [<section> ...]" >&2
    exit 2
fi
source "$(dirname "$0")/julia.sh"

sync_sources
sync_env
prefetch "$@"
stamp "submitting ${*} on $GPUS GPU(s) for $TIME"
julia "sbatch --qos=normal --gres=gpu:L40s:$GPUS --time=$TIME $CLUSTER/batch.sbatch $*"
