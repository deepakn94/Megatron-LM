#!/bin/bash

# Benchmark Newton-Schulz distribution strategies over the EGTP axis.
#
# EGTP=4 fits on one aws-cmh node (4 GB300s), but in a real job the group can span the
# scale-out fabric, InfiniBand or RoCE depending on the cluster. NVLink, SHM and NVLS are
# disabled below so the collectives take the network path and the timings reflect that.

#SBATCH -p batch
#SBATCH --account=nemotron_n4_pre
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 0:06:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=bench_muon_ns_egtp

# Force the EGTP group off NVLink so it measures the IB path.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

ROOT_DIR="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/dnarayanan/muon_ns_bench"
REPO_DIR="${ROOT_DIR}/code"
IMAGE_PATH="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/nemotron4_prep/images/nvidia+pytorch+26.04-py3+dependencies+mamba.sqsh"

NAME="egtp"
LOGS_DIR="${ROOT_DIR}/${NAME}/logs"
mkdir -p "${LOGS_DIR}"

DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

# The script sweeps every rank profile itself and reports the slowest, so nothing to pick.
options=" \
    --group egtp \
    --modelled-world-size 12288 \
    --tp 1 --gtp 64 --ep 64 --etp 1 --egtp 4 \
    --shard-latent-proj \
    --num-ns-steps 16 \
    --coefficient-type polar_express \
    --iters 20 \
    --warmup 5 "

run_cmd="python -u ${REPO_DIR}/tools/muon_analysis/bench_ns_strategies.py ${options}"

srun -l \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/scratch:/scratch" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
