#!/bin/bash

# Benchmark Newton-Schulz distribution strategies over the GTP axis.
#
# GTP shards dense weights. At the production degree of 64 the group fits inside one
# GB200/GB300 NVLink domain, so NVLink is left enabled here: that is the transport a
# real run uses. On aws-cmh (4 GPUs/node) a 64-rank group needs 16 nodes.
#
# GTP degree and node count must agree: nodes * 4 (GPUs/node) == GTP. To try a smaller
# degree first, override both, e.g.
#   sbatch --nodes=2 --export=ALL,GTP=8 tools/muon_analysis/bench_ns_gtp.sbatch

#SBATCH -p batch
#SBATCH --nodes=16
#SBATCH --account=nemotron_n4_pre
#SBATCH --exclusive
#SBATCH -t 0:06:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --job-name=bench_muon_ns_gtp

export NCCL_DEBUG=WARN

GTP=${GTP:-64}

ROOT_DIR="/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/dnarayanan/muon_ns_bench"
REPO_DIR="${ROOT_DIR}/code"
IMAGE_PATH="/scratch/fsw/portfolios/nemotron/projects/nemotron_sw_pre/nemotron4_prep/images/nvidia+pytorch+26.04-py3+dependencies+mamba.sqsh"

NAME="gtp"
LOGS_DIR="${ROOT_DIR}/${NAME}/logs"
mkdir -p "${LOGS_DIR}"

DATETIME=$(date +"date_%y-%m-%d_time_%H-%M-%S")

options=" \
    --group gtp \
    --modelled-world-size 12288 \
    --tp 1 --gtp ${GTP} --ep 64 --etp 1 --egtp 4 \
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
