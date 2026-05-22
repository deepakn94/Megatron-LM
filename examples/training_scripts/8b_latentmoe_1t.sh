#!/bin/bash

#SBATCH -p batch
#SBATCH --account=llmservice_fm_text
#SBATCH --nodes=96
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH --mem=0
# GB200/GB300 have 4 GPUs/node; set --ntasks-per-node=4, --gpus-per-node=4, and
# add --segment=4 (or --segment=16 for 16-node segments) on those platforms.
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --dependency=singleton
#SBATCH --job-name=8b_latentmoe_1t

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0  # Disable cuDNN fused attention.
export TORCHINDUCTOR_WORKER_START=fork
export TRITON_CACHE_DIR="/tmp/triton_cache/"

# Set this to a path you have write permission on; it must already contain all
# required assets (code, image, tokenizer, blend files, etc.). On OCI-HSG,
# "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/dnarayanan/bf16rs_technical_report"
# is one such path.
ROOT_DIR=""
REPO_DIR="${ROOT_DIR}/code"
# Run name; change this per experiment.
NAME="8b_latentmoe_1t"
IMAGE_PATH="${ROOT_DIR}/images/nvidia+pytorch+25.06-py3+dependencies+mamba.sqsh"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

RUN_DIR="${ROOT_DIR}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${ROOT_DIR}/data_cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}


# Tokenizer model.
TOKENIZER_MODEL="${ROOT_DIR}/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"

# Data blend.
BLEND_PATH="${ROOT_DIR}/blend_files/1t_singlephase.json"


options=" \
    --is-hybrid-model \
    --use-mcore-models \
    --num-layers 61 \
    --hybrid-override-pattern MEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEME \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --hidden-size 4608 \
    --num-attention-heads 40 \
    --group-query-attention \
    --num-query-groups 8 \
    --mamba-num-heads 128 \
    --ffn-hidden-size 3072 \
    --kv-channels 128 \
    --squared-relu \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.0132 \
    --position-embedding-type none \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --normalization RMSNorm \
    \
    --num-experts 512 \
    --moe-router-topk 6 \
    --moe-shared-expert-intermediate-size 6144 \
    --moe-latent-size 1152 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    \
    --mtp-spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --mtp-num-layers 2 \
    --mtp-num-layers-per-layer 2 \
    --mtp-hybrid-override-pattern *E \
    --mtp-loss-scaling-factor 0.3 \
    --calculate-per-token-loss \
    \
    --bf16 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --train-samples 122070313 \
    --lr-decay-style WSD \
    --lr-decay-samples 122070313 \
    --lr-warmup-samples 3051758 \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-samples 24414063 \
    --micro-batch-size 1 \
    --global-batch-size 3072 \
    --lr 8e-4 \
    --min-lr 8e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --eval-interval 1000 \
    --eval-iters 14 \
    \
    --enable-cuda-graph \
    --cuda-graph-scope mamba attn moe_router \
    --te-rng-tracker \
    --no-load-rng \
    \
    --per-split-data-args-path ${BLEND_PATH} \
    --data-cache-path ${DATACACHE_DIR} \
    --tokenizer-type TikTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tiktoken-pattern v2 \
    --no-mmap-bin-files \
    --num-workers 1 \
    --no-create-attention-mask-in-dataloader \
    \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --expert-model-parallel-size 16 \
    --expert-tensor-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --high-priority-stream-groups ep \
    --ddp-num-buckets 8 \
    --attention-backend flash \
    --recompute-granularity selective \
    --recompute-modules moe \
    \
    --ckpt-format torch_dist \
    --load ${CHECKPOINT_DIR} \
    --save ${CHECKPOINT_DIR} \
    --save-interval 500 \
    --save-retain-interval 2000 \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --async-save \
    --use-persistent-ckpt-worker \
    --ckpt-assume-constant-structure \
    \
    --log-interval 100 \
    --log-memory-interval 1000 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-progress \
    --log-energy \
    --logging-level 20 \
    --timing-log-option minmax \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --check-weight-hash-across-dp-replicas-interval 20000 \
    \
    --manual-gc \
    --manual-gc-interval 10 \
    --distributed-timeout-minutes 10 \
    --exit-duration-in-mins 235 \
    --disable-gloo-process-groups \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 "

run_cmd="python -u ${REPO_DIR}/pretrain_mamba.py ${options}"

# Adjust --container-mounts below if ROOT_DIR lives on a different filesystem
# (e.g. "/scratch:/scratch" on clusters where assets live under /scratch).
srun -l \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
