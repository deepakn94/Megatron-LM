#!/bin/bash

#SBATCH -p batch
#SBATCH --account=nemotron_sw_pre
#SBATCH --nodes=1536
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH --mem=0
# Targets GB200/GB300 (4 GPUs/node); H100 is impractical at this scale.
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --segment=16
#SBATCH --dependency=singleton
#SBATCH --job-name=ultra

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0  # Disable cuDNN fused attention.
export TORCHINDUCTOR_WORKER_START=fork
export TRITON_CACHE_DIR="/tmp/triton_cache/"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce allocator fragmentation.
# HybridEP token dispatcher (flex backend = hybridep).
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64
export USE_MNNVL=1
# Activation offloading (used with --fine-grained-activation-offloading).
export NVTE_CPU_OFFLOAD_V1=1

# Set this to a path you have write permission on; it must already contain all
# required assets (code, image, tokenizer, blend files, etc.). On OCI-HSG,
# "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/dnarayanan/bf16rs_technical_report"
# is one such path.
ROOT_DIR=""
REPO_DIR="${ROOT_DIR}/code"
# Run name; change this per experiment.
NAME="ultra"
# This recipe uses HybridEP (--moe-flex-dispatcher-backend hybridep); the
# container image must include the HybridEP runtime.
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

# Data blend (Ultra, phase 1, 25T tokens). On OCI-HSG, this lives at
# /lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/blend_files/ultra/25t_phase1.json;
# adjust path on other clusters.
BLEND_PATH="/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/blend_files/ultra/25t_phase1.json"

# TransformerEngine precision config for FP4 quantization. On OCI-HSG, this
# lives at
# /lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/code_ultra/te_quant.cfg;
# adjust path on other clusters.
TE_PRECISION_CONFIG="/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/nemotron6/code_ultra/te_quant.cfg"


# HybridEP fallback: if the container image lacks the HybridEP runtime, replace
# the --moe-token-dispatcher-type / --moe-flex-dispatcher-backend /
# --moe-hybridep-num-sms flags below with a single
# "--moe-token-dispatcher-type alltoall" and drop the
# NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN / USE_MNNVL exports above.
options=" \
    --use-mcore-models \
    --hybrid-layer-pattern MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME/*E/*E \
    --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 2 \
    --mamba-num-heads 256 \
    --ffn-hidden-size 5120 \
    --kv-channels 128 \
    --squared-relu \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.0099 \
    --position-embedding-type none \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --normalization RMSNorm \
    \
    --num-experts 512 \
    --moe-router-topk 22 \
    --moe-shared-expert-intermediate-size 10240 \
    --moe-latent-size 2048 \
    --moe-token-dispatcher-type flex \
    --moe-flex-dispatcher-backend hybridep \
    --moe-hybridep-num-sms 32 \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 5.0 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --cross-entropy-loss-fusion \
    --cross-entropy-fusion-impl native \
    \
    --mtp-use-repeated-layer \
    --mtp-loss-scaling-factor 0.1 \
    --calculate-per-token-loss \
    \
    --first-last-layers-bf16 \
    --num-layers-at-start-in-bf16 0 \
    --num-layers-at-end-in-bf16 16 \
    --fp4-format e2m1 \
    --fp4-recipe nvfp4 \
    --te-precision-config-file ${TE_PRECISION_CONFIG} \
    \
    --fine-grained-activation-offloading \
    --offload-modules moe_act \
    \
    --bf16 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --train-samples 3051757813 \
    --lr-decay-style WSD \
    --lr-decay-samples 3048706055 \
    --lr-warmup-samples 24414063 \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-samples 610351563 \
    --phase-transition-iterations 800000 \
    --micro-batch-size 2 \
    --global-batch-size 3072 \
    --lr 2.5e-4 \
    --min-lr 2.5e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --eval-interval 1000 \
    --eval-iters 14 \
    --override-opt_param-scheduler \
    \
    --cuda-graph-impl local \
    --cuda-graph-modules mamba attn moe_router \
    --te-rng-tracker \
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
    --tensor-model-parallel-size 8 \
    --sequence-parallel \
    --expert-model-parallel-size 64 \
    --expert-tensor-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --high-priority-stream-groups ep \
    --ddp-num-buckets 10 \
    --ddp-pad-buckets-for-high-nccl-busbw \
    --attention-backend flash \
    \
    --ckpt-format torch_dist \
    --load ${CHECKPOINT_DIR} \
    --save ${CHECKPOINT_DIR} \
    --save-interval 125 \
    --save-retain-interval 1000 \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --async-save \
    --use-persistent-ckpt-worker \
    --ckpt-assume-constant-structure \
    --result-rejected-tracker-filename ${CHECKPOINT_DIR}/result_rejected_tracker.txt \
    --rerun-mode disabled \
    \
    --log-interval 10 \
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
    --distributed-timeout-minutes 10 \
    --exit-duration-in-mins 5750 \
    --disable-gloo-process-groups \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 "

run_cmd="python -u ${REPO_DIR}/pretrain_hybrid.py ${options}"

# Adjust --container-mounts below if ROOT_DIR lives on a different filesystem
# (e.g. "/scratch:/scratch" on clusters where assets live under /scratch).
srun -l \
    --mpi=none \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
