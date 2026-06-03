# Nemotron-3 recipes (Nano / Super / Ultra)

Hybrid Mamba + attention + MoE pretraining recipes for the Nano, Super, and
Ultra models, each trained on 25 T tokens (`--train-samples 3 051 757 813`
at 8192 sequence length).

## Common scaffolding

- `nemotron_sw_pre` account, `batch` partition. Nano defaults to 384 H100
  nodes (8 GPUs/node, 3072 GPUs); Super defaults to 768 GB200 nodes
  (4 GPUs/node, 3072 GPUs); Ultra defaults to 1536 GB200 nodes
  (4 GPUs/node, 6144 GPUs). Super and Ultra both use `--segment=16`. H100
  is impractical for Super and Ultra at this scale.
- `--global-batch-size 3072`, `--seq-length 8192`, WSD LR schedule, BF16,
  TikTokenizer (`multiMixV8.gpt4o_nc_sd.500000.128k`),
  `--lr-warmup-samples 1 024 000` (Nano) / `24 414 063` (Super, Ultra).
- Distributed optimizer with `--overlap-grad-reduce` and
  `--overlap-param-gather`; `--attention-backend flash`; squared-relu MLPs;
  `--num-query-groups 2`, `--kv-channels 128`,
  `--pipeline-model-parallel-size 1`, `--expert-tensor-parallel-size 1`.
- `pretrain_hybrid.py` entrypoint.
- Container launched via
  `srun --mpi=none --container-image=... --container-mounts /lustre:/lustre --no-container-mount-home`.

Before submitting, set `ROOT_DIR` at the top of each script; see the OCI-HSG
example in the `ROOT_DIR` comment. `BLEND_PATH` and (for Super / Ultra)
`TE_PRECISION_CONFIG` are already hard-coded to their OCI-HSG locations and
need to be adjusted on other clusters.

## Per-script comparison

### Model architecture

| | Nano | Super | Ultra |
|---|---:|---:|---:|
| Main pattern length | 52 | 88 | 108 |
| `--hidden-size` | 2688 | 4096 | 8192 |
| `--num-attention-heads` | 32 | 32 | 64 |
| `--mamba-num-heads` | 64 | 128 | 256 |
| `--ffn-hidden-size` | 1856 | 2688 | 5120 |
| `--init-method-std` | 0.0173 | 0.014 | 0.0099 |

### MoE

| | Nano | Super | Ultra |
|---|---:|---:|---:|
| `--num-experts` | 128 | 512 | 512 |
| `--moe-router-topk` | 6 | 22 | 22 |
| `--moe-router-topk-scaling-factor` | 2.5 | 5.0 | 5.0 |
| `--moe-shared-expert-intermediate-size` | 3712 | 5376 | 10240 |
| `--moe-latent-size` | — | 1024 | 2048 |
| Dispatcher | `alltoall` | `flex` (HybridEP backend, 32 SMs) | `flex` (HybridEP backend, 32 SMs) |

### MTP

| | Nano | Super | Ultra |
|---|---|---|---|
| MTP enabled (`/*E/*E` suffix on `--hybrid-layer-pattern`) | — | yes | yes |
| `--mtp-use-repeated-layer` (shared MTP weights) | — | — | yes |
| `--mtp-loss-scaling-factor` | — | 0.3 | 0.1 |

### Precision / FP4 quantization

| | Nano | Super | Ultra |
|---|---|---|---|
| FP4 block (`--fp4-format e2m1 --fp4-recipe nvfp4`) | — | yes | yes |
| `--num-layers-at-end-in-bf16` | — | 16 | 16 |
| `--te-precision-config-file` | — | OCI-HSG `te_quant.cfg` | OCI-HSG `te_quant.cfg` |

### Offloading (Ultra only)

| | Nano | Super | Ultra |
|---|---|---|---|
| `--fine-grained-activation-offloading` + `--offload-modules moe_act` | — | — | yes |
| `export NVTE_CPU_OFFLOAD_V1=1` | — | — | yes |

### CUDA graphs

| | Nano | Super | Ultra |
|---|---|---|---|
| `--cuda-graph-impl local` + `--cuda-graph-modules mamba attn moe_router` + `--te-rng-tracker` | — | yes | yes |

### Parallelism

| | Nano | Super | Ultra |
|---|---:|---:|---:|
| `--tensor-model-parallel-size` | 2 | 2 | 8 |
| `--expert-model-parallel-size` | 32 | 64 | 64 |
| `--micro-batch-size` | 1 | 1 | 2 |
| `--tp-comm-overlap` | yes | — | — |
| `--high-priority-stream-groups ep` | — | yes | yes |
| `--ddp-num-buckets` | 8 | 10 | 10 |

### Training hyperparameters

| | Nano | Super | Ultra |
|---|---:|---:|---:|
| `--lr` / `--min-lr` | 1e-3 / 1e-5 | 4.5e-4 / 4.5e-6 | 2.5e-4 / 2.5e-6 |
| `--lr-warmup-samples` | 1 024 000 | 24 414 063 | 24 414 063 |
| `--lr-decay-samples` | 3 050 733 813 | 3 048 706 055 | 3 048 706 055 |
| `--phase-transition-iterations` | — | — | 800 000 |
| `--eval-interval` | 2000 | 1000 | 1000 |

### Checkpointing

| | Nano | Super | Ultra |
|---|---:|---:|---:|
| `--save-interval` | 1000 | 250 | 125 |
| `--save-retain-interval` | 10 000 | 1 000 | 1 000 |
| `--result-rejected-tracker-filename` | — | yes | yes |
| `--rerun-mode disabled` | — | — | yes |

### Data loader

| | Nano | Super | Ultra |
|---|---|---|---|
| Blend | `…/nemotron6/blend_files/nano/25t_phase1.json` (CW-DFW) | `…/nemotron6/blend_files/super/25t_phase1.json` (OCI-HSG) | `…/nemotron6/blend_files/ultra/25t_phase1.json` (OCI-HSG) |

### Logging / runtime

| | Nano | Super | Ultra |
|---|---:|---:|---:|
| `--log-interval` | 100 | 100 | 10 |
| `--log-memory-interval` | 500 | 500 | 1000 |
| `--distributed-timeout-minutes` | 10 | 10 | 10 |

### Top-of-script env vars

| | Nano | Super | Ultra |
|---|---|---|---|
| Standard (`CUDA_DEVICE_MAX_CONNECTIONS`, `NVTE_*`, `TORCHINDUCTOR_WORKER_START`, `TRITON_CACHE_DIR`) | yes | yes | yes |
| HybridEP (`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64`, `USE_MNNVL=1`) | — | yes | yes |
| CPU activation offload (`NVTE_CPU_OFFLOAD_V1=1`) | — | — | yes |

## Container requirements

- Nano: any image with TransformerEngine and Mamba2 support
  (the default `nvidia+pytorch+25.06-py3+dependencies+mamba.sqsh` works).
- **Super and Ultra require the HybridEP runtime** — both use the FlexDispatcher
  `hybridep` backend. A comment to that effect is inline above `IMAGE_PATH` in
  each script.

## Smaller-scale experiments

To run on fewer GPUs, weak-scale the global batch: drop `--nodes` and
`--global-batch-size` by the same factor, leaving `--micro-batch-size`,
`--train-samples`, and the LR-schedule samples unchanged. See the parent
directory's README for details and a worked example. Make sure the new
`--global-batch-size` stays divisible by `DP × micro_batch`, where
`DP = WORLD_SIZE / (TP × PP)`.
