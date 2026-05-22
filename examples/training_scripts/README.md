# Training scripts

SLURM sbatch scripts for pretraining a few Megatron-LM model recipes on 1T
tokens (122M samples × 8192 seq len). All scripts target the same shared
scaffolding:

- 96 nodes × 8 GPUs/node, `llmservice_fm_text` account, `batch` partition.
- bf16, WSD LR schedule, TikTokenizer (`multiMixV8.gpt4o_nc_sd.500000.128k`),
  blend file `1t_singlephase.json`, distributed optimizer with
  `--overlap-grad-reduce` / `--overlap-param-gather`, `--ddp-num-buckets 8`.
- Container launched via `srun --container-image=... --container-mounts /lustre:/lustre --no-container-mount-home`.

Before submitting, set `ROOT_DIR` at the top of each script to a path you have
write permission on; it must already contain `code/`, `images/`, `tokenizers/`,
and `blend_files/` (see the comment above `ROOT_DIR` for an OCI-HSG example).

## Models

| Script | Arch | Layers | Hidden | MoE | Latent MoE | MTP | Parallelism (TP/EP/PP) | Global batch | Peak LR |
|---|---|---:|---:|---|---|---|---|---:|---:|
| `8b_1t.sh` | Dense Transformer (RoPE) | 32 | 4096 | — | — | — | 4 / — / 1 | 1536 | 8e-4 |
| `8b_latentmoe_1t.sh` | Hybrid Mamba + attention | 61 | 4608 | 512 experts, top-6, shared-expert 6144 | 1152 | 2 layers, `*E` pattern | 2 / 16 / 1 | 3072 | 8e-4 |
| `3b_moe_1t.sh` | Hybrid Mamba + attention | 52 | 2688 | 128 experts, top-6, shared-expert 3712 | — | — | 2 / 32 / 1 | 768 | 1.2e-3 |

Notes:

- "Hybrid" means a Mamba/attention pattern (`--hybrid-override-pattern`) routed
  through `mamba_stack_spec`; `--num-layers` equals the length of the pattern
  string including `*` separators.
- "Latent MoE" refers to `--moe-latent-size` (latent compression on the MoE
  hidden path); only `8b_latentmoe_1t.sh` enables it.
- "MTP" refers to the Multi-Token-Prediction block
  (`--mtp-num-layers`, `--mtp-hybrid-override-pattern`,
  `--mtp-loss-scaling-factor`); only `8b_latentmoe_1t.sh` enables it.
- Both MoE scripts use `--moe-router-score-function sigmoid`,
  `--moe-router-load-balancing-type seq_aux_loss`,
  `--moe-router-topk-scaling-factor 2.5`, and `--moe-router-dtype fp32`.
- Both hybrid-MoE scripts (`8b_latentmoe_1t.sh`, `3b_moe_1t.sh`) enable CUDA
  graphs (`--enable-cuda-graph --cuda-graph-scope mamba attn moe_router`);
  `8b_latentmoe_1t.sh` additionally uses selective recompute of MoE modules.
  `8b_1t.sh` (dense) does not enable CUDA graphs.

## GB200 / GB300

The SBATCH header assumes 8 GPUs/node. On GB200/GB300 (4 GPUs/node), set
`--ntasks-per-node=4`, `--gpus-per-node=4`, and add `--segment=4`
(or `--segment=16` for 16-node segments). A comment to that effect is inline in
each script.
