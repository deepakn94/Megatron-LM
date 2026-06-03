# Training scripts

SLURM sbatch scripts for pretraining a few Megatron-LM model recipes on 1T
tokens (122M samples × 8192 seq len). All scripts target the same shared
scaffolding:

- 96 nodes × 8 GPUs/node, `nemotron_sw_pre` account, `batch` partition.
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
| `a8b_120b_latentmoe_1t.sh` | Hybrid Mamba + attention | 61 | 4608 | 512 experts, top-6, shared-expert 6144 | 1152 | 2 layers, `*E` pattern | 2 / 16 / 1 | 3072 | 8e-4 |
| `a3b_30b_moe_1t.sh` | Hybrid Mamba + attention | 52 | 2688 | 128 experts, top-6, shared-expert 3712 | — | — | 2 / 32 / 1 | 768 | 1.2e-3 |
| `a3b_30b_transformer_moe_1t.sh` | Transformer + MoE | 52 | 2688 | 128 experts, top-6, shared-expert 3712 | — | — | 2 / 32 / 1 | 768 | 1.2e-3 |

Notes:

- "Hybrid" means a Mamba/attention pattern (`--hybrid-layer-pattern`) routed
  through `hybrid_stack_spec`; the layer count is derived from the pattern
  string length, including `*` separators.
- File names follow `a{active}_{total}_{variant}_{horizon}.sh` — the `a`
  prefix marks the active-parameter count, total params follow, variant
  (e.g. `moe`, `latentmoe`, `transformer_moe`) next, token horizon last.
  `8b_1t.sh` is dense (active = total), so it carries no `a` prefix and no
  total. The `scaling_ladder/` subdirectory follows the same convention.
- `a3b_30b_transformer_moe_1t.sh` is the same recipe as `a3b_30b_moe_1t.sh`
  but with the Mamba layers replaced by attention (`M`→`*` in the pattern); it
  uses `hybrid_stack_spec` and `pretrain_hybrid.py`. Total/active params remain
  roughly the same (an attention layer is somewhat smaller than a Mamba2 layer
  at this hidden size, so non-MoE params drop by ~0.5% of total).
- "Latent MoE" refers to `--moe-latent-size` (latent compression on the MoE
  hidden path); only `a8b_120b_latentmoe_1t.sh` enables it.
- "MTP" refers to the Multi-Token-Prediction block (folded into the unified
  `--hybrid-layer-pattern` via `/`-separated MTP depths); only
  `a8b_120b_latentmoe_1t.sh` enables it among the scripts in this directory.
- Both MoE scripts use `--moe-router-score-function sigmoid`,
  `--moe-router-load-balancing-type seq_aux_loss`,
  `--moe-router-topk-scaling-factor 2.5`, and `--moe-router-dtype fp32`.
- All hybrid scripts (`a8b_120b_latentmoe_1t.sh`, `a3b_30b_moe_1t.sh`,
  `a3b_30b_transformer_moe_1t.sh`) enable CUDA graphs
  (`--cuda-graph-impl local` with `--cuda-graph-modules mamba attn moe_router`);
  `a8b_120b_latentmoe_1t.sh` additionally uses selective recompute of MoE
  modules. `8b_1t.sh` (dense) does not enable CUDA graphs.

## Sub-directories

- [`scaling_ladder/`](scaling_ladder/README.md) — six AdamW scaling-ladder
  recipes (315 M–3 B active, 1 B–30 B total, 88 B–1 T tokens). All hybrid
  Mamba + MoE, used to fit scaling laws. Follows the same `a{active}_{total}_moe_{horizon}.sh`
  filename convention.
- [`nemotron3/`](nemotron3/README.md) — three production Nemotron-3 recipes
  (`nano.sh`, `super.sh`, `ultra.sh`), each on 25 T tokens. Nano on H100 (384
  nodes / 3072 GPUs); Super on GB200 (768 nodes / 3072 GPUs); Ultra on GB200
  (1536 nodes / 6144 GPUs). Super and Ultra use FP4 quantization with a
  TransformerEngine precision config and MTP; Ultra additionally uses the
  HybridEP flex dispatcher and CPU activation offload.

## Smaller-scale experiments

To run any of these recipes on fewer GPUs, **weak-scale the global batch
size**: drop `--nodes` (or `--gpus-per-node`) and reduce `--global-batch-size`
by the same factor so the per-GPU work stays constant. Don't change
`--micro-batch-size`; just shrink the global batch.

Example with `a3b_30b_moe_1t.sh` (ships at 96 nodes × 8 GPUs = 768 GPUs,
`--global-batch-size 768`):

| Nodes | GPUs | `--global-batch-size` |
|---:|---:|---:|
| 96 | 768 | 768 (default) |
| 48 | 384 | 384 |
| 24 | 192 | 192 |
| 12 | 96 | 96 |

Keep `--train-samples`, `--lr-decay-samples`, and `--lr-wsd-decay-samples`
unchanged — the token horizon is the same, you just take more (smaller) steps.
Make sure the new `--global-batch-size` stays divisible by `DP × micro_batch`
(DP = `WORLD_SIZE / (TP × PP)`).

## GB200 / GB300

The SBATCH header assumes 8 GPUs/node. On GB200/GB300 (4 GPUs/node), set
`--ntasks-per-node=4`, `--gpus-per-node=4`, and add `--segment=4`
(or `--segment=16` for 16-node segments). A comment to that effect is inline in
each script.
