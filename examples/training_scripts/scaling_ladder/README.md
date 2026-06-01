# Scaling-ladder recipes (AdamW)

Hybrid Mamba+attention MoE pretraining recipes ported from
`ssh://git@gitlab-master.nvidia.com:12051/ADLR/nemotron-pretraining-scaling-ladder.git`
(`recipes/adamw/`). All scripts share the same scaffolding as
`../README.md` describes (96-node assumption replaced per-script with
the scaling-ladder team's node counts; same image, container mount,
account, optimizer, MoE config, CUDA-graph block, checkpointing, and
logging).

## File-name convention

`a{active_params}_{total_params}_moe_{token_horizon}.sh` — all lowercase.
The `a` prefix marks the active-parameter count, total params follow,
token horizon comes last. The parent directory uses the same convention
but drops the total (since most parent scripts are one-offs, not part of
a ladder), e.g. `a3b_moe_1t.sh`. The dense `8b_1t.sh` carries no `a`
prefix because active = total.

## Models

| Script | Nodes | Active / total | Tokens | Hidden | Attn heads | Mamba heads | FFN (per expert) | Init std | Shared expert | Pattern (main) | LR | Train samples | TP / EP | MTP |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| `a315m_1b_moe_88b.sh` | 8 | 315 M / 1 B | 88 B | 768 | 6 | 24 | 512 | 0.0325 | 960 | `MEMEM*EMEME*EMEM*EMEME` (22) | 2.2e-3 | 10.77 M | 1 / 8 | — |
| `a500m_2b_moe_140b.sh` | 32 | 500 M / 2 B | 140 B | 1024 | 8 | 32 | 640 | 0.028 | 1280 | `MEMEM*EMEMEM*EMEMEM*EMEMEME` (26) | 2.0e-3 | 17.09 M | 1 / 8 | — |
| `a770m_4b_moe_215b.sh` | 32 | 770 M / 4 B | 215 B | 1280 | 12 | 40 | 768 | 0.025 | 1536 | `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEME` (32) | 1.8e-3 | 26.25 M | 1 / 8 | — |
| `a1b_7b_moe_310b.sh` | 32 | 1 B / 7 B | 310 B | 1536 | 12 | 48 | 1024 | 0.0229 | 2048 | `MEMEM*EMEMEM*EMEMEMEM*EMEMEM*EMEMEME` (35) | 1.6e-3 | 37.84 M | 1 / 8 | — |
| `a2b_14b_moe_560b.sh` | 32 | 2 B / 14 B | 560 B | 2048 | 16 | 64 | 1280 | 0.0198 | 2560 | `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEME` (43) | 1.4e-3 | 68.36 M | 1 / 16 | — |
| `a3b_30b_moe_1t.sh` | 32 | 3 B / 30 B | 1 T | 2688 | 32 | 64 | 1856 | 0.0173 | 3712 | `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` (52) + `/*E/*E` | 1.2e-3 | 122.07 M | 2 / 32 | yes (2 depths, shared weights) |

Notes:

- `--num-experts 128`, `--moe-router-topk 6`, `--global-batch-size 768`,
  `--seq-length 8192`, `--lr-warmup-samples 1024000`, `--num-query-groups 2`,
  `--kv-channels 128`, `--pipeline-model-parallel-size 1`,
  `--expert-tensor-parallel-size 1` are identical across all six.
- `--micro-batch-size` is 4 for `a315m_1b_moe_88b.sh` and 1 for the rest.
- `--lr-decay-samples` equals `--train-samples` (full WSD decay); the
  `lr-wsd-decay` tail is 15% of train samples in each case.
- Only `a3b_30b_moe_1t.sh` enables MTP: pattern gets a `/*E/*E` suffix
  (two depths, identical `*E` pattern per depth), plus `--mtp-spec`,
  `--mtp-use-repeated-layer` (shared weights across depths),
  `--mtp-loss-scaling-factor 0.1`, `--calculate-per-token-loss`.
