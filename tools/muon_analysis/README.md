# Muon / LayerWise optimizer analysis tools

Standalone tools for reasoning about `LayerWiseDistributedOptimizer` (Muon) memory and
Newton-Schulz performance on real model configurations. Nothing here is imported by
training code, and none of it runs in CI.

| file | what it answers | needs |
|---|---|---|
| `padding_estimate.py` | How much of the optimizer's buffers is shard-imbalance padding, per buffer and per GPU count? | stdlib only |
| `bench_ns_strategies.py` | Which Newton-Schulz distribution strategy is fastest for the shapes a rank actually owns? | GPUs, `torch`, `emerging_optimizers` |
| `bench_ns_egtp.sbatch` | Runs the benchmark over the EGTP axis on one node, NVLink disabled. | Slurm, aws-cmh |
| `bench_ns_gtp.sbatch` | Runs the benchmark over the GTP axis, 16 nodes for GTP=64. | Slurm, aws-cmh |

## Why both

They answer the two halves of the same question. The layer-wise optimizer assigns whole
matrices to data-parallel shards, so a rank's cost depends on *which* matrices it drew.
`padding_estimate.py` models the memory consequence of that assignment; `bench_ns_strategies.py`
measures the time consequence. Both derive their shapes from the same model description
and the same compute-balanced assignment, so their rank profiles line up.

## Estimating padding

Pure arithmetic, so it runs anywhere:

```bash
python tools/muon_analysis/padding_estimate.py \
    --hybrid-layer-pattern 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME/*E/*E' \
    --hidden-size 2688 --ffn-hidden-size 3712 --moe-latent-size 672 \
    --num-experts 256 --moe-shared-expert-intermediate-size 3712 \
    --kv-channels 192 --num-attention-heads 32 --num-query-groups 1 \
    --mamba-num-heads 16 --mamba-num-groups 16 \
    --mtp-num-layers 2 --mtp-use-repeated-layer \
    --world-size 512 --expert-model-parallel-size 64 \
    --ddp-num-buckets 8 --total-params-per-rank 2846797792
```

Validated against a real 512-GPU run: bucket count and `dp_size` match the layout Megatron
logs, and padding agrees to within 0.2%.

## Benchmarking Newton-Schulz

The world size must equal the sharding degree being modelled, so the two axes need
different allocations:

```bash
sbatch tools/muon_analysis/bench_ns_egtp.sbatch                       # EGTP=4,  1 node
sbatch tools/muon_analysis/bench_ns_gtp.sbatch                        # GTP=64, 16 nodes
sbatch --nodes=2 --export=ALL,GTP=8 tools/muon_analysis/bench_ns_gtp.sbatch   # smaller trial
```

The EGTP wrapper disables NVLink, SHM and NVLS so the collectives take the scale-out
fabric, which is where an EGTP group sits in a real job. The GTP wrapper leaves NVLink
enabled, because a 64-rank GTP group fits inside one GB200/GB300 NVLink domain.

### Reading the output

`useful` FLOPs are the irreducible share: orthogonalizing the full matrix once, divided
across the group. `issued` FLOPs are what one GPU actually executes. Two consequences
worth keeping straight:

- `duplicated` recomputes the whole matrix on every rank, so its `useful TF/s` is already
  its `issued TF/s` divided by the group size. Do not discount it twice.
- `blockwise` is off by default. It orthogonalizes each block rather than the matrix, so
  it changes the update rather than distributing the same one, and it is not used in
  practice. Add it with `--modes blockwise duplicated distributed` if you want a floor;
  it issues *less* than useful, so its throughput is not comparable to the other two.

Every rank orthogonalizes concurrently and the step ends when the slowest finishes, so the
reported step cost is the max over rank profiles, not the mean.

## Measured on Nemotron-4 (152 layers, GB300, 16 NS steps, bf16 GEMMs)

Newton-Schulz time per optimizer step, per GPU. Each figure is the slowest rank profile
for that buffer, so the total assumes a GPU drawing the worst profile on both axes.

| mode | dense (GTP=64, NVLink) | expert (EGTP=4, network) | total |
|---|---|---|---|
| `duplicated` | 123.6 ms | 141.0 ms | **264.6 ms** |
| `distributed` | 223.8 ms | 1990.8 ms | 2214.6 ms |

`duplicated` wins on both axes despite paying exactly `group_size` redundancy, because
`distributed`'s replicated `A @ A` term does not shrink with the group. Whenever the
sharded dimension is the shorter one, Newton-Schulz runs along the long dimension instead
and that term explodes: redundancy reaches 104x and 312x on some dense shapes, and 22x on
the expert `3072x10240`. This is a compute problem, not a communication one, so a faster
interconnect does not fix it.

## Drift warning

Both tools reimplement `LayerWiseDistributedOptimizer._compute_per_buffer_param_layout`
rather than importing it, which is what keeps `padding_estimate.py` dependency-free. They
will go stale if that function changes. Re-check against the reference numbers in
`padding_estimate.py`'s module docstring after touching the packer.

The model description in both files is the 152-layer Nemotron-4 hybrid Mamba-MoE. Other
models need the constants updated; `padding_estimate.py` takes them as flags, while
`bench_ns_strategies.py` has them as module constants.
