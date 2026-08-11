# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Estimate LayerWise (Muon) optimizer buffer padding for a given model and parallelism.

The layer-wise optimizer assigns each Muon-managed matrix *whole* to one data-parallel
shard, because Newton-Schulz needs the full 2D weight. Every shard in a bucket is then
padded up to the largest one, so padding has a hard floor of ``dp_size * max_param_numel``
per bucket regardless of how many parameters there are. When a buffer holds fewer tensors
than it has shards, most shards are pure padding.

Pure stdlib, so config sweeps run anywhere without torch or a GPU.

DRIFT WARNING
-------------
This mirrors ``LayerWiseDistributedOptimizer._compute_per_buffer_param_layout`` in
``megatron/core/optimizer/layer_wise_optimizer.py`` (branch ``dnarayanan/gtp_plus_muon_v2``,
which carries the compute-balanced LPT and the padding-absorb bucket cutter). It is a
reimplementation, not an import, and will go stale if that function changes. Re-check it
against the reference below after touching the packer.

Reference, from a real 512-GPU run (a3b_30b_gdp_latentmoe, TP=1, no GTP, EP=64), where
Megatron logged::

    Layerwise param layout: 158 params, 1 buckets, dp_size=512,
      total_param_numel=898351104, total_buffer_numel=8984199168,
      total_padding=8085848064, overhead=900.1%

reproduced by::

    python tools/muon_analysis/padding_estimate.py \\
        --hybrid-layer-pattern 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME/*E/*E' \\
        --hidden-size 2688 --ffn-hidden-size 3712 --moe-latent-size 672 \\
        --num-experts 256 --moe-shared-expert-intermediate-size 3712 \\
        --kv-channels 192 --num-attention-heads 32 --num-query-groups 1 \\
        --mamba-num-heads 16 --mamba-num-groups 16 \\
        --mtp-num-layers 2 --mtp-use-repeated-layer \\
        --world-size 512 --expert-model-parallel-size 64 \\
        --ddp-num-buckets 8 --total-params-per-rank 2846797792

which gives 157 dense tensors, 1 bucket, dp_size=512 and 8100302848 padding: the bucket
count and dp_size match, and the padding is within 0.2%. Two known gaps, both small.
The tool is one dense tensor short because it does not model MTP's own projection, and
it reports zero expert padding against a measured 19955712, since the expert group here
is uniformly shaped and packs perfectly under this reimplementation.

LIMITATIONS
-----------
Buffers are split on ``is_expert_parallel`` only. The real ``BufferKey`` also keys on
``param_dtype`` and ``grad_dtype``, so under ``--fp8-param-gather`` the fp32 routers split
off into their own buffer while every other dense weight becomes uint8. Padding is
computed per buffer, so merging them here understates it for fp8 configs; the a3b
reference above is bf16, where the split does not arise. Non-Muon parameters (embeddings,
biases, norms, and any mixer weight tagged ``use_muon=False``) are excluded throughout:
they live in the DistributedOptimizer buffers, which use a byte-level layout with no
shard-imbalance padding at all.
"""

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Layer symbols, mirroring megatron.core.models.hybrid.hybrid_layer_allocation.Symbols.
MIXER_SYMBOLS = ("M", "G")
ATTENTION_SYMBOLS = ("*", "D", "+")
MOE_SYMBOL = "E"
MLP_SYMBOL = "-"

# Constants lifted from _compute_per_buffer_param_layout.
PADDING_FLOOR = 0.9
NUMEL_EPSILON = 0.3


@dataclass
class Param:
    """A Muon-managed 2D weight as the layout code sees it."""

    shape: Tuple[int, int]
    gtp_size: int = 1
    is_expert: bool = False

    @property
    def numel(self) -> int:
        """Element count of this rank's local shard."""
        return self.shape[0] * self.shape[1]

    def ns_compute_cost(self) -> int:
        """Newton-Schulz cost on the full post-AllGather shape (GTP shards dim 0)."""
        rows, cols = self.shape[0] * self.gtp_size, self.shape[1]
        big, small = max(rows, cols), min(rows, cols)
        return big * small * small


@dataclass
class BufferResult:
    """Padding outcome for one buffer."""

    tensors: int
    dp_size: int
    buckets: int
    real: int
    padding: int
    max_tensor: int
    shard_compute_loads: List[int] = field(default_factory=list)

    @property
    def overhead(self) -> float:
        """Padding as a fraction of the buffer's real parameters."""
        return self.padding / max(self.real, 1)

    @property
    def compute_imbalance(self) -> float:
        """Max shard Newton-Schulz load over the mean, i.e. the step's stretch factor."""
        if not self.shard_compute_loads:
            return 0.0
        mean = sum(self.shard_compute_loads) / len(self.shard_compute_loads)
        return max(self.shard_compute_loads) / mean if mean else 0.0


def pad_to_divisor(value: int, divisor: int) -> int:
    """Round *value* up to the nearest multiple of *divisor*."""
    return int(math.ceil(value / divisor) * divisor)


def pad_param_start(index: int) -> int:
    """Align a parameter start index to a 64-element boundary."""
    return pad_to_divisor(index, 64)


def shard_divisor(dp_size: int, pad_for_high_nccl_busbw: bool = False) -> int:
    """Per-shard alignment divisor, so ``dp_size * shard_size`` meets bucket-end alignment."""
    bucket_end = (
        math.lcm(dp_size, 128, 2**16) if pad_for_high_nccl_busbw else math.lcm(dp_size, 128)
    )
    return math.lcm(64, bucket_end // dp_size)


def compute_layout(
    params: List[Param],
    bucket_size: Optional[int],
    dp_size: int,
    pad_for_high_nccl_busbw: bool = False,
) -> BufferResult:
    """Bin-pack *params* into ``dp_size`` shards per bucket and total the padding.

    Mirrors ``_compute_per_buffer_param_layout``: walk parameters in reverse model order
    accumulating a chunk, close the chunk once it meets a soft-minimum size *and* the next
    parameter would grow the bucket, then place each chunk by compute-balanced LPT.
    """
    divisor = shard_divisor(dp_size, pad_for_high_nccl_busbw)
    total_padding = 0
    buckets = 0
    # Compute loads persist across buckets so expensive params spread out globally.
    shard_compute_loads = [0] * dp_size

    def emit_bucket(chunk: List[Param]) -> None:
        nonlocal total_padding, buckets
        if not chunk:
            return
        shard_cursors = [0] * dp_size
        chunk_numel = sum(p.numel for p in chunk)
        # Per-bucket numel cap bounds the padding the compute-first ordering can introduce.
        max_shard_numel = chunk_numel / dp_size * (1 + NUMEL_EPSILON)
        for param in sorted(chunk, key=lambda p: -p.ns_compute_cost()):
            candidates = [
                s
                for s in range(dp_size)
                if pad_param_start(shard_cursors[s]) + param.numel <= max_shard_numel
            ]
            if candidates:
                shard = min(candidates, key=lambda s: shard_compute_loads[s])
            else:
                shard = min(range(dp_size), key=lambda s: shard_cursors[s])
            shard_cursors[shard] = pad_param_start(shard_cursors[shard]) + param.numel
            shard_compute_loads[shard] += param.ns_compute_cost()
        padded_shard_size = pad_to_divisor(max(shard_cursors), divisor)
        total_padding += sum(padded_shard_size - cursor for cursor in shard_cursors)
        buckets += 1

    chunk: List[Param] = []
    chunk_numel = 0
    chunk_max_param = 0
    # Mirror the LPT placement incrementally to decide, per param, whether it still fits.
    shard_loads = [0] * dp_size

    def absorbs(numel: int) -> bool:
        """True if *numel* fills existing shard padding rather than growing the bucket."""
        target = pad_to_divisor(max(shard_loads), divisor)
        return pad_param_start(min(shard_loads)) + numel <= target

    def place(numel: int) -> None:
        shard = min(range(dp_size), key=lambda s: shard_loads[s])
        shard_loads[shard] = pad_param_start(shard_loads[shard]) + numel

    for param in reversed(params):
        if bucket_size is not None and chunk:
            threshold = max(bucket_size, int(dp_size * chunk_max_param * PADDING_FLOOR))
            if chunk_numel >= threshold and not absorbs(param.numel):
                emit_bucket(chunk)
                chunk, chunk_numel, chunk_max_param = [], 0, 0
                shard_loads[:] = [0] * dp_size
        place(param.numel)
        chunk.append(param)
        chunk_numel += param.numel
        chunk_max_param = max(chunk_max_param, param.numel)
    emit_bucket(chunk)

    return BufferResult(
        tensors=len(params),
        dp_size=dp_size,
        buckets=buckets,
        real=sum(p.numel for p in params),
        padding=total_padding,
        max_tensor=max((p.numel for p in params), default=0),
        shard_compute_loads=shard_compute_loads,
    )


def build_muon_params(args) -> List[Param]:
    """Build one rank's Muon-managed parameters, in model order.

    Only 2D non-embedding weights are emitted; embeddings, biases, norms and the SSM
    scalars go to the scalar optimizer and live in other buffers entirely.
    """
    hidden = args.hidden_size
    moe_ffn = args.moe_ffn_hidden_size or args.ffn_hidden_size
    latent = args.moe_latent_size
    tp, gtp = args.tensor_model_parallel_size, args.gtp_size
    etp, egtp = args.expert_tensor_parallel_size, args.expert_gtp_size
    local_experts = args.num_experts // args.expert_model_parallel_size if args.num_experts else 0
    d_inner = args.mamba_num_heads * args.mamba_head_dim

    # Bucketing walks parameters in reverse model order, so keep the symbol sequence.
    segments = args.hybrid_layer_pattern.split("/")
    symbols = list(segments[0])
    if args.mtp_num_layers:
        # --mtp-use-repeated-layer builds one layer's params and applies it N times.
        blocks = 1 if args.mtp_use_repeated_layer else args.mtp_num_layers
        for segment in segments[1 : 1 + blocks]:
            symbols.extend(segment)

    params: List[Param] = []
    for symbol in symbols:
        if symbol in MIXER_SYMBOLS:
            # out_proj [hidden, d_inner]: row-parallel over TP, GTP shards dim 0.
            params.append(Param((hidden // gtp, d_inner // tp), gtp))
            if args.include_mixer_in_proj:
                params.append(Param((args.mixer_in_proj_out_features // tp // gtp, hidden), gtp))
        elif symbol in ATTENTION_SYMBOLS:
            qkv_out = (args.num_attention_heads + 2 * args.num_query_groups) * args.kv_channels
            proj_in = args.num_attention_heads * args.kv_channels
            params.append(Param((qkv_out // tp // gtp, hidden), gtp))
            params.append(Param((hidden // gtp, proj_in // tp), gtp))
        elif symbol == MOE_SYMBOL:
            # The router is a bare nn.Parameter, so neither TP nor GTP shards it.
            params.append(Param((args.num_experts, hidden)))
            if latent:
                if args.shard_latent_proj:
                    # --gtp-remat-opt-in-modules moe_latent_proj (NVIDIA/Megatron-LM#6383).
                    params.append(Param((latent // gtp, hidden), gtp))
                    params.append(Param((hidden // gtp, latent), gtp))
                else:
                    # parallel_mode="duplicated" and no gtp_remat_group: fully replicated.
                    params.append(Param((latent, hidden)))
                    params.append(Param((hidden, latent)))
            if args.moe_shared_expert_intermediate_size:
                shared = args.moe_shared_expert_intermediate_size
                params.append(Param((shared // tp // gtp, hidden), gtp))
                params.append(Param((hidden // gtp, shared // tp), gtp))
            expert_in = latent or hidden
            for _ in range(local_experts):
                # TEGroupedLinear registers one weight per local expert (weight0..N).
                params.append(Param((moe_ffn // etp // egtp, expert_in), egtp, is_expert=True))
                params.append(Param((expert_in // etp // egtp, moe_ffn), egtp, is_expert=True))
        elif symbol == MLP_SYMBOL:
            params.append(Param((args.ffn_hidden_size // tp // gtp, hidden), gtp))
            params.append(Param((hidden // gtp, args.ffn_hidden_size // tp), gtp))
        else:
            raise ValueError(f"Unrecognized layer symbol {symbol!r} in --hybrid-layer-pattern")
    return params


def estimate(args) -> Dict[str, BufferResult]:
    """Split params into the expert / non-expert buffers and lay each one out."""
    params = build_muon_params(args)
    total = args.total_params_per_rank or sum(p.numel for p in params)
    bucket_size = total // args.ddp_num_buckets

    # GTP is carved out of the DP axis, so it divides the shard count rather than adding
    # to it: dense DP = world/(TP*GTP), expert DP = world/(ETP*EGTP*EP).
    groups = {
        "dense (ep=False)": (
            [p for p in params if not p.is_expert],
            args.world_size // (args.tensor_model_parallel_size * args.gtp_size),
        ),
        "expert (ep=True)": (
            [p for p in params if p.is_expert],
            args.world_size
            // (
                args.expert_tensor_parallel_size
                * args.expert_gtp_size
                * args.expert_model_parallel_size
            ),
        ),
    }

    results: Dict[str, BufferResult] = {}
    for name, (group, dp_size) in groups.items():
        if not group:
            continue
        if dp_size < 1:
            raise ValueError(
                f"Derived dp_size={dp_size} for the {name} buffer: --world-size is too "
                "small for the requested parallelism."
            )
        results[name] = compute_layout(group, bucket_size, dp_size)
    return results


def _parse_args():
    """Build the CLI, mirroring the Megatron flags the estimate depends on."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    model = parser.add_argument_group("model")
    model.add_argument("--hybrid-layer-pattern", required=True, help="e.g. 'MEMEM*EMEME'")
    model.add_argument("--hidden-size", type=int, required=True)
    model.add_argument("--ffn-hidden-size", type=int, required=True)
    model.add_argument("--moe-ffn-hidden-size", type=int, default=None)
    model.add_argument("--moe-latent-size", type=int, default=None)
    model.add_argument("--num-experts", type=int, default=0)
    model.add_argument("--moe-shared-expert-intermediate-size", type=int, default=0)
    model.add_argument("--kv-channels", type=int, default=128)
    model.add_argument("--num-attention-heads", type=int, default=32)
    model.add_argument("--num-query-groups", type=int, default=8)
    model.add_argument("--mamba-num-heads", type=int, default=16)
    model.add_argument("--mamba-head-dim", type=int, default=64)
    model.add_argument("--mamba-num-groups", type=int, default=16)
    model.add_argument("--mtp-num-layers", type=int, default=0)
    model.add_argument("--mtp-use-repeated-layer", action="store_true")
    model.add_argument(
        "--include-mixer-in-proj",
        action="store_true",
        help="Count the mixer in_proj as Muon-managed. GatedDeltaProduct tags it "
        "use_muon=False, so it is excluded by default.",
    )
    model.add_argument("--mixer-in-proj-out-features", type=int, default=0)

    par = parser.add_argument_group("parallelism")
    par.add_argument("--world-size", type=int, required=True)
    par.add_argument("--tensor-model-parallel-size", type=int, default=1)
    par.add_argument("--expert-model-parallel-size", type=int, default=1)
    par.add_argument("--expert-tensor-parallel-size", type=int, default=1)
    par.add_argument(
        "--tensor-parallel-num-weight-shards",
        type=int,
        default=None,
        help="Total TP x GTP_remat shards per dense weight; GTP degree is this over TP.",
    )
    par.add_argument("--expert-tensor-parallel-num-weight-shards", type=int, default=None)
    par.add_argument("--ddp-num-buckets", type=int, default=8)
    par.add_argument(
        "--total-params-per-rank",
        type=int,
        default=None,
        help="Rank parameter count used for bucket sizing (bucket_size = this // "
        "num_buckets). Defaults to the Muon params alone, which understates it; pass the "
        "value Megatron logs for an exact match.",
    )
    par.add_argument(
        "--shard-latent-proj",
        action="store_true",
        help="Model --gtp-remat-opt-in-modules moe_latent_proj (NVIDIA/Megatron-LM#6383).",
    )

    args = parser.parse_args()
    args.gtp_size = (
        args.tensor_parallel_num_weight_shards // args.tensor_model_parallel_size
        if args.tensor_parallel_num_weight_shards
        else 1
    )
    args.expert_gtp_size = (
        args.expert_tensor_parallel_num_weight_shards // args.expert_tensor_parallel_size
        if args.expert_tensor_parallel_num_weight_shards
        else 1
    )
    return args


def main() -> None:
    """Run the estimate and print a per-buffer table."""
    args = _parse_args()
    results = estimate(args)
    width = max(len(name) for name in results)
    header = (
        f"{'buffer':<{width}} {'tensors':>8} {'dp':>6} {'buckets':>8} {'real':>12} "
        f"{'padding':>12} {'overhead':>9} {'max tensor':>11} {'NS imbal':>9}"
    )
    print(header)
    print("-" * len(header))
    real = padding = 0
    for name, result in results.items():
        real += result.real
        padding += result.padding
        print(
            f"{name:<{width}} {result.tensors:>8} {result.dp_size:>6} {result.buckets:>8} "
            f"{result.real / 1e9:>11.4f}B {result.padding / 1e9:>11.4f}B "
            f"{result.overhead * 100:>8.1f}% {result.max_tensor / 1e6:>10.2f}M "
            f"{result.compute_imbalance:>8.2f}x"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<{width}} {'':>8} {'':>6} {'':>8} {real / 1e9:>11.4f}B "
        f"{padding / 1e9:>11.4f}B {padding / max(real, 1) * 100:>8.1f}%"
    )


if __name__ == "__main__":
    main()
