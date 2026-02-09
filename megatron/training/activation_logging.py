# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Forward activation logging using forward hooks."""

from collections import defaultdict
import torch
import torch.nn as nn

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

from .checkpointing import save_state_dict
from .utils import unwrap_model


def _get_linear_types():
    """Build tuple of linear layer types to capture activations from."""
    types = [nn.Linear, nn.Embedding, ColumnParallelLinear, RowParallelLinear]

    # Add Transformer Engine layers if available.
    try:
        from megatron.core.extensions.transformer_engine import (
            TELinear,
            TEColumnParallelLinear,
            TERowParallelLinear,
            TELayerNormColumnParallelLinear,
        )
        types.extend([TELinear, TEColumnParallelLinear, TERowParallelLinear,
                      TELayerNormColumnParallelLinear])
    except ImportError:
        pass

    try:
        from megatron.core.extensions.transformer_engine import (
            TEGroupedLinear,
            TEColumnParallelGroupedLinear,
            TERowParallelGroupedLinear,
        )
        if TEGroupedLinear is not None:
            types.extend([TEGroupedLinear, TEColumnParallelGroupedLinear,
                          TERowParallelGroupedLinear])
    except ImportError:
        pass

    return tuple(types)


LINEAR_TYPES = _get_linear_types()


class ActivationLogger:
    """Captures and saves forward activations from all linear layers using forward hooks."""

    def __init__(self):
        self._activations_state_dict = defaultdict(dict)
        self._hooks = []

    def _make_hook(self, model_chunk_name: str, module_name: str):
        """Create a forward hook for a named module."""
        def hook(_, input, output):
            input_tuple = input if isinstance(input, tuple) else (input,)
            for idx, inp in enumerate(input_tuple):
                if inp is not None and isinstance(inp, torch.Tensor):
                    name = f"{module_name}/input{idx}"
                    self._activations_state_dict[model_chunk_name][name] = inp.detach().cpu()
            output_tuple = output if isinstance(output, tuple) else (output,)
            for idx, out in enumerate(output_tuple):
                if out is not None and isinstance(out, torch.Tensor):
                    name = f"{module_name}/output{idx}"
                    self._activations_state_dict[model_chunk_name][name] = out.detach().cpu()
        return hook

    def save(self, iteration: int):
        """Save captured activations to disk and clear the buffer."""
        if not self._activations_state_dict:
            return
        save_state_dict(self._activations_state_dict, iteration, "activations")
        self._activations_state_dict.clear()

    def register_hooks(self, model: torch.nn.Module):
        """Find and register forward hooks on all linear layers."""
        assert len(self._hooks) == 0
        for model_chunk_id, model_chunk in enumerate(model):
            unwrapped_model_chunk = unwrap_model(model_chunk)
            for module_name, module in unwrapped_model_chunk.named_modules():
                if isinstance(module, LINEAR_TYPES):
                    model_chunk_name = f"model_chunk{model_chunk_id}"
                    handle = module.register_forward_hook(
                        self._make_hook(model_chunk_name, module_name)
                    )
                    self._hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


_LOGGER = None


def enable_activation_logging(model: torch.nn.Module):
    """Enable activation logging on a model."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = ActivationLogger()
    _LOGGER.register_hooks(model)


def disable_activation_logging():
    """Disable activation logging on a model."""
    global _LOGGER
    assert _LOGGER is not None
    _LOGGER.remove_hooks()


def save_activations(iteration: int):
    """Save activations to disk."""
    global _LOGGER
    assert _LOGGER is not None
    _LOGGER.save(iteration)
