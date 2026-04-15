"""
tojax: PyTorch to JAX Model Translation Library

This package provides tools for translating PyTorch models and operations to JAX.
The main functionality includes:

- Automatic registration of PyTorch data structures as JAX pytrees
- Function translation from PyTorch to JAX equivalents
- Model wrapper classes for seamless PyTorch-to-JAX conversion

This module automatically registers PyTorch's immutable collections and script
wrapper types as JAX pytrees to enable proper tree traversal and transformation.
"""

import jax
import jax._src.tree_util
import torch
import torch.fx.immutable_collections
import torch.jit._script

from .scatter import ScatterMode
from .tojax import RNGMode, tojax

# Register PyTorch immutable_list as a JAX pytree if not already registered
if torch.fx.immutable_collections.immutable_list not in jax._src.tree_util._registry:
    jax.tree_util.register_pytree_node(
        torch.fx.immutable_collections.immutable_list,
        torch.fx.immutable_collections._immutable_list_flatten,
        lambda ctx, values: list(values),  # type: ignore
    )

# Register PyTorch immutable_dict as a JAX pytree if not already registered
if torch.fx.immutable_collections.immutable_dict not in jax._src.tree_util._registry:
    jax.tree_util.register_pytree_node(
        torch.fx.immutable_collections.immutable_dict,
        torch.fx.immutable_collections._immutable_dict_flatten,
        lambda ctx, values: dict(zip(ctx, values)),  # type: ignore
    )

# Register PyTorch OrderedDictWrapper as a JAX pytree if not already registered
if torch.jit._script.OrderedDictWrapper not in jax._src.tree_util._registry:

    def _unwrap_ordered_dict(x: torch.jit._script.OrderedDictWrapper):
        """
        Unwrap PyTorch's OrderedDictWrapper for JAX pytree registration.

        Args:
            x: OrderedDictWrapper instance to unwrap

        Returns:
            Tuple of (keys, values) for pytree reconstruction
        """
        data = x._c
        keys, values = [], []
        for key, value in data.items():
            keys.append(key)
            values.append(value)
        return tuple(keys), tuple(values)

    jax.tree_util.register_pytree_node(
        torch.jit._script.OrderedDictWrapper,
        _unwrap_ordered_dict,
        lambda ctx, values: dict(zip(ctx, values)),  # type: ignore
    )


__all__ = ["tojax", "RNGMode", "ScatterMode"]
