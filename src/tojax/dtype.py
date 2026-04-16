# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""
Data type conversion utilities between PyTorch and JAX.

This module provides bidirectional conversion between PyTorch and JAX data types,
ensuring compatibility when translating models and operations between the two frameworks.
"""

import jax.numpy as jnp
import torch

# Mapping from PyTorch dtypes to JAX dtypes
_TJ_DTYPE = {
    torch.float16: jnp.float16,
    torch.float32: jnp.float32,
    torch.float64: jnp.float64,
    torch.int8: jnp.int8,
    torch.int16: jnp.int16,
    torch.int32: jnp.int32,
    torch.int64: jnp.int64,
    torch.uint8: jnp.uint8,
    torch.bool: jnp.bool_,
    torch.complex64: jnp.complex64,
    torch.complex128: jnp.complex128,
    torch.bfloat16: jnp.bfloat16,
}

# Reverse mapping from JAX dtypes to PyTorch dtypes
_JT_DTYPE = {v: k for k, v in _TJ_DTYPE.items()}


def jax_dtype(dtype) -> jnp.dtype:
    """
    Convert a PyTorch dtype to the corresponding JAX dtype.

    Args:
        dtype: PyTorch dtype to convert

    Returns:
        Corresponding JAX dtype

    Raises:
        KeyError: If the PyTorch dtype is not supported
    """
    if dtype in _JT_DTYPE:
        return dtype
    return _TJ_DTYPE[dtype]


def torch_dtype(dtype) -> torch.dtype:
    """
    Convert a JAX dtype to the corresponding PyTorch dtype.

    This function first attempts to resolve the dtype by name using torch's
    attribute lookup, falling back to the reverse mapping dictionary.

    Args:
        dtype: JAX dtype to convert (can be dtype object or string)

    Returns:
        Corresponding PyTorch dtype

    Raises:
        KeyError: If the JAX dtype is not supported and can't be resolved by name
    """
    try:
        return getattr(torch, str(dtype))
    except AttributeError:
        return _JT_DTYPE[dtype]
