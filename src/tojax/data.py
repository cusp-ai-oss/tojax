# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import jax
import jax.numpy as jnp
import torch

from tojax.dtype import jax_dtype
from tojax.wrapper import TensorWrapper


def tojax_data(x):
    """
    Convert PyTorch tensors and data types to JAX equivalents.

    This function recursively traverses an object and converts:
    - PyTorch tensors to JAX arrays (via numpy conversion)
    - PyTorch dtypes to JAX dtypes
    - Other objects are left unchanged

    Args:
        x: Object potentially containing PyTorch tensors/dtypes

    Returns:
        Object with PyTorch data converted to JAX format

    Note:
        Uses force=True on tensor.numpy() to handle tensors that require gradients.
    """

    def translate(x):
        if isinstance(x, torch.Tensor):
            return jnp.asarray(x.numpy(force=True))
        elif isinstance(x, torch.dtype):
            return jax_dtype(x)
        else:
            return x

    return jax.tree.map(translate, x, is_leaf=lambda x: isinstance(x, TensorWrapper))
