# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""
Main tojax module for converting PyTorch objects to JAX equivalents.

This module provides the core functionality for translating PyTorch models,
functions, and data to JAX, enabling seamless migration between the two
deep learning frameworks.
"""

from __future__ import annotations

import enum
from copy import deepcopy
from typing import Any, Callable

import jax
import torch

from tojax.data import tojax_data
from tojax.mode import TojaxMode
from tojax.patches import patch_module
from tojax.rng import RNGContext
from tojax.scatter import ScatterContext, ScatterMode
from tojax.wrapper import TensorWrapper, unwrap, wrap


def _convert_module(module: torch.nn.Module):
    """
    Convert a PyTorch module to use JAX arrays for parameters and buffers.

    This function creates a deep copy of the module and recursively replaces
    all PyTorch tensors (parameters and buffers) with TensorWrapper instances
    containing equivalent JAX arrays.

    Args:
        module: PyTorch module to convert

    Returns:
        Converted module with JAX-backed parameters and buffers

    Note:
        The original module structure and hierarchy are preserved, but all
        tensor data is converted to JAX format for compatibility.
    """
    # This function returns a cloned module where we recursively replace all torch tensors
    # with TensorWrappers that contain the original parameters in JAX arrays.
    module = deepcopy(module)
    module = patch_module(module)

    def traverse_module(module: torch.nn.Module):
        module._parameters = jax.tree.map(
            lambda x: TensorWrapper(tojax_data(x)),
            module._parameters,
            is_leaf=lambda x: isinstance(x, (torch.Tensor,)),
        )
        module._buffers = jax.tree.map(
            lambda x: TensorWrapper(tojax_data(x)),
            module._buffers,
            is_leaf=lambda x: isinstance(x, torch.Tensor),
        )
        for child in module.children():
            traverse_module(child)

    traverse_module(module)
    return module


def tojax_fn[**P, R](fn: Callable[P, R], rng_mode: RNGMode, scatter_mode: ScatterMode):
    """Convert a PyTorch function to work with JAX arrays.

    Wraps a function to automatically convert inputs to TensorWrapper format,
    execute within TojaxMode, and convert results back to JAX arrays.

    Args:
        fn: Function to convert for JAX compatibility.
        rng_mode: RAISE or FIXED.
        scatter_mode: Out-of-bounds index behavior for scatter ops.

    Returns:
        Wrapped function that operates on JAX arrays.
    """
    assert rng_mode in (RNGMode.RAISE, RNGMode.FIXED), (
        "rng_mode must be either RAISE or FIXED"
    )

    def wrapper(*args: P.args, **kwargs: P.kwargs):
        args = wrap(args)
        kwargs = wrap(kwargs)
        with ScatterContext(scatter_mode):
            if rng_mode == RNGMode.FIXED:
                with RNGContext(jax.random.PRNGKey(0)):
                    with TojaxMode() as ctx:
                        result = fn(*args, **kwargs)
            else:
                with TojaxMode() as ctx:
                    result = fn(*args, **kwargs)
        return unwrap(tojax(ctx.find_replacements(result)))

    return wrapper


def tojax_fn_with_rng[**P, R](fn: Callable[P, R], scatter_mode: ScatterMode):
    """Wrap a function for JAX compatibility with RNG support.

    Similar to ``tojax_fn``, but additionally manages a JAX random key
    context for functions that require random number generation.

    Args:
        fn: Function to convert for JAX compatibility.
        scatter_mode: Out-of-bounds index behavior for scatter ops.

    Returns:
        Wrapped function that takes a JAX random key as its first argument,
        followed by the original function arguments.
    """

    def wrapper(key: jax.Array, *args: P.args, **kwargs: P.kwargs):
        args = wrap(args)
        kwargs = wrap(kwargs)
        with ScatterContext(scatter_mode):
            with RNGContext(key):
                with TojaxMode() as ctx:
                    result = fn(*args, **kwargs)
        return unwrap(tojax(ctx.find_replacements(result)))

    return wrapper


class RNGMode(enum.Enum):
    RAISE = "raise"
    FIXED = "fixed"
    EXPLICIT = "explicit"


def tojax(
    obj,
    rng_mode: RNGMode = RNGMode.RAISE,
    scatter_mode: ScatterMode = ScatterMode.PROMISE_IN_BOUNDS,
) -> Any:
    """
    Main tojax function - converts PyTorch objects to JAX equivalents.

    This is the primary entry point for converting PyTorch models, functions,
    or data to JAX format. It automatically detects the type of object and
    applies the appropriate conversion strategy.

    Args:
        obj: PyTorch object to convert (Module, function, or data)
        rng_mode: Strategy for handling random number generation in functions
            - RAISE: Raise an error if RNG is used (default)
            - FIXED: Use a fixed random key for all RNG operations
            - EXPLICIT: Expect an explicit random key argument for RNG operations
        scatter_mode: Controls out-of-bounds index behavior for JAX scatter
            (indexed update) operations. See ``ScatterMode`` for details.
            Defaults to ``PROMISE_IN_BOUNDS``.

    Returns:
        JAX-compatible equivalent of the input object

    Conversion types:
    - torch.nn.Module: Converts to JAX-backed module function
    - Callable: Wraps function for JAX compatibility
    - Other data: Converts tensors/dtypes to JAX format
    """
    if isinstance(obj, torch.nn.Module):
        obj = _convert_module(obj)
    if callable(obj):
        match rng_mode:
            case RNGMode.EXPLICIT:
                return tojax_fn_with_rng(obj, scatter_mode)
            case _:
                return tojax_fn(obj, rng_mode, scatter_mode)
    else:
        return tojax_data(obj)
