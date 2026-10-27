# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Scatter mode configuration for JAX indexed update operations.

JAX's indexed update operations (e.g., ``jax.Array.at[].set()``) accept a ``mode``
parameter that controls out-of-bounds index behavior. This module provides a
context manager to configure that mode globally during jaxification.

Modes:
    - ``CLIP``: Clamp out-of-bounds indices to the nearest valid index.
    - ``FILL_OR_DROP``: Drop out-of-bounds updates (for writes) or fill with
      a default value (for reads).
    - ``PROMISE_IN_BOUNDS``: Assert that indices are always in bounds (fastest,
      but undefined behavior if violated).
"""

from __future__ import annotations

import enum
from typing import ClassVar, Literal

import jax.numpy as jnp
from jax import lax


class ScatterMode(enum.StrEnum):
    """Mode controlling JAX indexed-update out-of-bounds behavior.

    This enum maps directly to JAX's ``mode`` parameter on
    ``jax.Array.at[].set/add/max/...`` operations.
    """

    CLIP = "clip"
    FILL_OR_DROP = "drop"
    PROMISE_IN_BOUNDS = "promise_in_bounds"


def get_scatter_mode() -> str:
    """Return the active scatter mode string for use in JAX ``.at[]`` calls.

    Returns:
        The mode string (e.g. ``"promise_in_bounds"``).

    Raises:
        AssertionError: If no ``ScatterContext`` is active.
    """
    return ScatterContext.current().value.value


class ScatterContext:
    """Thread-local-style context manager for the active ``ScatterMode``.

    Typically used inside ``tojax()`` rather than directly by end users.

    Example::

        with ScatterContext(ScatterMode.CLIP):
            # all JAX scatter ops inside here use CLIP mode
            ...
    """

    _INSTANCE: ClassVar[ScatterContext | None] = None

    def __init__(self, value: ScatterMode):
        self.value = value

    @classmethod
    def current(cls) -> ScatterContext:
        """Return the currently active context.

        Raises:
            AssertionError: If no context is active.
        """
        assert cls._INSTANCE is not None, (
            "ScatterMode is not active, please check the scatter_mode."
        )
        return cls._INSTANCE

    def __enter__(self):
        self._prev = ScatterContext._INSTANCE
        ScatterContext._INSTANCE = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ScatterContext._INSTANCE = self._prev
        return False


_SCATTER_OPS = {
    "set": lax.scatter,
    "add": lax.scatter_add,
    "multiply": lax.scatter_mul,
    "max": lax.scatter_max,
    "min": lax.scatter_min,
}


def scatter_along_dim(
    input,
    dim: int,
    index,
    src,
    op: Literal["set", "add", "multiply", "max", "min"],
):
    """``torch.Tensor.scatter`` semantics with the reduction ``op``.

    For a 3-D tensor and ``dim == 0``, ``input[index[i][j][k]][j][k] op= src[i][j][k]``;
    ``src`` is read over ``index.shape`` only, and a scalar ``src`` is broadcast. The axes
    of ``index`` other than ``dim`` are scatter batching dimensions, so the update
    carries ``index`` alone rather than one coordinate array per axis. Those arrays are
    ``index.size`` long: a raveled ``index.size >= 2**31`` overflows the int32 shape
    arithmetic of a symbolic export, as in MACE's ``scatter_sum`` over (edges, channels).
    """
    dim = dim % input.ndim
    src = jnp.asarray(src, dtype=input.dtype)
    if src.ndim == 0:
        src = jnp.broadcast_to(src, index.shape)
    else:
        src = src[tuple(slice(0, n) for n in index.shape)]
    # torch allows index to be shorter than input on the other axes: scatter into that block
    block = tuple(
        slice(None) if d == dim else slice(0, n) for d, n in enumerate(index.shape)
    )
    batch = tuple(d for d in range(index.ndim) if d != dim)
    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(dim,),
        scatter_dims_to_operand_dims=(dim,),
        operand_batching_dims=batch,
        scatter_indices_batching_dims=batch,
    )
    updated = _SCATTER_OPS[op](
        input[block], index[..., None], src, dnums, mode=get_scatter_mode()
    )
    return input.at[block].set(updated)
