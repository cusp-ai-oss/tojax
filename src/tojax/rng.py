# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Random number generation context for tojax-translated functions.

Provides a context manager that feeds JAX PRNG keys to translated PyTorch
random operations (e.g., ``torch.randn``, ``torch.rand``).
"""

from __future__ import annotations

from typing import ClassVar

import jax


class RNGContext:
    """Context manager that supplies JAX PRNG keys to random operations.

    Each call to ``next_key`` splits the current key and returns a fresh
    sub-key, ensuring reproducible random sequences within a tojax-translated
    function.

    Args:
        key: Initial JAX PRNG key.

    Example::

        with RNGContext(jax.random.PRNGKey(0)):
            key = RNGContext.current().next_key()
    """

    _INSTANCE: ClassVar[RNGContext | None] = None

    def __init__(self, key: jax.Array):
        self._key = key

    @classmethod
    def current(cls) -> RNGContext:
        """Return the currently active context.

        Raises:
            AssertionError: If no context is active.
        """
        assert cls._INSTANCE is not None, (
            "RNGContext is not active, please check the RNGMode."
        )
        return cls._INSTANCE

    def next_key(self) -> jax.Array:
        """Split the current key and return a fresh sub-key."""
        self._key, next_key = jax.random.split(self._key)
        return next_key

    def __enter__(self):
        self._prev = RNGContext._INSTANCE
        RNGContext._INSTANCE = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        RNGContext._INSTANCE = self._prev
        return False
