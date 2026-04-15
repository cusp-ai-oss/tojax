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
from typing import ClassVar


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
