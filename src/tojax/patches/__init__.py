# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""
Module patching system for tojax compatibility.

This module provides a framework for registering and applying patches to PyTorch
modules that require special handling for JAX conversion. Patches enable fixing
compatibility issues or optimizing specific modules for JAX execution.
"""

from typing import Callable, overload

import torch.nn as nn

# Global registry mapping module types to their patch functions
_PATCHES: dict[type[nn.Module], Callable[[nn.Module], nn.Module]] = {}


def patch_module[T: nn.Module](module: T) -> T:
    """
    Apply registered patches to a PyTorch module and all its submodules.

    This function traverses the module hierarchy and applies any registered
    patch functions to modules of matching types.

    Args:
        module: PyTorch module to patch

    Returns:
        The patched module (modified in-place)

    Note:
        Patches are applied to all submodules recursively, not just the root module.
    """
    for m in module.modules():
        _PATCHES.get(type(m), lambda x: x)(m)
    return module


@overload
def register_patch[T: nn.Module](
    module_type: type[T], patch_fn: Callable[[T], T]
) -> Callable[[T], T]: ...


@overload
def register_patch[T: nn.Module](
    module_type: type[T], patch_fn: None = None
) -> Callable[[Callable[[T], T]], Callable[[T], T]]: ...


def register_patch[T: nn.Module](
    module_type: type[T], patch_fn: Callable[[T], T] | None = None
) -> Callable[[Callable[[T], T]], Callable[[T], T]] | Callable[[T], T]:
    """
    Register a patch function for a specific PyTorch module type.

    This function can be used as a decorator or called directly to register
    patch functions that will be applied when `patch_module` is called.

    Args:
        module_type: The PyTorch module class to patch
        patch_fn: Optional patch function. If None, returns a decorator.

    Returns:
        If patch_fn is provided: Returns the patch function
        If patch_fn is None: Returns a decorator function

    Raises:
        ValueError: If a patch for the module type is already registered

    Example:
        @register_patch(nn.Linear)
        def patch_linear(module):
            # Modify the linear module for JAX compatibility
            return module
    """

    def decorator(patch_fn: Callable[[T], T]):
        if module_type in _PATCHES:
            raise ValueError(f"Module {module_type} already has a registered patch.")
        _PATCHES[module_type] = patch_fn
        return patch_fn

    if patch_fn is None:
        return decorator
    return decorator(patch_fn)


# Import patch modules to register their patches
from . import e3nn  # noqa
