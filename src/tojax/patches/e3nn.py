# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""
E3NN (Euclidean Neural Networks) compatibility patches for tojax.

This module provides patches for e3nn modules to enable JAX compatibility,
including specialized handling for spherical harmonics and tensor products
that are core to equivariant neural networks.
"""

import os

# Required by e3nn
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import jax
import jax.numpy as jnp
import torch.fx
import torch.nn as nn

from tojax.graph import make_jax_fn_from_torch_fx_graph
from tojax.patches import register_patch
from tojax.wrapper import TensorWrapper, unwrap, wrap


class PatchModule(nn.Module):
    """
    Wrapper module that enables JAX execution of PyTorch FX graphs.

    This module can execute either PyTorch or JAX versions of a computation
    graph depending on the input tensor types (TensorWrapper vs torch.Tensor).
    """

    def __init__(self, graph_module: torch.fx.GraphModule):
        super().__init__()
        self.graph_module = jitable(graph_module)

    def forward(self, *args, **kwargs):
        """
        Execute the graph with either JAX or PyTorch backend.

        Args:
            *args: Input arguments
            **kwargs: Keyword arguments

        Returns:
            Output tensor(s) in the same format as inputs
        """
        if any(isinstance(arg, TensorWrapper) for arg in args):
            return wrap(
                make_jax_fn_from_torch_fx_graph(self.graph_module.graph)(
                    *unwrap(args), **unwrap(kwargs)
                )[0]
            )
        return self.graph_module(*args, **kwargs)


try:
    import e3nn.o3
    import e3nn.o3._linear
    import e3nn.o3._tensor_product
    import e3nn.o3._tensor_product._codegen
    import e3nn_jax._src.spherical_harmonics
    from opt_einsum_fx import jitable

    e3nn._OPT_DEFAULTS["jit_mode"] = False

    # Store original spherical harmonics implementation
    _og_sph_harm = e3nn.o3._spherical_harmonics._spherical_harmonics

    def _spherical_harmonics[T: torch.Tensor | TensorWrapper](
        lmax: int, x: T, y: T, z: T
    ) -> T:
        """
        JAX-compatible spherical harmonics computation.

        This function replaces e3nn's spherical harmonics with a JAX implementation
        when TensorWrapper inputs are detected, falling back to the original
        PyTorch implementation for regular tensors.

        Args:
            lmax: Maximum spherical harmonic degree
            x, y, z: Coordinate tensors

        Returns:
            Spherical harmonics in the same tensor format as inputs
        """
        if isinstance(x, torch.Tensor):
            return _og_sph_harm(lmax, x, y, z)
        x, y, z = unwrap((x, y, z))
        result = e3nn_jax._src.spherical_harmonics._jited_spherical_harmonics(
            tuple(range(lmax + 1)),
            jnp.stack((x, y, z), axis=-1),
            normalization="norm",
            algorithm=("recursive", "dense", "custom_jvp"),
        )
        result = jax.numpy.concatenate(result, axis=-1)
        return wrap(result)

    # Replace e3nn's spherical harmonics with JAX-compatible version
    e3nn.o3._spherical_harmonics._spherical_harmonics = _spherical_harmonics

    @register_patch(e3nn.o3.Linear)
    def patch_e3nn_linear(module: e3nn.o3.Linear):
        """
        Patch e3nn Linear modules for JAX compatibility.

        Replaces the compiled main function with a PatchModule that can
        execute the same computation graph using JAX when needed.
        """
        # TODO: Determine f_in and f_out.
        module._compiled_main = PatchModule(
            e3nn.o3._linear._codegen_linear(
                module.irreps_in,
                module.irreps_out,
                module.instructions,
                None,
                None,
                module.shared_weights,
                module._optimize_einsums,
            )[0]
        )
        return module

    @register_patch(e3nn.o3.FullTensorProduct)
    @register_patch(e3nn.o3.ElementwiseTensorProduct)
    @register_patch(e3nn.o3.FullyConnectedTensorProduct)
    @register_patch(e3nn.o3.TensorProduct)
    def patch_tensor_product[T: e3nn.o3.TensorProduct](module: T) -> T:
        """
        Patch e3nn TensorProduct modules for JAX compatibility.

        Replaces the compiled computation graphs with PatchModules that can
        execute using either PyTorch or JAX backends depending on input types.

        Args:
            module: e3nn TensorProduct module to patch

        Returns:
            Patched module with JAX-compatible computation graphs
        """
        module._compiled_main_left_right = PatchModule(
            e3nn.o3._tensor_product._codegen.codegen_tensor_product_left_right(
                module.irreps_in1,
                module.irreps_in2,
                module.irreps_out,
                module.instructions,
                module.shared_weights,
                module._specialized_code,
                module._optimize_einsums,
            )
        )
        module._compiled_main_right = PatchModule(
            e3nn.o3._tensor_product._codegen.codegen_tensor_product_right(
                module.irreps_in1,
                module.irreps_in2,
                module.irreps_out,
                module.instructions,
                module.shared_weights,
                module._specialized_code,
                module._optimize_einsums,
            )
        )
        return module

except ImportError:
    # e3nn or e3nn_jax not available - patches will not be registered
    pass
