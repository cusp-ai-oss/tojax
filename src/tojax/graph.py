# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch FX Graph to JAX function conversion.

This module provides functionality to convert PyTorch FX computation graphs
into equivalent JAX functions, enabling the translation of traced PyTorch
models to JAX for improved performance and compatibility.
"""

import jax
import torch
import torch.fx

from tojax.functions import jax_function
from tojax.wrapper import TensorWrapper, unwrap, wrap


def make_jax_fn_from_torch_fx_graph(
    graph: torch.fx.Graph, module: torch.nn.Module | None = None
):
    """
    Convert a PyTorch FX graph into a JAX function.

    This function takes a PyTorch FX computation graph and creates an equivalent
    JAX function that can be compiled and executed using JAX's infrastructure.

    Args:
        graph: PyTorch FX graph representing the computation
        module: Optional PyTorch module that owns the graph, used for attribute access
                when the graph's owning_module is None. Required when the graph contains
                get_attr nodes that reference module parameters or buffers.

    Returns:
        A JAX function that performs the same computation as the FX graph

    The conversion process:
    1. Iterates through each node in the FX graph
    2. Maps PyTorch operations to their JAX equivalents
    3. Handles placeholders, function calls, method calls, and attribute access
    4. Maintains an environment mapping node names to their computed values
    5. Returns the final outputs as unwrapped JAX arrays

    Supported FX node operations:
    - placeholder: Input tensors
    - call_function: Function calls (translated via jax_function)
    - call_method: Method calls on tensor objects
    - get_attr: Attribute access from the owning module
    - output: Final output collection

    Note:
        In-place operations are automatically removed as they're not supported in JAX.
        The function operates within a TojaxMode context for proper tensor handling.
    """
    from tojax.tojax import TojaxMode

    def jax_fn(*inps):
        args_iter = iter(inps)
        env: dict[str, TensorWrapper] = {}
        result: list[TensorWrapper] = []
        with TojaxMode() as ctx:
            for node in graph.nodes:
                args = jax.tree.map(
                    lambda x: env[x.name] if isinstance(x, torch.fx.Node) else x,
                    node.args,
                )
                kwargs = dict(node.kwargs)
                if "inplace" in kwargs:  # not supported in JAX
                    del kwargs["inplace"]
                try:
                    match node.op:
                        case "placeholder":
                            out = wrap(next(args_iter))
                        case "call_function":
                            out = jax_function(node.target)(*args, **kwargs)
                        case "call_method":
                            out = getattr(args[0], node.target)(*args[1:], **kwargs)
                        case "get_attr":
                            owner = (
                                graph.owning_module if graph.owning_module else module
                            )
                            if owner is None:
                                raise ValueError(
                                    "No owning module found, please pass the module parameter."
                                )
                            out = wrap(getattr(owner, node.target))
                        case "output":
                            result += list(args)
                        case _:
                            raise NotImplementedError(f"Unsupported node op: {node.op}")
                except Exception as e:
                    raise RuntimeError(
                        f"Error in node {node.name} with op {node.op} ({node.target}) ({node.args}) ({node.kwargs}): {e}"
                    ) from e
                env[node.name] = out
        result = ctx.find_replacements(result)
        return unwrap(tuple(result))

    return jax_fn
