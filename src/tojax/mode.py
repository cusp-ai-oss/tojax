from __future__ import annotations

import operator

import jax
import jax.numpy as jnp
import torch
import torch.overrides

from tojax.data import tojax_data
from tojax.functions import RANDOM_FUNCTIONS, get_matmul_precision, jax_function
from tojax.wrapper import SymbolicIntWrapper, TensorWrapper, wrap

# Set of all tensor method functions from PyTorch's C++ implementation
_TENSOR_FUNCTIONS = set(torch._C.TensorBase.__dict__.values())


def to_jax_compatible(obj):
    """
    Convert PyTorch tensors and TensorWrappers to JAX-compatible format.

    This function recursively traverses the input object and converts:
    - TensorWrapper instances to their underlying JAX arrays
    - PyTorch tensors to JAX arrays
    - Other objects are left unchanged

    Args:
        obj: Object potentially containing tensors to convert

    Returns:
        Object with tensors converted to JAX arrays
    """

    def _convert(x):
        if (mode := TojaxMode.current()) is not None:
            x = mode.find_replacements(x)
        if isinstance(x, TensorWrapper):
            return x.data
        elif isinstance(x, torch.Tensor):
            return jnp.asarray(x.numpy(force=True))
        else:
            return x

    return jax.tree.map(
        _convert,
        obj,
        is_leaf=lambda x: isinstance(x, (jax.Array, torch.Tensor, TensorWrapper)),
    )


class TojaxMode(torch.overrides.TorchFunctionMode):
    """
    Context manager that intercepts PyTorch function calls and converts them to JAX.

    This class extends PyTorch's TorchFunctionMode to provide automatic translation
    of PyTorch operations to JAX equivalents. It maintains tracking of in-place
    operations and tensor replacements to ensure proper semantics.

    Functions are only translated when at least one of the arguments is a JAX array
    indicating input dependence. If that is not the case, we execute all computations
    in PyTorch. This has the advantage of being very permissive.

    Attributes:
        _inplace_replacements: Mapping from original tensor IDs to their JAX replacements.
        _og_tensors: List of original PyTorch tensors for reference to avoid the GC deleting them.
    """

    _instance: TojaxMode | None = None

    def __init__(self):
        super().__init__()
        self._inplace_replacements = {}
        self._og_tensors = []
        self._ctx = jax.default_matmul_precision(get_matmul_precision().name.lower())
        TojaxMode._instance = self

    @classmethod
    def current(cls) -> TojaxMode | None:
        """Get the current active TojaxMode instance, if any."""
        return cls._instance

    def __enter__(self):
        self._ctx.__enter__()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ctx.__exit__(exc_type, exc_val, exc_tb)
        self._inplace_replacements.clear()
        self._og_tensors.clear()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def find_replacements(self, x):
        """
        Find and apply tensor replacements from in-place operations.

        Args:
            x: Object potentially containing tensors that need replacement

        Returns:
            Object with tensors replaced by their in-place operation results
        """

        def find_replacement(t):
            return self._inplace_replacements.get(id(t), t)

        return jax.tree.map(find_replacement, x)

    def register_tensors(self, og_tensor, operand):
        """
        Register tensor replacements for in-place operation tracking.

        Args:
            og_tensor: Original PyTorch tensor(s)
            operand: Replacement tensor(s) (wrapped JAX arrays)
        """

        def register(a, b):
            if not isinstance(a, torch.Tensor):
                return
            if id(a) in self._inplace_replacements:
                return
            self._inplace_replacements[id(a)] = b
            self._og_tensors.append(a)

        return jax.tree.map(
            register, og_tensor, operand, is_leaf=lambda x: isinstance(x, torch.Tensor)
        )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """
        Intercept PyTorch function calls and translate them to JAX.

        This method is called whenever a PyTorch function is invoked while
        TojaxMode is active. It determines whether to apply JAX translation
        based on the types of arguments involved.

        Args:
            func: PyTorch function being called
            types: Set of types involved in the operation
            args: Positional arguments to the function
            kwargs: Keyword arguments to the function

        Returns:
            Result of the JAX equivalent operation, or original PyTorch result
            if no JAX arrays are involved
        """
        kwargs = kwargs or {}
        og_args = args
        args, kwargs = self.find_replacements((args, kwargs))
        operates_on_arrays = jax.tree.reduce(
            operator.or_,
            jax.tree.map(
                lambda x: isinstance(x, (jax.Array, SymbolicIntWrapper, TensorWrapper)),
                (args, kwargs),
                is_leaf=lambda x: isinstance(
                    x, (jax.Array, SymbolicIntWrapper, TensorWrapper)
                ),
            ),
            False,
        )
        operates_on_arrays |= any(t is TensorWrapper for t in types)
        if operates_on_arrays or func in RANDOM_FUNCTIONS:
            wrapped_args, wrapped_kwargs = wrap(tojax_data((args, kwargs)))
            self.register_tensors((args, kwargs), (wrapped_args, wrapped_kwargs))

            if (
                types in ((torch.Tensor,), (torch.nn.parameter.Parameter,))
                and func.__name__ == "__get__"
            ):
                # we assume meta data like shape and dtype remain unchanged
                return func(og_args[0])
            if func in _TENSOR_FUNCTIONS:  # we are operating on a Tensor method
                return getattr(wrapped_args[0], func.__name__)(
                    *wrapped_args[1:], **wrapped_kwargs
                )
            else:
                return jax_function(func)(*wrapped_args, **wrapped_kwargs)
        else:
            return func(*args, **kwargs)
