# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""
Tensor wrapper for PyTorch-JAX compatibility.

This module provides the core TensorWrapper class that enables JAX arrays to behave
like PyTorch tensors, including support for PyTorch's method dispatch system,
in-place operations, and automatic function translation.
"""

import abc
import functools
import logging
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import torch
from jax import export

from tojax.dtype import jax_dtype, torch_dtype
from tojax.scatter import get_scatter_mode
from tojax.utils import get_caller

# fmt: off
_MATH_UNARY_DUNDERS = {
    "__neg__", "__pos__", "__abs__", "__invert__",
}
_MATH_BINARY_DUNDERS = {
    "__add__", "__sub__", "__mul__", "__matmul__", "__truediv__", "__floordiv__", "__mod__", "__divmod__", "__pow__",
}
_MATH_REFLECTED_DUNDERS = {
    "__radd__", "__rsub__", "__rmul__", "__rmatmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rdivmod__", "__rpow__",
    "__rand__", "__ror__", "__rxor__", "__rlshift__", "__rrshift__",
}
_MATH_INPLACE_DUNDERS = {
    "__iadd__", "__isub__", "__imul__", "__imatmul__", "__itruediv__", "__ifloordiv__", "__imod__", "__ipow__",
    "__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__"
}
_MATH_LOGICAL_DUNDERS = {
    "__and__", "__or__", "__xor__"
}
_MATH_BITSHIFT_DUNDERS = {
    "__lshift__", "__rshift__"
}
_MATH_COMPARISON_DUNDERS = {
    "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__"
}
_MATH_DUNDERS = (
    _MATH_UNARY_DUNDERS
    | _MATH_BINARY_DUNDERS
    | _MATH_REFLECTED_DUNDERS
    | _MATH_INPLACE_DUNDERS
    | _MATH_LOGICAL_DUNDERS
    | _MATH_BITSHIFT_DUNDERS
    | _MATH_COMPARISON_DUNDERS
)
# fmt: on


def _allow_axis[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that translates 'axis' keyword argument to 'dim' for PyTorch compatibility.

    Some PyTorch functions accept both 'axis' and 'dim' parameters. This decorator
    automatically converts 'axis' to 'dim' to maintain consistent PyTorch semantics.

    Args:
        func: Function to wrap with axis-to-dim translation

    Returns:
        Wrapped function that accepts 'axis' and converts it to 'dim'

    Raises:
        ValueError: If both 'axis' and 'dim' are specified simultaneously
    """

    # Some functions in torch accept "axis" as argument instead of "dim". This decorator
    # translates axis to dim
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "axis" in kwargs and "dim" in kwargs:
            raise ValueError(
                "Cannot specify both 'axis' and 'dim'. Use 'axis' instead."
            )
        if "axis" in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        return func(*args, **kwargs)

    return wrapper


def wrap(data):
    def is_leaf(x):
        return isinstance(
            x, (jax.Array, SymbolicIntWrapper, TensorWrapper)
        ) or export.is_symbolic_dim(x)

    def _wrap(x):
        if isinstance(x, (torch.Tensor, jax.Array)):
            return TensorWrapper(to_jax_compatible(x))
        elif export.is_symbolic_dim(x):
            return SymbolicIntWrapper(x)
        return x

    return jax.tree.map(_wrap, data, is_leaf=is_leaf)


def unwrap(data):
    def is_leaf(x):
        return isinstance(
            x, (TensorWrapper, SymbolicIntWrapper)
        ) or export.is_symbolic_dim(x)

    def _unwrap(x):
        if isinstance(x, TensorWrapper):
            return x.data
        elif isinstance(x, SymbolicIntWrapper):
            return x.dim
        return x

    return jax.tree.map(_unwrap, data, is_leaf=is_leaf)


def _make_wrapped_operator(op):
    def f(a, b):
        a, b = unwrap((a, b))
        f = getattr(operator, op, getattr(operator, op.replace("__r", "__")))
        return wrap(f(a, b))

    return f


class TojaxDispatchable(abc.ABC):
    """
    Abstract base class for wrapped objects where the torch dispatch will be
    translated to JAX operations.
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        PyTorch function dispatch protocol implementation.

        This method is called whenever a PyTorch function is applied to a TensorWrapper,
        enabling automatic translation from PyTorch operations to JAX equivalents.

        Args:
            func: PyTorch function being called
            types: Types involved in the operation
            args: Positional arguments to the function
            kwargs: Keyword arguments to the function

        Returns:
            Result of the JAX equivalent function, wrapped as TensorWrapper
        """
        from tojax.functions import jax_function

        kwargs = kwargs or {}
        args, kwargs = unwrap((args, kwargs))
        return wrap(jax_function(func)(*args, **kwargs))


class SymbolicIntWrapper(TojaxDispatchable, int):
    """
    A symbolic dimension for representing dynamic shapes.

    This class is used to represent dimensions that are not statically known
    at compile time, allowing for dynamic shape handling in JAX. It must
    inherit from int since torch.arange checks for number types before
    checking for custom __torch_function__. When inheriting from int, one must
    use the __new__ method since int is final.

    Attributes:
        dim: Name of the symbolic dimension
    """

    dim: Any

    def __new__(cls, dim):
        obj = int.__new__(cls, 1)
        obj.dim = dim
        return obj

    def __int__(self):
        return int(self.dim)

    def __str__(self):
        return f"<SymbolicIntWrapper: {self.dim}>"

    def __repr__(self):
        return f"SymbolicIntWrapper({repr(self.dim)})"


# For math operators, we simply add an unwrap and wrap around them and
# dispatch them as usual. Unsupported operators will raise errors as usual.
for op in _MATH_DUNDERS:
    setattr(SymbolicIntWrapper, op, _make_wrapped_operator(op))


@dataclass(frozen=False)  # Mutable since torch has in-place operations
class TensorWrapper(TojaxDispatchable):
    """
    A wrapper that makes JAX arrays behave like PyTorch tensors.

    This class enables seamless integration between JAX and PyTorch by:
    - Implementing PyTorch's tensor interface and semantics
    - Supporting in-place operations through data mutation
    - Providing automatic function dispatch via __torch_function__
    - Maintaining compatibility with PyTorch's method chaining patterns

    Attributes:
        data: The underlying JAX array

    Note:
        This class is mutable (frozen=False) to support PyTorch's in-place
        operations, even though JAX arrays are typically immutable.
    """

    data: jax.Array

    def __hash__(self):
        """
        Return a hash based on object identity.

        Note:
            This is not a valid hash for the tensor data, but allows
            TensorWrapper instances to be used in sets/dicts based on identity.
        """
        # This is obviously not a valid hash but it allows us to identify identical instances
        return id(self)

    @property
    def is_nested(self):
        """Expected by torch - indicates this is not a nested tensor."""
        # Expected by torch
        return False

    @property
    def dtype(self):
        """Return the PyTorch-equivalent dtype of the underlying JAX array."""
        return torch_dtype(self.data.dtype)

    @property
    def ndim(self):
        """Return the number of dimensions."""
        return self.data.ndim

    @property
    def shape(self):
        """Return the shape of the tensor."""
        return wrap(self.data.shape)

    @property
    def T(self):
        """Return the transpose of the tensor."""
        return TensorWrapper(self.data.T)

    @property
    def mT(self):
        """Return the matrix transpose of the tensor."""
        return TensorWrapper(self.data.mT)

    def cpu(self):
        return TensorWrapper(self.data)

    def cuda(self):
        return TensorWrapper(self.data)

    @property
    def device(self):
        """Return default device (JAX manages device placement automatically)."""
        return torch.get_default_device()

    def dim(self):
        """Return the number of dimensions (alias for ndim)."""
        return self.ndim

    def numel(self):
        """Return the total number of elements in the tensor."""
        return wrap(self.data.size)

    def size(self, dim=None):
        """
        Return the size of the tensor.

        Args:
            dim: If specified, return size of that dimension. If None, return full shape.

        Returns:
            Size of specified dimension or full shape tuple
        """
        if dim is None:
            return self.shape
        return self.shape[dim]

    def requires_grad_(self, requires_grad: bool = True):
        """
        Change the requires_grad flag in-place.

        Args:
            requires_grad: Whether gradients should be computed for this tensor

        Returns:
            Self for method chaining

        Note:
            If requires_grad is False, applies stop_gradient to the underlying JAX array.
        """
        if requires_grad:
            return self
        else:
            self.data = jax.lax.stop_gradient(self.data)
            return self

    @property
    def requires_grad(self):
        return True

    def detach(self):
        return TensorWrapper(jax.lax.stop_gradient(self.data))

    @requires_grad.setter
    def requires_grad(self, value):
        if not value:
            self.data = jax.lax.stop_gradient(self.data)

    def to(self, dtype, **_):
        if isinstance(dtype, torch.dtype):
            return TensorWrapper(
                jax.lax.convert_element_type(self.data, jax_dtype(dtype))
            )
        return TensorWrapper(self.data)

    def double(self):
        return TensorWrapper(self.data.astype(jnp.float64))

    def float(self):
        return TensorWrapper(self.data.astype(jnp.float32))

    def int(self):
        return TensorWrapper(self.data.astype(jnp.int32))

    def long(self):
        return TensorWrapper(self.data.astype(jnp.int64))

    def bool(self):
        return TensorWrapper(self.data.astype(jnp.bool_))

    def contiguous(self, *_, **__):
        return self

    def view(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], int):
            shape = shape[0]
        return torch.reshape(self, shape)

    reshape = view

    def permute(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], int):
            shape = shape[0]
        return torch.permute(self, tuple(shape))

    def expand_as(self, other):
        return self.expand(*other.shape)

    def expand(self, *shape):
        return TensorWrapper(expand(self.data, *unwrap(shape)))

    def __add__(self, other):
        return torch.add(self, other)

    def __lt__(self, other):
        return torch.lt(self, other)

    def __le__(self, other):
        return torch.le(self, other)

    def __eq__(self, other):
        return torch.eq(self, other)

    def __ne__(self, other):
        return torch.ne(self, other)

    def __gt__(self, other):
        return torch.gt(self, other)

    def __ge__(self, other):
        return torch.ge(self, other)

    def __matmul__(self, other):
        return torch.matmul(self, other)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __truediv__(self, other):
        return torch.true_divide(self, other)

    def __floordiv__(self, other):
        return torch.floor_divide(self, other)

    def __pow__(self, other):
        return torch.pow(self, other)

    def __radd__(self, other):
        return torch.add(other, self)

    def __rmatmul__(self, other):
        return torch.matmul(other, self)

    def __rmul__(self, other):
        return torch.mul(other, self)

    def __rsub__(self, other):
        return torch.sub(other, self)

    def __rtruediv__(self, other):
        return torch.true_divide(other, self)

    def __rfloordiv__(self, other):
        return torch.floor_divide(other, self)

    def __rmod__(self, other):
        return torch.remainder(other, self)

    def __rdivmod__(self, other):
        return (torch.floor_divide(other, self), torch.remainder(other, self))

    def __rpow__(self, other):
        return torch.pow(other, self)

    def __rlshift__(self, other):
        return torch.bitwise_left_shift(other, self)

    def __rrshift__(self, other):
        return torch.bitwise_right_shift(other, self)

    def __rand__(self, other):
        return torch.logical_and(other, self)

    def __rxor__(self, other):
        return torch.logical_xor(other, self)

    def __ror__(self, other):
        return torch.logical_or(other, self)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __neg__(self):
        return torch.neg(self)

    def __len__(self):
        return self.data.shape[0] if self.ndim > 0 else 1

    def __and__(self, other):
        return torch.logical_and(self, other)

    def __or__(self, other):
        return torch.logical_or(self, other)

    def __xor__(self, other):
        return torch.logical_xor(self, other)

    def __mod__(self, other):
        return torch.remainder(self, other)

    def __invert__(self):
        # not covered by automatic construction
        return torch.logical_not(self)

    def __getattr__(self, name: str):
        if dunder_method := re.match(r"^\_\_(.*)\_\_$", name):
            pure_fn_name = dunder_method.group(1)
            is_inplace = False
        elif in_place_op := re.match(r"^(.*)\_$", name):
            pure_fn_name = in_place_op.group(1)
            is_inplace = True
        else:
            pure_fn_name = name
            is_inplace = False

        if pure_fn_name in self.__dict__:  # hasattr but without recursion
            pure_fn = getattr(self, pure_fn_name)
        elif hasattr(torch, pure_fn_name):
            torch_fn = getattr(torch, pure_fn_name)

            def pure_fn(*args, **kwargs):
                from tojax.data import tojax_data
                from tojax.functions import jax_function

                try:
                    jax_fn = jax_function(torch_fn)
                except NotImplementedError:
                    return torch_fn(self, *args, **kwargs)
                args_w = wrap(tojax_data((self, *args)))
                kwargs_w = wrap(tojax_data(kwargs))
                return jax_fn(*args_w, **kwargs_w)
        else:
            raise AttributeError(
                f"TensorWrapper has no attribute '{name}' {pure_fn_name} {is_inplace}"
            )

        if not is_inplace:
            return pure_fn

        def inplace_fn(*args, **kwargs):
            result = pure_fn(*args, **kwargs)
            if isinstance(result, TensorWrapper):
                self.data = result.data
                return self
            else:
                raise TypeError(
                    f"Inplace operation '{name}' did not return a TensorWrapper"
                )

        return inplace_fn

    def __bool__(self):
        raise ValueError("Data dependent control flow is not supported!")

    def __iter__(self):
        # Required for tuple-unpacking patterns like `a, b, c = x.T` that show
        # up in equivariant libraries. Without an explicit __iter__, Python
        # falls back to __getitem__-with-IndexError iteration which JAX arrays
        # don't terminate cleanly under.
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key):
        return TensorWrapper(self.data.__getitem__(to_jax_compatible(key)))

    def __setitem__(self, key, value):
        key, value = to_jax_compatible((key, value))
        value = jnp.asarray(value)
        if isinstance(key, jax.Array) and key.dtype == jnp.bool_:
            if key.ndim + value.ndim <= self.ndim:
                self.data = jnp.where(
                    key[..., *[None] * (self.ndim - key.ndim)], value, self.data
                )
            else:
                raise ValueError(
                    "__setitem__ with dynamic values are not yet supported for boolean masks."
                )
        else:
            self.data = self.data.at[key].set(value, mode=get_scatter_mode())

    def tolist(self):
        def _list(x):
            return x.tolist()

        _struct = jax.ShapeDtypeStruct((), self.data.dtype)

        def _create_output(shape):
            if len(shape) == 0:
                return _struct
            else:
                return [_create_output(shape[1:]) for _ in range(shape[0])]

        return jax.tree.map(
            TensorWrapper,
            jax.pure_callback(_list, _create_output(self.shape), self.data),
        )

    def new_tensor(self, data, *, dtype=None, **_):
        if isinstance(data, TensorWrapper):
            return TensorWrapper(
                jnp.array(
                    to_jax_compatible(data),
                    dtype=self.data.dtype if dtype is None else jax_dtype(dtype),
                )
            )
        else:
            return torch.tensor(data, dtype=self.dtype if dtype is None else dtype)

    def new_zeros(self, *size, dtype=None, **_):
        if isinstance(size[0], tuple):
            size = size[0]
        return torch.zeros(size, dtype=self.dtype if dtype is None else dtype)

    def new_ones(self, *size, dtype=None, **_):
        if isinstance(size[0], tuple):
            size = size[0]
        return torch.ones(size, dtype=self.dtype if dtype is None else dtype)

    def new_full(self, size, fill_value, *, dtype=None, **_):
        return torch.full(size, fill_value, dtype=dtype or self.dtype)

    sum = _allow_axis(torch.sum)
    mean = _allow_axis(torch.mean)

    def scatter(self, dim, index, src, *, reduce=None):
        """Scatter src values into self at indices specified by index along dim."""
        index = to_jax_compatible(index)
        src = to_jax_compatible(src)
        data = self.data
        ndim = data.ndim

        # Build index arrays for each dimension
        indices = []
        for d in range(ndim):
            if d == dim:
                indices.append(index)
            else:
                shape = [1] * ndim
                shape[d] = index.shape[d]
                idx = jnp.broadcast_to(
                    jnp.arange(index.shape[d]).reshape(shape), index.shape
                )
                indices.append(idx)

        if reduce is None:
            data = data.at[tuple(indices)].set(src, mode=get_scatter_mode())
        elif reduce == "add":
            data = data.at[tuple(indices)].add(src, mode=get_scatter_mode())
        elif reduce == "multiply":
            data = data.at[tuple(indices)].multiply(src, mode=get_scatter_mode())
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")
        return TensorWrapper(data)

    def scatter_(self, dim, index, src, *, reduce=None):
        """Scatter src values into self at indices specified by index along dim."""
        self.data = self.scatter(dim, index, src, reduce=reduce).data
        return self

    def get_device(self):
        return 0

    def item(self):
        if self.data.size != 1:
            raise ValueError(
                "Only tensors with one element can be converted to a Python scalar"
            )
        logging.warning(
            f"Calling item() on a TensorWrapper will not be traced. Instead, 0 will be emitted. Called from:\n{get_caller()}"
        )
        if self.data.dtype == jnp.bool_:
            return False
        elif self.data.dtype in (jnp.float32, jnp.float64):
            return 0.0
        elif self.data.dtype in (
            jnp.uint8,
            jnp.int8,
            jnp.uint16,
            jnp.int16,
            jnp.int32,
            jnp.int64,
        ):
            return 0
        elif self.data.dtype in (jnp.complex64, jnp.complex128):
            return 0.0 + 0.0j
        return self.data.item()

    def repeat(self, *sizes):
        if (
            len(sizes) == 1
            and not isinstance(sizes[0], int)
            and not export.is_symbolic_dim(sizes[0])
        ):
            sizes = sizes[0]
        sizes = unwrap(tuple(sizes))
        if len(sizes) < self.data.ndim:
            raise RuntimeError(
                "Number of dimensions of repeat dims can not be smaller than "
                "number of dimensions of tensor"
            )
        return TensorWrapper(jnp.tile(self.data, sizes))


def to_jax_compatible(x):
    from tojax.mode import to_jax_compatible as _to_jax_compatible

    return _to_jax_compatible(x)


def expand(input, *sizes):
    """
    Expand a JAX array to a new shape using broadcasting.

    This function mimics PyTorch's expand behavior by using JAX's broadcast_to.
    It handles -1 values in the size specification to preserve existing dimensions.

    Args:
        input: JAX array to expand
        *sizes: Target shape. Can be a single tuple or individual dimension sizes.
                Use -1 to preserve the size of that dimension.

    Returns:
        Expanded JAX array with the specified shape

    Note:
        This is a JAX implementation of PyTorch's tensor.expand() method.
    """
    inp_shape = input.shape
    if (
        len(sizes) == 1
        and not isinstance(sizes[0], int)
        and not export.is_symbolic_dim(sizes[0])
    ):
        sizes = sizes[0]
    len_diff = len(sizes) - len(inp_shape)
    new_shape = sizes[:len_diff] + tuple(
        i if i != -1 else j for i, j in zip(sizes[len_diff:], inp_shape)
    )
    return jnp.broadcast_to(input, new_shape)
