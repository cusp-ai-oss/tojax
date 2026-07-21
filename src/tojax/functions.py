# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch to JAX function translation module.

This module provides a comprehensive translation system for converting PyTorch functions
to their JAX equivalents. It includes:

- A decorator system for registering PyTorch-to-JAX function mappings
- Support for in-place operations and output parameters
- Automatic keyword argument translation (e.g., dim -> axis, keepdim -> keepdims)
- Implementations for neural network operations, tensor operations, and mathematical functions

The main entry point is the `jax_function` which looks up the appropriate JAX
translation for a given PyTorch function.
"""

import _operator
import logging
from dataclasses import dataclass
from typing import Callable, overload

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.special
import numpy as np
import torch
from jax import export

from tojax.rng import RNGContext
from tojax.scatter import get_scatter_mode
from tojax.wrapper import TensorWrapper, expand, unwrap, wrap

# Global registry mapping PyTorch functions to their JAX equivalents
TRANSLATED_FNS: dict[Callable, Callable] = {}


@dataclass
class JaxFnWithInplace:
    """
    Wrapper for JAX functions that handles PyTorch-style in-place operations and output parameters.

    This class enables JAX functions to support PyTorch semantics for operations like:
    - `out` parameter: Stores result in a pre-allocated tensor
    - `inplace` parameter: Modifies the input tensor in-place

    Args:
        jax_fn: The underlying JAX function to wrap

    The wrapper automatically unwraps TensorWrapper objects, calls the JAX function,
    and re-wraps the result while handling in-place semantics.
    """

    jax_fn: Callable

    def __call__(self, *args: TensorWrapper, **kwargs):
        if (out := kwargs.pop("out", None)) is not None:
            result = self(*args, **kwargs)  # call without out argument
            out.data = result.data
            return out
        elif kwargs.pop("inplace", False):
            unwrapped_args, unwrapped_kwargs = unwrap((args, kwargs))
            assert len(unwrapped_args) == 1, (
                f"Expected 1 argument for inplace operation found {len(unwrapped_args)}"
            )
            result = self.jax_fn(*unwrapped_args, **unwrapped_kwargs)
            args[0].data = result
            return args[0]
        unwrapped_args, unwrapped_kwargs = unwrap((args, kwargs))
        return wrap(self.jax_fn(*unwrapped_args, **unwrapped_kwargs))


@overload
def translates(torch_fn: Callable) -> Callable[[Callable], Callable]: ...


@overload
def translates(torch_fn: Callable, jax_fn: Callable) -> Callable: ...


def translates(
    torch_fn: Callable, jax_fn: Callable | None = None
) -> Callable[[Callable], Callable] | Callable:
    """
    Decorator to register a PyTorch function translation to JAX.

    This function can be used in two ways:
    1. As a decorator: @translates(torch.add) def add_impl(...): ...
    2. Direct mapping: translates(torch.add, jnp.add)

    Args:
        torch_fn: The PyTorch function to translate
        jax_fn: Optional JAX function to map to. If provided, creates direct mapping.
                If None, expects to be used as a decorator.

    Returns:
        If jax_fn is provided: Returns the jax_fn wrapped with in-place support
        If jax_fn is None: Returns a decorator function

    The registered function is automatically wrapped with `JaxFnWithInplace` to
    support PyTorch-style in-place operations and output parameters.
    """

    def decorator(func: Callable) -> Callable:
        TRANSLATED_FNS[torch_fn] = JaxFnWithInplace(func)
        return func

    if jax_fn is not None:
        return decorator(jax_fn)
    else:
        return decorator


def translate_kwargs(jax_fn: Callable):
    """
    Automatically translates common PyTorch keyword arguments to JAX equivalents.

    This function wraps a JAX function to automatically translate PyTorch-style
    keyword arguments to their JAX counterparts:
    - `dims` -> `axes`
    - `dim` -> `axis`
    - `keepdim` -> `keepdims`

    Args:
        jax_fn: The JAX function to wrap with keyword translation

    Returns:
        A wrapped function that accepts PyTorch-style keyword arguments

    Example:
        translate_kwargs(jnp.sum) will accept both dim=0 and axis=0
    """

    def fn(*args, **kwargs):
        if "dims" in kwargs:
            kwargs["axes"] = kwargs.pop("dims")
        if "dim0" in kwargs:
            kwargs["axis1"] = kwargs.pop("dim0")
        if "dim1" in kwargs:
            kwargs["axis2"] = kwargs.pop("dim1")
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
        if "keepdim" in kwargs:
            kwargs["keepdims"] = kwargs.pop("keepdim")
        return jax_fn(*args, **kwargs)

    return fn


def translate_mapped(
    jax_fn: Callable,
    param_map: dict[str, str] | None = None,
    drop: tuple[str, ...] = (),
):
    """Translate torch→JAX with custom parameter name mapping."""
    param_map = param_map or {}

    def fn(*args, **kwargs):
        for src, dst in param_map.items():
            if src in kwargs:
                kwargs[dst] = kwargs.pop(src)
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
        if "keepdim" in kwargs:
            kwargs["keepdims"] = kwargs.pop("keepdim")
        for key in drop:
            kwargs.pop(key, None)
        return jax_fn(*args, **kwargs)

    return fn


def remove_unused_kwargs(func: Callable) -> Callable:
    def fn(*args, **kwargs):
        if "device" in kwargs:
            del kwargs["device"]
        return func(*args, **kwargs)

    return fn


def translate_constructor(torch_fn: Callable, jax_fn: Callable):
    """
    Creates a JAX translation for PyTorch tensor constructor functions.

    This function wraps JAX tensor creation functions to match PyTorch's constructor
    signature, filtering out PyTorch-specific arguments that don't apply to JAX
    (like device, requires_grad, etc.).

    Args:
        torch_fn: The PyTorch constructor function (e.g., torch.zeros, torch.ones)
        jax_fn: The corresponding JAX function (e.g., jnp.zeros, jnp.ones)

    Returns:
        A wrapped function that accepts PyTorch constructor arguments but only
        passes relevant ones (size and dtype) to the JAX function

    Note:
        PyTorch-specific arguments like layout, device, requires_grad, pin_memory,
        and memory_format are ignored as they don't apply to JAX arrays.
    """

    def fn(
        *size,
        out=None,
        dtype=None,
        layout=None,
        device=None,
        requires_grad=False,
        pin_memory=False,
        memory_format=None,
    ):
        if (
            len(size) == 1
            and not isinstance(size[0], int)
            and not export.is_symbolic_dim(size[0])
        ):
            size = size[0]
        return jax_fn(size, dtype=dtype)

    TRANSLATED_FNS[torch_fn] = JaxFnWithInplace(fn)
    return fn


def get_matmul_precision():
    if not torch.backends.cuda.matmul.allow_tf32:
        return jax.lax.Precision.HIGHEST
    match torch.get_float32_matmul_precision():
        case "highest":
            return jax.lax.Precision.HIGHEST
        case "high":
            return jax.lax.Precision.HIGH
        case "medium":
            return jax.lax.Precision.DEFAULT
        case _:
            raise ValueError(
                f"Unsupported PyTorch matmul precision: {torch.get_float32_matmul_precision()}"
            )


def inherit_precision(fn: Callable):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs, precision=get_matmul_precision())

    return wrapper


# =============================================================================
# Basic Function Translations
# =============================================================================
# Direct mappings between PyTorch and JAX functions

translates(torch.concatenate, translate_kwargs(jnp.concatenate))
translates(torch.concat, translate_kwargs(jnp.concatenate))
translates(torch.cat, translate_kwargs(jnp.concatenate))
translates(torch._assert, lambda condition, message: None)

# Tensor constructors
translate_constructor(torch.zeros, jnp.zeros)
translate_constructor(torch.zeros_like, jnp.zeros_like)
translate_constructor(torch.ones, jnp.ones)
translate_constructor(torch.ones_like, jnp.ones_like)
translate_constructor(torch.empty, jnp.empty)
translate_constructor(torch.empty_like, jnp.empty_like)


@translates(torch.full)
def full(
    size,
    fill_value,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
):
    if (
        len(size) == 1
        and not isinstance(size[0], int)
        and not export.is_symbolic_dim(size[0])
    ):
        size = size[0]
    return jnp.full(size, fill_value, dtype=dtype)


@translates(torch.full_like)
def full_like(
    input,
    fill_value,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    memory_format=torch.preserve_format,
):
    return jnp.full_like(input, fill_value, dtype=dtype)


@translates(torch.tensor)
def tensor(data, dtype=None, device=None, requires_grad=False):
    if not requires_grad:
        return jax.lax.stop_gradient(jnp.asarray(data, dtype=dtype))
    return jnp.asarray(data, dtype=dtype)


@translates(torch.as_tensor)
def as_tensor(data, dtype=None, device=None, requires_grad=False):
    if not requires_grad:
        return jax.lax.stop_gradient(jnp.asarray(data, dtype=dtype))
    return jnp.asarray(data, dtype=dtype)


# Mathematical and tensor operations
translates(torch.atleast_1d, jnp.atleast_1d)
translates(torch.atleast_2d, jnp.atleast_2d)
translates(torch.atleast_3d, jnp.atleast_3d)
translates(torch.matmul, inherit_precision(jnp.matmul))
translates(
    torch.mm,
    lambda a, b: jnp.einsum("ab,bc->ac", a, b, precision=get_matmul_precision()),
)
translates(torch.arange, remove_unused_kwargs(jnp.arange))
translates(torch.mul, jnp.multiply)
translates(torch.add, jnp.add)
translates(torch.sub, jnp.subtract)
translates(torch.div, jnp.divide)
translates(torch.dot, inherit_precision(jnp.dot))
translates(torch.vdot, inherit_precision(jnp.vdot))
translates(torch.inner, inherit_precision(jnp.inner))
translates(torch.outer, inherit_precision(jnp.outer))
translates(torch.linalg.multi_dot, inherit_precision(jnp.linalg.multi_dot))
translates(torch.linalg.vecdot, inherit_precision(jnp.vecdot))
translates(torch.floor_divide, jnp.floor_divide)
translates(torch.true_divide, jnp.true_divide)
translates(torch.pow, jnp.pow)
translates(torch.neg, jnp.negative)
translates(torch.lt, jnp.less)
translates(torch.less, jnp.less)
translates(torch.le, jnp.less_equal)
translates(torch.less_equal, jnp.less_equal)
translates(torch.eq, jnp.equal)
translates(torch.equal, jnp.equal)
translates(torch.gt, jnp.greater)
translates(torch.greater, jnp.greater)
translates(torch.ge, jnp.greater_equal)
translates(torch.greater_equal, jnp.greater_equal)
translates(torch.logical_and, jnp.logical_and)
translates(torch.logical_or, jnp.logical_or)
translates(torch.logical_xor, jnp.logical_xor)
translates(torch.logical_not, jnp.logical_not)
translates(torch.ne, jnp.not_equal)
translates(torch.not_equal, jnp.not_equal)
translates(torch.isnan, jnp.isnan)
translates(torch.isinf, jnp.isinf)
translates(torch.isfinite, jnp.isfinite)
translates(torch.maximum, jnp.maximum)
translates(torch.minimum, jnp.minimum)
translates(torch.isclose, jnp.isclose)
translates(torch.abs, jnp.abs)
translates(torch.sin, jnp.sin)
translates(torch.cos, jnp.cos)
translates(torch.tan, jnp.tan)
translates(torch.asin, jnp.arcsin)
translates(torch.acos, jnp.arccos)
translates(torch.atan, jnp.arctan)
translates(torch.tanh, jnp.tanh)
translates(torch.exp, jnp.exp)
translates(torch.log, jnp.log)
translates(torch.log2, jnp.log2)
translates(torch.log10, jnp.log10)
translates(torch.sqrt, jnp.sqrt)
translates(torch.ceil, jnp.ceil)
translates(torch.floor, jnp.floor)
translates(torch.round, jnp.round)
translates(torch.remainder, jnp.remainder)
translates(torch.reciprocal, jnp.reciprocal)
translates(torch.rsqrt, jax.lax.rsqrt)
translates(torch.sign, jnp.sign)
translates(torch.log1p, jnp.log1p)
translates(torch.expm1, jnp.expm1)
translates(torch.square, jnp.square)
translates(torch.sinh, jnp.sinh)
translates(torch.cosh, jnp.cosh)
translates(torch.asinh, jnp.arcsinh)
translates(torch.acosh, jnp.arccosh)
translates(torch.atanh, jnp.arctanh)
translates(torch.sinc, jnp.sinc)
translates(torch.exp2, jnp.exp2)
translates(torch.trunc, jnp.trunc)
translates(torch.deg2rad, jnp.deg2rad)
translates(torch.rad2deg, jnp.rad2deg)
translates(torch.hypot, jnp.hypot)
translates(torch.signbit, jnp.signbit)
translates(torch.negative, jnp.negative)
translates(torch.positive, jnp.positive)
translates(torch.vstack, jnp.vstack)
translates(torch.hstack, jnp.hstack)
translates(torch.diag, jnp.diag)
translates(torch.tril, jnp.tril)
translates(torch.triu, jnp.triu)
translates(torch.tile, jnp.tile)
translates(torch.functional.broadcast_tensors, jnp.broadcast_arrays)
translates(torch.broadcast_to, lambda input, shape: expand(input, *shape))
translates(torch.einsum, inherit_precision(jnp.einsum))
translates(torch._C._nn.gelu, jax.nn.gelu)
translates(torch._C.TensorBase.to, lambda x, *args, **kwargs: x)
translates(
    torch.bmm,
    lambda x, y: inherit_precision(jnp.einsum)("bij,bjk->bik", x, y),
)
translates(torch.where, jnp.where)
translates(torch.clamp, jnp.clip)
translates(torch.atan, jnp.arctan)
translates(torch.atan2, jnp.arctan2)
translates(torch.sin, jnp.sin)
translates(torch.cos, jnp.cos)
translates(torch.tan, jnp.tan)
translates(torch.asin, jnp.arcsin)
translates(torch.acos, jnp.arccos)
translates(torch.transpose, translate_kwargs(jnp.swapaxes))
translates(torch.permute, translate_kwargs(jnp.transpose))
translates(torch.sigmoid, jax.nn.sigmoid)


# Functions requiring keyword argument translation
translates(torch.sum, translate_kwargs(jnp.sum))
translates(torch.mean, translate_kwargs(jnp.mean))
translates(torch.argmax, translate_kwargs(jnp.argmax))
translates(torch.argmin, translate_kwargs(jnp.argmin))
translates(torch.tensordot, translate_kwargs(inherit_precision(jnp.tensordot)))
translates(torch.unsqueeze, translate_kwargs(jnp.expand_dims))
translates(torch.stack, translate_kwargs(jnp.stack))
translates(torch.cumsum, translate_kwargs(jnp.cumsum))
translates(torch.cross, translate_kwargs(jnp.cross))
translates(torch.prod, translate_kwargs(jnp.prod))
translates(torch.all, translate_kwargs(jnp.all))
translates(torch.any, translate_kwargs(jnp.any))
translates(torch.amax, translate_kwargs(jnp.max))
translates(torch.amin, translate_kwargs(jnp.min))
translates(torch.count_nonzero, translate_kwargs(jnp.count_nonzero))
translates(torch.cumprod, translate_kwargs(jnp.cumprod))
translates(torch.nanmean, translate_kwargs(jnp.nanmean))
translates(torch.nansum, translate_kwargs(jnp.nansum))
translates(torch.flip, translate_mapped(jnp.flip, {"dims": "axis"}))
translates(torch.roll, translate_mapped(jnp.roll, {"dims": "axis"}))

translates(torch.det, jnp.linalg.det)
translates(torch.linalg.det, jnp.linalg.det)

# =============================================================================
# Operator Translations
# =============================================================================
# Python operators that need to be translated

translates(getattr, getattr)
translates(_operator.iadd, jax.numpy.add)
translates(_operator.isub, jax.numpy.subtract)
translates(_operator.imul, jax.numpy.multiply)
translates(_operator.imatmul, inherit_precision(jax.numpy.matmul))
translates(_operator.itruediv, jax.numpy.true_divide)
translates(_operator.ifloordiv, jax.numpy.floor_divide)
translates(_operator.imod, jax.numpy.remainder)
translates(_operator.ipow, jax.numpy.power)
translates(_operator.getitem, _operator.getitem)
translates(_operator.add, _operator.add)
translates(_operator.sub, _operator.sub)
translates(_operator.mul, _operator.mul)
translates(_operator.truediv, _operator.truediv)
translates(_operator.pow, _operator.pow)
translates(_operator.neg, _operator.neg)
translates(_operator.eq, _operator.eq)
translates(_operator.ne, _operator.ne)
translates(_operator.lt, _operator.lt)
translates(_operator.le, _operator.le)
translates(_operator.gt, _operator.gt)
translates(_operator.ge, _operator.ge)
translates(_operator.is_, _operator.is_)
translates(_operator.is_not, _operator.is_not)

translates(torch.nn.functional.relu, jax.nn.relu)
translates(torch.nn.functional.silu, jax.nn.silu)


@translates(torch.squeeze)
def squeeze(input, dim=None):
    if dim is None:
        return jnp.squeeze(input)
    # torch.squeeze is a no-op for dims whose size != 1; jnp.squeeze raises.
    if isinstance(dim, int):
        axis = dim if dim >= 0 else input.ndim + dim
        if input.shape[axis] != 1:
            return input
        return jnp.squeeze(input, axis=axis)
    # tuple of dims: filter to only those with size 1
    axes = tuple(d if d >= 0 else input.ndim + d for d in dim)
    axes = tuple(d for d in axes if input.shape[d] == 1)
    if not axes:
        return input
    return jnp.squeeze(input, axis=axes)


@translates(torch.min)
def min(input, dim=None, keepdim=False):
    """
    JAX implementation of PyTorch's min function.

    Args:
        input: Input tensor
        dim: Dimension along which to find minimum. If None, returns global minimum.
        keepdim: Whether to keep the reduced dimensions

    Returns:
        If dim is None: Returns scalar minimum value
        If dim is specified: Returns tuple of (values, indices) like PyTorch
    """
    if dim is not None:
        return jnp.min(input, axis=dim, keepdims=keepdim), jnp.argmin(
            input, axis=dim, keepdims=keepdim
        )
    return jnp.min(input, axis=dim, keepdims=keepdim)


@translates(torch.max)
def max(input, dim=None, keepdim=False):
    """
    JAX implementation of PyTorch's max function.

    Args:
        input: Input tensor
        dim: Dimension along which to find maximum. If None, returns global maximum.
        keepdim: Whether to keep the reduced dimensions

    Returns:
        If dim is None: Returns scalar maximum value
        If dim is specified: Returns tuple of (values, indices) like PyTorch
    """
    if dim is not None:
        return jnp.max(input, axis=dim, keepdims=keepdim), jnp.argmax(
            input, axis=dim, keepdims=keepdim
        )
    return jnp.max(input, axis=dim, keepdims=keepdim)


@translates(torch.sort)
def sort(input, dim=-1, descending=False, stable=False):
    """
    JAX implementation of PyTorch's sort function.

    Args:
        input: Input tensor to sort
        dim: Dimension along which to sort (default: -1)
        descending: If True, sort in descending order (default: False)
        stable: Ignored (JAX sort is stable by default)

    Returns:
        Tuple of (sorted_values, indices) matching PyTorch's behavior
    """
    idx = jnp.argsort(input, axis=dim, descending=descending)
    sorted_input = jnp.take_along_axis(input, idx, axis=dim)
    return sorted_input, idx


@translates(torch.Tensor.split)
@translates(torch.split)
def split(input, split_size_or_sections, dim=0):
    """
    JAX implementation of PyTorch's split function.

    Args:
        input: Input tensor to split
        split_size_or_sections: Either an integer (split size) or list of section sizes
        dim: Dimension along which to split (default: 0)

    Returns:
        List of tensor sections

    Note:
        If split_size_or_sections is an integer, splits into chunks of that size.
        If it's a list, splits into sections of specified sizes.
    """
    if isinstance(split_size_or_sections, int):
        n = input.shape[dim]
        if n - split_size_or_sections < 0:
            return [input]
        else:
            result = []
            idx = [slice(None)] * input.ndim
            idx[dim] = slice(0, split_size_or_sections)
            inv_idx = [slice(None)] * input.ndim
            inv_idx[dim] = slice(split_size_or_sections, None)
            while input.shape[dim] > 0:
                result.append(input[tuple(idx)])
                input = input[tuple(inv_idx)]
            return result
    return jnp.split(input, np.cumsum(split_size_or_sections[:-1]), axis=dim)


@translates(torch.unbind)
def unbind(input, dim):
    """
    JAX implementation of PyTorch's unbind function.

    Args:
        input: Input tensor to unbind
        dim: Dimension along which to unbind

    Returns:
        Tuple of tensors split along the specified dimension
    """
    return tuple(jnp.take(input, i, axis=dim) for i in range(input.shape[dim]))


@translates(torch.index_select)
def index_select(input, dim, index):
    """
    JAX implementation of PyTorch's index_select function.

    Args:
        input: Input tensor
        dim: Dimension along which to select
        index: 1-D tensor containing indices to select

    Returns:
        Tensor with selected indices along the specified dimension
    """
    return jnp.take(input, index, axis=dim)


@translates(torch.nn.functional.max_pool2d)
def max_pool2d(
    input, kernel_size, stride, padding, dilation, ceil_mode, return_indices
):
    """
    JAX implementation of PyTorch's 2D max pooling.

    Args:
        input: Input tensor of shape (N, C, H, W)
        kernel_size: Size of pooling kernel (int or tuple)
        stride: Stride of pooling operation (int or tuple)
        padding: Padding to apply (int or tuple)
        dilation: Must be 1 (dilation not supported)
        ceil_mode: Must be False (ceil mode not supported)
        return_indices: Must be False (return indices not supported)

    Returns:
        Pooled tensor with same layout as input (N, C, H, W)

    Note:
        This implementation converts between PyTorch's NCHW and JAX's NHWC layouts
        internally for compatibility with Flax's pooling operations.
    """
    assert not ceil_mode, "Ceil mode is not supported."
    assert not return_indices, "Return indices is not supported."
    assert dilation == 1, "Dilation is not supported."
    match kernel_size:
        case kernel_size if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        case kernel_size if isinstance(kernel_size, tuple):
            kernel_size = (kernel_size[0], kernel_size[1])
        case _:
            raise ValueError(f"Unsupported kernel_size type: {kernel_size}")
    match stride:
        case stride if isinstance(stride, int):
            stride = (stride, stride)
        case stride if isinstance(stride, tuple):
            stride = (stride[0], stride[1])
        case _:
            raise ValueError(f"Unsupported stride type: {stride}")
    match padding:
        case padding if isinstance(padding, int):
            padding = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        case (padding_h, padding_w):
            padding = ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w))
        case _:
            raise ValueError(f"Unsupported padding type: {padding}")
    input = jnp.pad(input, padding, mode="constant", constant_values=-jnp.inf)
    input = jnp.transpose(input, (0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
    result = nn.max_pool(input, kernel_size, strides=stride, padding="VALID")
    result = jnp.transpose(result, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
    return result


@translates(torch.nn.functional.avg_pool2d)
def avg_pool2d(
    input, kernel_size, stride, padding, dilation, ceil_mode, return_indices
):
    assert not ceil_mode, "Ceil mode is not supported."
    assert not return_indices, "Return indices is not supported."
    assert dilation == 1, "Dilation is not supported."
    match kernel_size:
        case kernel_size if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        case kernel_size if isinstance(kernel_size, tuple):
            kernel_size = (kernel_size[0], kernel_size[1])
        case _:
            raise ValueError(f"Unsupported kernel_size type: {kernel_size}")
    match stride:
        case stride if isinstance(stride, int):
            stride = (stride, stride)
        case stride if isinstance(stride, tuple):
            stride = (stride[0], stride[1])
        case _:
            raise ValueError(f"Unsupported stride type: {stride}")
    if padding > 0:
        padding = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        input = jnp.pad(input, padding, mode="constant", constant_values=-jnp.inf)
    input = jnp.transpose(input, (0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
    result = nn.avg_pool(input, kernel_size, strides=stride, padding="VALID")
    result = jnp.transpose(result, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
    return result


@translates(torch.nn.functional.adaptive_avg_pool2d)
def adaptive_avg_pool2d(input, output_size):
    # TODO: this is not 100% correct, but it works for now.
    inp_size = input.shape[-2:]
    kernel_size = (
        inp_size[0] // output_size[0] + inp_size[0] % output_size[0],
        inp_size[1] // output_size[1] + inp_size[1] % output_size[1],
    )
    stride = (
        inp_size[0] // output_size[0],
        inp_size[1] // output_size[1],
    )
    return avg_pool2d(input, kernel_size, stride, 0, 1, False, False)


_INTERP_MODES = frozenset(
    {"nearest", "nearest-exact", "linear", "bilinear", "trilinear", "bicubic", "area"}
)
_INTERP_LINEAR_MODES = frozenset({"linear", "bilinear", "trilinear"})
_INTERP_ALIGN_MODES = _INTERP_LINEAR_MODES | {"bicubic"}
# Modes restricted to a specific number of spatial dims.
_INTERP_MODE_DIMS = {"linear": 1, "bilinear": 2, "bicubic": 2, "trilinear": 3}


def _interp_pixel_scale(in_size, out_size, align_corners, scale_factor):
    """Source-to-destination scale for a spatial axis (torch coordinate transform)."""
    if align_corners:
        if isinstance(out_size, int) and out_size <= 1:
            return 0.0
        return (in_size - 1) / (out_size - 1)
    if scale_factor is not None and scale_factor > 0:
        return 1.0 / scale_factor
    return in_size / out_size


def _interp_source_index(scale, dst, align_corners, cubic, xp):
    """Map output coordinates ``dst`` to fractional source coordinates."""
    if align_corners:
        return scale * dst
    src = scale * (dst + 0.5) - 0.5
    return src if cubic else xp.maximum(src, 0.0)


def _interp_cubic_coeffs(t, a):
    """Cubic convolution coefficients for the 4 taps at offsets [-1, 0, 1, 2]."""

    def c1(x):
        return ((a + 2) * x - (a + 3)) * x * x + 1

    def c2(x):
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

    return [c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)]


def _interp_aa_filter(dist, mode):
    """Antialiasing separable filter (triangle for bilinear, a=-0.5 cubic)."""
    dist = np.abs(dist)
    if mode == "bilinear":
        return np.where(dist < 1.0, 1.0 - dist, 0.0)
    a = -0.5
    r1 = ((a + 2) * dist - (a + 3)) * dist * dist + 1
    r2 = ((a * dist - 5 * a) * dist + 8 * a) * dist - 4 * a
    return np.where(dist < 1.0, r1, np.where(dist < 2.0, r2, 0.0))


def _interp_out_size(in_size, scale_factor):
    """Output size for a spatial axis given ``scale_factor`` (torch uses floor).

    For a symbolic input size only integer scale factors yield a representable
    symbolic output dimension; fractional factors raise.
    """
    if isinstance(in_size, int):
        return int(np.floor(in_size * scale_factor))
    if float(scale_factor).is_integer():
        return in_size * int(scale_factor)
    raise NotImplementedError(
        "interpolate with a symbolic input size requires an integer scale_factor "
        "or an explicit `size`."
    )


def _interp_index_weights(
    in_size, out_size, mode, align_corners, scale_factor, antialias, xp
):
    """Gather indices and combination weights for resizing one spatial axis.

    Args:
        in_size: Input size along the axis (int or symbolic dim).
        out_size: Output size along the axis (int or symbolic dim).
        mode: Interpolation mode.
        align_corners: Whether corner pixels are aligned (linear/cubic only).
        scale_factor: Scale factor passed to the coordinate transform, or None.
        antialias: Whether to apply the antialiasing filter (bilinear/bicubic).
        xp: Array module to build the arrays with. ``numpy`` bakes constant
            indices/weights for static sizes; ``jax.numpy`` supports symbolic
            sizes (fixed-tap modes only — not area/antialias).

    Returns:
        Tuple ``(indices, weights)`` of arrays with shape ``(out_size, k)``.
        ``weights`` is None for the nearest modes, which reduce to a pure gather.
    """
    int_t = np.int64 if xp is np else jnp.int32
    dst = xp.arange(out_size)
    if mode in ("nearest", "nearest-exact"):
        scale = _interp_pixel_scale(in_size, out_size, False, scale_factor)
        coord = dst if mode == "nearest" else dst + 0.5
        src = xp.minimum(xp.floor(coord * scale).astype(int_t), in_size - 1)
        return src[:, None], None

    scale = _interp_pixel_scale(in_size, out_size, align_corners, scale_factor)

    # area and antialias have a per-output tap count that depends on the (down)
    # sampling ratio, so their gather width must be a Python int — supported for
    # static sizes only (xp is numpy); callers gate symbolic sizes out.
    if antialias:  # bilinear / bicubic
        interp_size = 2 if mode == "bilinear" else 4
        support = interp_size * 0.5 * scale if scale >= 1.0 else interp_size * 0.5
        inv = 1.0 / scale if scale >= 1.0 else 1.0
        center = scale * (dst + 0.5)
        xmin = np.maximum(np.floor(center - support + 0.5).astype(np.int64), 0)
        xmax = np.minimum(np.floor(center + support + 0.5).astype(np.int64), in_size)
        k = int((xmax - xmin).max())
        idx = xmin[:, None] + np.arange(k)[None, :]
        valid = idx < xmax[:, None]
        w = _interp_aa_filter((idx - center[:, None] + 0.5) * inv, mode) * valid
        w = w / w.sum(axis=1, keepdims=True)
        return np.where(valid, idx, in_size - 1), w

    if mode in _INTERP_LINEAR_MODES:
        real = _interp_source_index(scale, dst, align_corners, cubic=False, xp=xp)
        i0 = xp.clip(xp.floor(real).astype(int_t), 0, in_size - 1)
        i1 = xp.where(i0 < in_size - 1, i0 + 1, i0)
        lam1 = real - i0
        return xp.stack([i0, i1], -1), xp.stack([1.0 - lam1, lam1], -1)

    if mode == "bicubic":
        real = _interp_source_index(scale, dst, align_corners, cubic=True, xp=xp)
        i = xp.floor(real).astype(int_t)
        coeffs = _interp_cubic_coeffs(real - i, -0.75)
        indices = xp.stack([xp.clip(i + o, 0, in_size - 1) for o in (-1, 0, 1, 2)], -1)
        return indices, xp.stack(coeffs, -1)

    # area: adaptive average pooling over the corresponding source region.
    starts = np.floor(dst * in_size / out_size).astype(np.int64)
    ends = np.ceil((dst + 1) * in_size / out_size).astype(np.int64)
    counts = ends - starts
    k = int(counts.max())
    idx = starts[:, None] + np.arange(k)[None, :]
    valid = idx < ends[:, None]
    return np.where(valid, idx, in_size - 1), valid / counts[:, None]


def _interp_resize_axis(x, axis, indices, weights):
    """Resize ``x`` along ``axis`` via per-tap gather-and-weighted-sum.

    ``indices``/``weights`` have shape ``(out_size, k)``. Taps are accumulated
    with ``k`` (statically known) 1-D gathers rather than a single 2-D gather
    plus reduction — the latter miscompiles under JAX symbolic shapes.
    """
    if weights is None:  # nearest: pure gather, preserves dtype exactly
        return jnp.take(x, indices[:, 0], axis=axis)
    bshape = (1,) * axis + (weights.shape[0],) + (1,) * (x.ndim - 1 - axis)

    def tap(j):
        w = jnp.asarray(weights[:, j], dtype=x.dtype).reshape(bshape)
        return jnp.take(x, indices[:, j], axis=axis) * w

    out = tap(0)
    for j in range(1, indices.shape[1]):
        out = out + tap(j)
    return out


@translates(torch.nn.functional.interpolate)
def interpolate(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
    antialias=False,
):
    """JAX implementation of PyTorch's ``interpolate`` / upsampling.

    Args:
        input: Tensor of shape ``(N, C, *spatial)`` with 1-3 spatial dims.
        size: Output spatial size (int or per-dim tuple). Mutually exclusive
            with ``scale_factor``.
        scale_factor: Spatial multiplier (float or per-dim tuple).
        mode: One of nearest, nearest-exact, linear, bilinear, bicubic,
            trilinear, area.
        align_corners: Corner-alignment for linear/bilinear/bicubic/trilinear.
        recompute_scale_factor: If True, derive the interpolation scale from the
            computed output size rather than from ``scale_factor``.
        antialias: Apply an antialiasing filter (bilinear/bicubic only).

    Returns:
        The resized tensor of shape ``(N, C, *out_spatial)``.

    Note:
        Symbolic (``jax.export``) spatial sizes are supported for the fixed-tap
        modes (nearest, nearest-exact, linear, bilinear, trilinear, bicubic)
        when the output size is concrete or given by an integer ``scale_factor``.
        The ``area`` mode and ``antialias`` need a per-output tap count derived
        from the sampling ratio, so they require concrete spatial sizes.
    """
    spatial = input.ndim - 2
    if spatial < 1:
        raise ValueError("interpolate expects input with shape (N, C, *spatial).")
    if mode not in _INTERP_MODES:
        raise NotImplementedError(f"interpolate mode '{mode}' is not supported.")
    if mode in _INTERP_MODE_DIMS and spatial != _INTERP_MODE_DIMS[mode]:
        raise ValueError(
            f"interpolate mode '{mode}' expects {_INTERP_MODE_DIMS[mode]} spatial "
            f"dim(s), got {spatial}."
        )
    if align_corners is not None and mode not in _INTERP_ALIGN_MODES:
        raise ValueError(
            "align_corners can only be set for the linear, bilinear, bicubic and "
            "trilinear modes."
        )
    if antialias and mode not in ("bilinear", "bicubic"):
        raise ValueError("antialias is only supported for bilinear and bicubic modes.")
    if (size is None) == (scale_factor is None):
        raise ValueError("Exactly one of size or scale_factor must be set.")

    align = bool(align_corners) if align_corners is not None else False

    in_sizes = [input.shape[2 + i] for i in range(spatial)]
    if size is not None:
        out_sizes = list(size) if isinstance(size, (list, tuple)) else [size] * spatial
        scale_factors = [None] * spatial
    else:
        sf = (
            list(scale_factor)
            if isinstance(scale_factor, (list, tuple))
            else [scale_factor] * spatial
        )
        out_sizes = [_interp_out_size(in_sizes[i], sf[i]) for i in range(spatial)]
        scale_factors = [None] * spatial if recompute_scale_factor else list(sf)
    if len(out_sizes) != spatial or len(scale_factors) != spatial:
        raise ValueError("size / scale_factor length must match the spatial dims.")

    # Weighted modes need float arithmetic; promote integer inputs and restore.
    orig_dtype = input.dtype
    weighted = mode not in ("nearest", "nearest-exact")
    if weighted and not jnp.issubdtype(input.dtype, jnp.floating):
        input = input.astype(jnp.float32)

    for i in range(spatial):
        axis = 2 + i
        # Static sizes use numpy (constant-folded, all modes). Symbolic sizes
        # use jax.numpy, which supports only the fixed-tap modes.
        concrete = isinstance(in_sizes[i], int) and isinstance(out_sizes[i], int)
        if not concrete and (mode == "area" or antialias):
            raise NotImplementedError(
                "interpolate with symbolic shapes supports only the nearest, "
                "nearest-exact, linear, bilinear, trilinear and bicubic modes "
                "(not area or antialias)."
            )
        indices, weights = _interp_index_weights(
            in_sizes[i],
            out_sizes[i],
            mode,
            align,
            scale_factors[i],
            antialias,
            np if concrete else jnp,
        )
        input = _interp_resize_axis(input, axis, indices, weights)

    if weighted and not jnp.issubdtype(orig_dtype, jnp.floating):
        input = jnp.round(input).astype(orig_dtype)
    return input


@translates(torch.nn.functional.conv2d)
@translates(torch.conv2d)
def conv2d(input, weight, bias, stride, padding, dilation, groups):
    assert groups == 1, "Group convolution is not supported."
    if isinstance(stride, int):
        stride = (stride, stride)
    match stride:
        case stride if isinstance(stride, int):
            stride = (stride, stride)
        case stride if isinstance(stride, tuple):
            stride = (stride[0], stride[1])
        case _:
            raise ValueError(f"Unsupported stride type: {stride}")
    match padding:
        case "valid":
            padding = "valid"
        case "same":
            padding = "same"
        case padding if isinstance(padding, int):
            padding = ((padding, padding), (padding, padding))
        case padding if isinstance(padding, tuple):
            padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        case _:
            raise ValueError(f"Unsupported padding type: {padding}")
    match dilation:
        case dilation if isinstance(dilation, int):
            dilation = (dilation, dilation)
        case dilation if isinstance(dilation, tuple):
            dilation = (dilation[0], dilation[1])
        case _:
            raise ValueError(f"Unsupported dilation type: {dilation}")

    result = jax.lax.conv_general_dilated(
        input,
        weight,
        window_strides=stride,
        padding=padding,
        lhs_dilation=None,
        rhs_dilation=dilation,
        precision=get_matmul_precision(),
    )
    if bias is not None:
        result += bias[..., None, None]  # spatial dimensions
    return result


@translates(torch.nn.functional.batch_norm)
def batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps):
    # The default torch batch norm is for images with 4 dimensions, (batch, channels, height, width)
    # We don't support training
    del momentum
    if training:
        running_mean = input.mean((0, 2, 3))
        running_var = input.var((0, 2, 3))

    def normalize(x, mean, var, w, b):
        return (x - mean) / jnp.sqrt(var + eps) * w + b

    # Add spatial dimensions to the input
    normalize = jax.vmap(normalize, in_axes=(-1, None, None, None, None), out_axes=-1)
    normalize = jax.vmap(normalize, in_axes=(-1, None, None, None, None), out_axes=-1)

    return normalize(input, running_mean, running_var, weight, bias)


@translates(torch.nn.functional.layer_norm)
def layer_norm(input, normalized_shape, weight, bias, eps):
    n_dims = len(normalized_shape)
    x = input - input.mean(axis=tuple(range(-n_dims, 0)), keepdims=True)
    x = x / jnp.sqrt(input.var(axis=tuple(range(-n_dims, 0)), keepdims=True) + eps)
    x = x * weight + bias
    return x


@translates(torch.nn.functional.embedding)
def embedding(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    assert padding_idx is None, "padding_idx is not supported yet"
    assert max_norm is None, "max_norm is not supported yet"
    assert norm_type == 2.0, "norm_type other than 2.0 is not supported yet"
    assert not scale_grad_by_freq, "scale_grad_by_freq is not supported yet"
    assert not sparse, "sparse is not supported yet"
    del padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    return jnp.take(weight, input, axis=0)


@translates(torch.index_reduce)
def index_reduce(input, dim, index, source, reduce, *, include_self=True, out=None):
    """JAX equivalent of torch.index_reduce."""
    del out  # Not supported in JAX
    idx = [slice(None)] * input.ndim
    idx[dim] = index

    if not include_self:
        # Initialize with identity element for the reduction
        identity = {
            "prod": 1,
            "mean": 0,
            "amax": -jnp.inf,
            "amin": jnp.inf,
        }
        input = input.at[tuple(idx)].set(identity[reduce], mode=get_scatter_mode())

    if reduce == "prod":
        return input.at[tuple(idx)].multiply(source, mode=get_scatter_mode())
    elif reduce == "mean":
        # Mean requires counting elements and dividing
        counts = (
            jnp.ones(input.shape, dtype=input.dtype)
            if include_self
            else jnp.zeros(input.shape, dtype=input.dtype)
        )
        counts = counts.at[tuple(idx)].add(
            jnp.ones_like(source), mode=get_scatter_mode()
        )
        sums = input.at[tuple(idx)].add(source, mode=get_scatter_mode())
        # Only apply mean to indexed positions
        return jnp.where(counts > (1 if include_self else 0), sums / counts, input)
    elif reduce == "amax":
        return input.at[tuple(idx)].max(source, mode=get_scatter_mode())
    elif reduce == "amin":
        return input.at[tuple(idx)].min(source, mode=get_scatter_mode())
    else:
        raise ValueError(f"Unsupported reduce operation: {reduce}")


@translates(torch.nn.functional.linear)
def linear(input, weight, bias):
    """
    JAX implementation of PyTorch's linear layer.

    Args:
        input: Input tensor of shape (..., in_features)
        weight: Weight matrix of shape (out_features, in_features)
        bias: Optional bias vector of shape (out_features,)

    Returns:
        Output tensor of shape (..., out_features)

    Note:
        Uses Einstein summation for efficient matrix multiplication
        compatible with arbitrary batch dimensions.
    """
    match weight.ndim:
        case 1:
            result = jnp.einsum(
                "...d,d->...", input, weight, precision=get_matmul_precision()
            )
        case 2:
            result = jnp.einsum(
                "...d,cd->...c", input, weight, precision=get_matmul_precision()
            )
        case _:
            raise ValueError(f"Unsupported weight shape: {weight.shape}")
    if bias is not None:
        result += bias
    return result


@translates(torch.nn.functional.dropout)
def dropout(input, p, training):
    if not training:
        return input
    assert p == 0.0, "Dropout is not supported in JAX."
    return input


@translates(torch.flatten)
def flatten(input, start_dim=0, end_dim=-1):
    """
    JAX implementation of PyTorch's flatten function.

    Args:
        input: Input tensor to flatten
        start_dim: First dimension to flatten (default: 0)
        end_dim: Last dimension to flatten (default: -1)

    Returns:
        Flattened tensor with specified dimensions collapsed

    Note:
        Dimensions from start_dim to end_dim (inclusive) are flattened
        into a single dimension while preserving other dimensions.
    """
    out_shape = list(input.shape)
    end_dim = end_dim % len(out_shape)
    for i in range(end_dim, start_dim - 1, -1):
        del out_shape[i]
    out_shape.insert(start_dim, -1)
    return jnp.reshape(input, out_shape)


@translates(torch.narrow)
def narrow(input, dim, start, length):
    size = input.shape[dim]
    index = [slice(None)] * input.ndim
    start = start % size
    index[dim] = slice(start, start + length, None)
    return input[*index]


def dot_product_attention(
    query, key, value, bias=None, mask=None, *, scale=None, is_causal=False
):
    """Precision-preserving drop-in for ``jax.nn.dot_product_attention``.

    For float32 queries this dispatches to ``jax.nn.dot_product_attention`` to
    use its hardware-optimized kernels. For other dtypes it uses a custom
    implementation whose softmax runs in the input dtype, since
    ``jax.nn.dot_product_attention`` always carries the softmax out in fp32 and
    would lose precision (e.g. for float64).

    Grouped-query attention (fewer key/value heads than query heads) is
    supported when the query head count is a multiple of the key/value head
    count. Query rows whose keys are all masked out produce a zero output
    (safe softmax), matching ``torch.nn.functional.scaled_dot_product_attention``.

    Args:
        query: Query array of shape ``(..., T, N, H)``.
        key: Key array of shape ``(..., S, K, H)`` with ``N`` a multiple of ``K``.
        value: Value array of shape ``(..., S, K, H)``.
        bias: Optional additive bias broadcastable to ``(..., N, T, S)``.
        mask: Optional boolean mask broadcastable to ``(..., N, T, S)`` where
            ``True`` marks positions that take part in attention.
        scale: Logit scale. Defaults to ``1 / sqrt(H)``.
        is_causal: If ``True``, applies a causal (lower-triangular) mask.

    Returns:
        Attention output with the same shape as ``query``.
    """
    if query.dtype == jnp.float32:
        out = jax.nn.dot_product_attention(
            query, key, value, bias=bias, mask=mask, scale=scale, is_causal=is_causal
        )
    else:
        head_dim = query.shape[-1]
        scale = 1.0 / np.sqrt(head_dim) if scale is None else scale
        n_q, n_kv = query.shape[-2], key.shape[-2]
        if n_q != n_kv:
            key = jnp.repeat(key, n_q // n_kv, axis=-2)
            value = jnp.repeat(value, n_q // n_kv, axis=-2)
        einsum = inherit_precision(jnp.einsum)
        logits = einsum("...TNH,...SNH->...NTS", query, key)
        logits = logits * jnp.asarray(scale, logits.dtype)
        if bias is not None:
            logits = logits + bias
        if mask is not None or is_causal:
            keep = jnp.ones(logits.shape, dtype=bool)
            if mask is not None:
                keep = keep & mask
            if is_causal:
                t, s = logits.shape[-2], logits.shape[-1]
                keep = keep & jnp.tril(jnp.ones((t, s), dtype=bool))
            neg = jnp.asarray(-0.7 * jnp.finfo(logits.dtype).max, logits.dtype)
            logits = jnp.where(keep, logits, neg)
        probs = jax.nn.softmax(logits, axis=-1)
        out = einsum("...NTS,...SNH->...TNH", probs, value)

    # Safe softmax: zero the output of query rows whose keys are all masked,
    # which would otherwise be NaN (or uniform). is_causal alone never fully
    # masks a row, so this only matters when a bias or mask is present.
    if bias is not None or mask is not None:
        t, s = query.shape[-3], key.shape[-3]
        allowed = jnp.ones((t, s), dtype=bool)
        if mask is not None:
            allowed = allowed & mask
        if bias is not None:
            allowed = allowed & (bias != -jnp.inf)
        if is_causal:
            allowed = allowed & jnp.tril(jnp.ones((t, s), dtype=bool))
        masked_rows = jnp.swapaxes(~jnp.any(allowed, axis=-1), -1, -2)[..., None]
        out = jnp.where(masked_rows, jnp.zeros_like(out), out)
    return out


def _mha_canonical_mask(mask, dtype):
    """Canonicalize a MultiheadAttention mask to an additive float mask.

    Boolean masks are converted so ``True`` positions become ``-inf`` (masked
    out) and ``False`` positions become ``0``; float masks are returned
    unchanged and used as additive biases.

    Args:
        mask: Boolean or float mask, or ``None``.
        dtype: Target float dtype for converted boolean masks.

    Returns:
        Additive float mask, or ``None`` if ``mask`` is ``None``.
    """
    if mask is None:
        return None
    if mask.dtype == jnp.bool_:
        return jnp.where(mask, jnp.asarray(-jnp.inf, dtype), jnp.asarray(0.0, dtype))
    return mask


@translates(torch.nn.functional.multi_head_attention_forward)
def multi_head_attention_forward(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: jax.Array | None,
    in_proj_bias: jax.Array | None,
    bias_k: jax.Array | None,
    bias_v: jax.Array | None,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: jax.Array,
    out_proj_bias: jax.Array | None,
    training: bool = True,
    key_padding_mask: jax.Array | None = None,
    need_weights: bool = True,
    attn_mask: jax.Array | None = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: jax.Array | None = None,
    k_proj_weight: jax.Array | None = None,
    v_proj_weight: jax.Array | None = None,
    static_k: jax.Array | None = None,
    static_v: jax.Array | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
):
    """JAX implementation of PyTorch's ``multi_head_attention_forward``.

    Faithfully mirrors PyTorch semantics for query/key/value projection,
    attention masking, and output projection. When ``need_weights`` is ``False``
    the attention core is computed via :func:`dot_product_attention`; otherwise
    it is computed explicitly so the attention weights can be returned.

    Args:
        query: Query of shape ``(L, E)`` (unbatched) or ``(L, N, E)``.
        key: Key of shape ``(S, E)`` or ``(S, N, E)``.
        value: Value of shape ``(S, E)`` or ``(S, N, E)``.
        embed_dim_to_check: Expected embedding dimension ``E``.
        num_heads: Number of attention heads.
        in_proj_weight: Packed ``(3E, E)`` input projection weight, or ``None``
            when ``use_separate_proj_weight`` is set.
        in_proj_bias: Packed ``(3E,)`` input projection bias, or ``None``.
        bias_k: Optional ``(1, 1, E)`` bias appended to the key sequence.
        bias_v: Optional ``(1, 1, E)`` bias appended to the value sequence.
        add_zero_attn: If ``True``, append a zero key/value along the source dim.
        dropout_p: Attention dropout probability; only ``0`` is supported.
        out_proj_weight: Output projection weight of shape ``(E, E)``.
        out_proj_bias: Optional output projection bias of shape ``(E,)``.
        training: If ``False``, dropout is disabled.
        key_padding_mask: Optional ``(N, S)`` (or ``(S,)`` unbatched) mask.
        need_weights: If ``True``, also return the attention weights.
        attn_mask: Optional 2D ``(L, S)`` or 3D ``(N*num_heads, L, S)`` mask.
        use_separate_proj_weight: If ``True``, use ``q/k/v_proj_weight`` instead
            of ``in_proj_weight``.
        q_proj_weight: Query projection weight when using separate weights.
        k_proj_weight: Key projection weight when using separate weights.
        v_proj_weight: Value projection weight when using separate weights.
        static_k: Optional precomputed key of shape ``(N*num_heads, S, E/num_heads)``.
        static_v: Optional precomputed value of shape ``(N*num_heads, S, E/num_heads)``.
        average_attn_weights: If ``True``, average returned weights over heads.
        is_causal: If ``True``, apply a causal mask (requires ``attn_mask``).

    Returns:
        Tuple ``(attn_output, attn_output_weights)`` where the weights are
        ``None`` when ``need_weights`` is ``False``.
    """
    assert dropout_p == 0.0 or not training, "Dropout is not supported."

    is_batched = query.ndim == 3
    if not is_batched:
        query, key, value = query[:, None], key[:, None], value[:, None]
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[None]

    tgt_len, bsz, embed_dim = query.shape
    dtype = query.dtype
    assert embed_dim == embed_dim_to_check, (
        f"was expecting embedding dimension of {embed_dim_to_check}, got {embed_dim}"
    )
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, (
        f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    )

    key_padding_mask = _mha_canonical_mask(key_padding_mask, dtype)
    if is_causal and attn_mask is None:
        raise RuntimeError("Need attn_mask if specifying the is_causal hint.")
    if is_causal and key_padding_mask is None and not need_weights:
        attn_mask = None
    else:
        attn_mask = _mha_canonical_mask(attn_mask, dtype)
        if key_padding_mask is not None:
            is_causal = False

    # In-projection.
    if in_proj_bias is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = jnp.split(in_proj_bias, 3)
    if not use_separate_proj_weight:
        w_q, w_k, w_v = jnp.split(in_proj_weight, 3)
    else:
        w_q, w_k, w_v = q_proj_weight, k_proj_weight, v_proj_weight
    q = linear(query, w_q, b_q)
    k = linear(key, w_k, b_k)
    v = linear(value, w_v, b_v)

    # Ensure attn_mask is at least 3D: (1, L, S) or (N*num_heads, L, S).
    if attn_mask is not None and attn_mask.ndim == 2:
        attn_mask = attn_mask[None]

    # Append bias_k/bias_v along the source sequence dimension.
    if bias_k is not None and bias_v is not None:
        k = jnp.concatenate([k, jnp.tile(bias_k, (1, bsz, 1))], axis=0)
        v = jnp.concatenate([v, jnp.tile(bias_v, (1, bsz, 1))], axis=0)
        if attn_mask is not None:
            attn_mask = jnp.pad(attn_mask, ((0, 0), (0, 0), (0, 1)))
        if key_padding_mask is not None:
            key_padding_mask = jnp.pad(key_padding_mask, ((0, 0), (0, 1)))

    # Reshape to (bsz*num_heads, seq, head_dim).
    q = q.reshape(tgt_len, bsz * num_heads, head_dim).swapaxes(0, 1)
    if static_k is None:
        k = k.reshape(k.shape[0], bsz * num_heads, head_dim).swapaxes(0, 1)
    else:
        k = static_k
    if static_v is None:
        v = v.reshape(v.shape[0], bsz * num_heads, head_dim).swapaxes(0, 1)
    else:
        v = static_v

    # Append zero key/value for add_zero_attn.
    if add_zero_attn:
        zero_shape = (bsz * num_heads, 1, head_dim)
        k = jnp.concatenate([k, jnp.zeros(zero_shape, k.dtype)], axis=1)
        v = jnp.concatenate([v, jnp.zeros(zero_shape, v.dtype)], axis=1)
        if attn_mask is not None:
            attn_mask = jnp.pad(attn_mask, ((0, 0), (0, 0), (0, 1)))
        if key_padding_mask is not None:
            key_padding_mask = jnp.pad(key_padding_mask, ((0, 0), (0, 1)))

    src_len = k.shape[1]

    # Merge key_padding_mask into the additive attention mask.
    if key_padding_mask is not None:
        kpm = jnp.broadcast_to(
            key_padding_mask.reshape(bsz, 1, 1, src_len),
            (bsz, num_heads, 1, src_len),
        ).reshape(bsz * num_heads, 1, src_len)
        attn_mask = kpm if attn_mask is None else attn_mask + kpm

    mm = inherit_precision(jnp.matmul)
    if need_weights:
        q_scaled = q * float(np.sqrt(1.0 / head_dim))
        scores = mm(q_scaled, k.swapaxes(-2, -1))
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = jax.nn.softmax(scores, axis=-1)
        attn_output = mm(weights, v).swapaxes(0, 1).reshape(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.reshape(tgt_len, bsz, -1)
        weights = weights.reshape(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            weights = weights.mean(axis=1)
        if not is_batched:
            attn_output, weights = attn_output[:, 0], weights[0]
        return attn_output, weights

    # need_weights is False: use the fused attention core.
    if attn_mask is not None:
        bias = (
            attn_mask[None]
            if attn_mask.shape[0] == 1
            else attn_mask.reshape(bsz, num_heads, -1, src_len)
        )
    else:
        bias = None
    qh = q.reshape(bsz, num_heads, tgt_len, head_dim).swapaxes(1, 2)
    kh = k.reshape(bsz, num_heads, src_len, head_dim).swapaxes(1, 2)
    vh = v.reshape(bsz, num_heads, src_len, head_dim).swapaxes(1, 2)
    attn_output = dot_product_attention(
        qh, kh, value=vh, bias=bias, is_causal=is_causal and attn_mask is None
    )
    attn_output = attn_output.swapaxes(0, 1).reshape(tgt_len, bsz, -1)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if not is_batched:
        attn_output = attn_output[:, 0]
    return attn_output, None


@translates(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """JAX implementation of PyTorch's ``scaled_dot_product_attention``.

    Dispatches to :func:`dot_product_attention`, which preserves precision for
    non-float32 dtypes. PyTorch's ``(..., num_heads, L, E)`` layout is converted
    to the ``(..., L, num_heads, E)`` layout expected by that helper.

    Args:
        query: Query of shape ``(..., num_heads, L, E)``.
        key: Key of shape ``(..., num_heads, S, E)``.
        value: Value of shape ``(..., num_heads, S, E)``.
        attn_mask: Optional boolean (``True`` attends) or additive float mask
            broadcastable to ``(..., num_heads, L, S)``.
        dropout_p: Dropout probability; only ``0`` is supported.
        is_causal: If ``True``, applies a causal mask.
        scale: Logit scale. Defaults to ``1 / sqrt(E)``.
        enable_gqa: If ``True``, enables grouped-query attention.

    Returns:
        Attention output in the same layout as ``query``.
    """
    del enable_gqa  # Grouped-query attention is inferred from head counts.
    assert dropout_p == 0.0, "Dropout is not supported."
    q, k, v = (jnp.swapaxes(x, -3, -2) for x in (query, key, value))
    bias = mask = None
    if attn_mask is not None:
        if attn_mask.dtype == jnp.bool_:
            mask = attn_mask
        else:
            bias = attn_mask
    out = dot_product_attention(
        q, k, v, bias=bias, mask=mask, scale=scale, is_causal=is_causal
    )
    return jnp.swapaxes(out, -3, -2)


@translates(torch._C._linalg.linalg_norm)
def linalg_norm(input, ord=None, dim=None, keepdim=False, *, out=None, dtype=None):
    return jnp.linalg.norm(
        input,
        ord=ord,
        axis=dim,
        keepdims=keepdim,
    )


@translates(torch.scatter_add)
def scatter_add(input, dim, index, src):
    idx_l = jnp.meshgrid(*[jnp.arange(s) for s in index.shape][::-1])
    idx_l = [i.ravel() for i in idx_l[::-1]]
    idx_l[dim] = index.ravel()
    return input.at[*idx_l].add(src.ravel(), mode=get_scatter_mode())


@translates(torch.index_add)
@translates(torch._C.TensorBase.index_add)
def index_add(self, dim: int, index, source, *, alpha=1):
    idx = [slice(None)] * self.ndim
    idx[dim] = index
    return self.at[tuple(idx)].add(source * alpha, mode=get_scatter_mode())


@translates(torch._C.TensorBase.view)
@translates(torch._C.TensorBase.reshape)
@translates(torch.reshape)
def reshape(input, *shape):
    if len(shape) == 1 and not isinstance(shape[0], int):
        shape = shape[0]
    return jnp.reshape(input, shape)


@translates(torch.nn.functional.normalize)
def normalize(input, p=2, dim=1, eps=1e-12, out=None):
    norm = jnp.linalg.norm(input, ord=p, axis=dim, keepdims=True)
    norm = jnp.maximum(norm, eps)
    return input / norm


@translates(torch.clone)
def clone(input, *, memory_format=None):
    return jnp.copy(input)


@translates(torch.autograd.grad)
def grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph=None,
    create_graph=False,
    only_inputs=True,
    allow_unused=None,
    is_grads_batched=False,
    materialize_grads=False,
):
    """
    Placeholder implementation for PyTorch's autograd.grad function.

    Args:
        outputs: Outputs with respect to which gradients are computed
        inputs: Inputs with respect to which gradients are computed
        grad_outputs: Gradient of outputs (ignored)
        retain_graph: Whether to retain computation graph (ignored)
        create_graph: Whether to create graph for higher-order derivatives (ignored)
        only_inputs: Whether to return only input gradients (ignored)
        allow_unused: Whether to allow unused inputs (ignored)
        is_grads_batched: Whether gradients are batched (ignored)
        materialize_grads: Whether to materialize gradients (ignored)

    Returns:
        Zero tensors with same shapes as inputs

    Warning:
        This is a placeholder that returns zeros. Automatic differentiation
        in JAX should be handled through jax.grad or jax.value_and_grad
        rather than relying on PyTorch's autograd system.
    """
    logging.warning("torch.autograd.grad cannot be tojax-translated.")
    return jax.tree.map(jnp.zeros_like, inputs)


@translates(torch.nn.functional.pad)
def pad(input, pad, mode="constant", value=None):
    if value is None:
        value = 0

    # Convert PyTorch pad format to JAX pad_width format
    # PyTorch: (left, right, top, bottom, ...) from last dim
    # JAX: ((before_0, after_0), (before_1, after_1), ...) from first dim
    ndim = input.ndim
    pad_width = [(0, 0)] * ndim
    for i in range(0, len(pad), 2):
        dim = ndim - 1 - (i // 2)
        pad_width[dim] = (pad[i], pad[i + 1])

    # Map PyTorch modes to JAX modes
    mode_map = {
        "constant": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }
    jax_mode = mode_map.get(mode)
    if jax_mode is None:
        raise ValueError(f"Unsupported padding mode: {mode}")

    if jax_mode == "constant":
        return jnp.pad(input, pad_width, mode=jax_mode, constant_values=value)
    return jnp.pad(input, pad_width, mode=jax_mode)


@translates(torch.bucketize)
def bucketize(input, boundaries, *, out_int32=False, right=False, out=None):
    del out
    dtype = jnp.int32 if out_int32 else jnp.int64
    side = "right" if right else "left"

    # Check if boundaries has symbolic shape
    has_symbolic = not all(isinstance(d, int) for d in boundaries.shape)
    if has_symbolic:
        # Fallback: comparison-based counting for symbolic shape compatibility
        # (searchsorted requires concrete array length for binary search)
        input_expanded = jnp.expand_dims(input, axis=-1)
        if right:
            result = jnp.sum(boundaries <= input_expanded, axis=-1)
        else:
            result = jnp.sum(boundaries < input_expanded, axis=-1)
    else:
        result = jnp.searchsorted(boundaries, input, side=side)
    return result.astype(dtype)


@translates(torch.scatter_reduce)
def scatter_reduce(input, dim, index, src, reduce, *, include_self=True):
    ndim = input.ndim

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
    indices = tuple(indices)

    if include_self:
        # Standard case: include input values in reduction
        if reduce == "sum":
            return input.at[indices].add(src, mode=get_scatter_mode())
        elif reduce == "prod":
            return input.at[indices].multiply(src, mode=get_scatter_mode())
        elif reduce == "amax":
            return input.at[indices].max(src, mode=get_scatter_mode())
        elif reduce == "amin":
            return input.at[indices].min(src, mode=get_scatter_mode())
        elif reduce == "mean":
            sum_result = input.at[indices].add(src, mode=get_scatter_mode())
            counts = jnp.ones_like(input)
            counts = counts.at[indices].add(jnp.ones_like(src), mode=get_scatter_mode())
            return sum_result / counts
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")
    else:
        # include_self=False: only reduce src values at touched indices,
        # keep original input at untouched indices
        # Track which indices are touched
        touched = jnp.zeros(input.shape, dtype=jnp.bool_)
        touched = touched.at[indices].set(True)

        # Get identity value for reduce operation
        if reduce == "sum":
            identity = jnp.zeros_like(input)
            reduced = identity.at[indices].add(src, mode=get_scatter_mode())
        elif reduce == "prod":
            identity = jnp.ones_like(input)
            reduced = identity.at[indices].multiply(src, mode=get_scatter_mode())
        elif reduce == "amax":
            identity = jnp.full_like(input, -jnp.inf)
            reduced = identity.at[indices].max(src, mode=get_scatter_mode())
        elif reduce == "amin":
            identity = jnp.full_like(input, jnp.inf)
            reduced = identity.at[indices].min(src, mode=get_scatter_mode())
        elif reduce == "mean":
            identity = jnp.zeros_like(input)
            sum_result = identity.at[indices].add(src, mode=get_scatter_mode())
            counts = jnp.zeros_like(input)
            counts = counts.at[indices].add(jnp.ones_like(src), mode=get_scatter_mode())
            # Avoid division by zero at untouched indices
            reduced = sum_result / jnp.maximum(counts, 1)
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")

        # Use reduced values where touched, original input elsewhere
        return jnp.where(touched, reduced, input)


@translates(torch.repeat_interleave)
def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    return jnp.repeat(input, repeats, axis=dim, total_repeat_length=output_size)


@translates(torch.diff)
def diff(input, n=1, dim=-1, prepend=None, append=None):
    if prepend is not None:
        input = jnp.concatenate([prepend, input], axis=dim)
    if append is not None:
        input = jnp.concatenate([input, append], axis=dim)
    return jnp.diff(input, n=n, axis=dim)


@translates(torch.isin)
def isin(elements, test_elements, *, assume_unique=False, invert=False):
    result = jnp.isin(
        elements, test_elements, assume_unique=assume_unique, invert=invert
    )
    return result


@translates(torch.softmax)
@translates(torch.nn.functional.softmax)
def softmax(input, dim=None, dtype=None, **kwargs):
    if dtype is not None:
        input = input.astype(dtype)
    return jax.nn.softmax(input, axis=dim)


@translates(torch.nn.functional.one_hot)
def one_hot(input, num_classes=-1):
    if num_classes == -1:
        num_classes = input.max() + 1
    return jax.nn.one_hot(input, num_classes=num_classes, dtype=jnp.int64)


@translates(torch.searchsorted)
def searchsorted(
    sorted_sequence,
    values,
    out_int32=False,
    right=False,
    side=None,
    out=None,
    sorter=None,
):
    if side is not None:
        effective_side = side
    else:
        effective_side = "right" if right else "left"
    if sorter is not None:
        sorted_sequence = sorted_sequence[sorter]
    result = _searchsorted_via_scan(sorted_sequence, values, side=effective_side)
    if out_int32:
        result = result.astype(jnp.int32)
    else:
        result = result.astype(jnp.int64)
    return result


def _searchsorted_via_scan(sorted_arr, query, side="left"):
    """Binary search via scan — works with symbolic shapes unlike jnp.searchsorted."""
    dtype = jnp.int32

    def body_fun(state, _):
        low, high = state
        mid = low.astype(jnp.uint32) + high.astype(jnp.uint32)
        mid = jax.lax.div(mid, jnp.uint32(2)).astype(dtype)
        go_left = (
            query <= sorted_arr[mid] if side == "left" else query < sorted_arr[mid]
        )
        return (jnp.where(go_left, low, mid), jnp.where(go_left, mid, high)), ()

    # Use fixed 32 iterations (sufficient for any int32-indexable array) so that
    # scan length is concrete even when sorted_arr has a symbolic shape.
    init = (
        jnp.zeros_like(query, dtype=dtype),
        jnp.full_like(query, sorted_arr.shape[0], dtype=dtype),
    )
    carry, _ = jax.lax.scan(body_fun, init, (), length=32, unroll=True)
    return carry[1]


@translates(torch.matrix_exp)
@translates(torch.linalg.matrix_exp)
def matrix_exp(A):
    return jax.scipy.linalg.expm(A)


@translates(torch.norm)
@translates(torch.linalg.norm)
def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None, ord=None):
    if dtype is not None:
        input = input.astype(dtype)
    if ord is not None:
        # torch.linalg.norm path: matrix-aware norms
        return jnp.linalg.norm(input, ord=ord, axis=dim, keepdims=keepdim)
    # torch.norm path: always vector norm when dim is None (flatten first)
    if dim is None and p != "fro" and input.ndim > 1:
        input = input.ravel()
    # "fro" is invalid for vector norms in JAX, map to None (2-norm)
    effective_ord = None if p == "fro" and isinstance(dim, int) else p
    return jnp.linalg.norm(input, ord=effective_ord, axis=dim, keepdims=keepdim)


translates(torch.linalg.inv, jnp.linalg.inv)
translates(torch.linalg.solve, jnp.linalg.solve)
translates(torch.linalg.eig, jnp.linalg.eig)
translates(torch.linalg.svd, jnp.linalg.svd)
translates(torch.linalg.cholesky, jnp.linalg.cholesky)
translates(torch.linalg.eigh, jnp.linalg.eigh)
translates(torch.linalg.eigvals, jnp.linalg.eigvals)
translates(torch.linalg.eigvalsh, jnp.linalg.eigvalsh)
translates(torch.linalg.slogdet, jnp.linalg.slogdet)
translates(torch.linalg.pinv, jnp.linalg.pinv)
translates(torch.linalg.svdvals, jnp.linalg.svdvals)
translates(torch.linalg.matrix_rank, jnp.linalg.matrix_rank)
translates(torch.linalg.matrix_power, jnp.linalg.matrix_power)
translates(torch.linalg.cond, jnp.linalg.cond)
translates(torch.linalg.tensorinv, jnp.linalg.tensorinv)
translates(torch.linalg.tensorsolve, jnp.linalg.tensorsolve)
translates(torch.linalg.matmul, inherit_precision(jnp.matmul))
translates(torch.linalg.qr, jnp.linalg.qr)
translates(torch.linalg.cross, translate_kwargs(jnp.linalg.cross))


@translates(torch.linalg.vector_norm)
def linalg_vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None):
    return jnp.linalg.vector_norm(x, ord=ord, axis=dim, keepdims=keepdim)


@translates(torch.linalg.matrix_norm)
def linalg_matrix_norm(
    x, ord="fro", dim=(-2, -1), keepdim=False, *, dtype=None, out=None
):
    # JAX matrix_norm always operates on last 2 dims, no dim param
    if dim == (-2, -1) or dim == (-1, -2):
        return jnp.linalg.matrix_norm(x, ord=ord, keepdims=keepdim)
    # For custom dims, fall back to jnp.linalg.norm
    return jnp.linalg.norm(x, ord=ord, axis=dim, keepdims=keepdim)


@translates(torch.linalg.solve_triangular)
def linalg_solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None):
    if not left:
        # XA = B -> A^T X^T = B^T
        result = jax.scipy.linalg.solve_triangular(
            A.T, B.T, lower=upper, unit_diagonal=unitriangular
        )
        return result.T
    return jax.scipy.linalg.solve_triangular(
        A, B, lower=not upper, unit_diagonal=unitriangular
    )


@translates(torch.linalg.lstsq)
def linalg_lstsq(A, B, rcond=None, *, driver=None):
    return jnp.linalg.lstsq(A, B, rcond=rcond)


@translates(torch.linalg.lu)
def linalg_lu(A, *, pivot=True, out=None):
    # torch returns (P, L, U); jax.scipy.linalg.lu also returns (P, L, U) when permute_l=False
    return jax.scipy.linalg.lu(A, permute_l=False)


@translates(torch.linalg.lu_factor)
def linalg_lu_factor(A, *, pivot=True, out=None):
    return jax.scipy.linalg.lu_factor(A)


translates(torch.erf, jax.scipy.special.erf)
translates(torch.erfc, jax.scipy.special.erfc)
translates(torch.erfinv, jax.scipy.special.erfinv)
translates(torch.lgamma, jax.scipy.special.gammaln)


@translates(torch.nn.functional.rms_norm)
def rms_norm(input, normalized_shape, weight=None, eps=None):
    axes = tuple(range(-len(normalized_shape), 0))
    if eps is None:
        eps = 1e-8
    rms = jnp.sqrt(jnp.mean(input**2, axis=axes, keepdims=True) + eps)
    x = input / rms
    if weight is not None:
        x = x * weight
    return x


@translates(torch.nn.functional.leaky_relu)
def leaky_relu(input, negative_slope=0.01, inplace=False):
    return jax.nn.leaky_relu(input, negative_slope=negative_slope)


@translates(torch.nn.functional.elu)
def elu(input, alpha=1.0, inplace=False):
    return jax.nn.elu(input, alpha=alpha)


@translates(torch.nn.functional.selu)
def selu(input, inplace=False):
    return jax.nn.selu(input)


@translates(torch.nn.functional.celu)
def celu(input, alpha=1.0, inplace=False):
    return jax.nn.celu(input, alpha=alpha)


@translates(torch.nn.functional.relu6)
def relu6(input, inplace=False):
    return jnp.clip(input, 0, 6)


@translates(torch.nn.functional.softplus)
def softplus(input, beta=1, threshold=20):
    return jnp.where(
        input * beta > threshold, input, jax.nn.softplus(input * beta) / beta
    )


@translates(torch.nn.functional.mish)
def mish(input, inplace=False):
    return input * jnp.tanh(jax.nn.softplus(input))


@translates(torch.nn.functional.hardswish)
def hardswish(input, inplace=False):
    return input * jnp.clip(input / 6 + 0.5, 0, 1)


@translates(torch.nn.functional.hardsigmoid)
def hardsigmoid(input, inplace=False):
    return jnp.clip(input / 6 + 0.5, 0, 1)


@translates(torch.nn.functional.log_softmax)
def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    return jax.nn.log_softmax(input, axis=dim)


@translates(torch.nn.functional.glu)
def glu(input, dim=-1):
    a, b = jnp.split(input, 2, axis=dim)
    return a * jax.nn.sigmoid(b)


@translates(torch.nn.functional.cosine_similarity)
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    dot = (x1 * x2).sum(axis=dim)
    norm1 = jnp.sqrt((x1**2).sum(axis=dim))
    norm2 = jnp.sqrt((x2**2).sum(axis=dim))
    return dot / jnp.maximum(norm1 * norm2, eps)


@translates(torch.nn.functional.group_norm)
def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    N, C = input.shape[:2]
    group_shape = (N, num_groups, C // num_groups) + input.shape[2:]
    x = input.reshape(group_shape)
    reduce_axes = tuple(range(2, len(group_shape)))
    mean = x.mean(axis=reduce_axes, keepdims=True)
    var = x.var(axis=reduce_axes, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x.reshape(input.shape)
    if weight is not None:
        shape = [1, -1] + [1] * (input.ndim - 2)
        x = x * weight.reshape(shape)
    if bias is not None:
        shape = [1, -1] + [1] * (input.ndim - 2)
        x = x + bias.reshape(shape)
    return x


@translates(torch.nn.functional.sigmoid)
def f_sigmoid(input):
    return jax.nn.sigmoid(input)


@translates(torch.nn.functional.tanh)
def f_tanh(input):
    return jnp.tanh(input)


@translates(torch.clamp_min)
def clamp_min(input, min=None, *, out=None):
    return jnp.clip(input, min=min)


@translates(torch.clamp_max)
def clamp_max(input, max=None, *, out=None):
    return jnp.clip(input, max=max)


@translates(torch.argsort)
def argsort(input, dim=-1, descending=False, stable=False):
    if descending:
        return jnp.flip(jnp.argsort(input, axis=dim, stable=stable), axis=dim)
    return jnp.argsort(input, axis=dim, stable=stable)


@translates(torch.std)
def std(input, dim=None, unbiased=True, keepdim=False, *, correction=None, out=None):
    if correction is None:
        correction = 1 if unbiased else 0
    return jnp.std(input, axis=dim, keepdims=keepdim, ddof=correction)


@translates(torch.var)
def var(input, dim=None, unbiased=True, keepdim=False, *, correction=None, out=None):
    if correction is None:
        correction = 1 if unbiased else 0
    return jnp.var(input, axis=dim, keepdims=keepdim, ddof=correction)


@translates(torch.eye)
def eye(
    n,
    m=None,
    *,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    return jnp.eye(n, M=m, dtype=dtype)


@translates(torch.linspace)
def linspace(
    start,
    end,
    steps,
    *,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    return jnp.linspace(start, end, steps, dtype=dtype)


@translates(torch.logspace)
def logspace(
    start,
    end,
    steps,
    base=10.0,
    *,
    out=None,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
):
    return jnp.logspace(start, end, steps, base=base, dtype=dtype)


@translates(torch.chunk)
def chunk(input, chunks, dim=0):
    return tuple(jnp.array_split(input, chunks, axis=dim))


@translates(torch.meshgrid)
def meshgrid(*tensors, indexing=None):
    if indexing is None:
        indexing = "ij"
    return jnp.meshgrid(*tensors, indexing=indexing)


@translates(torch.frac)
def frac(input):
    return input - jnp.trunc(input)


@translates(torch.lerp)
def lerp(input, end, weight):
    return input + weight * (end - input)


@translates(torch.masked_fill)
def masked_fill(input, mask, value):
    return jnp.where(mask, value, input)


@translates(torch.diagonal)
def diagonal(input, offset=0, dim1=0, dim2=1):
    return jnp.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)


@translates(torch.unflatten)
def unflatten(input, dim, sizes):
    shape = list(input.shape)
    shape = shape[:dim] + list(sizes) + shape[dim + 1 :]
    return input.reshape(shape)


@translates(torch.gather)
def gather(input, dim, index, *, sparse_grad=False):
    ndim = input.ndim
    indices = []
    for d in range(ndim):
        if d == dim:
            indices.append(index)
        else:
            shape = [1] * ndim
            shape[d] = index.shape[d]
            indices.append(
                jnp.broadcast_to(jnp.arange(index.shape[d]).reshape(shape), index.shape)
            )
    return input[tuple(indices)]


@translates(torch.topk)
def topk(input, k, dim=-1, largest=True, sorted=True):
    if dim < 0:
        dim = input.ndim + dim
    needs_transpose = dim != input.ndim - 1
    if needs_transpose:
        input = jnp.moveaxis(input, dim, -1)
    if not largest:
        values, indices = jax.lax.top_k(-input, k)
        values = -values
    else:
        values, indices = jax.lax.top_k(input, k)
    if needs_transpose:
        values = jnp.moveaxis(values, -1, dim)
        indices = jnp.moveaxis(indices, -1, dim)
    return values, indices


@translates(torch.logsumexp)
def logsumexp(input, dim, keepdim=False):
    return jax.scipy.special.logsumexp(input, axis=dim, keepdims=keepdim)


@translates(torch.is_floating_point)
def is_floating_point(input):
    return jnp.issubdtype(input.dtype, jnp.floating)


# =============================================================================
# Random Function Implementations
# =============================================================================
RANDOM_FUNCTIONS = set()


def translate_random_constructor(torch_fn: Callable, jax_fn: Callable):
    RANDOM_FUNCTIONS.add(torch_fn)

    def wrapped(
        *size,
        generator=None,
        out=None,
        dtype=None,
        layout=None,
        device=None,
        requires_grad=False,
        pin_memory=False,
    ):
        del generator, out, layout, device, requires_grad, pin_memory
        key = RNGContext.current().next_key()
        if dtype is None:
            dtype = jnp.float32
        return jax_fn(key, shape=size, dtype=dtype)

    return translates(torch_fn, wrapped)


def translate_random_like_constructor(torch_fn: Callable, jax_fn: Callable):
    RANDOM_FUNCTIONS.add(torch_fn)

    def wrapped(
        input,
        *,
        generator=None,
        dtype=None,
        layout=None,
        device=None,
        requires_grad=False,
        memory_format=None,
    ):
        del generator, layout, device, requires_grad, memory_format
        RANDOM_FUNCTIONS.add(torch_fn)
        key = RNGContext.current().next_key()
        if dtype is None:
            dtype = input.dtype
        return jax_fn(key, shape=input.shape, dtype=dtype)

    return translates(torch_fn, wrapped)


translate_random_constructor(torch.randn, jax.random.normal)
translate_random_like_constructor(torch.randn_like, jax.random.normal)
translate_random_constructor(torch.rand, jax.random.uniform)
translate_random_like_constructor(torch.rand_like, jax.random.uniform)


def jax_function(fn: Callable) -> Callable:
    """
    Main function lookup for PyTorch to JAX function translation.

    This function attempts to find a JAX equivalent for a given PyTorch function
    using the following lookup strategy:
    1. Check the TRANSLATED_FNS registry for explicit translations
    2. Look for the function in JAX's standard modules (numpy, nn, lax)
    3. Raise NotImplementedError if no translation is found

    Args:
        fn: PyTorch function to translate

    Returns:
        JAX function wrapper that handles PyTorch-style arguments

    Raises:
        NotImplementedError: If no JAX equivalent is found

    Note:
        Functions found in JAX modules are automatically wrapped with
        keyword argument translation (dim->axis, keepdim->keepdims, etc.)
    """
    # Fallback - check if the function is in the JAX library
    name = getattr(fn, "__name__", str(fn))
    if fn in TRANSLATED_FNS:
        return TRANSLATED_FNS[fn]
    elif jax_fn := getattr(
        jax.numpy, name, getattr(jax.nn, name, getattr(jax.lax, name, None))
    ):
        logging.warning(f"{fn} not found, using default JAX translation.")
        return JaxFnWithInplace(translate_kwargs(jax_fn))
    else:
        raise NotImplementedError(
            f"Function '{fn.__module__}.{getattr(fn, '__name__', str(fn))}' not found in translation map."
        )
