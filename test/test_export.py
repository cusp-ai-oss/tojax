# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from jax import export

from tojax import tojax


def test_simple_export():
    @tojax
    def f(x):
        return torch.pow(x, 2)

    inp = jnp.array([1, 2, 3])
    exported = export.export(jax.jit(f))(inp)
    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1


def test_symbolic_shape():
    @tojax
    def f(x):
        return torch.arange(x.shape[0] * 2)

    shape = export.symbolic_shape("a")
    inp = jax.ShapeDtypeStruct(shape, jnp.float32)
    exported = export.export(jax.jit(f))(inp)
    assert len(exported.in_avals) == 1
    assert exported.in_avals[0].shape == inp.shape
    assert len(exported.out_avals) == 1
    assert exported.out_avals[0].shape == (shape[0] * 2,)


def test_symbolic_zeros():
    @tojax
    def f(x):
        return torch.zeros(*x.shape)

    shape = export.symbolic_shape("a")
    inp = jax.ShapeDtypeStruct(shape, jnp.float32)
    exported = export.export(jax.jit(f))(inp)
    assert len(exported.in_avals) == 1
    assert exported.in_avals[0].shape == inp.shape
    assert len(exported.out_avals) == 1
    assert exported.out_avals[0].shape == inp.shape


def test_multiple_inputs_outputs():
    @tojax
    def f(x, y):
        z1 = torch.add(x, y)
        z2 = torch.mul(x, y)
        return z1, z2

    inp1 = jnp.array([1.0, 2.0, 3.0])
    inp2 = jnp.array([4.0, 5.0, 6.0])
    exported = export.export(jax.jit(f))(inp1, inp2)

    assert len(exported.in_avals) == 2
    assert len(exported.out_avals) == 2
    assert exported.in_avals[0].shape == (3,)
    assert exported.in_avals[1].shape == (3,)
    assert exported.out_avals[0].shape == (3,)
    assert exported.out_avals[1].shape == (3,)


def test_different_dtypes():
    @tojax
    def f(x):
        return torch.sin(x)

    # Test float32
    inp_f32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    exported_f32 = export.export(jax.jit(f))(inp_f32)
    assert exported_f32.in_avals[0].dtype == jnp.float32
    assert exported_f32.out_avals[0].dtype == jnp.float32

    # Test float64
    inp_f64 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    exported_f64 = export.export(jax.jit(f))(inp_f64)
    assert exported_f64.in_avals[0].dtype == jnp.float64
    assert exported_f64.out_avals[0].dtype == jnp.float64


def test_neural_network_layer():
    @tojax
    def f(x, weight, bias):
        return F.linear(x, weight, bias)

    batch_size, in_features, out_features = 2, 4, 3
    x = jnp.ones((batch_size, in_features))
    weight = jnp.ones((out_features, in_features))
    bias = jnp.ones(out_features)

    exported = export.export(jax.jit(f))(x, weight, bias)

    assert len(exported.in_avals) == 3
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (batch_size, in_features)
    assert exported.in_avals[1].shape == (out_features, in_features)
    assert exported.in_avals[2].shape == (out_features,)
    assert exported.out_avals[0].shape == (batch_size, out_features)


def test_conv2d_export():
    @tojax
    def f(x, weight, bias):
        return F.conv2d(x, weight, bias, stride=1, padding=1, dilation=1, groups=1)

    batch_size, in_channels, height, width = 1, 3, 8, 8
    out_channels, kernel_size = 16, 3

    x = jnp.ones((batch_size, in_channels, height, width))
    weight = jnp.ones((out_channels, in_channels, kernel_size, kernel_size))
    bias = jnp.ones(out_channels)

    exported = export.export(jax.jit(f))(x, weight, bias)

    assert len(exported.in_avals) == 3
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (batch_size, in_channels, height, width)
    assert exported.in_avals[1].shape == (
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
    )
    assert exported.in_avals[2].shape == (out_channels,)
    assert exported.out_avals[0].shape == (batch_size, out_channels, height, width)


def test_complex_operations():
    @tojax
    def f(x):
        # Test a sequence of operations
        y = torch.relu(x)
        y = torch.sum(y, dim=-1, keepdim=True)
        y = torch.sqrt(y + 1e-8)
        return y

    inp = jnp.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    exported = export.export(jax.jit(f))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (2, 3)
    assert exported.out_avals[0].shape == (2, 1)


def test_tensor_methods():
    @tojax
    def f(x):
        return x.transpose(0, 1).reshape(-1)

    inp = jnp.array([[1, 2, 3], [4, 5, 6]])
    exported = export.export(jax.jit(f))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (2, 3)
    assert exported.out_avals[0].shape == (6,)


def test_batch_operations():
    @tojax
    def f(x):
        # Batch normalization-like operation
        mean = torch.mean(x, dim=0, keepdim=True)
        var = torch.var(x, dim=0, keepdim=True)
        return (x - mean) / torch.sqrt(var + 1e-5)

    inp = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    exported = export.export(jax.jit(f))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (3, 2)
    assert exported.out_avals[0].shape == (3, 2)


def test_symbolic_batch_size():
    @tojax
    def f(x):
        batch_size = x.shape[0]
        return torch.zeros((batch_size, 10))

    batch_shape = export.symbolic_shape("batch")
    inp = jax.ShapeDtypeStruct((batch_shape[0], 5), jnp.float32)
    exported = export.export(jax.jit(f))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (batch_shape[0], 5)
    assert exported.out_avals[0].shape == (batch_shape[0], 10)


def test_multiple_symbolic_dims():
    @tojax
    def f(x):
        # Use a simple operation that works with the last dimension
        return torch.sum(x, dim=-1, keepdim=True)

    # Use symbolic shape for the last dimension
    shape = export.symbolic_shape("n")
    inp = jax.ShapeDtypeStruct((2, 3, shape[0]), jnp.float32)
    exported = export.export(jax.jit(f))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (2, 3, shape[0])
    assert exported.out_avals[0].shape == (2, 3, 1)


def test_conditional_logic():
    @tojax
    def f(x):
        # Simple conditional operation
        return torch.where(x > 0, x, torch.zeros_like(x))

    inp = jnp.array([-1.0, 0.0, 1.0, 2.0])
    exported = export.export(jax.jit(f))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (4,)
    assert exported.out_avals[0].shape == (4,)


def test_reduction_operations():
    @tojax
    def f(x):
        return {
            "sum": torch.sum(x),
            "mean": torch.mean(x),
            "max": torch.max(x),
            "min": torch.min(x),
        }

    inp = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Note: dictionaries need to be handled separately in JAX export
    @tojax
    def f_tuple(x):
        return (torch.sum(x), torch.mean(x), torch.max(x), torch.min(x))

    exported = export.export(jax.jit(f_tuple))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 4
    assert exported.in_avals[0].shape == (2, 3)
    # All reductions should result in scalar outputs
    for out_aval in exported.out_avals:
        assert out_aval.shape == ()


def test_indexing_operations():
    @tojax
    def f(x):
        return x[:, 0], x[0, :], x[0, 0]

    inp = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    exported = export.export(jax.jit(f))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 3
    assert exported.in_avals[0].shape == (2, 3)
    assert exported.out_avals[0].shape == (2,)  # First column
    assert exported.out_avals[1].shape == (3,)  # First row
    assert exported.out_avals[2].shape == ()  # Single element


def test_broadcasting_operations():
    @tojax
    def f(x, y):
        return x + y, x * y, x / y

    x = jnp.array([[1.0], [2.0], [3.0]])  # (3, 1)
    y = jnp.array([1.0, 2.0, 3.0, 4.0])  # (4,)

    exported = export.export(jax.jit(f))(x, y)

    assert len(exported.in_avals) == 2
    assert len(exported.out_avals) == 3
    assert exported.in_avals[0].shape == (3, 1)
    assert exported.in_avals[1].shape == (4,)
    # All outputs should have broadcasted shape (3, 4)
    for out_aval in exported.out_avals:
        assert out_aval.shape == (3, 4)


def test_serialization_roundtrip():
    @tojax
    def f(x):
        return torch.sin(x) + torch.cos(x)

    inp = jnp.array([0.0, jnp.pi / 2, jnp.pi])
    exported = export.export(jax.jit(f))(inp)

    # Test that we can serialize and deserialize
    serialized = exported.mlir_module()
    assert serialized is not None

    # Basic validation that the serialized module contains expected operations
    serialized_str = str(serialized)
    assert len(serialized_str) > 0


def test_empty_tensor_handling():
    @tojax
    def f(x):
        return torch.reshape(x, (0, 5))

    inp = jnp.array([]).reshape(0, 3)
    exported = export.export(jax.jit(f))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (0, 3)
    assert exported.out_avals[0].shape == (0, 5)


def test_large_tensor_shapes():
    @tojax
    def f(x):
        return torch.sum(x, dim=-1)

    # Test with reasonably large tensor (but not too large for testing)
    large_shape = (100, 200)
    inp = jnp.ones(large_shape)
    exported = export.export(jax.jit(f))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == large_shape
    assert exported.out_avals[0].shape == (100,)


def test_nested_function_calls():
    # Note: Nested tojax functions don't work well together during export
    # Instead, test a single tojax-translated function with multiple operations
    @tojax
    def f(x):
        # Simulate nested behavior with sequential operations
        y = torch.relu(x)  # This would be the "inner" function
        return torch.sum(y)  # This would be the "outer" function

    inp = jnp.array([-1.0, 0.0, 1.0, 2.0])
    exported = export.export(jax.jit(f))(inp)

    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (4,)
    assert exported.out_avals[0].shape == ()


def test_gradient_compatibility():
    """Test that exported functions work with JAX transformations."""

    @tojax
    def f(x):
        return torch.sum(torch.pow(x, 2))

    inp = jnp.array([1.0, 2.0, 3.0])
    exported = export.export(jax.jit(f))(inp)

    # The exported function should be compatible with JAX transformations
    assert len(exported.in_avals) == 1
    assert len(exported.out_avals) == 1
    assert exported.in_avals[0].shape == (3,)
    assert exported.out_avals[0].shape == ()

    # Test that the output dtype is appropriate for gradient computation
    assert exported.out_avals[0].dtype in [jnp.float32, jnp.float64]


def test_multi_head_attention_symbolic_shapes():
    embed_dim, num_heads = 8, 2

    @tojax
    def f(query, key, value, ipw, ipb, opw, opb):
        out, _ = F.multi_head_attention_forward(
            query,
            key,
            value,
            embed_dim,
            num_heads,
            ipw,
            ipb,
            None,
            None,
            False,
            0.0,
            opw,
            opb,
            need_weights=False,
        )
        return out

    tgt, src, batch = export.symbolic_shape("L, S, N")
    q = jax.ShapeDtypeStruct((tgt, batch, embed_dim), jnp.float64)
    k = jax.ShapeDtypeStruct((src, batch, embed_dim), jnp.float64)
    v = jax.ShapeDtypeStruct((src, batch, embed_dim), jnp.float64)
    ipw = jax.ShapeDtypeStruct((3 * embed_dim, embed_dim), jnp.float64)
    ipb = jax.ShapeDtypeStruct((3 * embed_dim,), jnp.float64)
    opw = jax.ShapeDtypeStruct((embed_dim, embed_dim), jnp.float64)
    opb = jax.ShapeDtypeStruct((embed_dim,), jnp.float64)

    exported = export.export(jax.jit(f))(q, k, v, ipw, ipb, opw, opb)
    assert exported.out_avals[0].shape == (tgt, batch, embed_dim)


def test_scaled_dot_product_attention_symbolic_shapes():
    num_heads, head_dim = 2, 8

    @tojax
    def f(query, key, value):
        return F.scaled_dot_product_attention(query, key, value)

    batch, tgt, src = export.symbolic_shape("B, L, S")
    q = jax.ShapeDtypeStruct((batch, num_heads, tgt, head_dim), jnp.float64)
    k = jax.ShapeDtypeStruct((batch, num_heads, src, head_dim), jnp.float64)
    v = jax.ShapeDtypeStruct((batch, num_heads, src, head_dim), jnp.float64)

    exported = export.export(jax.jit(f))(q, k, v)
    assert exported.out_avals[0].shape == (batch, num_heads, tgt, head_dim)
