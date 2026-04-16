# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.nn as nn
from pytest import fixture

from tojax.functions import get_matmul_precision
from tojax.tojax import RNGMode, tojax

torch.set_default_dtype(torch.float64)


@fixture
def img_input():
    return torch.randn(1, 3, 16, 16)


_UNIVARIATE_LAYERS = []


def register_univariate_layer(layer):
    _UNIVARIATE_LAYERS.append(layer.__name__)
    return fixture(layer)


@register_univariate_layer
def conv2d_layer():
    return nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        bias=True,
    )


@register_univariate_layer
def batch_norm_layer():
    return nn.BatchNorm2d(
        num_features=3,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    )


@register_univariate_layer
def max_pool2d_layer():
    return nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


# @register_univariate_layer # This is failing due to ceil mode not being supported
def avg_pool2d_layer():
    return nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


@register_univariate_layer
def layer_norm_layer():
    return nn.LayerNorm(normalized_shape=[3, 16, 16], eps=1e-5, elementwise_affine=True)


@register_univariate_layer
def linear():
    return nn.Linear(in_features=16, out_features=16, bias=True)


@register_univariate_layer
def flatten():
    return nn.Flatten(start_dim=1, end_dim=-1)


@register_univariate_layer
def relu():
    return nn.ReLU(inplace=False)


@fixture(params=_UNIVARIATE_LAYERS)
def layer(request):
    return request.getfixturevalue(request.param)


def test_univariate_layer(img_input, layer):
    torch_out = layer(img_input).detach().cpu().numpy()
    jax_fn = tojax(layer)
    jax_inp = tojax(img_input)
    jax_out = np.asarray(jax_fn(jax_inp))
    npt.assert_allclose(jax_out, torch_out, rtol=1e-5, atol=1e-5)


# Test fixtures for matrix operations
@fixture
def matrix_a():
    """Small 2D matrix for basic operations"""
    return torch.randn(4, 5, dtype=torch.float64)


@fixture
def matrix_b():
    """Compatible 2D matrix for matrix multiplication"""
    return torch.randn(5, 3, dtype=torch.float64)


@fixture
def batch_matrix_a():
    """3D batched matrix"""
    return torch.randn(2, 4, 5, dtype=torch.float64)


@fixture
def batch_matrix_b():
    """3D batched matrix compatible with batch_matrix_a"""
    return torch.randn(2, 5, 3, dtype=torch.float64)


@fixture
def vector_a():
    """1D vector"""
    return torch.randn(5, dtype=torch.float64)


@fixture
def vector_b():
    """1D vector of same size"""
    return torch.randn(5, dtype=torch.float64)


class TestMatrixOperations:
    """Test suite for matrix multiplication operations"""

    def test_matmul_2d_matrices(self, matrix_a, matrix_b):
        """Test 2D matrix multiplication using torch.matmul"""
        torch_result = torch.matmul(matrix_a, matrix_b)

        # Use tojax to create a function that performs the operation
        def matmul_op(a, b):
            return torch.matmul(a, b)

        jax_matmul = tojax(matmul_op)
        jax_result = jax_matmul(matrix_a, matrix_b)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_matmul_operator(self, matrix_a, matrix_b):
        """Test matrix multiplication using @ operator"""
        torch_result = matrix_a @ matrix_b

        def matmul_op(a, b):
            return a @ b

        jax_matmul = tojax(matmul_op)
        jax_result = jax_matmul(matrix_a, matrix_b)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_bmm_batched_matrices(self, batch_matrix_a, batch_matrix_b):
        """Test batched matrix multiplication using torch.bmm"""
        torch_result = torch.bmm(batch_matrix_a, batch_matrix_b)

        def bmm_op(a, b):
            return torch.bmm(a, b)

        jax_bmm = tojax(bmm_op)
        jax_result = jax_bmm(batch_matrix_a, batch_matrix_b)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_mm_2d_matrices(self, matrix_a, matrix_b):
        """Test 2D matrix multiplication using torch.mm"""
        torch_result = torch.mm(matrix_a, matrix_b)

        def mm_op(a, b):
            return torch.mm(a, b)

        jax_mm = tojax(mm_op)
        jax_result = jax_mm(matrix_a, matrix_b)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_vector_dot_product(self, vector_a, vector_b):
        """Test vector dot product using torch.matmul"""
        torch_result = torch.matmul(vector_a, vector_b)

        def dot_op(a, b):
            return torch.matmul(a, b)

        jax_dot = tojax(dot_op)
        jax_result = jax_dot(vector_a, vector_b)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_matrix_vector_multiply(self, matrix_a, vector_a):
        """Test matrix-vector multiplication"""
        torch_result = torch.matmul(matrix_a, vector_a)

        def matvec_op(m, v):
            return torch.matmul(m, v)

        jax_matvec = tojax(matvec_op)
        jax_result = jax_matvec(matrix_a, vector_a)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )


class TestBasicTensorOperations:
    """Test suite for basic tensor operations"""

    def test_add(self, matrix_a):
        """Test tensor addition"""
        other = torch.randn_like(matrix_a)
        torch_result = torch.add(matrix_a, other)

        def add_op(a, b):
            return torch.add(a, b)

        jax_add = tojax(add_op)
        jax_result = jax_add(matrix_a, other)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_sub(self, matrix_a):
        """Test tensor subtraction"""
        other = torch.randn_like(matrix_a)
        torch_result = torch.sub(matrix_a, other)

        def sub_op(a, b):
            return torch.sub(a, b)

        jax_sub = tojax(sub_op)
        jax_result = jax_sub(matrix_a, other)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_mul(self, matrix_a):
        """Test element-wise multiplication"""
        other = torch.randn_like(matrix_a)
        torch_result = torch.mul(matrix_a, other)

        def mul_op(a, b):
            return torch.mul(a, b)

        jax_mul = tojax(mul_op)
        jax_result = jax_mul(matrix_a, other)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_div(self, matrix_a):
        """Test element-wise division"""
        other = torch.randn_like(matrix_a) + 1e-6  # Avoid division by zero
        torch_result = torch.div(matrix_a, other)

        def div_op(a, b):
            return torch.div(a, b)

        jax_div = tojax(div_op)
        jax_result = jax_div(matrix_a, other)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_pow(self, matrix_a):
        """Test element-wise power operation"""
        exponent = 2.0
        torch_result = torch.pow(matrix_a, exponent)

        def pow_op(a, exp):
            return torch.pow(a, exp)

        jax_pow = tojax(pow_op)
        jax_result = jax_pow(matrix_a, exponent)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )


class TestMathematicalFunctions:
    """Test suite for mathematical functions"""

    def test_sin(self, matrix_a):
        """Test sine function"""
        torch_result = torch.sin(matrix_a)

        def sin_op(a):
            return torch.sin(a)

        jax_sin = tojax(sin_op)
        jax_result = jax_sin(matrix_a)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_cos(self, matrix_a):
        """Test cosine function"""
        torch_result = torch.cos(matrix_a)

        def cos_op(a):
            return torch.cos(a)

        jax_cos = tojax(cos_op)
        jax_result = jax_cos(matrix_a)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_tan(self, matrix_a):
        """Test tangent function"""
        torch_result = torch.tan(matrix_a)

        def tan_op(a):
            return torch.tan(a)

        jax_tan = tojax(tan_op)
        jax_result = jax_tan(matrix_a)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_tanh(self, matrix_a):
        """Test hyperbolic tangent function"""
        torch_result = torch.tanh(matrix_a)

        def tanh_op(a):
            return torch.tanh(a)

        jax_tanh = tojax(tanh_op)
        jax_result = jax_tanh(matrix_a)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_exp(self, matrix_a):
        """Test exponential function"""
        # Use smaller values to avoid overflow
        small_matrix = matrix_a * 0.1
        torch_result = torch.exp(small_matrix)

        def exp_op(a):
            return torch.exp(a)

        jax_exp = tojax(exp_op)
        jax_result = jax_exp(small_matrix)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_log(self, matrix_a):
        """Test natural logarithm function"""
        # Use positive values for log
        positive_matrix = torch.abs(matrix_a) + 1e-6
        torch_result = torch.log(positive_matrix)

        def log_op(a):
            return torch.log(a)

        jax_log = tojax(log_op)
        jax_result = jax_log(positive_matrix)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_sqrt(self, matrix_a):
        """Test square root function"""
        # Use positive values for sqrt
        positive_matrix = torch.abs(matrix_a)
        torch_result = torch.sqrt(positive_matrix)

        def sqrt_op(a):
            return torch.sqrt(a)

        jax_sqrt = tojax(sqrt_op)
        jax_result = jax_sqrt(positive_matrix)

        npt.assert_allclose(
            np.asarray(jax_result), torch_result.numpy(), rtol=1e-5, atol=1e-5
        )


class TestInPlaceOperations:
    """Test suite for in-place operations where results would differ from out-of-place"""

    def test_add_inplace_vs_outplace_shared_memory(self):
        """Test that in-place add affects shared tensors while out-of-place doesn't"""
        # Create original tensor
        original = torch.randn(3, 3, dtype=torch.float64)

        # Test out-of-place operation
        view1 = original.clone()
        view2 = view1  # Same tensor, not a copy
        other = torch.ones_like(view1)

        # Out-of-place: view2 should remain unchanged
        result_outplace = torch.add(view1, other)
        assert not torch.equal(result_outplace, view1), (
            "Out-of-place should create new tensor"
        )
        assert torch.equal(view1, view2), (
            "Original tensor should be unchanged in out-of-place"
        )

        # Test with JAX equivalent
        def add_op(a, b):
            return torch.add(a, b)

        jax_add = tojax(add_op)
        jax_result_outplace = jax_add(view1.clone(), other)

        npt.assert_allclose(
            np.asarray(jax_result_outplace),
            result_outplace.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_add_inplace_behavior(self):
        """Test in-place addition behavior"""
        original = torch.randn(3, 3, dtype=torch.float64)
        other = torch.ones_like(original)

        # Store original values
        original_copy = original.clone()

        # In-place operation should modify the original tensor
        torch.add(original, other, out=original)
        expected_result = original_copy + other

        # Test with JAX - using a function that tests out parameter
        def add_with_out(a, b, out_tensor):
            torch.add(a, b, out=out_tensor)
            return out_tensor

        jax_add_out = tojax(add_with_out)
        jax_original = original_copy.clone()
        jax_other = other.clone()
        jax_out = torch.zeros_like(original)
        jax_out = jax_add_out(jax_original, jax_other, jax_out)

        npt.assert_allclose(
            np.asarray(jax_out), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_mul_inplace_vs_outplace(self):
        """Test in-place vs out-of-place multiplication with different outcomes"""
        # Create a scenario where in-place and out-of-place would show different behavior
        base = torch.ones(2, 2, dtype=torch.float64)
        multiplier = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64)

        # Out-of-place: original remains unchanged
        base_copy1 = base.clone()
        result_outplace = torch.mul(base_copy1, multiplier)
        assert torch.equal(base_copy1, base), "Original should be unchanged"

        # Test out parameter behavior
        base_copy2 = base.clone()
        output_tensor = torch.zeros_like(base)
        torch.mul(base_copy2, multiplier, out=output_tensor)
        assert torch.equal(base_copy2, base), "Input should be unchanged when using out"
        assert torch.equal(output_tensor, result_outplace), (
            "Output tensor should contain result"
        )

        # Test with JAX
        def mul_with_out(a, b, out_tensor):
            torch.mul(a, b, out=out_tensor)
            return out_tensor

        jax_mul_out = tojax(mul_with_out)
        jax_base = base.clone()
        jax_multiplier = multiplier.clone()
        jax_out = torch.zeros_like(base)

        jax_out = jax_mul_out(jax_base, jax_multiplier, jax_out)
        npt.assert_allclose(
            np.asarray(jax_out), result_outplace.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_pow_inplace_accumulation(self):
        """Test power operation with out parameter for accumulation scenarios"""
        base = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        exponent = 2.0

        # Create accumulator tensor
        accumulator = torch.ones_like(base)

        # Use out parameter to store result in accumulator
        torch.pow(base, exponent, out=accumulator)
        expected = torch.pow(base, exponent)

        # Test with JAX
        def pow_with_out(a, exp, out_tensor):
            torch.pow(a, exp, out=out_tensor)
            return out_tensor

        jax_pow_out = tojax(pow_with_out)
        jax_base = base.clone()
        jax_accumulator = torch.ones_like(base)
        jax_accumulator = jax_pow_out(jax_base, exponent, jax_accumulator)

        npt.assert_allclose(
            np.asarray(jax_accumulator), expected.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_matmul_with_out_parameter(self):
        """Test matrix multiplication with out parameter for memory efficiency"""
        a = torch.randn(3, 4, dtype=torch.float64)
        b = torch.randn(4, 5, dtype=torch.float64)

        # Pre-allocate output tensor
        output = torch.zeros(3, 5, dtype=torch.float64)

        # Use out parameter
        torch.matmul(a, b, out=output)
        expected = torch.matmul(a, b)

        assert torch.allclose(output, expected), "Out parameter should work correctly"

        # Test with JAX
        def matmul_with_out(x, y, out_tensor):
            torch.matmul(x, y, out=out_tensor)
            return out_tensor

        jax_matmul_out = tojax(matmul_with_out)
        jax_a = a.clone()
        jax_b = b.clone()
        jax_output = torch.zeros(3, 5, dtype=torch.float64)

        jax_output = jax_matmul_out(jax_a, jax_b, jax_output)
        npt.assert_allclose(
            np.asarray(jax_output), expected.numpy(), rtol=1e-5, atol=1e-5
        )


class TestComplexOutParameterScenarios:
    """Test complex scenarios where out parameter behavior is important"""

    def test_chained_operations_with_out(self):
        """Test chained operations using out parameter for memory efficiency"""
        x = torch.randn(4, 4, dtype=torch.float64)
        y = torch.randn(4, 4, dtype=torch.float64)

        # Pre-allocate intermediate and final result tensors
        intermediate = torch.zeros_like(x)
        final_result = torch.zeros_like(x)

        # Chain operations: (x + y) * x, storing intermediates
        torch.add(x, y, out=intermediate)
        torch.mul(intermediate, x, out=final_result)

        # Expected result computed normally
        expected = (x + y) * x

        # Test with JAX - create a function that performs both operations with out parameters
        def chained_ops_with_out(a, b, intermediate_out, final_out):
            torch.add(a, b, out=intermediate_out)
            torch.mul(intermediate_out, a, out=final_out)
            return final_out

        jax_chained = tojax(chained_ops_with_out)
        jax_x = x.clone()
        jax_y = y.clone()
        jax_intermediate = torch.zeros_like(x)
        jax_final = torch.zeros_like(x)

        jax_final = jax_chained(jax_x, jax_y, jax_intermediate, jax_final)

        npt.assert_allclose(
            np.asarray(jax_final), expected.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_broadcasting_with_out_parameter(self):
        """Test broadcasting behavior with out parameter"""
        matrix = torch.randn(3, 4, dtype=torch.float64)
        vector = torch.randn(4, dtype=torch.float64)

        # Pre-allocate output with correct broadcast shape
        output = torch.zeros(3, 4, dtype=torch.float64)

        # Add with broadcasting
        torch.add(matrix, vector, out=output)
        expected = matrix + vector

        # Test with JAX
        def broadcast_add_with_out(m, v, out_tensor):
            torch.add(m, v, out=out_tensor)
            return out_tensor

        jax_broadcast_add = tojax(broadcast_add_with_out)
        jax_matrix = matrix.clone()
        jax_vector = vector.clone()
        jax_output = torch.zeros(3, 4, dtype=torch.float64)

        jax_output = jax_broadcast_add(jax_matrix, jax_vector, jax_output)
        npt.assert_allclose(
            np.asarray(jax_output), expected.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_reduction_operations_accumulated(self):
        """Test reduction operations where results are accumulated"""
        data = torch.randn(5, 6, dtype=torch.float64)

        # Test different reduction operations
        # Note: This tests the pattern, even if specific reductions aren't implemented yet
        expected_sum = torch.sum(data, dim=1)

        # Test with JAX (if sum is implemented)
        def sum_op(tensor, dim):
            return torch.sum(tensor, dim=dim)

        jax_sum = tojax(sum_op)
        jax_result = jax_sum(data, 1)
        npt.assert_allclose(
            np.asarray(jax_result), expected_sum.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_tensor_indexing_with_out(self):
        """Test tensor operations with indexing and out parameter"""
        large_tensor = torch.randn(10, 10, dtype=torch.float64)
        small_tensor = torch.randn(5, 5, dtype=torch.float64)

        # Extract a slice and perform operation with out parameter
        extracted = large_tensor[:5, :5]
        result = torch.zeros_like(extracted)

        torch.add(extracted, small_tensor, out=result)
        expected = extracted + small_tensor

        # Test with JAX - create a function that tests slicing with out parameter
        def slice_add_with_out(large, small, out_tensor):
            extracted_slice = large[:5, :5]
            torch.add(extracted_slice, small, out=out_tensor)
            return out_tensor

        jax_slice_add = tojax(slice_add_with_out)
        jax_large = large_tensor.clone()
        jax_small = small_tensor.clone()
        jax_result = torch.zeros(5, 5, dtype=torch.float64)

        jax_result = jax_slice_add(jax_large, jax_small, jax_result)
        npt.assert_allclose(
            np.asarray(jax_result), expected.numpy(), rtol=1e-5, atol=1e-5
        )


# Extensible list of random functions to test
# Format: (id, random_fn, needs_template)
# To add a new function, append to this list
RANDOM_FUNCTIONS = [
    pytest.param(torch.randn, False, id="randn"),
    pytest.param(torch.randn_like, True, id="randn_like"),
    pytest.param(torch.rand, False, id="rand"),
    pytest.param(torch.rand_like, True, id="rand_like"),
]


class TestRandomFunctions:
    """Test suite for random functions with different RNG modes."""

    @pytest.fixture
    def template_tensor(self):
        return torch.ones(3, 4, dtype=torch.float64)

    def _make_op(self, fn, needs_template, template_tensor):
        """Create an op function for the given random function."""
        if needs_template:

            def op():
                return fn(template_tensor)
        else:

            def op():
                return fn(3, 4)

        return op

    @pytest.mark.parametrize("fn,needs_template", RANDOM_FUNCTIONS)
    def test_raise_mode(self, fn, needs_template, template_tensor):
        """RAISE mode should error when RNG is used."""
        op = self._make_op(fn, needs_template, template_tensor)
        jax_fn = tojax(op, rng_mode=RNGMode.RAISE)
        with pytest.raises(AssertionError):
            jax_fn()

    @pytest.mark.parametrize("fn,needs_template", RANDOM_FUNCTIONS)
    def test_fixed_mode(self, fn, needs_template, template_tensor):
        """FIXED mode should produce deterministic results."""
        op = self._make_op(fn, needs_template, template_tensor)
        jax_fn = tojax(op, rng_mode=RNGMode.FIXED)
        result1 = np.asarray(jax_fn())
        result2 = np.asarray(jax_fn())
        npt.assert_array_equal(result1, result2)

    @pytest.mark.parametrize("fn,needs_template", RANDOM_FUNCTIONS)
    def test_explicit_mode(self, fn, needs_template, template_tensor):
        """EXPLICIT mode should accept key and produce deterministic results."""
        op = self._make_op(fn, needs_template, template_tensor)
        jax_fn = tojax(op, rng_mode=RNGMode.EXPLICIT)
        key = jax.random.PRNGKey(42)
        result1 = np.asarray(jax_fn(key))
        result2 = np.asarray(jax_fn(key))
        npt.assert_array_equal(result1, result2)
        # Different key should produce different results
        result3 = np.asarray(jax_fn(jax.random.PRNGKey(123)))
        assert not np.array_equal(result1, result3)


class TestScatterOperations:
    """Test suite for scatter operations."""

    def test_scatter_dim0(self):
        """Test scatter_ along dim 0."""

        def scatter_op(x):
            index = torch.tensor([[0, 1, 2], [0, 1, 2]])
            src = torch.ones(2, 3, dtype=torch.float64)
            result = x.clone()
            result.scatter_(0, index, src)
            return result

        x = torch.zeros(3, 3, dtype=torch.float64)
        torch_result = scatter_op(x)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(x)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())

    def test_scatter_dim1(self):
        """Test scatter_ along dim 1."""

        def scatter_op(x):
            index = torch.tensor([[0, 1, 2], [2, 1, 0]])
            src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
            result = x.clone()
            result.scatter_(1, index, src)
            return result

        x = torch.zeros(2, 3, dtype=torch.float64)
        torch_result = scatter_op(x)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(x)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())

    def test_scatter_add(self):
        """Test scatter_ with reduce='add'."""

        def scatter_op(x):
            index = torch.tensor([[0, 0, 0]])
            src = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
            result = x.clone()
            result.scatter_(0, index, src, reduce="add")
            return result

        x = torch.ones(2, 3, dtype=torch.float64)
        torch_result = scatter_op(x)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(x)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())

    def test_scatter_multiply(self):
        """Test scatter_ with reduce='multiply'."""

        def scatter_op(x):
            index = torch.tensor([[0, 0, 0]])
            src = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64)
            result = x.clone()
            result.scatter_(0, index, src, reduce="multiply")
            return result

        x = torch.ones(2, 3, dtype=torch.float64) * 2
        torch_result = scatter_op(x)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(x)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())


# Padding modes to test
PAD_MODES = [
    pytest.param("constant", id="constant"),
    pytest.param("reflect", id="reflect"),
    pytest.param("replicate", id="replicate"),
    pytest.param("circular", id="circular"),
]


class TestPadOperations:
    """Test suite for F.pad operations."""

    @pytest.mark.parametrize("mode", PAD_MODES)
    def test_pad_1d(self, mode):
        """Test 1D padding (last dim only)."""

        def pad_op(x):
            return nn.functional.pad(x, (2, 3), mode=mode)

        x = torch.randn(2, 3, 10, dtype=torch.float64)
        torch_result = pad_op(x)

        jax_fn = tojax(pad_op)
        jax_result = jax_fn(x)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy(), rtol=1e-5)

    @pytest.mark.parametrize("mode", PAD_MODES)
    def test_pad_2d(self, mode):
        """Test 2D padding (last 2 dims)."""

        def pad_op(x):
            return nn.functional.pad(x, (1, 2, 3, 4), mode=mode)

        x = torch.randn(2, 3, 10, 10, dtype=torch.float64)
        torch_result = pad_op(x)

        jax_fn = tojax(pad_op)
        jax_result = jax_fn(x)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy(), rtol=1e-5)

    def test_pad_constant_value(self):
        """Test constant padding with custom value."""

        def pad_op(x):
            return nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=5.0)

        x = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
        torch_result = pad_op(x)

        jax_fn = tojax(pad_op)
        jax_result = jax_fn(x)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())


class TestBucketize:
    """Test suite for torch.bucketize."""

    def test_bucketize_basic(self):
        """Test basic bucketize with default right=False."""

        def bucketize_op(x, boundaries):
            return torch.bucketize(x, boundaries)

        x = torch.tensor([1.5, 2.5, 3.5, 4.5])
        boundaries = torch.tensor([2.0, 3.0, 4.0])
        torch_result = bucketize_op(x, boundaries)

        jax_fn = tojax(bucketize_op)
        jax_result = jax_fn(x, boundaries)
        npt.assert_array_equal(np.asarray(jax_result), torch_result.numpy())

    def test_bucketize_right_true(self):
        """Test bucketize with right=True."""

        def bucketize_op(x, boundaries):
            return torch.bucketize(x, boundaries, right=True)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        boundaries = torch.tensor([2.0, 3.0, 4.0])
        torch_result = bucketize_op(x, boundaries)

        jax_fn = tojax(bucketize_op)
        jax_result = jax_fn(x, boundaries)
        npt.assert_array_equal(np.asarray(jax_result), torch_result.numpy())

    def test_bucketize_exact_matches(self):
        """Test bucketize behavior at exact boundary values."""

        def bucketize_left(x, boundaries):
            return torch.bucketize(x, boundaries, right=False)

        def bucketize_right(x, boundaries):
            return torch.bucketize(x, boundaries, right=True)

        x = torch.tensor([2.0, 3.0, 4.0])
        boundaries = torch.tensor([2.0, 3.0, 4.0])

        # right=False: value goes to first bucket where boundary > value
        torch_left = bucketize_left(x, boundaries)
        jax_left = tojax(bucketize_left)(x, boundaries)
        npt.assert_array_equal(np.asarray(jax_left), torch_left.numpy())

        # right=True: value goes to first bucket where boundary >= value
        torch_right = bucketize_right(x, boundaries)
        jax_right = tojax(bucketize_right)(x, boundaries)
        npt.assert_array_equal(np.asarray(jax_right), torch_right.numpy())

    def test_bucketize_out_int32(self):
        """Test bucketize with out_int32=True."""

        def bucketize_op(x, boundaries):
            return torch.bucketize(x, boundaries, out_int32=True)

        x = torch.tensor([1.5, 2.5, 3.5])
        boundaries = torch.tensor([2.0, 3.0])
        torch_result = bucketize_op(x, boundaries)

        jax_fn = tojax(bucketize_op)
        jax_result = jax_fn(x, boundaries)
        npt.assert_array_equal(np.asarray(jax_result), torch_result.numpy())
        assert jax_result.dtype == np.int32

    def test_bucketize_2d_input(self):
        """Test bucketize with 2D input tensor."""

        def bucketize_op(x, boundaries):
            return torch.bucketize(x, boundaries)

        x = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        boundaries = torch.tensor([2.0, 3.0, 4.0])
        torch_result = bucketize_op(x, boundaries)

        jax_fn = tojax(bucketize_op)
        jax_result = jax_fn(x, boundaries)
        npt.assert_array_equal(np.asarray(jax_result), torch_result.numpy())


class TestScatterReduce:
    """Test suite for torch.scatter_reduce."""

    def test_scatter_reduce_sum(self):
        """Test scatter_reduce with sum reduction."""

        def scatter_op(input, index, src):
            return torch.scatter_reduce(input, 0, index, src, reduce="sum")

        input = torch.zeros(3, 3, dtype=torch.float64)
        index = torch.tensor([[0, 1, 2], [0, 1, 2]])
        src = torch.ones(2, 3, dtype=torch.float64)
        torch_result = scatter_op(input, index, src)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(input, index, src)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())

    def test_scatter_reduce_prod(self):
        """Test scatter_reduce with prod reduction."""

        def scatter_op(input, index, src):
            return torch.scatter_reduce(input, 0, index, src, reduce="prod")

        input = torch.ones(3, 3, dtype=torch.float64) * 2
        index = torch.tensor([[0, 1, 2]])
        src = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float64)
        torch_result = scatter_op(input, index, src)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(input, index, src)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())

    def test_scatter_reduce_amax(self):
        """Test scatter_reduce with amax reduction."""

        def scatter_op(input, index, src):
            return torch.scatter_reduce(input, 0, index, src, reduce="amax")

        input = torch.zeros(3, 3, dtype=torch.float64)
        index = torch.tensor([[0, 0, 0], [0, 0, 0]])
        src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        torch_result = scatter_op(input, index, src)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(input, index, src)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())

    def test_scatter_reduce_amin(self):
        """Test scatter_reduce with amin reduction."""

        def scatter_op(input, index, src):
            return torch.scatter_reduce(input, 0, index, src, reduce="amin")

        input = torch.ones(3, 3, dtype=torch.float64) * 10
        index = torch.tensor([[0, 0, 0], [0, 0, 0]])
        src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
        torch_result = scatter_op(input, index, src)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(input, index, src)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())

    def test_scatter_reduce_mean(self):
        """Test scatter_reduce with mean reduction."""

        def scatter_op(input, index, src):
            return torch.scatter_reduce(input, 0, index, src, reduce="mean")

        input = torch.zeros(3, 3, dtype=torch.float64)
        index = torch.tensor([[0, 0, 0], [0, 0, 0]])
        src = torch.tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]], dtype=torch.float64)
        torch_result = scatter_op(input, index, src)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(input, index, src)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())

    def test_scatter_reduce_include_self_false(self):
        """Test scatter_reduce with include_self=False."""

        def scatter_op(input, index, src):
            return torch.scatter_reduce(
                input, 0, index, src, reduce="sum", include_self=False
            )

        input = torch.ones(3, 3, dtype=torch.float64) * 100  # Should be ignored
        index = torch.tensor([[0, 1, 2]])
        src = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        torch_result = scatter_op(input, index, src)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(input, index, src)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())

    def test_scatter_reduce_dim1(self):
        """Test scatter_reduce along dim 1."""

        def scatter_op(input, index, src):
            return torch.scatter_reduce(input, 1, index, src, reduce="sum")

        input = torch.zeros(2, 4, dtype=torch.float64)
        index = torch.tensor([[0, 1, 0, 1], [2, 3, 2, 3]])
        src = torch.ones(2, 4, dtype=torch.float64)
        torch_result = scatter_op(input, index, src)

        jax_fn = tojax(scatter_op)
        jax_result = jax_fn(input, index, src)
        npt.assert_allclose(np.asarray(jax_result), torch_result.numpy())


class TestRepeatInterleave:
    """Test suite for torch.repeat_interleave."""

    def test_scalar_repeats(self):
        """Test repeat_interleave with scalar repeats."""

        def op(x):
            return torch.repeat_interleave(x, 3)

        x = torch.tensor([1, 2, 3], dtype=torch.float64)
        torch_result = op(x)

        jax_fn = tojax(op)
        jax_result = jax_fn(x)
        npt.assert_array_equal(np.asarray(jax_result), torch_result.numpy())

    def test_tensor_repeats(self):
        """Test repeat_interleave with tensor repeats."""

        def op(x, repeats):
            return torch.repeat_interleave(x, repeats)

        x = torch.tensor([1, 2, 3], dtype=torch.float64)
        repeats = torch.tensor([1, 2, 3])
        torch_result = op(x, repeats)

        jax_fn = tojax(op)
        jax_result = jax_fn(x, repeats)
        npt.assert_array_equal(np.asarray(jax_result), torch_result.numpy())

    def test_with_dim(self):
        """Test repeat_interleave along a specific dimension."""

        def op(x):
            return torch.repeat_interleave(x, 2, dim=1)

        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
        torch_result = op(x)

        jax_fn = tojax(op)
        jax_result = jax_fn(x)
        npt.assert_array_equal(np.asarray(jax_result), torch_result.numpy())

    def test_with_dim_and_tensor_repeats(self):
        """Test repeat_interleave with tensor repeats along a dimension."""

        def op(x, repeats):
            return torch.repeat_interleave(x, repeats, dim=0)

        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
        repeats = torch.tensor([1, 2])
        torch_result = op(x, repeats)

        jax_fn = tojax(op)
        jax_result = jax_fn(x, repeats)
        npt.assert_array_equal(np.asarray(jax_result), torch_result.numpy())

    def test_output_size(self):
        """Test repeat_interleave with output_size parameter."""

        def op(x, repeats):
            return torch.repeat_interleave(x, repeats, output_size=6)

        x = torch.tensor([1, 2, 3], dtype=torch.float64)
        repeats = torch.tensor([1, 2, 3])
        torch_result = op(x, repeats)

        jax_fn = tojax(op)
        jax_result = jax_fn(x, repeats)
        npt.assert_array_equal(np.asarray(jax_result), torch_result.numpy())


def _collect_dot_general_precisions(jaxpr) -> list:
    """Extract precision values from all dot_general ops in a jaxpr."""
    precisions = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            precisions.append(eqn.params.get("precision"))
        if eqn.params.get("call_jaxpr"):
            precisions.extend(_collect_dot_general_precisions(eqn.params["call_jaxpr"]))
    return precisions


def _set_torch_precision(precision_str):
    """Context-manager-like helper to set torch precision and return cleanup info."""
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_prec = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision(precision_str)
    return old_tf32, old_prec


def _restore_torch_precision(old_tf32, old_prec):
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.set_float32_matmul_precision(old_prec)


PRECISION_PARAMS = [
    pytest.param("highest", jax.lax.Precision.HIGHEST, id="highest"),
    pytest.param("high", jax.lax.Precision.HIGH, id="high"),
    pytest.param("medium", jax.lax.Precision.DEFAULT, id="medium"),
]


class TestMatmulPrecision:
    """Test that PyTorch matmul precision settings are correctly inherited."""

    def test_get_matmul_precision_tf32_disabled(self):
        """When allow_tf32=False, precision should always be HIGHEST."""
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            assert get_matmul_precision() == jax.lax.Precision.HIGHEST
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @pytest.mark.parametrize("torch_prec, expected_jax", PRECISION_PARAMS)
    def test_get_matmul_precision_mapping(self, torch_prec, expected_jax):
        """Test mapping from torch precision string to JAX precision."""
        old_tf32, old_prec = _set_torch_precision(torch_prec)
        try:
            assert get_matmul_precision() == expected_jax
        finally:
            _restore_torch_precision(old_tf32, old_prec)

    @pytest.mark.parametrize("torch_prec, expected_jax", PRECISION_PARAMS)
    def test_matmul_precision_in_jaxpr(self, torch_prec, expected_jax):
        """Test torch.matmul propagates precision to dot_general."""
        old_tf32, old_prec = _set_torch_precision(torch_prec)
        try:
            jax_fn = tojax(lambda a, b: torch.matmul(a, b))
            a = jnp.ones((4, 5), dtype=jnp.float32)
            b = jnp.ones((5, 3), dtype=jnp.float32)
            jaxpr = jax.make_jaxpr(jax_fn)(a, b)
            precisions = _collect_dot_general_precisions(jaxpr)
            assert len(precisions) > 0
            assert all(expected_jax in p for p in precisions)
        finally:
            _restore_torch_precision(old_tf32, old_prec)

    @pytest.mark.parametrize("torch_prec, expected_jax", PRECISION_PARAMS)
    def test_mm_precision_in_jaxpr(self, torch_prec, expected_jax):
        """Test torch.mm propagates precision to dot_general."""
        old_tf32, old_prec = _set_torch_precision(torch_prec)
        try:
            jax_fn = tojax(lambda a, b: torch.mm(a, b))
            a = jnp.ones((4, 5), dtype=jnp.float32)
            b = jnp.ones((5, 3), dtype=jnp.float32)
            jaxpr = jax.make_jaxpr(jax_fn)(a, b)
            precisions = _collect_dot_general_precisions(jaxpr)
            assert len(precisions) > 0
            assert all(expected_jax in p for p in precisions)
        finally:
            _restore_torch_precision(old_tf32, old_prec)

    @pytest.mark.parametrize("torch_prec, expected_jax", PRECISION_PARAMS)
    def test_einsum_precision_in_jaxpr(self, torch_prec, expected_jax):
        """Test torch.einsum propagates precision to dot_general."""
        old_tf32, old_prec = _set_torch_precision(torch_prec)
        try:
            jax_fn = tojax(lambda a, b: torch.einsum("ij,jk->ik", a, b))
            a = jnp.ones((4, 5), dtype=jnp.float32)
            b = jnp.ones((5, 3), dtype=jnp.float32)
            jaxpr = jax.make_jaxpr(jax_fn)(a, b)
            precisions = _collect_dot_general_precisions(jaxpr)
            assert len(precisions) > 0
            assert all(expected_jax in p for p in precisions)
        finally:
            _restore_torch_precision(old_tf32, old_prec)

    @pytest.mark.parametrize("torch_prec, expected_jax", PRECISION_PARAMS)
    def test_bmm_precision_in_jaxpr(self, torch_prec, expected_jax):
        """Test torch.bmm propagates precision to dot_general."""
        old_tf32, old_prec = _set_torch_precision(torch_prec)
        try:
            jax_fn = tojax(lambda a, b: torch.bmm(a, b))
            a = jnp.ones((2, 4, 5), dtype=jnp.float32)
            b = jnp.ones((2, 5, 3), dtype=jnp.float32)
            jaxpr = jax.make_jaxpr(jax_fn)(a, b)
            precisions = _collect_dot_general_precisions(jaxpr)
            assert len(precisions) > 0
            assert all(expected_jax in p for p in precisions)
        finally:
            _restore_torch_precision(old_tf32, old_prec)

    @pytest.mark.parametrize("torch_prec, expected_jax", PRECISION_PARAMS)
    def test_tensordot_precision_in_jaxpr(self, torch_prec, expected_jax):
        """Test torch.tensordot propagates precision to dot_general."""
        old_tf32, old_prec = _set_torch_precision(torch_prec)
        try:
            jax_fn = tojax(lambda a, b: torch.tensordot(a, b, dims=1))
            a = jnp.ones((4, 5), dtype=jnp.float32)
            b = jnp.ones((5, 3), dtype=jnp.float32)
            jaxpr = jax.make_jaxpr(jax_fn)(a, b)
            precisions = _collect_dot_general_precisions(jaxpr)
            assert len(precisions) > 0
            assert all(expected_jax in p for p in precisions)
        finally:
            _restore_torch_precision(old_tf32, old_prec)

    @pytest.mark.parametrize("torch_prec, expected_jax", PRECISION_PARAMS)
    def test_linear_precision_in_jaxpr(self, torch_prec, expected_jax):
        """Test nn.Linear propagates precision to dot_general."""
        old_tf32, old_prec = _set_torch_precision(torch_prec)
        try:
            layer = nn.Linear(5, 3, dtype=torch.float32)
            jax_fn = tojax(layer)
            x = jnp.ones((2, 5), dtype=jnp.float32)
            jaxpr = jax.make_jaxpr(jax_fn)(x)
            precisions = _collect_dot_general_precisions(jaxpr)
            assert len(precisions) > 0
            assert all(expected_jax in p for p in precisions)
        finally:
            _restore_torch_precision(old_tf32, old_prec)


class TestDiff:
    def test_basic(self):
        x = torch.tensor([1.0, 3.0, 6.0, 10.0])
        expected = torch.diff(x).numpy()
        result = tojax(torch.diff)(jnp.asarray(x))
        npt.assert_allclose(np.asarray(result), expected)

    def test_n2(self):
        x = torch.tensor([1.0, 3.0, 6.0, 10.0, 15.0])
        expected = torch.diff(x, n=2).numpy()
        result = tojax(lambda t: torch.diff(t, n=2))(jnp.asarray(x))
        npt.assert_allclose(np.asarray(result), expected)

    def test_dim(self):
        x = torch.arange(12.0).reshape(3, 4)
        expected = torch.diff(x, dim=0).numpy()
        result = tojax(lambda t: torch.diff(t, dim=0))(jnp.asarray(x))
        npt.assert_allclose(np.asarray(result), expected)

    def test_prepend_append(self):
        x = torch.tensor([1.0, 3.0, 6.0])
        prepend = torch.tensor([0.0])
        append = torch.tensor([10.0])
        expected = torch.diff(x, prepend=prepend, append=append).numpy()
        result = tojax(lambda t, p, a: torch.diff(t, prepend=p, append=a))(
            jnp.asarray(x), jnp.asarray(prepend), jnp.asarray(append)
        )
        npt.assert_allclose(np.asarray(result), expected)


class TestIsin:
    def test_basic(self):
        elements = torch.tensor([1, 2, 3, 4, 5])
        test_elements = torch.tensor([2, 4])
        expected = torch.isin(elements, test_elements).numpy()
        result = tojax(torch.isin)(jnp.asarray(elements), jnp.asarray(test_elements))
        npt.assert_array_equal(np.asarray(result), expected)

    def test_invert(self):
        elements = torch.tensor([1, 2, 3, 4, 5])
        test_elements = torch.tensor([2, 4])
        expected = torch.isin(elements, test_elements, invert=True).numpy()
        result = tojax(lambda e, t: torch.isin(e, t, invert=True))(
            jnp.asarray(elements), jnp.asarray(test_elements)
        )
        npt.assert_array_equal(np.asarray(result), expected)

    def test_2d(self):
        elements = torch.arange(12).reshape(3, 4)
        test_elements = torch.tensor([0, 3, 7, 11])
        expected = torch.isin(elements, test_elements).numpy()
        result = tojax(torch.isin)(jnp.asarray(elements), jnp.asarray(test_elements))
        npt.assert_array_equal(np.asarray(result), expected)


class TestTensorCreation:
    """Test torch.tensor and torch.as_tensor with device argument."""

    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_tensor_with_device(self, device):
        data = [1.0, 2.0, 3.0]
        expected = torch.tensor(data, device=device).numpy()

        jax_fn = tojax(lambda d: torch.tensor(d, device=device))
        result = jax_fn(jnp.asarray(data))
        npt.assert_allclose(np.asarray(result), expected)

    @pytest.mark.parametrize("device", [None, "cpu"])
    def test_as_tensor_with_device(self, device):
        data = [1.0, 2.0, 3.0]
        expected = torch.as_tensor(data, device=device).numpy()

        jax_fn = tojax(lambda d: torch.as_tensor(d, device=device))
        result = jax_fn(jnp.asarray(data))
        npt.assert_allclose(np.asarray(result), expected)

    @pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64, torch.int64])
    def test_tensor_with_dtype_and_device(self, dtype):
        data = [1.0, 2.0, 3.0]
        expected = torch.tensor(data, dtype=dtype, device="cpu").numpy()

        jax_fn = tojax(lambda d: torch.tensor(d, dtype=dtype, device="cpu"))
        result = jax_fn(jnp.asarray(data))
        npt.assert_allclose(np.asarray(result), expected)


class TestSoftmax:
    """Test torch.softmax / torch.nn.functional.softmax translation."""

    def test_basic(self):
        x = torch.randn(3, 4)
        expected = torch.softmax(x, dim=-1).numpy()
        result = tojax(lambda t: torch.softmax(t, dim=-1))(jnp.asarray(x))
        npt.assert_allclose(np.asarray(result), expected, rtol=1e-5, atol=1e-5)

    def test_dim0(self):
        x = torch.randn(3, 4)
        expected = torch.softmax(x, dim=0).numpy()
        result = tojax(lambda t: torch.softmax(t, dim=0))(jnp.asarray(x))
        npt.assert_allclose(np.asarray(result), expected, rtol=1e-5, atol=1e-5)

    def test_functional(self):
        x = torch.randn(3, 4)
        expected = torch.nn.functional.softmax(x, dim=-1).numpy()
        result = tojax(lambda t: torch.nn.functional.softmax(t, dim=-1))(jnp.asarray(x))
        npt.assert_allclose(np.asarray(result), expected, rtol=1e-5, atol=1e-5)

    def test_with_dtype(self):
        x = torch.randn(3, 4)
        expected = torch.softmax(x, dim=-1, dtype=torch.float32).numpy()
        result = tojax(lambda t: torch.softmax(t, dim=-1, dtype=torch.float32))(
            jnp.asarray(x)
        )
        npt.assert_allclose(np.asarray(result), expected, rtol=1e-5, atol=1e-5)


class TestSearchSorted:
    def test_basic(self):
        def op(s, v):
            return torch.searchsorted(s, v)

        s = torch.tensor([1, 3, 5, 7, 9])
        v = torch.tensor([3, 6, 9])
        expected = op(s, v)
        result = tojax(op)(s, v)
        npt.assert_array_equal(np.asarray(result), expected.numpy())

    def test_right(self):
        def op(s, v):
            return torch.searchsorted(s, v, right=True)

        s = torch.tensor([1, 3, 5, 7, 9])
        v = torch.tensor([3, 6, 9])
        expected = op(s, v)
        result = tojax(op)(s, v)
        npt.assert_array_equal(np.asarray(result), expected.numpy())

    def test_side(self):
        def op_left(s, v):
            return torch.searchsorted(s, v, side="left")

        def op_right(s, v):
            return torch.searchsorted(s, v, side="right")

        s = torch.tensor([1, 3, 5, 7, 9])
        v = torch.tensor([3, 5, 7])
        for op in [op_left, op_right]:
            expected = op(s, v)
            result = tojax(op)(s, v)
            npt.assert_array_equal(np.asarray(result), expected.numpy())

    def test_out_int32(self):
        def op(s, v):
            return torch.searchsorted(s, v, out_int32=True)

        s = torch.tensor([1, 3, 5, 7, 9])
        v = torch.tensor([3, 6, 9])
        result = tojax(op)(s, v)
        assert result.dtype == np.int32


class TestNorm:
    def test_frobenius_default(self):
        x = torch.randn(3, 4)
        expected = torch.linalg.norm(x).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.norm(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_vector_norm_along_dim(self):
        x = torch.randn(3, 4)
        expected = torch.linalg.norm(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.norm(t, dim=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_nuclear_norm(self):
        x = torch.randn(3, 4)
        expected = torch.linalg.norm(x, ord="nuc").numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.norm(t, ord="nuc"))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_inf_norm(self):
        x = torch.randn(3, 4)
        expected = torch.linalg.norm(x, ord=float("inf"), dim=1).numpy()
        result = np.asarray(
            tojax(lambda t: torch.linalg.norm(t, ord=float("inf"), dim=1))(x)
        )
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_matrix_ord1(self):
        x = torch.randn(3, 4)
        expected = torch.linalg.norm(x, ord=1).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.norm(t, ord=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_keepdim(self):
        x = torch.randn(3, 4)
        expected = torch.linalg.norm(x, dim=0, keepdim=True).numpy()
        result = np.asarray(
            tojax(lambda t: torch.linalg.norm(t, dim=0, keepdim=True))(x)
        )
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_torch_norm_p2(self):
        x = torch.randn(3, 4)
        expected = torch.norm(x, p=2).numpy()
        result = np.asarray(tojax(lambda t: torch.norm(t, p=2))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)


class TestRmsNorm:
    """Test nn.functional.rms_norm translation."""

    def test_rms_norm_with_weight(self):
        x = torch.randn(2, 3, 4)
        w = torch.randn(4)
        expected = nn.functional.rms_norm(x, [4], weight=w, eps=1e-5).numpy()
        result = np.asarray(
            tojax(lambda t, wt: nn.functional.rms_norm(t, [4], weight=wt, eps=1e-5))(
                x, w
            )
        )
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_rms_norm_without_weight(self):
        x = torch.randn(2, 3, 4)
        expected = nn.functional.rms_norm(x, [4], eps=1e-6).numpy()
        result = np.asarray(
            tojax(lambda t: nn.functional.rms_norm(t, [4], eps=1e-6))(x)
        )
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_rms_norm_multi_dim(self):
        x = torch.randn(2, 3, 4)
        w = torch.randn(3, 4)
        expected = nn.functional.rms_norm(x, [3, 4], weight=w, eps=1e-5).numpy()
        result = np.asarray(
            tojax(lambda t, wt: nn.functional.rms_norm(t, [3, 4], weight=wt, eps=1e-5))(
                x, w
            )
        )
        npt.assert_allclose(result, expected, rtol=1e-5)


class TestComparison:
    """Test comparison and check operations."""

    def test_ne(self):
        x, y = torch.randn(3, 4), torch.randn(3, 4)
        expected = torch.ne(x, y).numpy()
        result = np.asarray(tojax(lambda a, b: torch.ne(a, b))(x, y))
        npt.assert_array_equal(result, expected)

    def test_maximum(self):
        x, y = torch.randn(3, 4), torch.randn(3, 4)
        expected = torch.maximum(x, y).numpy()
        result = np.asarray(tojax(lambda a, b: torch.maximum(a, b))(x, y))
        npt.assert_allclose(result, expected)

    def test_minimum(self):
        x, y = torch.randn(3, 4), torch.randn(3, 4)
        expected = torch.minimum(x, y).numpy()
        result = np.asarray(tojax(lambda a, b: torch.minimum(a, b))(x, y))
        npt.assert_allclose(result, expected)

    def test_isnan(self):
        x = torch.tensor([1.0, float("nan"), 3.0])
        expected = torch.isnan(x).numpy()
        result = np.asarray(tojax(lambda t: torch.isnan(t))(x))
        npt.assert_array_equal(result, expected)

    def test_isfinite(self):
        x = torch.tensor([1.0, float("inf"), float("nan")])
        expected = torch.isfinite(x).numpy()
        result = np.asarray(tojax(lambda t: torch.isfinite(t))(x))
        npt.assert_array_equal(result, expected)


class TestReductions:
    """Test reduction operations."""

    def test_prod(self):
        x = torch.randn(3, 4)
        expected = torch.prod(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.prod(t, dim=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_all(self):
        x = torch.tensor([[True, True], [True, False]])
        expected = torch.all(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.all(t, dim=1))(x))
        npt.assert_array_equal(result, expected)

    def test_any(self):
        x = torch.tensor([[False, False], [True, False]])
        expected = torch.any(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.any(t, dim=1))(x))
        npt.assert_array_equal(result, expected)

    def test_amax(self):
        x = torch.randn(3, 4)
        expected = torch.amax(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.amax(t, dim=1))(x))
        npt.assert_allclose(result, expected)

    def test_amin(self):
        x = torch.randn(3, 4)
        expected = torch.amin(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.amin(t, dim=1))(x))
        npt.assert_allclose(result, expected)

    def test_count_nonzero(self):
        x = torch.tensor([[0, 1, 2], [0, 0, 3]])
        expected = torch.count_nonzero(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.count_nonzero(t, dim=1))(x))
        npt.assert_array_equal(result, expected)


class TestMathOps:
    """Test math operations."""

    def test_rsqrt(self):
        x = torch.rand(3, 4) + 0.1
        expected = torch.rsqrt(x).numpy()
        result = np.asarray(tojax(lambda t: torch.rsqrt(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_sign(self):
        x = torch.randn(3, 4)
        expected = torch.sign(x).numpy()
        result = np.asarray(tojax(lambda t: torch.sign(t))(x))
        npt.assert_allclose(result, expected)

    def test_log1p(self):
        x = torch.rand(3, 4)
        expected = torch.log1p(x).numpy()
        result = np.asarray(tojax(lambda t: torch.log1p(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_sinh(self):
        x = torch.randn(3, 4)
        expected = torch.sinh(x).numpy()
        result = np.asarray(tojax(lambda t: torch.sinh(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_asinh(self):
        x = torch.randn(3, 4)
        expected = torch.asinh(x).numpy()
        result = np.asarray(tojax(lambda t: torch.asinh(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_sinc(self):
        x = torch.randn(3, 4)
        expected = torch.sinc(x).numpy()
        result = np.asarray(tojax(lambda t: torch.sinc(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_frac(self):
        x = torch.tensor([-1.5, 0.3, 2.7])
        expected = torch.frac(x).numpy()
        result = np.asarray(tojax(lambda t: torch.frac(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_lerp(self):
        x, y = torch.randn(3, 4), torch.randn(3, 4)
        expected = torch.lerp(x, y, 0.5).numpy()
        result = np.asarray(tojax(lambda a, b: torch.lerp(a, b, 0.5))(x, y))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_erf(self):
        x = torch.randn(3, 4)
        expected = torch.erf(x).numpy()
        result = np.asarray(tojax(lambda t: torch.erf(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_deg2rad(self):
        x = torch.tensor([0.0, 90.0, 180.0, 360.0])
        expected = torch.deg2rad(x).numpy()
        result = np.asarray(tojax(lambda t: torch.deg2rad(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)


class TestClamp:
    """Test clamp_min and clamp_max."""

    def test_clamp_min(self):
        x = torch.randn(3, 4)
        expected = torch.clamp_min(x, 0.0).numpy()
        result = np.asarray(tojax(lambda t: torch.clamp_min(t, 0.0))(x))
        npt.assert_allclose(result, expected)

    def test_clamp_max(self):
        x = torch.randn(3, 4)
        expected = torch.clamp_max(x, 0.5).numpy()
        result = np.asarray(tojax(lambda t: torch.clamp_max(t, 0.5))(x))
        npt.assert_allclose(result, expected)


class TestStdVar:
    """Test std and var with Bessel correction."""

    def test_std_default(self):
        x = torch.randn(4, 5)
        expected = torch.std(x).numpy()
        result = np.asarray(tojax(lambda t: torch.std(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_std_dim(self):
        x = torch.randn(4, 5)
        expected = torch.std(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.std(t, dim=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_var_correction(self):
        x = torch.randn(4, 5)
        expected = torch.var(x, dim=1, correction=0).numpy()
        result = np.asarray(tojax(lambda t: torch.var(t, dim=1, correction=0))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)


class TestTensorManipulation:
    """Test tensor manipulation operations."""

    def test_flip(self):
        x = torch.randn(3, 4)
        expected = torch.flip(x, [1]).numpy()
        result = np.asarray(tojax(lambda t: torch.flip(t, [1]))(x))
        npt.assert_allclose(result, expected)

    def test_roll(self):
        x = torch.randn(3, 4)
        expected = torch.roll(x, 2, dims=1).numpy()
        result = np.asarray(tojax(lambda t: torch.roll(t, 2, dims=1))(x))
        npt.assert_allclose(result, expected)

    def test_chunk(self):
        x = torch.randn(6, 4)
        expected = [c.numpy() for c in torch.chunk(x, 3, dim=0)]
        result = tojax(lambda t: torch.chunk(t, 3, dim=0))(x)
        for r, e in zip(result, expected):
            npt.assert_allclose(np.asarray(r), e)

    def test_argsort(self):
        x = torch.randn(3, 4)
        expected = torch.argsort(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.argsort(t, dim=1))(x))
        npt.assert_array_equal(result, expected)

    def test_argsort_descending(self):
        x = torch.randn(3, 4)
        expected = torch.argsort(x, dim=1, descending=True).numpy()
        result = np.asarray(
            tojax(lambda t: torch.argsort(t, dim=1, descending=True))(x)
        )
        npt.assert_array_equal(result, expected)

    def test_diag(self):
        x = torch.randn(4)
        expected = torch.diag(x).numpy()
        result = np.asarray(tojax(lambda t: torch.diag(t))(x))
        npt.assert_allclose(result, expected)

    def test_tril(self):
        x = torch.randn(3, 3)
        expected = torch.tril(x).numpy()
        result = np.asarray(tojax(lambda t: torch.tril(t))(x))
        npt.assert_allclose(result, expected)

    def test_triu(self):
        x = torch.randn(3, 3)
        expected = torch.triu(x).numpy()
        result = np.asarray(tojax(lambda t: torch.triu(t))(x))
        npt.assert_allclose(result, expected)

    def test_tile(self):
        x = torch.randn(2, 3)
        expected = torch.tile(x, (2, 3)).numpy()
        result = np.asarray(tojax(lambda t: torch.tile(t, (2, 3)))(x))
        npt.assert_allclose(result, expected)

    def test_diagonal(self):
        x = torch.randn(3, 4)
        expected = torch.diagonal(x).numpy()
        result = np.asarray(tojax(lambda t: torch.diagonal(t))(x))
        npt.assert_allclose(result, expected)

    def test_unflatten(self):
        x = torch.randn(2, 12)
        expected = torch.unflatten(x, 1, (3, 4)).numpy()
        result = np.asarray(tojax(lambda t: torch.unflatten(t, 1, (3, 4)))(x))
        npt.assert_allclose(result, expected)

    def test_eye(self):
        expected = torch.eye(3).numpy()
        result = np.asarray(tojax(lambda: torch.eye(3))())
        npt.assert_allclose(result, expected)

    def test_linspace(self):
        expected = torch.linspace(0, 1, 5).numpy()
        result = np.asarray(tojax(lambda: torch.linspace(0, 1, 5))())
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_meshgrid(self):
        x = torch.arange(3)
        y = torch.arange(4)
        expected = [g.numpy() for g in torch.meshgrid(x, y, indexing="ij")]
        result = tojax(lambda a, b: torch.meshgrid(a, b, indexing="ij"))(x, y)
        for r, e in zip(result, expected):
            npt.assert_array_equal(np.asarray(r), e)


class TestGather:
    """Test torch.gather."""

    def test_gather_dim0(self):
        x = torch.randn(3, 4)
        index = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 1]])
        expected = torch.gather(x, 0, index).numpy()
        result = np.asarray(tojax(lambda t, i: torch.gather(t, 0, i))(x, index))
        npt.assert_allclose(result, expected)

    def test_gather_dim1(self):
        x = torch.randn(3, 4)
        index = torch.tensor([[0, 1], [2, 3], [1, 0]])
        expected = torch.gather(x, 1, index).numpy()
        result = np.asarray(tojax(lambda t, i: torch.gather(t, 1, i))(x, index))
        npt.assert_allclose(result, expected)


class TestTopk:
    """Test torch.topk."""

    def test_topk(self):
        x = torch.randn(3, 10)
        exp_values, exp_indices = torch.topk(x, 3, dim=1)
        values, indices = tojax(lambda t: torch.topk(t, 3, dim=1))(x)
        npt.assert_allclose(np.asarray(values), exp_values.numpy(), rtol=1e-5)
        npt.assert_array_equal(np.asarray(indices), exp_indices.numpy())

    def test_topk_smallest(self):
        x = torch.randn(3, 10)
        exp_values, exp_indices = torch.topk(x, 3, dim=1, largest=False)
        values, indices = tojax(lambda t: torch.topk(t, 3, dim=1, largest=False))(x)
        npt.assert_allclose(np.asarray(values), exp_values.numpy(), rtol=1e-5)
        npt.assert_array_equal(np.asarray(indices), exp_indices.numpy())


class TestMaskedFill:
    """Test torch.masked_fill."""

    def test_masked_fill(self):
        x = torch.randn(3, 4)
        mask = torch.tensor([[True, False, True, False]] * 3)
        expected = torch.masked_fill(x, mask, -1.0).numpy()
        result = np.asarray(tojax(lambda t, m: torch.masked_fill(t, m, -1.0))(x, mask))
        npt.assert_allclose(result, expected)


class TestLogsumexp:
    """Test torch.logsumexp."""

    def test_logsumexp(self):
        x = torch.randn(3, 4)
        expected = torch.logsumexp(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.logsumexp(t, dim=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)


class TestActivations:
    """Test F.* activation functions."""

    def test_leaky_relu(self):
        x = torch.randn(3, 4)
        expected = nn.functional.leaky_relu(x, 0.1).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.leaky_relu(t, 0.1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_elu(self):
        x = torch.randn(3, 4)
        expected = nn.functional.elu(x, alpha=1.5).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.elu(t, alpha=1.5))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_selu(self):
        x = torch.randn(3, 4)
        expected = nn.functional.selu(x).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.selu(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_celu(self):
        x = torch.randn(3, 4)
        expected = nn.functional.celu(x, alpha=1.5).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.celu(t, alpha=1.5))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_relu6(self):
        x = torch.randn(3, 4) * 10
        expected = nn.functional.relu6(x).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.relu6(t))(x))
        npt.assert_allclose(result, expected)

    def test_softplus(self):
        x = torch.randn(3, 4)
        expected = nn.functional.softplus(x, beta=2).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.softplus(t, beta=2))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_mish(self):
        x = torch.randn(3, 4)
        expected = nn.functional.mish(x).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.mish(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_hardswish(self):
        x = torch.randn(3, 4) * 5
        expected = nn.functional.hardswish(x).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.hardswish(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_hardsigmoid(self):
        x = torch.randn(3, 4) * 5
        expected = nn.functional.hardsigmoid(x).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.hardsigmoid(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_log_softmax(self):
        x = torch.randn(3, 4)
        expected = nn.functional.log_softmax(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.log_softmax(t, dim=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_glu(self):
        x = torch.randn(3, 8)
        expected = nn.functional.glu(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.glu(t, dim=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_f_sigmoid(self):
        x = torch.randn(3, 4)
        expected = nn.functional.sigmoid(x).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.sigmoid(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_f_tanh(self):
        x = torch.randn(3, 4)
        expected = nn.functional.tanh(x).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.tanh(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)


class TestGroupNorm:
    """Test F.group_norm."""

    def test_group_norm(self):
        x = torch.randn(2, 6, 4, 4)
        weight = torch.randn(6)
        bias = torch.randn(6)
        expected = nn.functional.group_norm(x, 3, weight=weight, bias=bias).numpy()
        result = np.asarray(
            tojax(lambda t, w, b: nn.functional.group_norm(t, 3, weight=w, bias=b))(
                x, weight, bias
            )
        )
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_group_norm_no_affine(self):
        x = torch.randn(2, 4, 3, 3)
        expected = nn.functional.group_norm(x, 2).numpy()
        result = np.asarray(tojax(lambda t: nn.functional.group_norm(t, 2))(x))
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


class TestCosineSimilarity:
    """Test F.cosine_similarity."""

    def test_cosine_similarity(self):
        x, y = torch.randn(3, 4), torch.randn(3, 4)
        expected = nn.functional.cosine_similarity(x, y, dim=1).numpy()
        result = np.asarray(
            tojax(lambda a, b: nn.functional.cosine_similarity(a, b, dim=1))(x, y)
        )
        npt.assert_allclose(result, expected, rtol=1e-5)


class TestLinalg:
    """Test torch.linalg operations."""

    def test_inv(self):
        x = torch.randn(3, 3)
        x = x @ x.T + torch.eye(3)
        expected = torch.linalg.inv(x).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.inv(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_solve(self):
        A = torch.randn(3, 3)
        A = A @ A.T + torch.eye(3)
        b = torch.randn(3)
        expected = torch.linalg.solve(A, b).numpy()
        result = np.asarray(tojax(lambda a, b: torch.linalg.solve(a, b))(A, b))
        npt.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_svd(self):
        x = torch.randn(3, 3)
        U, S, Vh = torch.linalg.svd(x)
        rU, rS, rVh = tojax(lambda t: torch.linalg.svd(t))(x)
        reconstructed_torch = (U @ torch.diag(S) @ Vh).numpy()
        reconstructed_jax = np.asarray(rU) @ np.diag(np.asarray(rS)) @ np.asarray(rVh)
        npt.assert_allclose(
            reconstructed_jax, reconstructed_torch, rtol=1e-4, atol=1e-4
        )


class TestNanReductions:
    """Test nan-aware reductions."""

    def test_nanmean(self):
        x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
        expected = torch.nanmean(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.nanmean(t, dim=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_nansum(self):
        x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
        expected = torch.nansum(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.nansum(t, dim=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)


class TestCumprod:
    """Test torch.cumprod."""

    def test_cumprod(self):
        x = torch.randn(3, 4)
        expected = torch.cumprod(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.cumprod(t, dim=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)


class TestLinalgExtended:
    """Test extended torch.linalg operations."""

    def test_cholesky(self):
        x = torch.randn(3, 3)
        x = x @ x.T + torch.eye(3) * 3  # positive definite
        expected = torch.linalg.cholesky(x).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.cholesky(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_eigh(self):
        x = torch.randn(3, 3)
        x = x + x.T  # symmetric
        exp_vals, exp_vecs = torch.linalg.eigh(x)
        vals, vecs = tojax(lambda t: torch.linalg.eigh(t))(x)
        npt.assert_allclose(np.asarray(vals), exp_vals.numpy(), rtol=1e-5, atol=1e-5)
        # eigenvectors unique up to sign — compare via reconstruction
        reconstructed_torch = (exp_vecs @ torch.diag(exp_vals) @ exp_vecs.T).numpy()
        reconstructed_jax = (
            np.asarray(vecs) @ np.diag(np.asarray(vals)) @ np.asarray(vecs).T
        )
        npt.assert_allclose(
            reconstructed_jax, reconstructed_torch, rtol=1e-4, atol=1e-4
        )

    def test_eigvals(self):
        x = torch.randn(3, 3)
        expected = torch.linalg.eigvals(x)
        result = tojax(lambda t: torch.linalg.eigvals(t))(x)
        # eigenvalues may be in different order; compare sorted magnitudes
        npt.assert_allclose(
            np.sort(np.abs(np.asarray(result))),
            np.sort(np.abs(expected.numpy())),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_qr(self):
        x = torch.randn(4, 3)
        Q, R = torch.linalg.qr(x)
        rQ, rR = tojax(lambda t: torch.linalg.qr(t))(x)
        # QR unique up to sign of rows — compare reconstruction
        npt.assert_allclose(
            np.asarray(rQ) @ np.asarray(rR), (Q @ R).numpy(), rtol=1e-5, atol=1e-5
        )

    def test_slogdet(self):
        x = torch.randn(3, 3)
        x = x @ x.T + torch.eye(3)
        exp_sign, exp_logabsdet = torch.linalg.slogdet(x)
        r_sign, r_logabsdet = tojax(lambda t: torch.linalg.slogdet(t))(x)
        npt.assert_allclose(np.asarray(r_sign), exp_sign.numpy())
        npt.assert_allclose(np.asarray(r_logabsdet), exp_logabsdet.numpy(), rtol=1e-5)

    def test_pinv(self):
        x = torch.randn(3, 5)
        expected = torch.linalg.pinv(x).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.pinv(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_svdvals(self):
        x = torch.randn(3, 4)
        expected = torch.linalg.svdvals(x).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.svdvals(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_matrix_rank(self):
        x = torch.randn(3, 3)
        x[2] = x[0] + x[1]  # rank 2
        expected = torch.linalg.matrix_rank(x).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.matrix_rank(t))(x))
        npt.assert_array_equal(result, expected)

    def test_matrix_power(self):
        x = torch.randn(3, 3)
        expected = torch.linalg.matrix_power(x, 3).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.matrix_power(t, 3))(x))
        npt.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_cond(self):
        x = torch.randn(3, 3)
        x = x @ x.T + torch.eye(3)
        expected = torch.linalg.cond(x).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.cond(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-4)

    def test_vector_norm(self):
        x = torch.randn(3, 4)
        expected = torch.linalg.vector_norm(x, dim=1).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.vector_norm(t, dim=1))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_matrix_norm(self):
        x = torch.randn(3, 4)
        expected = torch.linalg.matrix_norm(x).numpy()
        result = np.asarray(tojax(lambda t: torch.linalg.matrix_norm(t))(x))
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_solve_triangular(self):
        A = torch.randn(3, 3)
        A = torch.triu(A) + torch.eye(3) * 2  # upper triangular, well-conditioned
        b = torch.randn(3, 2)
        expected = torch.linalg.solve_triangular(A, b, upper=True).numpy()
        result = np.asarray(
            tojax(lambda a, b: torch.linalg.solve_triangular(a, b, upper=True))(A, b)
        )
        npt.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_lstsq(self):
        A = torch.randn(5, 3)
        b = torch.randn(5, 2)
        exp_result = torch.linalg.lstsq(A, b)
        jax_result = tojax(lambda a, b: torch.linalg.lstsq(a, b))(A, b)
        # Compare the solution (first element of the returned tuple)
        npt.assert_allclose(
            np.asarray(jax_result[0]), exp_result.solution.numpy(), rtol=1e-4, atol=1e-4
        )


class TestSqueeze:
    """Test torch.squeeze with no-op semantics for non-size-1 dims."""

    def test_squeeze_size1_dim(self):
        x = torch.randn(2, 1, 3)
        expected = torch.squeeze(x, 1).numpy()
        result = np.asarray(tojax(lambda t: torch.squeeze(t, 1))(x))
        npt.assert_allclose(result, expected)
        assert result.shape == (2, 3)

    def test_squeeze_noop_nonsize1(self):
        x = torch.randn(2, 1, 3)
        expected = torch.squeeze(x, 0).numpy()
        result = np.asarray(tojax(lambda t: torch.squeeze(t, 0))(x))
        npt.assert_allclose(result, expected)
        assert result.shape == (2, 1, 3)

    def test_squeeze_no_dim(self):
        x = torch.randn(1, 2, 1, 3, 1)
        expected = torch.squeeze(x).numpy()
        result = np.asarray(tojax(lambda t: torch.squeeze(t))(x))
        npt.assert_allclose(result, expected)
        assert result.shape == (2, 3)

    def test_squeeze_negative_dim(self):
        x = torch.randn(2, 1, 3)
        expected = torch.squeeze(x, -2).numpy()
        result = np.asarray(tojax(lambda t: torch.squeeze(t, -2))(x))
        npt.assert_allclose(result, expected)
        assert result.shape == (2, 3)

    def test_squeeze_negative_dim_noop(self):
        x = torch.randn(2, 1, 3)
        expected = torch.squeeze(x, -1).numpy()
        result = np.asarray(tojax(lambda t: torch.squeeze(t, -1))(x))
        npt.assert_allclose(result, expected)
        assert result.shape == (2, 1, 3)

    def test_squeeze_tuple_dims(self):
        x = torch.randn(1, 2, 1, 3, 1)
        expected = torch.squeeze(x, (0, 2, 4)).numpy()
        result = np.asarray(tojax(lambda t: torch.squeeze(t, (0, 2, 4)))(x))
        npt.assert_allclose(result, expected)
        assert result.shape == (2, 3)

    def test_squeeze_tuple_mixed(self):
        x = torch.randn(1, 2, 1, 3, 1)
        expected = torch.squeeze(x, (0, 1)).numpy()
        result = np.asarray(tojax(lambda t: torch.squeeze(t, (0, 1)))(x))
        npt.assert_allclose(result, expected)
        assert result.shape == (2, 1, 3, 1)
