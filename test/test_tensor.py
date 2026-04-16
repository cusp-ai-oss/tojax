# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import numpy.testing as npt
import pytest
import torch
from pytest import fixture

from tojax.tojax import tojax

torch.set_default_dtype(torch.float64)


@fixture
def base_tensor():
    """Base tensor for in-place operations"""
    return torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)


@fixture
def matrix_tensor():
    """Matrix tensor for in-place operations"""
    return torch.randn(3, 4, dtype=torch.float64)


@fixture
def other_tensor():
    """Another tensor for binary operations"""
    return torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64)


@fixture
def matrix_other():
    """Another matrix tensor for binary operations"""
    return torch.randn(3, 4, dtype=torch.float64)


class TestTensorInPlaceOperations:
    """Test suite for in-place tensor operations using underscore methods"""

    def test_add_inplace(self, base_tensor, other_tensor):
        """Test in-place addition using tensor.add_()"""
        # PyTorch version
        torch_tensor = base_tensor.clone()
        original_id = id(torch_tensor)
        expected_result = base_tensor + other_tensor
        torch_tensor.add_(other_tensor)

        # Verify in-place behavior
        assert id(torch_tensor) == original_id, (
            "Tensor ID should remain the same for in-place ops"
        )
        assert torch.allclose(torch_tensor, expected_result), "Result should be correct"

        # JAX version
        def inplace_add_op(tensor, other):
            return tensor.add_(other)

        jax_inplace_add = tojax(inplace_add_op)
        jax_tensor = base_tensor.clone()
        result = jax_inplace_add(jax_tensor, other_tensor)

        npt.assert_allclose(
            np.asarray(result), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_sub_inplace(self, base_tensor, other_tensor):
        """Test in-place subtraction using tensor.sub_()"""
        # PyTorch version
        torch_tensor = base_tensor.clone()
        expected_result = base_tensor - other_tensor
        torch_tensor.sub_(other_tensor)

        # JAX version
        def inplace_sub_op(tensor, other):
            return tensor.sub_(other)

        jax_inplace_sub = tojax(inplace_sub_op)
        jax_tensor = base_tensor.clone()
        result = jax_inplace_sub(jax_tensor, other_tensor)

        npt.assert_allclose(
            np.asarray(result), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_mul_inplace(self, base_tensor, other_tensor):
        """Test in-place multiplication using tensor.mul_()"""
        # PyTorch version
        torch_tensor = base_tensor.clone()
        expected_result = base_tensor * other_tensor
        torch_tensor.mul_(other_tensor)

        # JAX version
        def inplace_mul_op(tensor, other):
            return tensor.mul_(other)

        jax_inplace_mul = tojax(inplace_mul_op)
        jax_tensor = base_tensor.clone()
        result = jax_inplace_mul(jax_tensor, other_tensor)

        npt.assert_allclose(
            np.asarray(result), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_div_inplace(self, base_tensor):
        """Test in-place division using tensor.div_()"""
        other = torch.tensor(
            [0.5, 2.0, 1.5, 0.8], dtype=torch.float64
        )  # Avoid division by zero

        # PyTorch version
        torch_tensor = base_tensor.clone()
        expected_result = base_tensor / other
        torch_tensor.div_(other)

        # JAX version
        def inplace_div_op(tensor, divisor):
            return tensor.div_(divisor)

        jax_inplace_div = tojax(inplace_div_op)
        jax_tensor = base_tensor.clone()
        result = jax_inplace_div(jax_tensor, other)

        npt.assert_allclose(
            np.asarray(result), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_pow_inplace(self, base_tensor):
        """Test in-place power operation using tensor.pow_()"""
        exponent = 2.0

        # PyTorch version
        torch_tensor = base_tensor.clone()
        expected_result = torch.pow(base_tensor, exponent)
        torch_tensor.pow_(exponent)

        # JAX version
        def inplace_pow_op(tensor, exp):
            return tensor.pow_(exp)

        jax_inplace_pow = tojax(inplace_pow_op)
        jax_tensor = base_tensor.clone()
        result = jax_inplace_pow(jax_tensor, exponent)

        npt.assert_allclose(
            np.asarray(result), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_neg_inplace(self, base_tensor):
        """Test in-place negation using tensor.neg_()"""
        # PyTorch version
        torch_tensor = base_tensor.clone()
        expected_result = -base_tensor
        torch_tensor.neg_()

        # JAX version
        def inplace_neg_op(tensor):
            return tensor.neg_()

        jax_inplace_neg = tojax(inplace_neg_op)
        jax_tensor = base_tensor.clone()
        result = jax_inplace_neg(jax_tensor)

        npt.assert_allclose(
            np.asarray(result), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )


class TestMatrixInPlaceOperations:
    """Test in-place operations on matrix tensors"""

    def test_matrix_add_inplace(self, matrix_tensor, matrix_other):
        """Test in-place matrix addition"""
        # PyTorch version
        torch_matrix = matrix_tensor.clone()
        expected_result = matrix_tensor + matrix_other
        torch_matrix.add_(matrix_other)

        # JAX version
        def matrix_inplace_add(tensor, other):
            return tensor.add_(other)

        jax_matrix_add = tojax(matrix_inplace_add)
        jax_matrix = matrix_tensor.clone()
        result = jax_matrix_add(jax_matrix, matrix_other)

        npt.assert_allclose(
            np.asarray(result), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_matrix_mul_inplace_scalar(self, matrix_tensor):
        """Test in-place matrix multiplication by scalar"""
        scalar = 2.5

        # PyTorch version
        torch_matrix = matrix_tensor.clone()
        expected_result = matrix_tensor * scalar
        torch_matrix.mul_(scalar)

        # JAX version
        def matrix_inplace_mul_scalar(tensor, s):
            return tensor.mul_(s)

        jax_matrix_mul = tojax(matrix_inplace_mul_scalar)
        jax_matrix = matrix_tensor.clone()
        result = jax_matrix_mul(jax_matrix, scalar)

        npt.assert_allclose(
            np.asarray(result), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )


class TestAdvancedInPlaceOperations:
    """Test advanced in-place operations and edge cases"""

    def test_chained_inplace_operations(self, base_tensor):
        """Test chaining multiple in-place operations"""
        other1 = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        other2 = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float64)

        # PyTorch version - chained operations
        torch_tensor = base_tensor.clone()
        torch_tensor.add_(other1).mul_(other2)
        expected_result = (base_tensor + other1) * other2

        # JAX version
        def chained_inplace_ops(tensor, add_val, mul_val):
            return tensor.add_(add_val).mul_(mul_val)

        jax_chained_ops = tojax(chained_inplace_ops)
        jax_tensor = base_tensor.clone()
        result = jax_chained_ops(jax_tensor, other1, other2)

        npt.assert_allclose(
            np.asarray(result), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_inplace_vs_outplace_memory_behavior(self, base_tensor, other_tensor):
        """Test that in-place operations modify original tensor while out-of-place don't"""
        # Create two copies
        tensor_for_inplace = base_tensor.clone()
        tensor_for_outplace = base_tensor.clone()

        # Out-of-place operation should not modify original
        outplace_result = tensor_for_outplace + other_tensor
        assert torch.equal(tensor_for_outplace, base_tensor), (
            "Out-of-place should not modify original"
        )

        # In-place operation should modify original
        tensor_for_inplace.add_(other_tensor)
        assert not torch.equal(tensor_for_inplace, base_tensor), (
            "In-place should modify original"
        )
        assert torch.equal(tensor_for_inplace, outplace_result), (
            "Results should be the same"
        )

        # Test with JAX
        def test_inplace_behavior(tensor, other):
            # This function tests the in-place semantics
            original_data = tensor.clone()
            tensor.add_(other)
            return tensor, original_data

        jax_inplace_test = tojax(test_inplace_behavior)
        jax_tensor = base_tensor.clone()
        jax_result, jax_original = jax_inplace_test(jax_tensor, other_tensor)

        # The result should match the expected value
        npt.assert_allclose(
            np.asarray(jax_result), outplace_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_inplace_with_broadcasting(self, matrix_tensor):
        """Test in-place operations with broadcasting"""
        # Create a vector that will broadcast with the matrix
        vector = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        # PyTorch version
        torch_matrix = matrix_tensor.clone()
        expected_result = matrix_tensor + vector
        torch_matrix.add_(vector)

        # JAX version
        def inplace_add_broadcast(matrix, vec):
            return matrix.add_(vec)

        jax_broadcast_add = tojax(inplace_add_broadcast)
        jax_matrix = matrix_tensor.clone()
        result = jax_broadcast_add(jax_matrix, vector)

        npt.assert_allclose(
            np.asarray(result), expected_result.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_inplace_mathematical_functions(self):
        """Test in-place mathematical functions like sqrt_, exp_, etc."""
        # Use positive values for sqrt
        positive_tensor = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)

        # Test sqrt_
        torch_tensor = positive_tensor.clone()
        expected_sqrt = torch.sqrt(positive_tensor)
        torch_tensor.sqrt_()

        def inplace_sqrt_op(tensor):
            return tensor.sqrt_()

        jax_inplace_sqrt = tojax(inplace_sqrt_op)
        jax_tensor = positive_tensor.clone()
        result = jax_inplace_sqrt(jax_tensor)

        npt.assert_allclose(
            np.asarray(result), expected_sqrt.numpy(), rtol=1e-5, atol=1e-5
        )

        # Test exp_ with smaller values to avoid overflow
        small_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)

        torch_exp_tensor = small_tensor.clone()
        expected_exp = torch.exp(small_tensor)
        torch_exp_tensor.exp_()

        def inplace_exp_op(tensor):
            return tensor.exp_()

        jax_inplace_exp = tojax(inplace_exp_op)
        jax_exp_tensor = small_tensor.clone()
        exp_result = jax_inplace_exp(jax_exp_tensor)

        npt.assert_allclose(
            np.asarray(exp_result), expected_exp.numpy(), rtol=1e-5, atol=1e-5
        )


class TestInPlaceOperationSemantics:
    """Test specific semantics of in-place operations"""

    def test_inplace_return_behavior(self, base_tensor, other_tensor):
        """Test that in-place operations return the modified tensor"""
        # PyTorch behavior: in-place operations return the tensor itself
        torch_tensor = base_tensor.clone()
        original_id = id(torch_tensor)
        result = torch_tensor.add_(other_tensor)

        assert id(result) == original_id, (
            "In-place operation should return same tensor object"
        )
        assert torch.equal(result, torch_tensor), (
            "Returned tensor should be the modified tensor"
        )

        # JAX version should have equivalent behavior
        def inplace_return_test(tensor, other):
            result = tensor.add_(other)
            return result, tensor  # Return both to compare

        jax_return_test = tojax(inplace_return_test)
        jax_tensor = base_tensor.clone()
        jax_result, jax_tensor_after = jax_return_test(jax_tensor, other_tensor)

        # Both should have the same values
        npt.assert_allclose(
            np.asarray(jax_result), np.asarray(jax_tensor_after), rtol=1e-10, atol=1e-10
        )

        # And should match the expected result
        expected = base_tensor + other_tensor
        npt.assert_allclose(
            np.asarray(jax_result), expected.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_inplace_with_aliasing(self):
        """Test in-place operations with tensor aliasing"""
        # Create a tensor and an alias (view) of it
        original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)

        # In PyTorch, modifying the original affects views of the same data
        original.add_(1.0)
        expected_alias = torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        # Test with JAX
        def inplace_aliasing_test(tensor):
            flat_view = tensor.view(-1)
            tensor.add_(1.0)
            return tensor, flat_view

        jax_aliasing_test = tojax(inplace_aliasing_test)
        jax_original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        jax_result_matrix, jax_result_flat = jax_aliasing_test(jax_original)

        # Check that both have been modified
        expected_matrix = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float64)

        # We expect these tests to fail since tojax does not handle views
        with pytest.raises(AssertionError):
            npt.assert_allclose(
                np.asarray(jax_result_matrix),
                expected_matrix.numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
            npt.assert_allclose(
                np.asarray(jax_result_flat),
                expected_alias.numpy(),
                rtol=1e-5,
                atol=1e-5,
            )


class TestReflectedOperators:
    """Test reflected (right-hand) operators with scalar on the left."""

    @pytest.fixture
    def tensor(self):
        return torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

    @pytest.fixture
    def int_tensor(self):
        return torch.tensor([1, 2, 3], dtype=torch.int64)

    @pytest.fixture
    def bool_tensor(self):
        return torch.tensor([True, False, True])

    def _check(self, func, tensor, expected):
        result = tojax(func)(tensor)
        npt.assert_allclose(np.asarray(result), expected.numpy(), rtol=1e-5, atol=1e-5)

    def test_radd(self, tensor):
        self._check(lambda t: 1.0 + t, tensor, 1.0 + tensor)

    def test_rsub(self, tensor):
        self._check(lambda t: 10.0 - t, tensor, 10.0 - tensor)

    def test_rmul(self, tensor):
        self._check(lambda t: 3.0 * t, tensor, 3.0 * tensor)

    def test_rtruediv(self, tensor):
        self._check(lambda t: 12.0 / t, tensor, 12.0 / tensor)

    def test_rfloordiv(self, tensor):
        self._check(lambda t: 10.0 // t, tensor, 10.0 // tensor)

    def test_rmod(self, tensor):
        self._check(lambda t: 10.0 % t, tensor, 10.0 % tensor)

    def test_rpow(self, tensor):
        self._check(lambda t: 2.0**t, tensor, 2.0**tensor)

    def test_rmatmul(self):
        mat = torch.randn(3, 4, dtype=torch.float64)
        left = torch.randn(2, 3, dtype=torch.float64)
        expected = left @ mat

        def func(m):
            return left @ m

        result = tojax(func)(mat)
        npt.assert_allclose(np.asarray(result), expected.numpy(), rtol=1e-5, atol=1e-5)

    def test_rand(self, bool_tensor):
        expected = torch.tensor([True, False, False]) & bool_tensor
        self._check(
            lambda t: torch.tensor([True, False, False]) & t, bool_tensor, expected
        )

    def test_ror(self, bool_tensor):
        expected = torch.tensor([False, True, False]) | bool_tensor
        self._check(
            lambda t: torch.tensor([False, True, False]) | t, bool_tensor, expected
        )

    def test_rxor(self, bool_tensor):
        expected = torch.tensor([True, True, False]) ^ bool_tensor
        self._check(
            lambda t: torch.tensor([True, True, False]) ^ t, bool_tensor, expected
        )


class TestForwardOperators:
    """Test forward (left-hand) dunder operators on TensorWrapper."""

    @pytest.fixture
    def a(self):
        return torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

    @pytest.fixture
    def b(self):
        return torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)

    @pytest.fixture
    def bool_a(self):
        return torch.tensor([True, False, True])

    @pytest.fixture
    def bool_b(self):
        return torch.tensor([False, True, True])

    def _check(self, func, *tensors, expected):
        result = tojax(func)(*tensors)
        npt.assert_allclose(np.asarray(result), expected.numpy(), rtol=1e-5, atol=1e-5)

    # Arithmetic
    def test_add(self, a, b):
        self._check(lambda a, b: a + b, a, b, expected=a + b)

    def test_sub(self, a, b):
        self._check(lambda a, b: a - b, a, b, expected=a - b)

    def test_mul(self, a, b):
        self._check(lambda a, b: a * b, a, b, expected=a * b)

    def test_truediv(self, a, b):
        self._check(lambda a, b: a / b, a, b, expected=a / b)

    def test_floordiv(self, a, b):
        self._check(lambda a, b: a // b, a, b, expected=a // b)

    def test_mod(self, a, b):
        self._check(lambda a, b: a % b, a, b, expected=a % b)

    def test_pow(self, a):
        self._check(lambda a: a**2.0, a, expected=a**2.0)

    def test_matmul(self):
        m1 = torch.randn(2, 3, dtype=torch.float64)
        m2 = torch.randn(3, 4, dtype=torch.float64)
        expected = m1 @ m2
        result = tojax(lambda a, b: a @ b)(m1, m2)
        npt.assert_allclose(np.asarray(result), expected.numpy(), rtol=1e-5, atol=1e-5)

    def test_neg(self, a):
        self._check(lambda a: -a, a, expected=-a)

    # Comparison
    def test_lt(self, a, b):
        self._check(lambda a, b: a < b, a, b, expected=a < b)

    def test_le(self, a, b):
        self._check(lambda a, b: a <= b, a, b, expected=a <= b)

    def test_eq(self, a, b):
        self._check(lambda a, b: a == b, a, b, expected=a == b)

    def test_ne(self, a, b):
        self._check(lambda a, b: a != b, a, b, expected=a != b)

    def test_gt(self, a, b):
        self._check(lambda a, b: a > b, a, b, expected=a > b)

    def test_ge(self, a, b):
        self._check(lambda a, b: a >= b, a, b, expected=a >= b)

    # Logical
    def test_and(self, bool_a, bool_b):
        self._check(lambda a, b: a & b, bool_a, bool_b, expected=bool_a & bool_b)

    def test_or(self, bool_a, bool_b):
        self._check(lambda a, b: a | b, bool_a, bool_b, expected=bool_a | bool_b)

    def test_xor(self, bool_a, bool_b):
        self._check(lambda a, b: a ^ b, bool_a, bool_b, expected=bool_a ^ bool_b)

    def test_invert(self, bool_a):
        self._check(lambda a: ~a, bool_a, expected=~bool_a)
