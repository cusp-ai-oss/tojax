"""
Tests for PyTorch FX Graph to JAX function conversion.

This module tests the make_jax_fn_from_torch_fx_graph function by creating
various PyTorch FX graphs and comparing their outputs with JAX equivalents.
"""

import numpy.testing as npt
import torch
import torch.fx
from pytest import fixture

from tojax.graph import make_jax_fn_from_torch_fx_graph

torch.set_default_dtype(torch.float64)


@fixture
def sample_input():
    """Standard input tensor for testing"""
    return torch.randn(2, 3, dtype=torch.float64)


@fixture
def sample_matrix_input():
    """Matrix input for testing matrix operations"""
    return torch.randn(3, 3, dtype=torch.float64)


@fixture
def sample_batch_matrices():
    """Batch of matrices for testing batch operations"""
    return torch.randn(2, 3, 3, dtype=torch.float64)


class TestFunctionCalls:
    """Test graphs that call PyTorch functions"""

    def test_simple_addition_function(self, sample_input):
        """Test FX graph with torch.add function call"""

        def torch_fn(x, y):
            return torch.add(x, y)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Test inputs
        x = sample_input
        y = torch.randn_like(x)

        # Compare outputs
        torch_result = torch_fn(x, y)
        jax_result = jax_fn(x.numpy(), y.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_matmul_function(self, sample_matrix_input):
        """Test FX graph with torch.matmul function call"""

        def torch_fn(a, b):
            return torch.matmul(a, b)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Test inputs
        a = sample_matrix_input
        b = torch.randn(3, 4, dtype=torch.float64)

        # Compare outputs
        torch_result = torch_fn(a, b)
        jax_result = jax_fn(a.numpy(), b.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_sin_function(self, sample_input):
        """Test FX graph with torch.sin function call"""

        def torch_fn(x):
            return torch.sin(x)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(sample_input)
        jax_result = jax_fn(sample_input.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_multiple_function_calls(self, sample_input):
        """Test FX graph with multiple function calls"""

        def torch_fn(x):
            y = torch.sin(x)
            z = torch.cos(y)
            return torch.add(y, z)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(sample_input)
        jax_result = jax_fn(sample_input.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_bmm_function(self, sample_batch_matrices):
        """Test FX graph with torch.bmm (batch matrix multiply) function"""

        def torch_fn(a, b):
            return torch.bmm(a, b)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Test inputs
        a = sample_batch_matrices
        b = torch.randn(2, 3, 4, dtype=torch.float64)

        # Compare outputs
        torch_result = torch_fn(a, b)
        jax_result = jax_fn(a.numpy(), b.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)


class TestMethodCalls:
    """Test graphs that call tensor methods"""

    def test_reshape_method(self, sample_input):
        """Test FX graph with tensor.reshape method call"""

        def torch_fn(x):
            return x.reshape(-1)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(sample_input)
        jax_result = jax_fn(sample_input.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_flatten_method(self, sample_input):
        """Test FX graph with tensor.flatten method call"""

        def torch_fn(x):
            return x.flatten()

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(sample_input)
        jax_result = jax_fn(sample_input.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)


class TestComplexGraphs:
    """Test complex graphs with mixed function and method calls"""

    def test_mixed_operations(self, sample_input):
        """Test FX graph with both function and method calls"""

        def torch_fn(x):
            y = torch.sin(x)  # function call
            z = y.flatten()  # method call
            w = torch.cos(z)  # function call
            return torch.sum(w)  # function call

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(sample_input)
        jax_result = jax_fn(sample_input.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_multiple_inputs_simple(self):
        """Test FX graph with multiple inputs and single output"""

        def torch_fn(x, y):
            z = torch.add(x, y)
            w = torch.sin(z)
            return torch.sum(w)

        # Create test inputs
        x = torch.randn(3, 4, dtype=torch.float64)
        y = torch.randn(3, 4, dtype=torch.float64)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(x, y)
        jax_result = jax_fn(x.numpy(), y.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_nested_operations(self, sample_input):
        """Test FX graph with nested operations"""

        def torch_fn(x):
            y = torch.sin(torch.cos(x))
            z = y.reshape(-1)
            w = torch.sum(z)
            return torch.exp(w)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(sample_input)
        jax_result = jax_fn(sample_input.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)


class TestSimpleFunctionGraphs:
    """Test simple function-based graphs without complex modules"""

    def test_elementwise_operations(self, sample_input):
        """Test FX graph with element-wise operations"""

        def torch_fn(x):
            y = torch.mul(x, 2.0)
            z = torch.add(y, 1.0)
            return torch.div(z, 3.0)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(sample_input)
        jax_result = jax_fn(sample_input.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_mathematical_functions(self, sample_input):
        """Test FX graph with mathematical functions"""

        def torch_fn(x):
            y = torch.abs(x)
            z = torch.sqrt(y + 1.0)
            return torch.log(z)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(sample_input)
        jax_result = jax_fn(sample_input.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)


class TestErrorHandling:
    """Test error handling in graph conversion"""

    def test_basic_operation_success(self):
        """Test that basic operations work without error"""

        def torch_fn(x):
            return x + 1

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Should not raise an error for supported operations
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)
        x = torch.randn(2, 3, dtype=torch.float64)
        result = jax_fn(x.numpy())
        assert result is not None
        assert len(result) == 1  # Single output wrapped in tuple

    def test_graph_execution_basic(self):
        """Test basic graph execution"""

        def torch_fn(x):
            return torch.sum(x)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Test with proper input
        x = torch.randn(2, 3, dtype=torch.float64)
        result = jax_fn(x.numpy())
        assert result is not None
        assert len(result) == 1  # Single output


class TestInPlaceOperationRemoval:
    """Test that in-place operations are properly removed from graphs"""

    def test_relu_operation(self):
        """Test that ReLU operations work (may have inplace args)"""

        def torch_fn(x):
            return torch.relu(x)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function - should handle any inplace kwargs
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Test execution
        x = torch.randn(2, 3, dtype=torch.float64)
        torch_result = torch_fn(x)
        jax_result = jax_fn(x.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)


class TestBasicConsistency:
    """Test basic consistency between graph conversion and PyTorch"""

    def test_simple_arithmetic(self, sample_input):
        """Test simple arithmetic operations"""

        def torch_fn(x):
            return x * 2 + 1

        # PyTorch reference
        torch_result = torch_fn(sample_input)

        # FX graph conversion
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)
        graph_fn = make_jax_fn_from_torch_fx_graph(graph)
        graph_result = graph_fn(sample_input.numpy())

        # Results should match
        npt.assert_allclose(graph_result[0], torch_result.numpy(), rtol=1e-5)

    def test_trigonometric_functions(self):
        """Test trigonometric function conversion"""

        def torch_fn(x):
            return torch.sin(x) + torch.cos(x)

        # Test with known input for reproducibility
        x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)

        # PyTorch reference
        torch_result = torch_fn(x)

        # FX graph conversion
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)
        graph_fn = make_jax_fn_from_torch_fx_graph(graph)
        graph_result = graph_fn(x.numpy())

        # Results should match
        npt.assert_allclose(graph_result[0], torch_result.numpy(), rtol=1e-5)


class TestAdvancedFunctions:
    """Test more advanced PyTorch functions"""

    def test_concatenation(self):
        """Test torch.cat function"""

        def torch_fn(x, y):
            return torch.cat([x, y], dim=0)

        # Test inputs
        x = torch.randn(2, 3, dtype=torch.float64)
        y = torch.randn(2, 3, dtype=torch.float64)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(x, y)
        jax_result = jax_fn(x.numpy(), y.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_where_function(self):
        """Test torch.where function"""

        def torch_fn(x):
            return torch.where(x > 0, x, torch.zeros_like(x))

        # Test input
        x = torch.randn(2, 3, dtype=torch.float64)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(x)
        jax_result = jax_fn(x.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_stacking_function(self):
        """Test torch.stack function"""

        def torch_fn(x, y):
            return torch.stack([x, y], dim=0)

        # Test inputs
        x = torch.randn(2, 3, dtype=torch.float64)
        y = torch.randn(2, 3, dtype=torch.float64)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(x, y)
        jax_result = jax_fn(x.numpy(), y.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_power_function(self):
        """Test torch.pow function"""

        def torch_fn(x):
            return torch.pow(x, 2.0)

        # Test input (use positive values to avoid complex numbers)
        x = torch.abs(torch.randn(2, 3, dtype=torch.float64)) + 0.1

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(x)
        jax_result = jax_fn(x.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)


class TestGraphEdgeCases:
    """Test edge cases and corner scenarios"""

    def test_single_constant_output(self):
        """Test graph that outputs a constant"""

        def torch_fn(x):
            # Use input but return constant
            _ = x + 1
            return torch.tensor(42.0, dtype=torch.float64)

        # Test input
        x = torch.randn(2, 3, dtype=torch.float64)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph, tracer.root)

        # Compare outputs
        torch_result = torch_fn(x)
        jax_result = jax_fn(x.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_scalar_operations(self):
        """Test operations that result in scalars"""

        def torch_fn(x):
            y = torch.sum(x)
            z = torch.sqrt(y)
            return z * 2.0

        # Test input
        x = torch.abs(torch.randn(2, 3, dtype=torch.float64)) + 0.1

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(x)
        jax_result = jax_fn(x.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_broadcasting_operations(self):
        """Test operations with broadcasting"""

        def torch_fn(x):
            # x is (2, 3), we add a (3,) tensor
            y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
            return x + y

        # Test input
        x = torch.randn(2, 3, dtype=torch.float64)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph, tracer.root)

        # Compare outputs
        torch_result = torch_fn(x)
        jax_result = jax_fn(x.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)


class TestComplexMathematicalOperations:
    """Test complex mathematical operations through graphs"""

    def test_norm_calculation(self):
        """Test L2 norm calculation"""

        def torch_fn(x):
            # Manual L2 norm: sqrt(sum(x^2))
            squared = torch.pow(x, 2)
            summed = torch.sum(squared)
            return torch.sqrt(summed)

        # Test input
        x = torch.randn(3, 4, dtype=torch.float64)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(x)
        jax_result = jax_fn(x.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_softmax_like_operation(self):
        """Test softmax-like operation"""

        def torch_fn(x):
            # Manual softmax: exp(x) / sum(exp(x))
            exp_x = torch.exp(x)
            sum_exp = torch.sum(exp_x, dim=-1, keepdim=True)
            return exp_x / sum_exp

        # Test input (smaller values to avoid overflow)
        x = torch.randn(2, 3, dtype=torch.float64) * 0.1

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(x)
        jax_result = jax_fn(x.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)

    def test_matrix_determinant_like(self):
        """Test operations similar to matrix determinant calculation"""

        def torch_fn(x):
            # For 2x2 matrix: det = a*d - b*c
            a, b = x[:, 0], x[:, 1]
            c, d = x[:, 2], x[:, 3]
            return a * d - b * c

        # Test input (reshape 2x4 to treat as batch of 2x2 matrices)
        x = torch.randn(2, 4, dtype=torch.float64)

        # Create FX graph
        tracer = torch.fx.Tracer()
        graph = tracer.trace(torch_fn)

        # Convert to JAX function
        jax_fn = make_jax_fn_from_torch_fx_graph(graph)

        # Compare outputs
        torch_result = torch_fn(x)
        jax_result = jax_fn(x.numpy())

        npt.assert_allclose(jax_result[0], torch_result.numpy(), rtol=1e-5)
