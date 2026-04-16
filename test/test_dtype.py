# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Tests for dtype conversion utilities."""

import jax.numpy as jnp
import pytest
import torch

from tojax.dtype import jax_dtype, torch_dtype


class TestDtypeConversion:
    """Test dtype conversion between PyTorch and JAX."""

    @pytest.mark.parametrize(
        "torch_dt,jax_dt",
        [
            (torch.float16, jnp.float16),
            (torch.float32, jnp.float32),
            (torch.float64, jnp.float64),
            (torch.int8, jnp.int8),
            (torch.int16, jnp.int16),
            (torch.int32, jnp.int32),
            (torch.int64, jnp.int64),
            (torch.uint8, jnp.uint8),
            (torch.bool, jnp.bool_),
            (torch.complex64, jnp.complex64),
            (torch.complex128, jnp.complex128),
            (torch.bfloat16, jnp.bfloat16),
        ],
    )
    def test_jax_dtype(self, torch_dt, jax_dt):
        """Test conversion from PyTorch to JAX dtype."""
        assert jax_dtype(torch_dt) == jax_dt

    @pytest.mark.parametrize(
        "torch_dt,jax_dt",
        [
            (torch.float16, jnp.float16),
            (torch.float32, jnp.float32),
            (torch.float64, jnp.float64),
            (torch.int8, jnp.int8),
            (torch.int16, jnp.int16),
            (torch.int32, jnp.int32),
            (torch.int64, jnp.int64),
            (torch.uint8, jnp.uint8),
            (torch.bool, jnp.bool_),
            (torch.complex64, jnp.complex64),
            (torch.complex128, jnp.complex128),
            (torch.bfloat16, jnp.bfloat16),
        ],
    )
    def test_torch_dtype(self, torch_dt, jax_dt):
        """Test conversion from JAX to PyTorch dtype."""
        assert torch_dtype(jax_dt) == torch_dt

    def test_torch_dtype_string(self):
        """Test torch_dtype with string inputs."""
        assert torch_dtype("float32") == torch.float32
        assert torch_dtype("int64") == torch.int64

    def test_bidirectional_conversion(self):
        """Test that conversions are bidirectional."""
        for torch_dt in [torch.float32, torch.int32, torch.bool]:
            jax_dt = jax_dtype(torch_dt)
            converted_back = torch_dtype(jax_dt)
            assert converted_back == torch_dt

    def test_unsupported_dtype_errors(self):
        """Test error handling for unsupported dtypes."""
        with pytest.raises(KeyError):
            jax_dtype("invalid_dtype")

        with pytest.raises(KeyError):
            torch_dtype("invalid_dtype")
