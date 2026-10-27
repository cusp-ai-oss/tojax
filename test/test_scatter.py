# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import torch
from jax import export

from tojax.scatter import ScatterContext, ScatterMode, get_scatter_mode
from tojax.tojax import tojax

torch.set_default_dtype(torch.float64)


class TestScatterMode:
    """Tests for ScatterMode enum."""

    def test_values(self):
        assert ScatterMode.CLIP == "clip"
        assert ScatterMode.FILL_OR_DROP == "drop"
        assert ScatterMode.PROMISE_IN_BOUNDS == "promise_in_bounds"

    def test_is_str_enum(self):
        assert isinstance(ScatterMode.CLIP, str)


class TestScatterContext:
    """Tests for ScatterContext manager."""

    def test_context_sets_and_restores(self):
        assert ScatterContext._INSTANCE is None
        with ScatterContext(ScatterMode.CLIP) as ctx:
            assert ScatterContext._INSTANCE is ctx
            assert ctx.value is ScatterMode.CLIP
        assert ScatterContext._INSTANCE is None

    def test_nested_contexts(self):
        with ScatterContext(ScatterMode.CLIP):
            assert get_scatter_mode() == "clip"
            with ScatterContext(ScatterMode.FILL_OR_DROP):
                assert get_scatter_mode() == "drop"
            assert get_scatter_mode() == "clip"

    def test_current_raises_without_context(self):
        with pytest.raises(AssertionError, match="ScatterMode is not active"):
            ScatterContext.current()

    def test_get_scatter_mode_returns_string(self):
        with ScatterContext(ScatterMode.PROMISE_IN_BOUNDS):
            assert get_scatter_mode() == "promise_in_bounds"

    def test_restores_on_exception(self):
        with pytest.raises(RuntimeError):
            with ScatterContext(ScatterMode.CLIP):
                raise RuntimeError("test")
        assert ScatterContext._INSTANCE is None


class TestScatterModeIntegration:
    """Test that scatter_mode propagates through tojax."""

    def test_scatter_add_default_mode(self):
        def fn(x):
            index = torch.tensor([[0, 1, 2]])
            src = torch.ones(1, 3, dtype=torch.float64)
            return torch.scatter_add(x, 0, index, src)

        x = torch.zeros(3, 3, dtype=torch.float64)
        expected = fn(x).numpy()

        jax_fn = tojax(fn)
        result = np.asarray(jax_fn(x))
        npt.assert_allclose(result, expected)

    def test_scatter_add_clip_mode(self):
        def fn(x):
            index = torch.tensor([[0, 1, 2]])
            src = torch.ones(1, 3, dtype=torch.float64)
            return torch.scatter_add(x, 0, index, src)

        x = torch.zeros(3, 3, dtype=torch.float64)
        expected = fn(x).numpy()

        jax_fn = tojax(fn, scatter_mode=ScatterMode.CLIP)
        result = np.asarray(jax_fn(x))
        npt.assert_allclose(result, expected)

    def test_index_add_with_scatter_mode(self):
        def fn(x):
            index = torch.tensor([0, 2])
            source = torch.ones(2, 3, dtype=torch.float64)
            return torch.index_add(x, 0, index, source)

        x = torch.zeros(3, 3, dtype=torch.float64)
        expected = fn(x).numpy()

        jax_fn = tojax(fn, scatter_mode=ScatterMode.CLIP)
        result = np.asarray(jax_fn(x))
        npt.assert_allclose(result, expected)

    def test_setitem_with_scatter_mode(self):
        def fn(x):
            result = x.clone()
            result[0] = 99.0
            return result

        x = torch.ones(3, dtype=torch.float64)
        expected = fn(x).numpy()

        jax_fn = tojax(fn, scatter_mode=ScatterMode.CLIP)
        result = np.asarray(jax_fn(x))
        npt.assert_allclose(result, expected)

    def test_scatter_reduce_sum_with_mode(self):
        def fn(input, index, src):
            return torch.scatter_reduce(input, 0, index, src, reduce="sum")

        input = torch.zeros(3, 3, dtype=torch.float64)
        index = torch.tensor([[0, 1, 2], [0, 1, 2]])
        src = torch.ones(2, 3, dtype=torch.float64)
        expected = fn(input, index, src).numpy()

        for mode in ScatterMode:
            jax_fn = tojax(fn, scatter_mode=mode)
            result = np.asarray(jax_fn(input, index, src))
            npt.assert_allclose(result, expected)

    def test_scatter_inplace_with_mode(self):
        def fn(x):
            result = x.clone()
            index = torch.tensor([[0, 1, 2]])
            src = torch.ones(1, 3, dtype=torch.float64) * 5
            result.scatter_(0, index, src)
            return result

        x = torch.zeros(3, 3, dtype=torch.float64)
        expected = fn(x).numpy()

        jax_fn = tojax(fn, scatter_mode=ScatterMode.CLIP)
        result = np.asarray(jax_fn(x))
        npt.assert_allclose(result, expected)


class TestScatterAlongDim:
    """Scatters against torch on 3-D tensors, every dim, with index blocks smaller than
    src and input on the other axes."""

    @staticmethod
    def _inputs(dim, seed=0):
        g = torch.Generator().manual_seed(seed)
        input = torch.randn(4, 5, 6, generator=g)
        src = torch.randn(5, 6, 7, generator=g)
        shape = [3, 4, 5]
        index = torch.randint(0, input.shape[dim], shape, generator=g)
        return input, index, src

    @pytest.mark.parametrize("dim", [0, 1, 2, -1])
    def test_scatter_add(self, dim):
        input, index, src = self._inputs(dim)
        expected = torch.scatter_add(input, dim, index, src).numpy()
        result = tojax(lambda x, i, s: torch.scatter_add(x, dim, i, s))(
            input, index, src
        )
        npt.assert_allclose(np.asarray(result), expected, rtol=1e-12)

    @pytest.mark.parametrize("reduce", ["sum", "prod", "amax", "amin", "mean"])
    @pytest.mark.parametrize("include_self", [True, False])
    @pytest.mark.parametrize("dim", [0, 2])
    def test_scatter_reduce(self, reduce, include_self, dim):
        input, index, src = self._inputs(dim)

        def fn(x, i, s):
            return torch.scatter_reduce(x, dim, i, s, reduce, include_self=include_self)

        expected = fn(input, index, src).numpy()
        npt.assert_allclose(
            np.asarray(tojax(fn)(input, index, src)), expected, rtol=1e-12
        )

    @pytest.mark.parametrize("reduce", [None, "add", "multiply"])
    def test_tensor_scatter(self, reduce):
        input, index, src = self._inputs(1)

        def fn(x, i, s):
            if reduce is None:
                return x.scatter(1, i, s)
            return x.scatter(1, i, s, reduce=reduce)

        expected = fn(input, index, src).numpy()
        npt.assert_allclose(
            np.asarray(tojax(fn)(input, index, src)), expected, rtol=1e-12
        )

    def test_tensor_scatter_scalar_value(self):
        input, index, _ = self._inputs(0)

        def fn(x, i):
            return x.scatter(0, i, 2.5)

        expected = fn(input, index).numpy()
        npt.assert_allclose(np.asarray(tojax(fn)(input, index)), expected)

    def test_broadcast_index_builds_no_flat_coordinates(self):
        """MACE's scatter_sum: a 1-D receiver index expanded over the channels. The
        translation must not build index.size-long coordinate arrays (they overflow int32
        shape arithmetic in a symbolic export once edges x channels reaches 2**31)."""

        def fn(nodes, messages, receivers):
            index = receivers.unsqueeze(-1).expand_as(messages)
            return torch.scatter_add(nodes, 0, index, messages)

        nodes, messages = torch.zeros(3, 4), torch.randn(7, 4)
        receivers = torch.randint(0, 3, (7,))
        expected = fn(nodes, messages, receivers).numpy()
        jax_fn = tojax(fn)
        npt.assert_allclose(np.asarray(jax_fn(nodes, messages, receivers)), expected)

        jaxpr = jax.make_jaxpr(jax_fn)(
            jnp.zeros((3, 4)), jnp.zeros((7, 4)), jnp.zeros(7, jnp.int32)
        )
        flat = [
            v.aval.shape
            for eq in jaxpr.eqns
            for v in eq.outvars
            if getattr(v.aval, "shape", None) == (7 * 4,)
        ]
        assert not flat, f"flattened coordinate arrays: {flat}"

    def test_symbolic_export(self):
        def fn(nodes, messages, receivers):
            index = receivers.unsqueeze(-1).expand_as(messages)
            return torch.scatter_add(nodes, 0, index, messages)

        n, e, c = export.symbolic_shape("n, e, c")
        exported = export.export(jax.jit(tojax(fn)))(
            jax.ShapeDtypeStruct((n, c), jnp.float32),
            jax.ShapeDtypeStruct((e, c), jnp.float32),
            jax.ShapeDtypeStruct((e,), jnp.int32),
        )
        assert exported.out_avals[0].shape == (n, c)
        nodes, messages = np.zeros((3, 4), np.float32), np.ones((7, 4), np.float32)
        receivers = np.array([0, 1, 2, 0, 1, 2, 0], np.int32)
        result = exported.call(nodes, messages, receivers)
        npt.assert_allclose(np.asarray(result), [[3] * 4, [2] * 4, [2] * 4])

    def test_compiles_past_int32_element_count(self):
        """A MACE-sized scatter past 2**31 elements (264192 x 9088). Shape refinement
        runs at compile time, so no arrays are needed."""

        def fn(nodes, messages, receivers):
            index = receivers.unsqueeze(-1).expand_as(messages)
            return torch.scatter_add(nodes, 0, index, messages)

        n, e, c = export.symbolic_shape("n, e, c")
        exported = export.export(jax.jit(tojax(fn)))(
            jax.ShapeDtypeStruct((n, c), jnp.float32),
            jax.ShapeDtypeStruct((e, c), jnp.float32),
            jax.ShapeDtypeStruct((e,), jnp.int32),
        )
        edges, channels, atoms = 264192, 9088, 2064
        jax.jit(exported.call).lower(
            jax.ShapeDtypeStruct((atoms, channels), jnp.float32),
            jax.ShapeDtypeStruct((edges, channels), jnp.float32),
            jax.ShapeDtypeStruct((edges,), jnp.int32),
        ).compile()
