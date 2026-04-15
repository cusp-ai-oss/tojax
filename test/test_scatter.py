import numpy as np
import numpy.testing as npt
import pytest
import torch

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
