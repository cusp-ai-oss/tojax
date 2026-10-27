# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import numpy.testing as npt
import pytest
import torch

from tojax.tojax import tojax  # imports the e3nn patch, which sets up e3nn's import

pytest.importorskip("e3nn_jax")
e3nn_o3 = pytest.importorskip("e3nn.o3")

from tojax.patches.e3nn import _raw_normalization  # noqa: E402

torch.set_default_dtype(torch.float64)


@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
@pytest.mark.parametrize("normalize", [True, False])
def test_spherical_harmonics_match_e3nn(normalization, normalize):
    """The patched spherical harmonics against e3nn's own, for every normalization."""
    sh = e3nn_o3.SphericalHarmonics(
        e3nn_o3.Irreps.spherical_harmonics(3), normalize=normalize, normalization=normalization
    )
    vectors = torch.randn(10, 3, generator=torch.Generator().manual_seed(0))
    expected = sh(vectors).numpy()
    npt.assert_allclose(np.asarray(tojax(sh)(vectors)), expected, rtol=1e-6, atol=1e-10)


def _fake_raw(scale_with_degree: bool):
    """A stand-in for e3nn's internal function: unit-norm degrees, or norm sqrt(2l + 1)."""

    def raw(lmax, x, y, z):
        parts = [torch.ones_like(x)]
        for degree in range(1, lmax + 1):
            v = torch.zeros(x.shape + (2 * degree + 1,))
            v[..., 0] = math.sqrt(2 * degree + 1) if scale_with_degree else 1.0
            parts.append(v)
        return torch.cat([parts[0][..., None], *parts[1:]], dim=-1)

    return raw


def test_raw_normalization_detects_both_conventions():
    """e3nn 0.4 returns 'norm' from its internal function, e3nn 0.5 'component'."""
    assert _raw_normalization(_fake_raw(scale_with_degree=False)) == "norm"
    assert _raw_normalization(_fake_raw(scale_with_degree=True)) == "component"
