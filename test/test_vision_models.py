# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
import jax
import numpy.testing as npt
import torch
from pytest import fixture
from torchvision.models import resnet18, vit_b_16

from tojax.tojax import tojax

torch.set_default_dtype(torch.float64)


@fixture
def img_input():
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224)


@fixture
def model_resnet18():
    return resnet18(weights=None)


@fixture
def model_vit_b_16():
    return vit_b_16(weights=None)


@fixture(params=["model_resnet18", "model_vit_b_16"])
def model(request):
    return request.getfixturevalue(request.param)


def test_models(img_input, model):
    model.eval()
    torch_out = model(img_input).detach().numpy()

    # Test forward pass
    jax_model = tojax(model)
    jax_inp = tojax(img_input)
    jax_out = jax_model(jax_inp)
    npt.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-4)

    # Test gradients
    jax_grad = jax.grad(lambda x: jax_model(x).sum())(jax_inp)
    img_input.requires_grad = True
    model(img_input).sum().backward()
    npt.assert_allclose(img_input.grad.detach().numpy(), jax_grad, rtol=1e-4, atol=1e-4)
