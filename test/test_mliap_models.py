import functools

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
import requests
import torch
from jax import export

from tojax import tojax

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="session")
def small_mace(tmp_path_factory):
    url = "https://github.com/ACEsuit/mace-off/raw/refs/heads/main/mace_off23/MACE-OFF23_small.model"
    model = tmp_path_factory.getbasetemp() / "mace.model"
    r = requests.get(url)
    model.write_bytes(r.content)
    return model


@pytest.fixture(scope="session")
def model(small_mace):
    return torch.load(small_mace, weights_only=False)


@pytest.fixture
def single_input(model):
    N = 5
    Zs = model.atomic_numbers
    torch.manual_seed(42)
    node_attrs = torch.nn.functional.one_hot(
        torch.randint(0, len(Zs), (N,)), len(Zs)
    ).to(torch.float64)
    positions = torch.randn(N, 3, dtype=torch.float64)
    edge_index = torch.stack(torch.where(torch.ones((N, N)) - torch.eye(N)), dim=0)
    batch = torch.zeros(N, dtype=torch.int64)
    shifts = torch.randn(edge_index.shape[1], 3, dtype=torch.float64)
    cell = torch.randn(1, 3, 3, dtype=torch.float64)
    return {
        "node_attrs": node_attrs,
        "positions": positions.requires_grad_(True),
        "edge_index": edge_index,
        "batch": batch,
        "cell": cell,
        "ptr": torch.tensor([0, N], dtype=torch.int64),
        "shifts": shifts,
    }


@pytest.fixture
def batched_input(single_input):
    batch = torch.repeat_interleave(torch.arange(3, dtype=torch.int64), repeats=5)
    ptr = torch.tensor([0, 5, 10, 15], dtype=torch.int64)
    return {
        "node_attrs": torch.cat([single_input["node_attrs"]] * 3, dim=0),
        "positions": torch.cat([single_input["positions"]] * 3, dim=0).requires_grad_(
            True
        ),
        "edge_index": torch.cat([single_input["edge_index"]] * 3, dim=1),
        "batch": batch,
        "cell": torch.cat([single_input["cell"]] * 3, dim=0),
        "ptr": ptr,
        "shifts": torch.cat([single_input["shifts"]] * 3, dim=0),
    }


@pytest.fixture(params=["single_input", "batched_input"])
def input_data(request):
    return request.getfixturevalue(request.param)


def test_mace(model, input_data):
    jax_model = tojax(model)
    jax_input = tojax(input_data)

    jax_out = jax_model(jax_input, compute_force=False)
    torch_out = model(input_data, compute_force=False)

    assert_fn = functools.partial(npt.assert_allclose, rtol=1e-4, atol=1e-4)
    jax.tree.map(assert_fn, jax_out, tojax(torch_out))

    def torch_forces(positions):
        x = {**input_data, "positions": positions}
        return -model(x, compute_force=True)["forces"]

    @jax.jit
    @jax.grad
    def jax_forces(positions):
        data = {**jax_input, "positions": positions}
        return jax_model(data, compute_force=False)["energy"].sum()

    jax.tree.map(
        assert_fn,
        jax_forces(jax_input["positions"]),
        tojax(torch_forces(input_data["positions"])),
    )


def test_mace_export(model, input_data):
    jax_model = tojax(model)
    jax_input = tojax(input_data)
    export.export(jax.jit(jax_model))(jax_input)


@pytest.fixture
def symbolic_input(model):
    # Get atomic numbers from the model
    num_atom_types = len(model.atomic_numbers)  # type: ignore
    # variadic dimensions
    num_atoms, num_edges, num_systems = export.symbolic_shape(
        "num_atoms, num_edges, num_systems"
    )
    # Create one-hot encoded node attributes based on atomic numbers
    node_attrs = jax.ShapeDtypeStruct((num_atoms, num_atom_types), jnp.float64)
    # Random atomic positions
    positions = jax.ShapeDtypeStruct((num_atoms, 3), jnp.float64)
    # Create fully connected edge index (excluding self-loops)
    edge_index = jax.ShapeDtypeStruct((2, num_edges), jnp.int64)
    # Batch indices (single batch)
    batch = jax.ShapeDtypeStruct((num_atoms,), jnp.int64)
    # Random edge shifts
    shifts = jax.ShapeDtypeStruct((num_edges, 3), jnp.float64)
    # Pointer array for batch boundaries
    ptr = jax.ShapeDtypeStruct((num_systems + 1,), jnp.int64)
    # Unitcell
    cell = jax.ShapeDtypeStruct((num_systems, 3, 3), jnp.float64)
    return {
        "node_attrs": node_attrs,
        "positions": positions,
        "edge_index": edge_index,
        "batch": batch,
        "cell": cell,
        "ptr": ptr,
        "shifts": shifts,
    }


def test_polyshape_export(model, symbolic_input):
    export.export(jax.jit(tojax(model)))(symbolic_input)


@pytest.fixture
def polyshape_jax_model(model, symbolic_input):
    return export.export(jax.jit(functools.partial(tojax(model), compute_force=False)))(
        symbolic_input
    )


def test_polyshape_inference(model, polyshape_jax_model, input_data):
    torch_out = model(input_data, compute_force=False)
    jax_out = polyshape_jax_model.call(tojax(input_data))

    assert_fn = functools.partial(npt.assert_allclose, rtol=1e-4, atol=1e-4)
    jax.tree.map(assert_fn, jax_out, tojax(torch_out))
