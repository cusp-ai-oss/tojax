# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
"""Unified export utilities for molecular dynamics models.

Provides a model-independent interface for exporting PyTorch models to JAX,
with parameter separation, symbolic shapes, and zip packaging.
"""

import argparse
import json
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypedDict

import jax
import jax.core
import jax.numpy as jnp
import msgpack
import numpy as np
import torch
from jax import Array, export

from tojax import RNGMode, tojax
from tojax.scatter import ScatterMode


class AtomGraphInput(TypedDict):
    """Universal input format for molecular dynamics models.

    All models receive this format; each adapter maps it to model-specific
    fields internally. ``natoms`` and ``nedges`` are derived from ``batch``
    and ``edge_index`` at runtime.

    Array fields are ``torch.Tensor`` before jaxification and ``jax.Array``
    after. The type is ``Any`` to support both contexts.
    """

    pos: Any  # (N, 3) float32
    atomic_numbers: Any  # (N,) int64
    cell: Any  # (B, 3, 3) float32
    pbc: Any  # (B, 3) bool
    edge_index: Any  # (2, E) int64
    cell_offsets: Any  # (E, 3) float32
    batch: Any  # (N,) int64
    charge: Any  # (B,) int64
    spin: Any  # (B,) int64


class EnergyFn(Protocol):
    """Protocol for model energy functions.

    Takes an ``AtomGraphInput`` dict and returns a scalar or per-system
    energy tensor.
    """

    def __call__(self, data: AtomGraphInput) -> torch.Tensor: ...


class PredictFn(Protocol):
    """Protocol for model predict functions returning energy, forces, stress.

    Returns a dict with ``"energy"`` (required), and optionally
    ``"forces"`` and ``"stress"``.
    """

    def __call__(self, data: AtomGraphInput) -> dict[str, torch.Tensor]: ...


# ---------------------------------------------------------------------------
# Symbolic shapes & abstract input
# ---------------------------------------------------------------------------
def make_abstract_input(
    n_atoms: int,
    n_batches: int,
    n_edges: int,
    *,
    symbolic: str = "",
    constraints: tuple[str, ...] = (),
) -> dict:
    """Create abstract input with symbolic shapes for JAX export.

    Args:
        n_atoms: Concrete number of atoms.
        n_batches: Concrete number of systems/batches.
        n_edges: Concrete number of edges.
        symbolic: Letters indicating which dims are symbolic:
            N=atoms, S=systems, E=edges (e.g. "NSE" for all).
        constraints: Export constraints, e.g. ("E >= 64",).
    """
    sym = symbolic.upper()
    sym_dims = []
    for letter, fixed in [("N", n_atoms), ("S", n_batches), ("E", n_edges)]:
        sym_dims.append(letter if letter in sym else str(fixed))
    shapes = export.symbolic_shape(", ".join(sym_dims), constraints=constraints)
    N, B, E = shapes
    return dict(
        pos=jax.ShapeDtypeStruct((N, 3), jnp.float32),
        atomic_numbers=jax.ShapeDtypeStruct((N,), jnp.int64),
        cell=jax.ShapeDtypeStruct((B, 3, 3), jnp.float32),
        pbc=jax.ShapeDtypeStruct((B, 3), jnp.bool_),
        edge_index=jax.ShapeDtypeStruct((2, E), jnp.int64),
        cell_offsets=jax.ShapeDtypeStruct((E, 3), jnp.float32),
        batch=jax.ShapeDtypeStruct((N,), jnp.int64),
        charge=jax.ShapeDtypeStruct((B,), jnp.int64),
        spin=jax.ShapeDtypeStruct((B,), jnp.int64),
    )


def make_dummy_data(n_atoms: int, n_batches: int, n_edges: int, dtype:torch.dtype) -> AtomGraphInput:
    """Create random dummy input data matching ``AtomGraphInput``."""
    natoms_per = [n_atoms // n_batches] * n_batches
    for i in range(n_atoms % n_batches):
        natoms_per[i] += 1
    natoms_t = torch.tensor(natoms_per)
    return AtomGraphInput(
        pos=torch.randn(n_atoms, 3, dtype=dtype),
        atomic_numbers=torch.randint(1, 10, (n_atoms,)),
        cell=torch.eye(3, dtype=dtype).unsqueeze(0).repeat(n_batches, 1, 1),
        pbc=torch.ones(n_batches, 3, dtype=torch.bool),
        edge_index=torch.randint(0, n_atoms, (2, n_edges)),
        cell_offsets=torch.randn(n_edges, 3, dtype=dtype),
        batch=torch.arange(n_batches).repeat_interleave(natoms_t),
        charge=torch.zeros(n_batches, dtype=torch.long),
        spin=torch.zeros(n_batches, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_model(
    energy_fn: EnergyFn,
    abstract_input: dict,
    platforms: list[str],
    *,
    rng_mode: RNGMode = RNGMode.FIXED,
    scatter_mode: ScatterMode = ScatterMode.FILL_OR_DROP,
) -> tuple[list[Array], export.Exported]:
    """Export a PyTorch energy function, separating parameters from graph.

    Args:
        energy_fn: Plain PyTorch callable matching ``EnergyFn`` protocol.
            Will be wrapped with ``tojax()`` internally.
        abstract_input: Dict of ``ShapeDtypeStruct`` (possibly symbolic).
        platforms: Target JAX platforms (e.g. ``["cuda"]``).
        rng_mode: RNG mode for tojax.
        scatter_mode: Scatter mode for tojax.

    Returns:
        ``(parameters, exported)`` where parameters are the extracted constants.
    """
    jax_fn = jax.jit(
        tojax(energy_fn, rng_mode=rng_mode, scatter_mode=scatter_mode),
        inline=True,
    )
    closed = jax.make_jaxpr(jax_fn)(abstract_input)
    out_treedef = jax.tree.structure(jax.eval_shape(jax_fn, abstract_input))

    @jax.jit
    def wrapped_fn(consts, tree):
        flat_out = jax.core.eval_jaxpr(closed.jaxpr, consts, *jax.tree.leaves(tree))
        return out_treedef.unflatten(flat_out)

    abstract_consts = [jax.ShapeDtypeStruct(c.shape, c.dtype) for c in closed.consts]
    return closed.consts, export.export(wrapped_fn, platforms=platforms)(
        abstract_consts, abstract_input
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def serialize_params(params: list[Array]) -> bytes:
    """Serialize JAX arrays to msgpack bytes."""
    entries = []
    for arr in params:
        np_arr = np.asarray(arr)
        entries.append(
            {
                "__np__": True,
                "shape": list(np_arr.shape),
                "dtype": np_arr.dtype.str,
                "data": np_arr.tobytes(),
            }
        )
    return msgpack.packb(entries)  # type: ignore[return-value]


def deserialize_params(raw: bytes) -> list[Array]:
    """Deserialize msgpack bytes to JAX arrays."""
    entries = msgpack.unpackb(raw, raw=False)
    return [
        jnp.array(
            np.frombuffer(e["data"], dtype=np.dtype(e["dtype"]))
            .reshape(e["shape"])
            .copy()
        )
        for e in entries
    ]


# ---------------------------------------------------------------------------
# Zip I/O
# ---------------------------------------------------------------------------
def save_exported_model(
    exported: export.Exported,
    parameters: list[Array],
    abstract_input: dict,
    path: Path,
    metadata: dict,
) -> None:
    """Save exported model and parameters to a zip archive.

    Archive contains: ``model.jax``, ``params.msgpack``, ``dtypes.json``,
    ``metadata.json``. The metadata dict must include at least a ``cutoff``
    key with the model's interaction cutoff radius.
    """
    if "cutoff" not in metadata:
        raise ValueError("metadata must include 'cutoff' (interaction cutoff radius)")
    serialized = exported.serialize(1)
    dtypes = {k: v.dtype.str for k, v in abstract_input.items()}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("model.jax", serialized)
        zf.writestr("params.msgpack", serialize_params(parameters))
        zf.writestr("dtypes.json", json.dumps(dtypes))
        zf.writestr("metadata.json", json.dumps(metadata))
    print(f"Exported model saved to {path}")


def load_exported_model(
    path: Path,
) -> tuple[export.Exported, list[Array], dict, dict]:
    """Load an exported model, parameters, dtypes, and metadata from a zip."""
    with zipfile.ZipFile(path) as zf:
        exp = export.deserialize(bytearray(zf.read("model.jax")))
        params = deserialize_params(zf.read("params.msgpack"))
        dtypes = json.loads(zf.read("dtypes.json"))
        metadata = json.loads(zf.read("metadata.json"))
    return exp, params, dtypes, metadata


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def make_predict_fn(
    exported: export.Exported,
    params: list[Array],
) -> Callable:
    """Create a jitted inference function with parameters closed over."""

    @jax.jit
    def predict(data: dict):
        return exported.call(params, data)

    return predict


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def benchmark_jax(fn: Callable, *args, n_iters: int = 10) -> float:
    """Benchmark a JAX function, returning min wall-clock time per call."""
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return min(times)


def benchmark_torch(fn: Callable, *args, n_iters: int = 10) -> float:
    """Benchmark a PyTorch function, returning min wall-clock time per call."""
    fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_export_args(parser: argparse.ArgumentParser) -> None:
    """Add common export CLI arguments to a parser."""
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output zip path.",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=["cuda"],
        help="JAX export platforms (default: cuda).",
    )
    parser.add_argument(
        "--symbolic",
        type=str,
        default="",
        help=(
            "Letters indicating which dims are symbolic: "
            "N=atoms, S=systems, E=edges (e.g. 'NSE' for all)."
        ),
    )
    parser.add_argument(
        "--n-atoms", type=int, default=256, help="Number of atoms in dummy data."
    )
    parser.add_argument(
        "--n-edges", type=int, default=6400, help="Number of edges in dummy data."
    )
    parser.add_argument(
        "--n-batches", type=int, default=1, help="Number of batches/systems."
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Random seed for dummy data."
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=10,
        help="Number of iterations for benchmarking.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip reload verification and benchmarking.",
    )


# ---------------------------------------------------------------------------
# Unified export pipeline
# ---------------------------------------------------------------------------
def _report_error(name: str, jax_val: np.ndarray, torch_val: np.ndarray) -> None:
    """Print absolute and relative error for a quantity."""
    abs_err = float(np.max(np.abs(jax_val - torch_val)))
    denom = float(np.max(np.abs(torch_val))) + 1e-12
    rel_err = 100.0 * abs_err / denom
    print(f"  {name}: max_abs_err={abs_err:.2e}, max_rel_err={rel_err:.4f}%")


def run_export(
    fn: EnergyFn | PredictFn,
    args: argparse.Namespace,
    metadata: dict,
    *,
    constraints: tuple[str, ...] = (),
) -> None:
    """End-to-end export pipeline: test -> export -> save -> verify -> benchmark.

    Args:
        fn: Either an ``EnergyFn`` (returns energy tensor) or a
            ``PredictFn`` (returns dict with "energy" and optionally
            "forces"/"stress"). If a ``PredictFn`` is given, it is
            wrapped to extract energy for the export, and forces/stress
            are validated against JAX gradients.
        args: Parsed CLI namespace (must contain the common export args
            added by ``add_export_args``).
        metadata: Model-specific metadata to include in the zip.
            Must include ``cutoff`` (interaction cutoff radius).
        constraints: Symbolic shape constraints, e.g. ``("E >= 64",)``.
    """
    torch.manual_seed(args.seed)
    dtype = torch.float32
    if hasattr(args, 'f64'):
        if args.f64:
            dtype = torch.float64
    data = make_dummy_data(args.n_atoms, args.n_batches, args.n_edges, dtype)

    # Probe the function to determine if it returns a dict or a tensor
    probe_out = fn(data)
    is_predict_fn = isinstance(probe_out, dict)

    if is_predict_fn:
        predict_fn: PredictFn = fn  # type: ignore[assignment]

        def energy_fn(data: AtomGraphInput) -> torch.Tensor:
            return predict_fn(data)["energy"]
    else:
        predict_fn = None  # type: ignore[assignment]
        energy_fn = fn  # type: ignore[assignment]

    print("PyTorch forward pass:", energy_fn(data))

    if hasattr(args, "benchmark_iters") and not getattr(args, "skip_verify", False):
        elapsed_pt = benchmark_torch(energy_fn, data, n_iters=args.benchmark_iters)
        print(f"PyTorch time: {elapsed_pt:.4f}s")

    # Export
    abstract_input = make_abstract_input(
        args.n_atoms,
        args.n_batches,
        args.n_edges,
        symbolic=args.symbolic,
        constraints=constraints,
    )
    print("Exporting model...")
    parameters, exported = export_model(
        energy_fn, abstract_input, platforms=args.platforms
    )
    print("Export complete!")

    # Save
    save_exported_model(
        exported, parameters, abstract_input, args.output, metadata=metadata
    )

    if args.skip_verify:
        return

    # Reload and verify
    exported, params, dtypes, _meta = load_exported_model(args.output)
    print("Reloaded dtypes:", dtypes)

    jax_predict = make_predict_fn(exported, params)
    jax_data = tojax(data)

    # Get PyTorch reference values
    torch_energy = energy_fn(data).numpy(force=True)
    torch_forces = None
    torch_stress = None
    if is_predict_fn:
        torch_out = predict_fn(data)
        if "forces" in torch_out and torch_out["forces"] is not None:
            torch_forces = torch_out["forces"].numpy(force=True)
        if "stress" in torch_out and torch_out["stress"] is not None:
            torch_stress = torch_out["stress"].numpy(force=True)

    # JAX energy + forces via value_and_grad
    @jax.jit
    @jax.value_and_grad
    def energy_and_grad(pos):
        inp = {**jax_data, "pos": pos}
        return jax_predict(inp).sum()

    jax_energy_val, jax_forces = energy_and_grad(jax_data["pos"])
    jax_energy = np.asarray(jax.block_until_ready(jax_energy_val))
    jax_forces = np.asarray(jax.block_until_ready(jax_forces))

    print("Verification:")
    _report_error("energy", jax_energy, torch_energy.sum())

    if torch_forces is not None:
        # Forces = -dE/dpos
        _report_error("forces", -jax_forces, torch_forces)

    # JAX stress via grad w.r.t. cell
    if torch_stress is not None:

        @jax.jit
        @jax.grad
        def energy_grad_cell(cell):
            inp = {**jax_data, "cell": cell}
            return jax_predict(inp).sum()

        jax_stress = np.asarray(
            jax.block_until_ready(energy_grad_cell(jax_data["cell"]))
        )
        # Convert full (…, 3, 3) tensor to Voigt (…, 6) if needed
        if jax_stress.size != torch_stress.size:
            jax_stress = np.stack(
                [
                    jax_stress[..., 0, 0],
                    jax_stress[..., 1, 1],
                    jax_stress[..., 2, 2],
                    jax_stress[..., 1, 2],
                    jax_stress[..., 0, 2],
                    jax_stress[..., 0, 1],
                ],
                axis=-1,
            )
        _report_error("stress", jax_stress.reshape(torch_stress.shape), torch_stress)

    # Benchmark
    elapsed = benchmark_jax(
        energy_and_grad, jax_data["pos"], n_iters=args.benchmark_iters
    )
    print(f"Exported JAX energy+forces time: {elapsed:.4f}s")
