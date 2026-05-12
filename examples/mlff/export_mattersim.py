#!/usr/bin/env python3
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "flatbuffers",
#     "tojax",
#     "jax[cuda12]",
#     "mattersim",
#     "msgpack",
# ]
#
# [tool.uv.sources]
# tojax = { path = "../../", editable = true }
# ///
"""Export MatterSim (M3GNet) to JAX.

M3GNet's reference three-body builder uses ``torch.where`` and boolean
indexing, so its output shape is data-dependent and not JAX-traceable.
This script replaces it with a dense ``N * K * (K - 1)`` construction
(K = ``--max-neighbors-per-atom``) plus a validity mask, and
monkey-patches ``M3Gnet.forward`` (UMA-style; no mattersim fork) so the
mask is applied to ``three_basis`` right after the spherical-basis call.
"""

import argparse
from pathlib import Path
from typing import Dict

import jax
import torch
from torch.jit._state import disable as torch_jit_disable

torch_jit_disable()  # SphericalBasisLayer uses @torch.jit.script; tojax needs eager Python

from export_common import (  # noqa: E402
    AtomGraphInput,
    add_export_args,
    benchmark_torch,
    make_dummy_data,
    run_export,
)
from mattersim.forcefield.m3gnet.m3gnet import M3Gnet  # noqa: E402
from mattersim.forcefield.m3gnet.modules.scatter import scatter_sum  # noqa: E402
from mattersim.forcefield.potential import Potential  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Monkey-patched M3Gnet forward
# ---------------------------------------------------------------------------
def _patched_m3gnet_forward(
    self: M3Gnet,
    input: Dict[str, torch.Tensor],
    dataset_idx: int = -1,
) -> torch.Tensor:
    """M3Gnet.forward with one extra line: ``three_basis *= three_body_mask``.

    Mirrors upstream forward exactly except for the masking step. Avoids
    forking mattersim. Unused ``num_*`` placeholders are tolerated.
    """
    pos = input["atom_pos"]
    cell = input["cell"]
    pbc_offsets = input["pbc_offsets"].to(pos.dtype)
    atom_attr = input["atom_attr"]
    edge_index = input["edge_index"].long()
    three_body_indices = input["three_body_indices"].long()
    three_body_mask = input["three_body_mask"].to(pos.dtype)
    batch = input["batch"]

    edge_batch = batch[edge_index[0]]
    edge_vector = pos[edge_index[0]] - (
        pos[edge_index[1]] + torch.einsum("ei,eij->ej", pbc_offsets, cell[edge_batch])
    )
    edge_length = torch.linalg.norm(edge_vector, dim=1)
    vij = edge_vector[three_body_indices[:, 0]]
    vik = edge_vector[three_body_indices[:, 1]]
    rij = edge_length[three_body_indices[:, 0]]
    rik = edge_length[three_body_indices[:, 1]]
    cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
    cos_jik = torch.clamp(cos_jik, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    triple_edge_length = rik.view(-1)
    edge_length = edge_length.unsqueeze(-1)
    atomic_numbers = atom_attr.squeeze(1).long()

    atom_attr = self.atom_embedding(self.one_hot_atoms(atomic_numbers))
    edge_attr = self.rbf(edge_length.view(-1))
    edge_attr_zero = edge_attr
    edge_attr = self.edge_encoder(edge_attr)
    three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))
    three_basis = three_basis * three_body_mask.unsqueeze(-1)

    for conv in self.graph_conv:
        atom_attr, edge_attr = conv(
            atom_attr=atom_attr,
            edge_attr=edge_attr,
            edge_attr_zero=edge_attr_zero,
            edge_index=edge_index,
            three_basis=three_basis,
            three_body_index=three_body_indices,
            edge_length=edge_length,
            num_edges=input.get("num_bonds"),
            num_triple_ij=input.get("num_triple_ij"),
            num_atoms=input.get("num_atoms"),
        )

    energies_i = self.final(atom_attr).view(-1)
    energies_i = self.normalizer(energies_i, atomic_numbers)
    return scatter_sum(energies_i, batch, dim=0, dim_size=cell.shape[0])


# ---------------------------------------------------------------------------
# Static-shape three-body construction
# ---------------------------------------------------------------------------
def _compute_threebody_dense(
    edge_index: torch.Tensor,
    n_atoms_total: int,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded three-body indices + validity mask.

    Args:
        edge_index: [2, E] sorted by ``edge_index[0]`` (central atom).
        n_atoms_total: Symbolic ``N`` (use ``pos.shape[0]``).
        max_neighbors: Static cap K. The actual maximum bonds-per-atom
            must not exceed this — excess neighbours are silently
            dropped from three-body contributions.

    Returns:
        three_body_indices ``[N*K*K, 2]`` and mask ``[N*K*K]``. The full
        ``K*K`` grid is enumerated and diagonal pairs (``u == v``) are
        masked out. ``K*(K-1)`` would be tighter but JAX's shape solver
        cannot derive ``K*(K-1) >= 0`` from a constraint on ``K``
        alone, so the K*K shape is used to keep symbolic export viable.
    """
    K = max_neighbors
    src = edge_index[0]
    ones = torch.ones_like(src)
    n_bond = torch.zeros(n_atoms_total, dtype=torch.long).scatter_add_(0, src, ones)
    starts = torch.cumsum(n_bond, dim=0) - n_bond

    slot = torch.arange(K, dtype=torch.long)
    slot_valid = slot[None, :] < n_bond[:, None]
    slot_idx = torch.where(
        slot_valid, starts[:, None] + slot[None, :], torch.zeros_like(starts[:, None])
    )

    pair_u = slot.repeat_interleave(K)
    pair_v = torch.tile(slot, (K,))

    tb_u = slot_idx[:, pair_u]
    tb_v = slot_idx[:, pair_v]
    valid = slot_valid[:, pair_u] & slot_valid[:, pair_v] & (pair_u != pair_v)[None, :]

    three_body_indices = torch.stack([tb_u.reshape(-1), tb_v.reshape(-1)], dim=1)
    return three_body_indices, valid.reshape(-1)


# ---------------------------------------------------------------------------
# MatterSim energy adapter
# ---------------------------------------------------------------------------
def make_mattersim_predict_fn(
    model: M3Gnet,
    max_neighbors: int | None,
    cube_num: int,
    scale: int,
):
    """Return a PyTorch callable ``AtomGraphInput -> {energy, forces, stress}``.

    ``forces = -dE/dpos`` and ``stress = dE/dcell`` (raw lattice gradient,
    matching what ``run_export`` compares against from ``jax.grad``).

    If ``max_neighbors`` is None, K is derived in the forward pass as
    ``K = (E * cube_num) // (N * scale)`` (integer arithmetic only, so
    the formula stays valid under symbolic N/E).
    """

    def predict(data: AtomGraphInput) -> dict[str, torch.Tensor]:
        edge_index = data["edge_index"].long()
        order = torch.argsort(edge_index[0], stable=True)
        edge_index = edge_index[:, order]

        # requires_grad_ is a no-op under tojax tracing (preserves gradient
        # flow), and arms autograd on the PyTorch side. Avoid .detach()/.clone()
        # — both translate to stop_gradient and would zero out jax.grad.
        pos = data["pos"].requires_grad_(True)
        cell = data["cell"].requires_grad_(True)
        cell_offsets = data["cell_offsets"][order].to(pos.dtype)
        batch = data["batch"]
        n_atoms_total = pos.shape[0]
        n_systems = cell.shape[0]
        n_edges = edge_index.shape[1]

        if max_neighbors is None:
            K = (n_edges * cube_num) // (n_atoms_total * scale)
        else:
            K = max_neighbors

        tb_idx, tb_mask = _compute_threebody_dense(edge_index, n_atoms_total, K)

        edge_batch = batch[edge_index[0]]
        num_atoms = torch.zeros(n_systems, dtype=torch.long).scatter_add_(
            0, batch, torch.ones_like(batch)
        )
        num_bonds = torch.zeros(n_systems, dtype=torch.long).scatter_add_(
            0, edge_batch, torch.ones_like(edge_batch)
        )
        num_triple_ij = torch.zeros(n_edges, dtype=torch.long).scatter_add_(
            0, tb_idx[:, 0], tb_mask.to(torch.long)
        )

        model_input = {
            "atom_pos": pos,
            "cell": cell,
            "pbc_offsets": cell_offsets,
            "atom_attr": data["atomic_numbers"].unsqueeze(-1).to(pos.dtype),
            "edge_index": edge_index,
            "three_body_indices": tb_idx,
            "three_body_mask": tb_mask,
            "num_bonds": num_bonds,
            "num_triple_ij": num_triple_ij.unsqueeze(-1),
            "num_atoms": num_atoms,
            "num_graphs": torch.tensor(n_systems, dtype=torch.long),
            "batch": batch,
        }
        energy = model.forward(model_input).view(-1)
        dE_dpos, dE_dcell = torch.autograd.grad(
            energy.sum(), [pos, cell], create_graph=False
        )
        return {"energy": energy, "forces": -dE_dpos, "stress": dE_dcell}

    return predict


# ---------------------------------------------------------------------------
# Patched-vs-original comparison (eager PyTorch)
# ---------------------------------------------------------------------------
def _compare_with_original(
    model: M3Gnet,
    original_forward,
    args: argparse.Namespace,
) -> None:
    """Run both forwards on the same physical graph and compare.

    Builds a uniform-K graph (``K = n_edges // n_atoms`` edges per atom)
    and feeds it to:

    * the original M3Gnet forward with compact ``K*(K-1)`` triples;
    * the patched forward with dense ``N*K*K`` triples plus a mask that
      zeros out diagonal (``u == v``) entries.

    Both representations enumerate the same set of valid triples, so
    energies and forces must agree up to floating-point summation order.
    Also reports wall-clock for each path as a side benefit.
    """
    N, B = args.n_atoms, args.n_batches
    K = args.n_edges // N
    if K < 2:
        print(f"\nSkipping --compare-original: K = E // N = {K} < 2 (no triples).")
        return

    # Reuse make_dummy_data for the bulk; override edge_index with uniform-K
    # edges (no self-loops, since pos[src] - pos[dst] = 0 would divide-by-zero
    # in cos_jik and produce NaN identically in both forwards).
    torch.manual_seed(args.seed)
    data = make_dummy_data(N, B, K * N)
    src = torch.arange(N).repeat_interleave(K)
    dst = torch.randint(0, N - 1, (K * N,))
    data["edge_index"] = torch.stack([src, dst + (dst >= src).long()])

    edge_index = data["edge_index"].long()
    order = torch.argsort(edge_index[0], stable=True)
    edge_index = edge_index[:, order]
    cell_offsets = data["cell_offsets"][order].to(data["pos"].dtype)
    batch = data["batch"]

    # Dense N*K*K + mask via the production builder; compact K*(K-1) inline.
    tb_d, mask_d = _compute_threebody_dense(edge_index, N, K)
    slot = torch.arange(K)
    pair_u = slot.repeat_interleave(K - 1)
    pair_v_tmp = slot[: K - 1].repeat(K)
    pair_v = pair_v_tmp + (pair_v_tmp >= pair_u).long()
    base = (torch.arange(N) * K)[:, None]
    tb_c = torch.stack(
        [(base + pair_u[None, :]).reshape(-1), (base + pair_v[None, :]).reshape(-1)],
        dim=1,
    )
    mask_c = torch.ones(tb_c.shape[0], dtype=torch.bool)

    edge_batch = batch[edge_index[0]]
    num_atoms = torch.zeros(B, dtype=torch.long).scatter_add_(
        0, batch, torch.ones_like(batch)
    )
    num_bonds = torch.zeros(B, dtype=torch.long).scatter_add_(
        0, edge_batch, torch.ones_like(edge_batch)
    )
    num_triple_ij = torch.full((edge_index.shape[1], 1), K - 1, dtype=torch.long)

    def run(forward_fn, triples, mask):
        pos = data["pos"].requires_grad_(True)
        cell = data["cell"].requires_grad_(True)
        energy = forward_fn(
            model,
            {
                "atom_pos": pos,
                "cell": cell,
                "pbc_offsets": cell_offsets,
                "atom_attr": data["atomic_numbers"].unsqueeze(-1).to(pos.dtype),
                "edge_index": edge_index,
                "three_body_indices": triples,
                "three_body_mask": mask,  # patched uses it; original ignores it
                "num_bonds": num_bonds,
                "num_triple_ij": num_triple_ij,
                "num_atoms": num_atoms,
                "num_graphs": torch.tensor(B, dtype=torch.long),
                "batch": batch,
            },
        ).view(-1)
        (dE_dpos,) = torch.autograd.grad(energy.sum(), [pos])
        return energy.detach(), (-dE_dpos).detach()

    e_o, f_o = run(original_forward, tb_c, mask_c)
    e_p, f_p = run(_patched_m3gnet_forward, tb_d, mask_d)

    def err(name, a, b):
        ae = float(torch.max(torch.abs(a - b)))
        re = 100 * ae / (float(torch.max(torch.abs(a))) + 1e-12)
        print(f"  {name}: max_abs_err={ae:.2e}, max_rel_err={re:.4f}%")

    print("\nPatched vs original M3Gnet forward (eager PyTorch, same graph):")
    err("energy", e_o, e_p)
    err("forces", f_o, f_p)

    iters = args.benchmark_iters
    t_o = benchmark_torch(lambda: run(original_forward, tb_c, mask_c), n_iters=iters)
    t_p = benchmark_torch(
        lambda: run(_patched_m3gnet_forward, tb_d, mask_d), n_iters=iters
    )
    print(f"  original (eager) energy+forces time: {t_o:.4f}s")
    print(f"  patched  (eager) energy+forces time: {t_p:.4f}s")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Export MatterSim model to JAX")
    add_export_args(parser)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="mattersim-v1.0.0-1m",
        help=(
            "Checkpoint name ('mattersim-v1.0.0-1m', 'mattersim-v1.0.0-5m') "
            "or a path to a .pth file. Named checkpoints are auto-downloaded "
            "to ~/.local/mattersim/pretrained_models/."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device.")
    parser.add_argument(
        "--compare-original",
        action="store_true",
        help=(
            "After exporting, run the original (unpatched) M3Gnet forward "
            "eagerly on the same physical graph and compare energy/forces "
            "against the patched dense+mask forward. Also reports eager "
            "wall-clock for each. Uses K = n_edges // n_atoms (uniform)."
        ),
    )
    parser.add_argument(
        "--max-neighbors-per-atom",
        type=int,
        default=None,
        help=(
            "Static cap on bonds-per-atom for three-body construction. If "
            "unset, derived in the forward pass from the tensor dimensions "
            "as 1.5 * (E/N) * (r_3b/r_cut)^3 (integer math). Override for "
            "surfaces, defects, or other non-uniform structures where the "
            "max exceeds the bulk average — excess triples are silently "
            "dropped."
        ),
    )
    args = parser.parse_args()

    if args.output is None:
        name = Path(args.checkpoint).name
        if name.endswith(".pth"):
            name = name[: -len(".pth")]
        args.output = Path(f"{name}.zip")

    _original_forward = M3Gnet.forward
    M3Gnet.forward = _patched_m3gnet_forward  # apply patch globally
    torch.set_default_device(args.device)
    potential = Potential.from_checkpoint(
        load_path=args.checkpoint,
        device=args.device,
        load_training_state=False,
    )
    model = potential.model.eval()
    cutoff = float(model.model_args["cutoff"])
    threebody_cutoff = float(model.model_args["threebody_cutoff"])

    # Integer fixed-point K ≈ 1.2 * (E / N) * (r_3b / r_cut)^3 — same
    # constants as the predict-time K derivation.
    scale = 1024
    cube_num = round((threebody_cutoff / cutoff) ** 3 * scale * 6 / 5)

    constraints: tuple[str, ...] = ("E >= 2 * N",)  # E/N >= 2 for a connected graph
    sym = args.symbolic.upper()
    if args.max_neighbors_per_atom is None and ("E" in sym or "N" in sym):
        # JAX's solver cannot propagate `E >= 2 * N` through floordiv, so
        # bound the floor expression directly to prove K >= 1.
        constraints += (f"floordiv({cube_num} * E, {scale} * N) >= 1",)

    run_export(
        make_mattersim_predict_fn(model, args.max_neighbors_per_atom, cube_num, scale),
        args,
        metadata=dict(
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            checkpoint=args.checkpoint,
            max_neighbors_per_atom=args.max_neighbors_per_atom,
        ),
        constraints=constraints,
    )
    print(f"\nMatterSim model successfully exported to {args.output}!")

    if args.compare_original:
        _compare_with_original(model, _original_forward, args)


if __name__ == "__main__":
    main()
