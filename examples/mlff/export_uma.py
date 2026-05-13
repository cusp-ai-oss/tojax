# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13, <3.14"
# dependencies = [
#     "fairchem-core",
#     "tojax",
#     "jax[cuda12]",
#     "flatbuffers",
#     "torchtnt",
#     "torch",
#     "msgpack",
# ]
#
# [tool.uv]
# override-dependencies = ["numpy>=2.0"]
# [tool.uv.sources]
# fairchem-core = { git = "https://github.com/n-gao/fairchem.git", branch = "ng/sybmolic_shapes", subdirectory = "packages/fairchem-core" }
# tojax = { path = "../../", editable = true }
# torchtnt = { git = "https://github.com/meta-pytorch/tnt" }
# ///
"""Export a UMA model from PyTorch to JAX and package it as a .zip archive."""

import argparse
import functools
import logging
from collections import defaultdict
from pathlib import Path

import fairchem.core.models.uma.triton.custom_ops  # noqa: F401 - registers torch.ops.fairchem.*
import jax
import jax.numpy as jnp
import torch
from export_common import AtomGraphInput, add_export_args, run_export
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
from fairchem.core.models.uma.nn.mole import MOLE
from fairchem.core.units.mlip_unit import MLIPPredictUnit, load_predict_unit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from jax import Array

from tojax.functions import TRANSLATED_FNS
from tojax.wrapper import TensorWrapper, to_jax_compatible

logging.basicConfig(level=logging.INFO)

jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)

AtomicData.validate = lambda self: None  # disable validation for jax export
_OG_RAGGED_DOT = MOLE.ragged_dot


# ---------------------------------------------------------------------------
# Monkey-patches for traceability
# ---------------------------------------------------------------------------
def collate_predictions(predict_fn):
    @functools.wraps(predict_fn)
    def collated_predict(
        predict_unit, data: AtomicData, undo_element_references: bool = True
    ):
        preds = predict_fn(predict_unit, data, undo_element_references)
        collated_preds = defaultdict(list)
        assert len(data.dataset) == 1, (
            "Collation currently only supports batches with a single dataset, found: {data.dataset}"
        )
        for i, dataset in enumerate(data.dataset):
            for task in predict_unit.dataset_to_tasks[dataset]:
                if task.level == "system" or task.level == "atom":
                    collated_preds[task.property].append(preds[task.name])
                else:
                    raise RuntimeError(
                        f"Unrecognized task level={task.level} found in data batch at position {i}"
                    )
        return {prop: torch.cat(val) for prop, val in collated_preds.items()}

    return collated_predict


MLIPPredictUnit.predict = collate_predictions(MLIPPredictUnit.predict.__wrapped__)
eSCNMDMoeBackbone.on_predict_check = lambda self, data: None
eSCNMDMoeBackbone._assert_all_mole_info_consistent = lambda self, data: None


# ---------------------------------------------------------------------------
# JAX translations for fairchem Triton kernels (forward only)
# ---------------------------------------------------------------------------
_L_TO_M = jnp.array([0, 2, 6, 3, 7, 1, 5, 8, 4])
_M_TO_L = jnp.array([0, 5, 1, 3, 8, 6, 2, 4, 7])


def _jax_node_to_edge_wigner_permute(x, edge_index, wigner, out, x_edge):
    x_d, ei_d, w_d = x.data, edge_index.data, wigner.data
    x_src = x_d[ei_d[0]]
    x_tgt = x_d[ei_d[1]]
    x_cat = jnp.concatenate([x_src, x_tgt], axis=-1)
    rotated = jnp.einsum("eij,ejc->eic", w_d, x_cat)
    out.data = rotated[:, _L_TO_M, :]
    x_edge.data = x_cat


def _jax_permute_wigner_inv_edge_to_node(x, wigner, out, x_l):
    x_d, w_d = x.data, wigner.data
    x_perm = x_d[:, _M_TO_L, :]
    out.data = jnp.einsum("eij,ejc->eic", w_d, x_perm)
    x_l.data = x_perm


TRANSLATED_FNS[torch.ops.fairchem._kernel_node_to_edge_wigner_permute] = (
    _jax_node_to_edge_wigner_permute
)
TRANSLATED_FNS[torch.ops.fairchem._kernel_permute_wigner_inv_edge_to_node] = (
    _jax_permute_wigner_inv_edge_to_node
)


# ---------------------------------------------------------------------------
# Segmented matmul for multi-system support
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(3,))
def _blocked_impl(
    x_sorted: Array, weights: Array, mask_sorted: Array, minimum_segment_size: int
) -> Array:
    N, K = x_sorted.shape
    S, K_w, D_out = weights.shape
    assert K == K_w, f"x dim 1 ({K}) != weights dim 1 ({K_w})"
    assert mask_sorted.shape == (N,), f"mask shape {mask_sorted.shape} != ({N},)"

    if S == 1:
        return x_sorted @ weights[0]

    pad_n = (-N) % minimum_segment_size
    xp = jnp.pad(x_sorted, [(0, pad_n), (0, 0)])
    mp = jnp.pad(mask_sorted, [(0, pad_n)], constant_values=0)
    N_pad = N + pad_n
    num_blocks = N_pad // minimum_segment_size

    x_blocks = xp.reshape(num_blocks, minimum_segment_size, K)
    m_blocks = mp.reshape(num_blocks, minimum_segment_size)
    s0 = m_blocks[:, 0]

    w_padded = jnp.pad(weights, [(0, 1), (0, 0), (0, 0)])
    w_both = jax.vmap(lambda i: jax.lax.dynamic_slice_in_dim(w_padded, i, 2, axis=0))(
        s0
    )
    p_both = jnp.matmul(x_blocks[:, None], w_both)
    idx = (m_blocks != s0[:, None]).astype(jnp.int32)
    block_idx = jnp.arange(num_blocks)[:, None]
    row_idx = jnp.arange(minimum_segment_size)[None, :]
    out = p_both[block_idx, idx, row_idx]
    return out.reshape(N_pad, D_out)[:N]


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
@functools.partial(jax.jit, static_argnums=(3,))
def segmented_matmul_blocked(
    x_sorted: Array, weights: Array, mask_sorted: Array, minimum_segment_size: int
) -> Array:
    """Pure JAX segmented matmul using the 2-segment-per-block assumption."""
    return _blocked_impl(x_sorted, weights, mask_sorted, minimum_segment_size)


def _blocked_fwd(x_sorted, weights, mask_sorted, minimum_segment_size):
    out = _blocked_impl(x_sorted, weights, mask_sorted, minimum_segment_size)
    return out, (x_sorted, weights, mask_sorted)


def _blocked_bwd(minimum_segment_size, res, g):
    x_sorted, weights, mask_sorted = res
    g_x = _blocked_impl(g, weights.mT, mask_sorted, minimum_segment_size)
    g_w = jnp.zeros_like(weights)  # TODO: implement weights grad
    return g_x, g_w, jnp.zeros_like(mask_sorted)


segmented_matmul_blocked.defvjp(_blocked_fwd, _blocked_bwd)
segmented_matmul_blocked = jax.jit(
    segmented_matmul_blocked, static_argnames=("minimum_segment_size",)
)  # type: ignore


def patch_ragged_dot(min_edges_per_system: int | None = None) -> None:
    """Monkey-patch MOLE.ragged_dot.

    If ``min_edges_per_system`` is None, all systems are assumed to share a
    single weight set (``rhs[0]``). Otherwise, segmented matmul is used with
    the given block size to support per-system weights.
    """

    def ragged_dot(self, lhs, rhs, group_sizes):
        if not isinstance(lhs, TensorWrapper):
            return _OG_RAGGED_DOT(self, lhs, rhs, group_sizes)
        if min_edges_per_system is None:
            return torch.einsum("...d,ed->...e", lhs, rhs[0])
        lh_in = to_jax_compatible(lhs)
        rh_in = to_jax_compatible(rhs)
        gs_in = to_jax_compatible(group_sizes)
        N = lh_in.shape[0]
        indices = jnp.arange(gs_in.shape[0]).repeat(gs_in, total_repeat_length=N)
        if lh_in.ndim == 3:
            T = lh_in.shape[1]
            lh_flat = lh_in.reshape(N * T, -1)
            indices_flat = jnp.repeat(indices, T)
            result = segmented_matmul_blocked(
                lh_flat, rh_in.mT, indices_flat, min_edges_per_system
            )
            result = result.reshape(N, T, -1)
        else:
            result = segmented_matmul_blocked(
                lh_in, rh_in.mT, indices, min_edges_per_system
            )
        return TensorWrapper(result)

    MOLE.ragged_dot = ragged_dot


# ---------------------------------------------------------------------------
# UMA energy adapter
# ---------------------------------------------------------------------------
def make_uma_predict_fn(predictor, dataset: str = "omat"):
    """Return a PyTorch callable ``AtomGraphInput -> dict``."""

    def predict(data: AtomGraphInput) -> dict[str, torch.Tensor]:
        batch = data["batch"]
        edge_index = data["edge_index"]
        n_batches = data["cell"].shape[0]
        natoms = torch.zeros(n_batches, dtype=torch.long).scatter_add_(
            0, batch, torch.ones_like(batch, dtype=torch.long)
        )
        src_batch = batch[edge_index[0]]
        nedges = torch.zeros(n_batches, dtype=torch.long).scatter_add_(
            0, src_batch, torch.ones_like(src_batch, dtype=torch.long)
        )
        uma_data = AtomicData(
            pos=data["pos"],
            atomic_numbers=data["atomic_numbers"],
            cell=data["cell"],
            pbc=data["pbc"],
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=data["cell_offsets"],
            nedges=nedges,
            charge=data["charge"],
            spin=data["spin"],
            fixed=torch.zeros_like(data["atomic_numbers"]),
            tags=torch.zeros_like(data["atomic_numbers"]),
            batch=batch,
            dataset=dataset,
        )
        result = predictor.predict(uma_data)
        out: dict[str, torch.Tensor] = {"energy": result["energy"]}
        if "forces" in result:
            out["forces"] = result["forces"]
        if "stress" in result:
            out["stress"] = result["stress"]
        return out

    return predict


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a UMA model from PyTorch to JAX."
    )
    add_export_args(parser)
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--dataset", default="omat", help="Dataset name.")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Cutoff radius.")
    parser.add_argument(
        "--min-edges-per-system",
        type=int,
        default=None,
        help=(
            "Segmented matmul block size. Used when --merge-mole is unset "
            "and either S is symbolic or --n-batches >= 2 (per-system MOLE "
            "weights). Higher values are more efficient, so pick as large "
            "as possible; if set larger than the minimum number of edges "
            "per system at runtime, results will be incorrect. If unset "
            "in the multi-system case (and --merge-mole is also unset), "
            "the first system's MOLE weights are used for all systems; "
            "results are only correct when all systems share MOLE gating "
            "(e.g. same composition / charge / spin; positions, cells, "
            "and random seeds may differ)."
        ),
    )
    parser.add_argument(
        "--merge-mole", action="store_true", help="Enable MOLE merging."
    )
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32.")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"{Path(args.checkpoint).stem}_{args.dataset}.zip")

    mss = args.min_edges_per_system
    s_dynamic = "S" in args.symbolic.upper()
    multi_systems_possible = s_dynamic or args.n_batches >= 2
    shape_desc = "S is symbolic" if s_dynamic else f"--n-batches={args.n_batches}"

    # --merge-mole collapses MOLE experts into a single weight set, which is
    # only valid when all atoms share the same MOLE gating (e.g. a single
    # system, or homogeneous compositions). Warn if the export admits more.
    if args.merge_mole and multi_systems_possible:
        logging.warning(
            "--merge-mole assumes all systems are the same kind, but %s. "
            "Results will be incorrect for batches with heterogeneous "
            "compositions at runtime.",
            shape_desc,
        )

    # Per-system weights are handled by segmented matmul (when mss is set) or
    # by falling back to the first system's weights (when mss is unset, on the
    # user's word that all systems share MOLE gating).
    needs_segmented = not args.merge_mole and multi_systems_possible and mss is not None

    if not args.merge_mole and multi_systems_possible and mss is None:
        logging.warning(
            "Neither --merge-mole nor --min-edges-per-system is set, but %s. "
            "MOLE will use the first system's weights for all systems; "
            "results will be incorrect at runtime unless all systems share "
            "MOLE gating (same composition / charge / spin; positions, "
            "cells, and random seeds may differ).",
            shape_desc,
        )
    if mss is not None and not needs_segmented:
        logging.warning(
            "--min-edges-per-system is ignored: segmented matmul is not "
            "used (either --merge-mole is set, or the export is for a "
            "single system with concrete shape)."
        )

    patch_ragged_dot(min_edges_per_system=mss if needs_segmented else None)
    torch.set_default_device(args.device)

    predictor = load_predict_unit(
        args.checkpoint,
        device=args.device,
        inference_settings=InferenceSettings(
            tf32=not args.no_tf32,
            activation_checkpointing=False,
            merge_mole=args.merge_mole,
            compile=False,
            external_graph_gen=True,
            internal_graph_gen_version=2,
        ),
    )
    predictor.model.module.backbone.otf_graph = False
    predictor.model.module.supports_single_atoms = True

    constraints = (
        (f"E >= {mss}",) if "E" in args.symbolic.upper() and mss is not None else ()
    )
    run_export(
        make_uma_predict_fn(predictor, dataset=args.dataset),
        args,
        metadata=dict(cutoff=args.cutoff),
        constraints=constraints,
    )


if __name__ == "__main__":
    main()
