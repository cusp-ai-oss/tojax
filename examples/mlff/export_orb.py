# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "dm-tree>=0.1.10",
#     "flatbuffers",
#     "tojax",
#     "jax[cuda12]",
#     "msgpack",
#     "orb-models>=0.6.2",
# ]
#
# [tool.uv]
# override-dependencies = ["dm-tree>=0.1.10"]
#
# [tool.uv.sources]
# tojax = { path = "../../", editable = true }
# orb-models = { git = "https://github.com/n-gao/orb-models.git", branch = "ng/traceable" }
# ///
# orb-models = { path = "../../orb-models", editable = true }
"""Export Orb model to JAX with parameter separation."""

import argparse
from pathlib import Path

import jax
import torch
from torch.jit._state import disable as torch_jit_disable

torch_jit_disable()
from export_common import (  # noqa: E402
    AtomGraphInput,
    add_export_args,
    run_export,
)
from orb_models.forcefield import pretrained  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)

PRETRAINED_MODELS = {
    "orb_v3_conservative_inf_omat": pretrained.orb_v3_conservative_inf_omat,
}


# ---------------------------------------------------------------------------
# Orb energy adapter
# ---------------------------------------------------------------------------
def make_orb_predict_fn(orbff):
    """Return a PyTorch callable ``AtomGraphInput -> dict``."""

    def predict(data: AtomGraphInput) -> dict[str, torch.Tensor]:
        from orb_models.common.atoms.batch.graph_batch import AtomGraphs

        batch = data["batch"]
        n_atoms = batch.shape[0]
        n_systems = data["cell"].shape[0]

        ones = torch.ones_like(batch)
        counts = torch.zeros(n_systems, dtype=torch.int64).scatter_add_(0, batch, ones)

        senders = data["edge_index"][0]
        receivers = data["edge_index"][1]
        src_batch = batch[senders]
        nedges = torch.zeros(n_systems, dtype=torch.int64).scatter_add_(
            0, src_batch, torch.ones_like(src_batch)
        )

        atomic_numbers = data["atomic_numbers"]
        atomic_numbers_embedding = torch.nn.functional.one_hot(
            atomic_numbers.long(), 118
        ).float()

        graph = AtomGraphs(
            senders=senders,
            receivers=receivers,
            n_node=counts,
            n_edge=nedges,
            node_features={
                "positions": data["pos"],
                "atomic_numbers": atomic_numbers,
                "atomic_numbers_embedding": atomic_numbers_embedding,
                "atom_identity": torch.arange(n_atoms, dtype=torch.int64),
            },
            system_features={"cell": data["cell"], "pbc": data["pbc"]},
            edge_features={"unit_shifts": data["cell_offsets"]},
            node_targets={},
            edge_targets={},
            system_targets={},
            system_id=None,
            fix_atoms=None,
            tags=None,
            radius=6.0,
            max_num_neighbors=nedges,
            half_supercell=False,
        )
        # vectors, stress_displacement, generator = graph.compute_differentiable_edge_vectors()
        # graph.edge_features["vectors"] = vectors
        # return graph.edge_features["vectors"].norm(dim=1)
        result = orbff.predict(graph, split=False)
        # return result
        out: dict[str, torch.Tensor] = {"energy": result["energy"]}
        if "grad_forces" in result:
            out["forces"] = result["grad_forces"]
        if "grad_stress" in result:
            out["stress"] = result["grad_stress"]
        return out

    return predict


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def main():
    import torch.nn as nn

    nn.Module.compile
    parser = argparse.ArgumentParser(description="Export Orb model to JAX")
    add_export_args(parser)
    parser.add_argument(
        "--model-name",
        type=str,
        default="orb_v3_conservative_inf_omat",
        choices=list(PRETRAINED_MODELS.keys()),
        help="Pretrained model name.",
    )
    parser.add_argument(
        "--precision", type=str, default="float32-high", help="Precision."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device.")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"{args.model_name}.zip")

    orbff, adapter = PRETRAINED_MODELS[args.model_name](
        device=args.device, precision=args.precision, compile=False
    )

    run_export(
        make_orb_predict_fn(orbff),
        args,
        metadata=dict(
            cutoff=float(adapter.radius),
            model_name=args.model_name,
            precision=args.precision,
        ),
    )
    print(f"\nOrb model successfully exported to {args.output}!")


if __name__ == "__main__":
    main()
