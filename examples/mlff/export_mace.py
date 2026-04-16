#!/usr/bin/env python3
# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "e3nn-jax",
#     "flatbuffers",
#     "tojax",
#     "jax[cuda12]",
#     "mace-torch",
#     "msgpack",
#     "requests",
# ]
#
# [tool.uv.sources]
# tojax = { path = "../../", editable = true }
# mace-torch = { git = "https://github.com/n-gao/mace.git", branch = "ng/remove_cast" }
# ///
"""Export MACE model to JAX with parameter separation."""

import argparse
import hashlib
import pathlib
from copy import deepcopy
from pathlib import Path
from typing import Optional

import jax
import numpy as np
import requests
import torch
from export_common import AtomGraphInput, add_export_args, run_export

from tojax.patches import patch_module

# Set precision
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Model downloading/caching
# ---------------------------------------------------------------------------
def get_cache_dir() -> pathlib.Path:
    cache_dir = pathlib.Path.home() / ".cache" / "tojax" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def url_to_cache_filename(url: str) -> str:
    filename = pathlib.Path(url).name
    if not filename or "." not in filename:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = f"model_{url_hash}.model"
    return filename


def download_mace_model(
    url: str, output_path: Optional[pathlib.Path] = None
) -> pathlib.Path:
    if output_path is None:
        output_path = get_cache_dir() / url_to_cache_filename(url)
    if output_path.exists():
        print(f"Using cached model from: {output_path}")
        return output_path
    print(f"Downloading MACE model from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    print(f"Model saved to: {output_path}")
    return output_path


def load_mace_model(model_path: pathlib.Path) -> torch.nn.Module:
    model = torch.load(model_path, weights_only=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# MACE energy adapter
# ---------------------------------------------------------------------------
def make_mace_predict_fn(torch_model: torch.nn.Module, f64: bool):
    """Return a PyTorch callable ``AtomGraphInput -> dict``."""
    torch_model = deepcopy(torch_model)
    torch_model = patch_module(torch_model)
    atomic_numbers_list: list[int] = list(torch_model.atomic_numbers)  # type: ignore[union-attr]
    num_atom_types = len(atomic_numbers_list)
    # Build lookup: atomic_number -> index in model's atomic_numbers list
    z_to_index = torch.zeros(max(atomic_numbers_list) + 1, dtype=torch.int64)
    for idx, z in enumerate(atomic_numbers_list):
        z_to_index[z] = idx

    def predict(data: AtomGraphInput) -> dict[str, torch.Tensor]:
        batch = data["batch"]
        n_systems = data["cell"].shape[0]
        counts = torch.zeros(n_systems, dtype=torch.int64).scatter_add_(
            0, batch, torch.ones_like(batch)
        )
        ptr = torch.zeros(n_systems + 1, dtype=torch.int64)
        ptr[1:] = torch.cumsum(counts, 0)

        fdtype = torch.float64 if f64 else torch.float32
        indices = z_to_index[data["atomic_numbers"]]
        node_attrs = torch.nn.functional.one_hot(indices, num_atom_types).to(fdtype)

        # cell_offsets are integer image indices; MACE expects real-space shifts
        edge_batch = batch[data["edge_index"][0]]
        shifts = torch.einsum(
            "ei,eij->ej", data["cell_offsets"], data["cell"][edge_batch]
        )

        mace_input = {
            "node_attrs": node_attrs,
            "positions": data["pos"],
            "edge_index": data["edge_index"],
            "batch": batch,
            "cell": None,
            "ptr": ptr,
            "shifts": shifts,
        }
        result = torch_model(mace_input, compute_force=True)
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
def main():
    parser = argparse.ArgumentParser(description="Export MACE model to JAX")
    add_export_args(parser)
    parser.add_argument(
        "--url",
        type=str,
        default="https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
        help="URL to download MACE model from",
    )
    parser.add_argument("--cache", type=str, default=None, help="Cache path.")
    parser.add_argument("--f64", action="store_true", help="64-bit precision.")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear cache and exit."
    )
    args = parser.parse_args()

    if args.clear_cache:
        import shutil

        cache_dir = get_cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cleared cache: {cache_dir}")
        return

    model_path = download_mace_model(
        args.url, pathlib.Path(args.cache) if args.cache else None
    )
    if args.output is None:
        suffix = "_64" if args.f64 else "_32"
        args.output = Path(f"{model_path.stem}{suffix}.zip")

    torch_model = load_mace_model(model_path)
    torch_model = torch_model.double() if args.f64 else torch_model.float()

    run_export(
        make_mace_predict_fn(torch_model, args.f64),
        args,
        metadata=dict(
            cutoff=float(torch_model.r_max),  # type: ignore[arg-type]
            atomic_numbers=np.asarray(torch_model.atomic_numbers).tolist(),  # type: ignore[arg-type]
        ),
    )
    print("\nMACE model successfully exported!")


if __name__ == "__main__":
    main()
