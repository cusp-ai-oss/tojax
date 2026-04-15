# MLFF Export Examples

Export PyTorch machine learning force field (MLFF) models to JAX using `tojax`.
The exported models separate parameters from the computation graph and package
everything into a single `.zip` file for deployment.

Three models are supported:

| Model | Script | Download |
|-------|--------|----------|
| [MACE](https://github.com/ACEsuit/mace) | `export_mace.py` | Automatic (cached in `~/.cache/tojax/models/`) |
| [Orb](https://github.com/orbital-materials/orb-models) | `export_orb.py` | Automatic (via `orb_models` package) |
| [UMA](https://github.com/FAIR-Chem/fairchem) | `export_uma.py` | **Manual** — download checkpoints from [Hugging Face](https://huggingface.co/facebook/uma) |

## Quick Start

Each script is self-contained. Use `--help` for a full list of options:

```bash
uv run python export_mace.py --help
uv run python export_orb.py --help
uv run python export_uma.py --help
```

### MACE

The MACE checkpoint is downloaded automatically from GitHub:

```bash
uv run python export_mace.py --output mace.zip --symbolic NSE
```

Use `--url` to point at a different checkpoint, `--cache` to change the
download directory, or `--f64` for double precision.

### Orb

Orb weights are fetched automatically by the `orb_models` package:

```bash
uv run python export_orb.py --output orb.zip --symbolic NSE
```

Use `--model-name` to select a different pretrained variant and `--precision`
to control floating-point accuracy.

### UMA

UMA checkpoints must be downloaded manually from
[Hugging Face](https://huggingface.co/facebook/uma) and passed via
`--checkpoint`:

```bash
uv run python export_uma.py --checkpoint /path/to/uma.pt --output uma.zip --symbolic NSE
```

UMA-specific flags include `--dataset` (inference head selection),
`--multi-system` (per-system weight matrices), and `--min-edges-per-system`.

## Symbolic Shapes (`--symbolic`)

This is the most important flag. By default, every dimension (number of atoms,
systems, edges) is **baked into the compiled model** — you get a graph that only
works for the exact sizes used during export and must recompile for anything
else.

`--symbolic` makes one or more of these dimensions **polymorphic**, so the
exported model accepts variable-sized inputs at runtime without recompilation:

| Letter | Dimension | Meaning |
|--------|-----------|---------|
| `N` | Atoms | Number of atoms can vary |
| `S` | Systems | Number of batched systems can vary |
| `E` | Edges | Number of graph edges can vary |

Combine freely — `--symbolic NSE` makes all three variable:

```bash
# Concrete: locked to 256 atoms, 1 system, 6400 edges
uv run python export_mace.py --output mace_fixed.zip

# Polymorphic: any size at runtime
uv run python export_mace.py --output mace_flex.zip --symbolic NSE
```

For production use, **always export with `--symbolic NSE`** unless you have a
specific reason to fix a dimension (e.g. single-system simulations where
`S = 1` is known at compile time).

## Common Options

These flags are shared across all three scripts:

```
--output PATH              Output zip file path
--platforms PLATFORM       Target platform: cuda, cpu, or tpu (default: cuda)
--symbolic STR             Symbolic dimensions (see above)
--n-atoms INT              Atom count for test data (default: 256)
--n-edges INT              Edge count for test data (default: 6400)
--n-batches INT            Batch/system count for test data (default: 1)
--seed INT                 Random seed (default: 2)
--benchmark-iters INT      Number of benchmark iterations (default: 10)
--skip-verify              Skip reload verification and benchmarking
```

## Output Format

Each export produces a `.zip` containing:

- `model.jax` — serialized JAX computation graph
- `params.msgpack` — model parameters (msgpack format)
- `dtypes.json` — input array dtypes
- `metadata.json` — model metadata (cutoff radius, etc.)

Load an exported model with `export_common.load_exported_model()` and create an
inference function with `export_common.make_predict_fn()`.
