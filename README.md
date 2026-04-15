# `tojax`

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-compatible-green.svg)](https://github.com/google/jax)
[![PyTorch](https://img.shields.io/badge/PyTorch-compatible-orange.svg)](https://pytorch.org/)

`tojax` is a powerful library that enables seamless translation of **pure** PyTorch functions and models to JAX, combining PyTorch's familiar API with JAX's performance advantages including XLA compilation and automatic differentiation.

## Key Features

- **Automatic Model Translation**: Convert PyTorch models to JAX with a single function call
- **Function-Level Translation**: Translate individual PyTorch operations to JAX equivalents
- **Tensor Compatibility**: Use PyTorch-style tensor operations backed by JAX arrays
- **In-Place Operation Support**: Handle PyTorch's mutable semantics in JAX's immutable world
- **Specialized Library Support**: Built-in patches for E3NN and FairChem models
- **Graph Translation**: Convert PyTorch FX computation graphs to JAX functions

## Installation

### Using uv
```bash
uv add tojax
```

### Using pip
```bash
pip install tojax
```

### Environment
In general, the prebuilt binaries of JAX and PyTorch do not work well within the same environment if both are installed with CUDA. To avoid this issue, please install only one of them with CUDA. Most likely, you want JAX to have the CUDA bindings since that is where computations are executed with this library.

## Examples

### 1. Function Translation

`tojax` automatically translates PyTorch functions to JAX equivalents:

```python
import torch
from tojax import tojax

# Get JAX equivalent of a PyTorch function
jax_add = tojax(torch.add)

# Use with JAX arrays
import jax.numpy as jnp
a = jnp.array([1, 2, 3])
b = jnp.array([4, 5, 6])
result = jax_add(a, b)  # Uses JAX implementation
```

### 2. Model Conversion

Convert entire PyTorch models to JAX functions:

```python
import torch.nn as nn
from tojax import tojax

# Define a PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

# Convert to JAX
model = SimpleModel()
jax_model = tojax(model)

# Use with JAX arrays
import jax.numpy as jnp
x = jnp.ones((32, 10))
output = jax_model(x)
```

### 3. JIT Compilation
Everything is JIT compatible:
```python
import jax
import torch.nn as nn
from tojax import tojax


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


# Convert model
model = SimpleModel()  # From earlier example
jax_model = tojax(model)


# JIT compile for performance
@jax.jit
def fast_inference(x):
    return jax_model(x)


# Benchmark
import time

x = jnp.ones((1000, 10))

# First call compiles
start = time.time()
result = jax.block_until_ready(fast_inference(x))
compile_time = time.time() - start

# Subsequent calls are fast
start = time.time()
result = jax.block_until_ready(fast_inference(x))
runtime = time.time() - start

print(f"Compile time: {compile_time:.4f}s")
print(f"Runtime: {runtime:.6f}s")
```

### 5. Gradient Computation
You can use standard JAX transformations like `jax.grad`:
```python
import jax
import jax.numpy as jnp
import torch.nn as nn
from tojax import tojax


# Define a simple model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearModel()
jax_model = tojax(model)


# Define loss function
def loss_fn(x, y):
    pred = jax_model(x)
    return jnp.mean((pred - y) ** 2)


# Compute gradients
x = jnp.zeros((100, 2))
y = jnp.zeros((100, 1))

grad_fn = jax.grad(loss_fn)
gradients = grad_fn(x, y)
print(f"Gradient shape: {gradients.shape}")
```

### 6. Export
Importantly, the resulting JAX functions can be exported and loaded without having the original source code or weights.
```py
import jax
import torch
from jax import export
from tojax import tojax

@tojax
def f(x):
    return torch.pow(x, 2)

inp = jnp.array([1, 2, 3])
exported = export.export(jax.jit(f))(inp)
with open("exported_fn.jax", "wb") as f:
    f.write(exported.serialize())
```
This even works with shape polymorphism if the original source code supports this
```py
import jax
import torch
from jax import export
from tojax import tojax

@tojax
def f(x):
    return torch.pow(x, 2)

poly_shape = export.symbolic_shape("batch_size")
exported = export.export(jax.jit(f))(jax.ShapeDtypeStruct(poly_shape, jnp.float32))
with open("exported_fn.jax", "wb") as f:
    f.write(exported.serialize())
```

## How does it work?
tojax works by swapping PyTorch function dispatches by equivalent JAX functions. Crucially, we only do this for operations that act on tensors that depend on the input of the function. This allows tojax to be very permissive and allow it to deal with the intertwined Python+PyTorch code that is frequently used for pre-processing.

```py
import torch
from tojax import tojax

@tojax
def f(x):
    # These are all executed by PyTorch and the result will be taken to JAX.
    a = torch.arange(10)
    a = torch.pow(a, 2)
    # The following operations depends on the function input x, thus, they get translated to JAX.
    y = torch.add(x, a)
    z = torch.sin(y + x)
    return y

f(jnp.zeros(()))
```

## Limitations and When `tojax` Won't Work

While `tojax` handles most PyTorch code seamlessly, there are important limitations due to JAX's functional programming model and XLA compilation requirements.

### Data-Dependent Control Flow

**`tojax` will fail** when your PyTorch code contains control flow that depends on tensor values (data-dependent control flow). This is because JAX requires all control flow to be traceable at compile time.

#### Examples That Won't Work

```python
import torch
import torch.nn as nn
from tojax import tojax

# This will FAIL - conditional based on tensor value
class ProblematicModel(nn.Module):
    def forward(self, x):
        if x.sum() > 0:  # Data-dependent condition
            return x * 2
        else:
            return x * 3

# This will FAIL - loop with data-dependent bounds
def problematic_function(x):
    result = x
    for i in range(int(x[0])):  # Loop bound depends on data
        result = result + 1
    return result

# This will FAIL - indexing with data-dependent values
def problematic_indexing(x, indices):
    # Advanced indexing with computed indices
    mask = x > 0.5
    return x[mask]  # Dynamic shape based on data
```

#### Examples That Work (Static Control Flow)

```python
import torch
import torch.nn as nn
import jax.numpy as jnp
from tojax import tojax


# Static control flow - works fine
class StaticModel(nn.Module):
    def __init__(self, use_layer=True):
        super().__init__()
        self.layer = nn.Linear(10, 10)
        self.use_layer = use_layer

    def forward(self, x):
        if self.use_layer:  # Condition based on static attribute
            x = self.layer(x)
        return x


# Fixed iteration count - works fine
def static_loop_function(x):
    result = x
    for i in range(5):  # Fixed number of iterations
        result = result * 2
    return result


# Use jnp.where for conditional operations
def conditional_with_where(x):
    # Use jnp.where instead of if/else on data
    return torch.where(x > 0, x * 2, x * 3)


# Fixed-size operations work fine
def fixed_operations(x):
    # All operations have predictable shapes
    mean_pooled = x.mean(dim=-1)
    reshaped = x.reshape(x.shape[0], -1)
    return reshaped @ mean_pooled.unsqueeze(-1)
```

### Dynamic Shapes
```python
# Operations that create dynamic output shapes
def dynamic_filter(x, threshold):
    return x[x > threshold]  # Output size depends on data

# Use fixed-size operations with padding/masking
def fixed_size_filter(x, threshold, max_size):
    mask = x > threshold
    # Pad to fixed size and use mask for downstream operations
    return torch.where(mask, x, 0)
```

### Symbolic Shape Tracing and `len()`
When using symbolic shape tracing (e.g., `jax.export` with polymorphic shapes), use `tensor.shape[0]` instead of `len(tensor)`. Python requires `__len__` to return a concrete `int`, so `len()` cannot propagate symbolic dimensions.
```python
# Will break symbolic shape tracing
def bad(x):
    n = len(x)            # returns a concrete int, raises an Exception
    return x.reshape(n, -1)

# Works with symbolic shapes
def good(x):
    n = x.shape[0]        # preserves symbolic dimension
    return x.reshape(n, -1)
```

### Views
```python
# Views always return copies in tojax, so the flat_view will not share the same data as tensor.
# This error is not raised since it is hard to notice.
def inplace_aliasing(tensor):
    flat_view = tensor.view(-1)
    tensor.add_(1.0)
    return tensor, flat_view
```

### Side effects

```python
# We only translate pure functions and the JAX function will always compute the same result as the torch function call would have.
i = 1

def f(x):
    nonlocal i
    i += 1
    return x + i

f(torch.zeros(())) # 1
f(torch.zeros(())) # 2
# A single increment during the first trace
jax_f = jax.jit(tojax)(f)
jax_f(jnp.zeros(())) # 3
jax_f(jnp.zeros(())) # 3
# Incrementing again
f(torch.zeros(())) # 4
```

## Advanced Features

### Custom Function Registration

Register your own PyTorch-to-JAX function mappings:

```python
from tojax.functions import translates
import jax.numpy as jnp
import torch

@translates(torch.sin)
def my_jax_implementation(x):
    return jnp.sin(x) * 10
```

### Module Patching

Create patches for custom modules:

```python
from tojax.patches import register_patch
import torch.nn as nn

@register_patch(MyCustomModule)
def patch_my_module(module):
    # Modify module for JAX compatibility
    module.some_incompatible_flag = False
    return module
```

## Tested Models
We have tested tojax on the following models:
* [MACE](https://github.com/ACEsuit/mace)
* [UMA](https://github.com/facebookresearch/fairchem)
* [Orb](https://github.com/orbital-materials/orb-models)
* `torchvision.models.resnet18`
* `torchvision.models.vit_b_16`

## Testing

Run the test suite:

```bash
# Using uv
uv run pytest

# Using pytest directly
pytest test/
```

## License

This project is licensed under the Apache License, Version 2.0

## Acknowledgements

- [JAX](https://github.com/google/jax) for the underlying array library and transformations
- [PyTorch](https://pytorch.org/) for the deep learning framework we're translating from
- [E3NN](https://github.com/e3nn/e3nn) and [E3NN-JAX](https://github.com/e3nn/e3nn-jax) for equivariant neural networks
- [Flax](https://github.com/google/flax) for neural network components
- [torch2jax](https://github.com/samuela/torch2jax) for inspiration

## Citation
If you use tojax in your research, please cite:

```
@software{tojax2026,
  title={tojax},
  author={Cusp AI},
  year={2026},
  url={https://github.com/cusp-ai-oss/tojax}
}
```
