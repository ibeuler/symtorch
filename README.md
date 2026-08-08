<div align="center">
  <h1>TorchSymPy</h1>
  <p><strong>SymPy-to-Torch Transcompilation for Massively Batched, GPU-Accelerated Numerical Integration</strong></p>

  [![PyPI - Version](https://img.shields.io/pypi/v/torchsympy)](https://pypi.org/project/torchsympy/)
  [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchsympy)](https://pypi.org/project/torchsympy/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

**TorchSymPy** bridges the gap between SymPy's symbolic manipulation and PyTorch's highly optimized batched tensor operations. You can transcompile symbolic integrals directly into callable PyTorch engines capable of extremely fast, batched evaluation on GPUs and CPUs.

*Note: this module was first developed for [libphysics](https://github.com/ferhatpy/libphysics) — then split out into a standalone library to tackle generalized parallel computational bottlenecks.*

---

## Why TorchSymPy?

When working with analytical integrals in computational physics or machine learning, researchers often hit a bottleneck: 
1. **SymPy** is great for exact manipulation but painfully slow (or fails) for heavy numeric evaluation.
2. **SciPy** (e.g. `scipy.integrate.nquad`) is highly accurate but inherently sequential and single-threaded. 
3. **PyTorch** thrives on massively parallel grid evaluations, but writing integrators by hand is tedious.

**TorchSymPy** gives you the best of all worlds. You write math in `SymPy`, and TorchSymPy transpiles it into highly optimized `TorchExpr` kernels that run up to **2,700x faster** than `SciPy` by leveraging `torchquad` and massively batched GPU architectures.

## Installation

To install the latest stable version from PyPI:
```bash
pip install torchsympy
```

To install from source (development):
```bash
git clone https://github.com/ibeuler/TorchSymPy.git
cd TorchSymPy
pip install -e .
```

> **Note on PyTorch:** For GPU acceleration, ensure you have a CUDA-compatible `torch` wheel installed (e.g., `torch==2.5.1+cu121`).

## Quickstart

The easiest path from a symbolic integral to a batched GPU evaluation:

```python
import torch
import torchsympy
import sympy as sp

# 1. Define your integrand symbolically
x = sp.Symbol("x", real=True)
p = sp.Symbol("p", real=True)
expr = sp.Integral(sp.exp(-p * x**2), (x, -sp.oo, sp.oo))

# 2. Compile to a TorchSymPy engine
lt = torchsympy.TorchSymPy()
texpr = lt.torchify(expr)

# 3. Evaluate massively batched parameter grids on accelerators
p_grid = torch.linspace(0.5, 100.0, 10000, dtype=torch.float64, device="cuda").unsqueeze(-1)
re, im = texpr.torch_integrate_batched(
    params_values=p_grid,
    method="gauss-legendre",
    N=501,                       # Quadrature nodes
    device="cuda",               # Target accelerator
    dtype=torch.float64,         
    chunk_size_params=4096       # Safely chunk massive batches to avoid OOM
)

print(f"Real part shape: {re.shape}") # Output: torch.Size([10000])
```

## Core Concepts: Integration Methods

Once you compile an expression, `TorchSymPy` provides three execution paths depending on your memory and scaling constraints:

### 1. Batched Path: `torch_integrate_batched()` (Recommended)
This is the primary workhorse for large parameter sweeps. It automatically handles shape broadcasting, batches execution in chunks to prevent Out-Of-Memory (OOM) errors, and manages device placement. It is the safest and most structured way to evaluate dense multidimensional grids.

### 2. Vectorized Path: `torchquad_integrate_vectorized()`
This is the raw, broadcasting-first path. It passes unstructured parameter tensors directly into the integrand. You are completely responsible for ensuring that the parameter grids broadcast correctly against the spatial integration domain. While riskier for OOM errors, it can yield slightly higher throughput on specific architectures by eliminating chunking overhead.

### 3. Loop-Driven Path: `torchquad_integrate()`
This is a simpler, unbatched evaluation method. Instead of projecting the entire parameter space onto the GPU at once, it accepts a simple 1D array of parameter combinations and internally loops through them. Use this when memory is severely constrained, or when you only need to evaluate a handful of distinct parameter points rather than a massive grid sweep.

## Benchmarks: Speed Gains & Accuracy vs. SciPy & SymPy

TorchSymPy evaluates parameterized integrals across vast grids immensely faster than traditional methods. In our benchmark suite evaluating a parameterized Damped Cosine $\int_{0}^{\infty} e^{-x} \cos(k x) dx$, we observe huge multi-order speedups on GPUs. 

The following table demonstrates the inherent trade-off between quadrature resolution ($N$) and accuracy/speed:

| Execution | Time per Point | Speedup vs SciPy | Accuracy (vs Analytical) |
| :--- | :---: | :---: | :---: | 
| **SciPy (nquad)** | 1.664 ms | 1.0x | $\sim 2.90 \times 10^{-9}$ |
| **TorchSymPy (Vectorized, N=121)** | 0.00048 ms | **3,467x** | $\sim 2.07 \times 10^{-1}$ (Low N) |
| **TorchSymPy (Batched, N=121)** | 0.00073 ms | **2,279x** | $\sim 2.07 \times 10^{-1}$ (Low N) |
| **TorchSymPy (Vectorized, N=2001)** | 0.00854 ms | **194x** | $\sim 5.72 \times 10^{-5}$ (Medium N) |
| **TorchSymPy (Batched, N=2001)** | 0.05164 ms | **32x** | $\sim 5.72 \times 10^{-5}$ (Medium N) |
| **TorchSymPy (Vectorized, N=5001)** | 0.02589 ms | **64x** | $\sim 2.90 \times 10^{-9}$ (High N) |
| **TorchSymPy (Batched, N=5001)** | 0.55701 ms | **3.0x** | $\sim 2.90 \times 10^{-9}$ (High N) |

*(Benchmarks run on an NVIDIA RTX GPU across a 10,000 parameter grid. `TorchSymPy` converges to parity with SciPy while remaining orders of magnitude faster at standard resolutions).*

### The "Hard Integrals" Problem (Experimental Analytical Check)

While `TorchSymPy` achieves numeric parity with `SciPy` for well-behaved integrals (like $\int x^{-x} dx$), evaluating conditionally convergent oscillatory integrals over infinite domains numerically pushes *all* quadrature engines to their breaking points. 

Consider the famously difficult oscillatory integral:
$$ \int_0^\infty \frac{\sin(x)}{\sqrt{x^2 + 1}} dx $$

The true, analytical exact value (calculated symbolically via SymPy hypergeometric functions) is `0.873084`. However, if we force pure numerical evaluation without symbolic reduction:

| Method | Output Value | Absolute Error | Notes |
| :--- | :---: | :---: | :--- |
| **SymPy (True Analytical)** | `0.873084` | **0.0** | Solved symbolically via Hypergeometric functions |
| **SymPy (Pure `evalf()`)** | `-4.000000` | `4.873` | Completely fails convergence natively |
| **SciPy (`nquad`)** | `1.550175` | `0.677` | Fails with `IntegrationWarning` (Divergent) |
| **TorchSymPy (`GaussLegendre`)** | `-1.343219` | `2.216` | Breaks due to mapped infinite oscillations |

**Takeaway:** `TorchSymPy` provides incredible performance scaling and accurate results matching `SciPy` on standard mapping domains. However, for pathological integrands (like conditionally convergent oscillations at infinity), you should rely on `SymPy`'s exact symbolic analytical integrations *before* attempting numerical grid sweeps.

## Running the Test Suite

```bash
pytest tests/ -v
```

## Examples & Tutorials

Check the [`examples/`](examples/) directory for specific physics applications and basic integration usage, including generating Wigner functions.

## License
Distributed under the MIT License. See `LICENSE` for more information.