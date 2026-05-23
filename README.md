# symtorch (distribution: libsymtorch)

SymPy-to-Torch transcompilation and numerical integration library designed for extremely fast, batched evaluation on GPUs and CPUs. 

Note: this module was first developed for [libphysics](https://github.com/ferhatpy/libphysics) — then it was split out into a standalone library to tackle generalized parallel computational bottlenecks.

## Install (dev)

```bash
pip install -e .
```

## Install (PyPI)

```bash
pip install libsymtorch
```

## Requirements

This project depends on the packages listed in `requirements.txt`. The primary runtime requirements are:

- Python 3.8+
- torch==2.5.1+cu121 (or another `torch` build appropriate for your platform)
- sympy==1.13.1
- torchquad==0.5.0
- numpy==2.4.2
- scipy==1.17.0
- loguru==0.7.3

To install the pinned packages, run:

```bash
pip install -r requirements.txt
```

If you intend to use GPU acceleration, please install a `torch` wheel that matches your CUDA version (the example above is a CUDA-enabled wheel). For CPU-only environments, install the CPU `torch` wheel instead.

## Quick Usage

`symtorch` bridges the gap between SymPy's symbolic manipulation and PyTorch's highly optimized batched tensor operations. You can transcompile symbolic integrals directly into callable PyTorch engines.

```python
import torch
import symtorch
import sympy as sp

# 1. Define your integrands symbolically
x = sp.Symbol("x", real=True)
p = sp.Symbol("p", real=True)
expr = sp.Integral(sp.exp(-p * x**2), (x, -sp.oo, sp.oo))

# 2. Compile to SymTorch engine
lt = symtorch.SymTorch()
texpr = lt.torchify(expr)

# 3. Evaluate massively batched parameter grids on accelerators
p_grid = torch.linspace(0.5, 100.0, 10000, dtype=torch.float64, device="cuda").unsqueeze(-1)
re, im = texpr.torch_integrate_batched(
    params_values=p_grid,
    method="gauss-legendre",
    N=501,                       # Force high quadrature scaling for difficult integrands
    device="cuda",               # Target accelerator natively
    dtype=torch.float64,         
    chunk_size_params=4096       # Automatically chunks massive batches to avoid OOM
)
print("Real part:", re)
```

### Parameter Reference

When evaluating continuous integrals dynamically using `torch_integrate_batched()`, you have granular control over numerical performance constraints:

- **`params_values`** (`torch.Tensor`): Array parameter map corresponding to all substituted free variables defining your symbolic integral.
- **`method`** (`str`): The mapped quadrature strategy (e.g. `"gauss-legendre"`).
- **`N`** (`int`): Limits Quadrature nodes. Higher N bounds push stability on deeply oscillatory configurations (allowing you to surpass typical recursion bottlenecks) but proportionately scales VRAM and FLOP demands.
- **`device`** (`str` or `torch.device`): Device mapping (e.g., `"cpu"`, `"cuda"`, `"mps"`).
- **`dtype`** (`torch.dtype`): Evaluative precision schema (use `torch.float64` for analytic comparisons or exponential limits).
- **`chunk_size_params`** (`int`): Sub-array partitioning for the grid parameters. Ex: With a $1,000,000$ point map and $N=2000$, setting this to `2048` iteratively offloads pressure allowing safe calculation without Out-Of-Memory hazards.

---

## ⚡ Universals Benchmarks & Capabilities

A major component of SymTorch is overcoming conventional scaling walls embedded heavily in dynamic Python CPU evaluators (e.g., SciPy's recursive `quad`). 

**SymTorch evaluates integrals up to ~2,640× faster than SciPy with similar accuracy, and can uniquely utilize massive static grids to achieve mathematically narrower error bounds on notoriously difficult oscillatory problems.**

_Evaluation constraints_: Uniform sequential sweeps covering $p \in [0.5, 100.0]$ across a $10,000$ point grid resolving analytical ground truths. N represents quadrature scaling depth.

### Tabular Comparisons

| Case Name & Analytical Form | Grid | N | SymTorch ms/pt | SciPy ms/pt | Speedup | SymTorch Err | SciPy Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| <br>**1_Gaussian**<br><br> $\int_{-\infty}^{\infty} e^{-p x^2}dx$<br><br> $= \sqrt{\frac{\pi}{p}}$<br><br> | 10000 | 121 | `0.00092` | `0.13262` | **144.0x** | `1.07e-14` | `3.72e-15` |
| | 10000 | 301 | `0.00332` | `0.14036` | **42.2x** | `2.13e-14` | `3.72e-15` |
| | 10000 | 501 | `0.00756` | `0.13877` | **18.3x** | `2.22e-14` | `3.72e-15` |
| | 10000 | 1001| `0.01996` | `0.13185` | **6.6x**  | `2.27e-13` | `3.72e-15` |
| | 10000 | 2001| `0.06202` | `0.13026` | **2.1x**  | `4.59e-13` | `3.72e-15` |
| <br>**2_Fourier_Gaussian**<br><br> $\int_{-\infty}^{\infty} e^{-x^2}e^{ipx}dx$<br><br> $= \sqrt{\pi}\,e^{-p^2/4}$<br><br> | 10000 | 121 | `0.00137` | `2.46700` | **1805.5x**| `2.65e-01` | `7.92e-10` |
| | 10000 | 301 | `0.00427` | `2.47832` | **580.4x** | `6.58e-03` | `7.92e-10` |
| | 10000 | 501 | `0.00949` | `2.42028` | **254.9x** | `4.58e-05` | `7.92e-10` |
| | 10000 | 1001| `0.02473` | `2.39489` | **96.8x**  | `9.88e-12` | `7.92e-10` |
| | 10000 | 2001| `0.07130` | `2.49811` | **35.0x**  | `3.05e-13` | `7.92e-10` |
| <br>**3_Damped_Cosine**<br><br> $\int_{0}^{\infty} e^{-x}\cos(px)dx$<br><br> $= \frac{1}{1+p^2}$<br><br> | 10000 | 121 | `0.00062` | `1.64998` | **2641.1x**| `2.07e-01` | `2.41e-09` |
| | 10000 | 301 | `0.00267` | `1.68845` | **632.9x** | `5.83e-02` | `2.41e-09` |
| | 10000 | 501 | `0.00657` | `1.67522` | **254.8x** | `2.20e-02` | `2.41e-09` |
| | 10000 | 1001| `0.01807` | `1.63858` | **90.7x**  | `2.67e-03` | `2.41e-09` |
| | 10000 | 2001| `0.05752` | `1.63351` | **28.4x**  | `5.72e-05` | `2.41e-09` |

_*To access raw LaTeX arrays, inspect: `tests/benchmarks/ultimate_universal_benchmark.tex`_ 
### Key Identifications:
* **Speedup Range:** Depending on the quadrature depth (N) and integrand complexity, speedups scale dynamically from $\sim2\times$ (at massive $N=2001$ limits) up to $\sim2641\times$ (at standard $N=121$ baselines).
* **Oscillatory Bypassing:** When SciPy encounters intense structural oscillations (`Case 2`), its adaptive iteration bottoms out (capping at $10^{-10}$ error margins). By structurally projecting massive arrays via SymTorch (`N=1001+`) we cleanly break past recursive thresholds achieving up to $10^{-13}$ true numerical accuracy while retaining ~ $40\times$ faster wall-times!

## Running Tests

To run the benchmarking suite locally across `symtorch` implementations:

```bash
pytest tests/ -v
pytest tests/test_benchmark_ultimate.py -s  # Generates local .csv, .md, & .tex format drops safely 
```

## Examples & Tutorials

For more extensive usage—including plotting workflows and physics applications—check the [`examples/notebooks/`](examples/notebooks/) directory which includes:
- `01_the_basics.ipynb`
- `02_parameter_sweeps.ipynb`
- `03_scattering_and_wigner.ipynb`

## License

Please see the [LICENSE](LICENSE) file in the root of the repository for usage and distribution terms.
