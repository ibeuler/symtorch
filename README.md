# symtorch (distribution: libsymtorch)

SymPy-to-Torch conversion and numerical integration helpers.

Note: this module was first developed for libphysics (and remains compatible): https://github.com/ferhatpy/libphysics — then it was split out into a standalone library.

## Install (dev)

pip install -e .

## Install (PyPI)

```bash
pip install libsymtorch
```

## Quick usage

```python
import symtorch
import sympy as sp

x = sp.Symbol("x", real=True)
expr = sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo))

lt = symtorch.SymTorch()
texpr = lt.torchify(expr)
re, im = texpr.torchquad_integrate(N=121)
print(re, im)
```
