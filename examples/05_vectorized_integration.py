import sys
from pathlib import Path
import torch
import sympy as sp

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

import torchsympy

torch.set_default_dtype(torch.float64)

def main():
    print("--- TorchSymPy Vectorized Integration Example ---")
    
    # 1. Define symbolic integral: \int_{-1}^{1} (a * x^2 + b) dx
    x, a, b = sp.symbols("x a b", real=True)
    integrand = a * x**2 + b
    expr = sp.Integral(integrand, (x, -1, 1))
    
    # Analytical Result: (2/3)*a + 2*b
    print(f"\nEvaluating: {expr}")
    
    # 2. Compile
    lt = torchsympy.TorchSymPy()
    texpr = lt.torchify(expr)
    
    # 3. Create parameter arrays
    # In vectorized mode, we manually broadcast. 
    # Let's say we have 3 values for 'a' and 4 values for 'b'.
    a_vals = torch.tensor([1.0, 2.0, 3.0]).view(-1, 1)  # shape (3, 1)
    b_vals = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, -1) # shape (1, 4)
    
    # We broadcast them against each other natively in PyTorch:
    # Notice we must also add a dimension for the integration variable 'x' internally.
    # torchquad_integrate_vectorized requires the user to match the shapes.
    # The simplest way is to pass tensors that broadcast to (N_a, N_b, 1)
    
    a_grid = a_vals.unsqueeze(-1) # shape (3, 1, 1)
    b_grid = b_vals.unsqueeze(-1) # shape (1, 4, 1)
    
    print("\nVectorized Evaluation...")
    # Signature matches alphabetical symbols: a, b
    re, im = texpr.torchquad_integrate_vectorized(
        params_values=[a_grid, b_grid],
        method="gauss-legendre",
        N=21
    )
    
    print(f"\nResult shape: {re.shape} -> Note that it returned shape (3, 4) properly!")
    print("Results matrix (a along rows, b along cols):")
    print(re)
    
    # Compare with analytical
    analytical = (2/3)*a_vals + 2*b_vals
    print("\nAnalytical Matrix:")
    print(analytical)
    
    diff = torch.max(torch.abs(re - analytical))
    print(f"\nMax Absolute Error: {diff.item():.2e}")

if __name__ == "__main__":
    with torch.no_grad():
        main()
