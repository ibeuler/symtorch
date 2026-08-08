import sys
import os
from pathlib import Path
import torch
import sympy as sp
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

import torchsympy
from torchquad import GaussLegendre

torch.set_default_dtype(torch.float64)

def main():
    print("Initializing Wigner function of a cat state...")
    
    # 1. Define symbolic variables
    q, p, x = sp.symbols("q p x", real=True)
    
    # Simple Cat state wave function: Psi(x) = C * (exp(-x^2/2) + exp(-(x-a)^2/2))
    # For simplicity, we just set a=2
    a_val = 2.0
    def psi(x_var):
        return sp.exp(-x_var**2 / 2) + sp.exp(-(x_var - a_val)**2 / 2)
    
    # Wigner function formula: W(q,p) = (1/pi) \int \Psi*(q+x) \Psi(q-x) exp(2 i p x) dx
    integrand = (1 / sp.pi) * psi(q + x) * psi(q - x) * sp.exp(2 * sp.I * p * x)
    
    print("Integrand defined. Compiling to TorchSymPy engine...")
    
    expr = sp.Integral(integrand, (x, -sp.oo, sp.oo))
    
    lt = torchsympy.TorchSymPy()
    texpr = lt.torchify(expr)
    
    print("Compiled successfully. Generating spatial grid...")
    
    # 2. Define spatial grids
    grid_size = 100
    q_vals = torch.linspace(-4, 4, grid_size, device="cpu")
    p_vals = torch.linspace(-4, 4, grid_size, device="cpu")
    
    q_grid, p_grid = torch.meshgrid(q_vals, p_vals, indexing="ij")
    
    q_flat = q_grid.flatten().unsqueeze(-1)
    p_flat = p_grid.flatten().unsqueeze(-1)
    
    # The signature of the compiled expression params matches alphabetical order of free symbols: p, q
    params_tensor = torch.cat([p_flat, q_flat], dim=1)
    
    print(f"Evaluating {grid_size}x{grid_size} = {grid_size**2} grid points on CPU...")
    
    # 3. Evaluate massively batched integral
    re, im = texpr.torch_integrate_batched(
        params_values=params_tensor,
        N=101,  
        method="gauss-legendre",
        device="cpu",
        chunk_size_params=4096
    )
    
    print("Evaluation complete. Plotting...")
    
    # 4. Plot
    W = re.reshape(grid_size, grid_size).numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(W.T, cmap='RdBu', origin='lower', extent=[-4, 4, -4, 4])
    plt.colorbar(label='W(q, p)')
    plt.title("Wigner Function of a Cat State (TorchSymPy)")
    plt.xlabel('Position (q)')
    plt.ylabel('Momentum (p)')
    plt.tight_layout()
    
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_file = os.path.join(plots_dir, "04_wigner_cat_state.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    with torch.no_grad():
        main()
