"""
Example 06: Batched Differentiation of Arbitrary Functions over Parameter Grids

This example demonstrates how to use the `torch_differentiate` method of a `TorchExpr`
to easily compute multi-dimensional batched derivatives (Jacobians) over a parameter 
grid on the GPU.

The `torch_differentiate` method provides simplicity of usage and completeness for 
users who want to leverage PyTorch's `vmap` and `jacrev` parallel computing capabilities 
without handling the boilerplate of batch dimension unrolling manually.
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import torchsympy
from sympy import symbols, exp

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1. We can differentiate an arbitrary Python/PyTorch function.
    # It does not need to be a SymPy integral expression.
    def custom_model(x, y, alpha, beta):
        return torch.exp(-alpha * x**2 - beta * y**2)

    lt = torchsympy.TorchSymPy()
    
    # We wrap the callable into a TorchExpr. 
    # Suppose it takes 2 variables (x, y) and 2 parameters (alpha, beta).
    # Since we won't integrate it, the domain values don't strictly matter for differentiation,
    # but we must provide a dummy domain of length 2 to indicate dim=2.
    texpr = lt.torchify_callable(
        integrand=custom_model,
        domain=[[-1.0, 1.0], [-1.0, 1.0]], 
        n_params=2
    )

    # 2. Create a multi-dimensional parameter grid
    # Let's create a 10x10 grid for alpha and beta, and evaluate at x=1.0, y=0.5
    alpha_grid = torch.linspace(0.1, 2.0, 10, device=device).unsqueeze(1).expand(10, 10)
    beta_grid = torch.linspace(0.1, 2.0, 10, device=device).unsqueeze(0).expand(10, 10)
    
    x_val = torch.tensor(1.0, device=device)
    y_val = torch.tensor(0.5, device=device)

    # 3. Compute batched differentiation over the grid
    # We want the Jacobian with respect to the parameters alpha and beta (indices 2 and 3).
    print("Computing multi-dimensional batched differentiation over the grid...")
    d_alpha, d_beta = texpr.torch_differentiate(
        domain_points=[x_val, y_val],
        params_values=[alpha_grid, beta_grid],
        argnums=(2, 3) # Differentiate w.r.t alpha (idx 2) and beta (idx 3)
    )

    print(f"Jacobian w.r.t alpha shape: {d_alpha.shape}") # Expected: (10, 10)
    print(f"Jacobian w.r.t beta shape: {d_beta.shape}\n") # Expected: (10, 10)

    # Let's inspect a specific point on the grid (e.g. index 5, 5)
    print(f"At alpha={alpha_grid[5,5]:.2f}, beta={beta_grid[5,5]:.2f}:")
    print(f"d(model)/d(alpha) = {d_alpha[5,5]:.4f}")
    print(f"d(model)/d(beta)  = {d_beta[5,5]:.4f}")

if __name__ == "__main__":
    main()
