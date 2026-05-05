# -*- coding: utf-8 -*-
"""
benchmark_gpu_vs_scipy.py

A performance showcase for the `symtorch` library.
Calculates the highly oscillatory Wigner Function of a Schrödinger Cat State.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import scipy.integrate as integrate
import sympy as sp
import torch
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

import symtorch

torch.set_default_dtype(torch.float64)  # Use double precision for better accuracy in integration


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark symtorch on a Wigner-function integral (cat state).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for symtorch evaluation.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg).strip().lower()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = _parse_args()

    print("==================================================")
    print("symtorch Benchmark: Wigner Function of a Cat State")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 1. Define the Physics (SymPy)
    # ---------------------------------------------------------
    print("[1/5] Defining symbolic expressions...")
    x, p, y = sp.symbols('x p y', real=True)
    
    # Separation of the cat state wave packets
    x0 = 2.0 
    
    # Cat state wavefunction (unnormalized for simplicity)
    # psi(x) = exp(-(x - x0)^2) + exp(-(x + x0)^2)
    psi_minus = sp.exp(-(x - y - x0)**2) + sp.exp(-(x - y + x0)**2)
    psi_plus  = sp.exp(-(x + y - x0)**2) + sp.exp(-(x + y + x0)**2)
    
    # Wigner integrand: (1/pi) * psi*(x-y) * psi(x+y) * exp(2*i*p*y)
    integrand = (1 / sp.pi) * psi_minus * psi_plus * sp.exp(sp.I * 2 * p * y)
    wigner_integral = sp.Integral(integrand, (y, -sp.oo, sp.oo))

    # ---------------------------------------------------------
    # 2. Setup SciPy Competitor
    # ---------------------------------------------------------
    print("[2/5] Preparing SciPy functions...")
    # SciPy quad doesn't handle complex numbers natively, so we split it
    integrand_np = sp.lambdify((y, x, p), integrand, 'numpy')
    
    def scipy_re(y_val, x_val, p_val):
        return np.real(integrand_np(y_val, x_val, p_val))
    
    def scipy_im(y_val, x_val, p_val):
        return np.imag(integrand_np(y_val, x_val, p_val))

    # ---------------------------------------------------------
    # 3. Setup symtorch Champion
    # ---------------------------------------------------------
    print("[3/5] Compiling SymPy to Torch via symtorch...")
    st = symtorch.SymTorch()
    texpr = st.torchify(wigner_integral)

    # ---------------------------------------------------------
    # 4. The SciPy CPU Benchmark (Extrapolated)
    # ---------------------------------------------------------
    grid_size = 400
    total_points = grid_size * grid_size
    print(f"\nTarget Grid: {grid_size}x{grid_size} ({total_points:,} integrals)")
    
    # SciPy takes too long for the full grid, so we sample a 10x10 grid to get an average.
    scipy_sample_size = 10
    print(f"\n--- SCIPY (CPU) ---")
    print(f"Benchmarking SciPy adaptive quadrature on a {scipy_sample_size}x{scipy_sample_size} sample...")
    
    x_sample = np.linspace(-4, 4, scipy_sample_size)
    p_sample = np.linspace(-4, 4, scipy_sample_size)
    
    t0_scipy = time.time()
    for x_val in x_sample:
        for p_val in p_sample:
            # We integrate over a wide domain. Pure oo often causes SciPy to fail on oscillatory functions
            integrate.quad(scipy_re, -10, 10, args=(x_val, p_val), limit=50)
            integrate.quad(scipy_im, -10, 10, args=(x_val, p_val), limit=50)
    t1_scipy = time.time()
    
    scipy_time_per_point = (t1_scipy - t0_scipy) / (scipy_sample_size**2)
    estimated_scipy_total = scipy_time_per_point * total_points
    
    print(f"SciPy time per integral: {scipy_time_per_point * 1000:.2f} ms")
    print(f"Estimated SciPy total time for {total_points:,} points: ~{estimated_scipy_total / 60:.2f} minutes")

    # ---------------------------------------------------------
    # 5. The symtorch Benchmark
    # ---------------------------------------------------------
    print(f"\n--- SYMTORCH ---")
    device = _resolve_device(args.device)
    print(f"Using device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    x_vals = torch.linspace(-4.0, 4.0, grid_size, device=device)
    p_vals = torch.linspace(-4.0, 4.0, grid_size, device=device)
    X, P = torch.meshgrid(x_vals, p_vals, indexing='ij')
    
    # Flatten into (x, p) pairs
    params_grid = torch.stack([X.flatten(), P.flatten()], dim=-1)

    print(f"Running batched integration for {total_points:,} points...")
    
    # Warmup (forces CUDA initialization if on GPU)
    _ = texpr.torch_integrate_batched(params_values=params_grid[:10], N=11, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0_torch = time.time()
    re_batched, _ = texpr.torch_integrate_batched(
        params_values=params_grid,
        N=1001, # High resolution for oscillatory stability
        device=device,
        chunk_size_params=4096 # Tune based on VRAM
    )
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1_torch = time.time()

    torch_total_time = t1_torch - t0_torch
    speedup = estimated_scipy_total / torch_total_time

    print(f"symtorch total time: {torch_total_time:.3f} seconds")
    print(f"Speedup Multiplier: {speedup:,.0f}x FASTER than SciPy")


    # -----------------------------------------------------
    # 6. Plotting the Result
    # -----------------------------------------------------
    print("\n[5/5] Plotting results...")
    W = re_batched.reshape(grid_size, grid_size).cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    # We transpose W so X is on the horizontal axis and P is on the vertical
    plt.imshow(W.T, cmap='RdBu', origin='lower', extent=[-4, 4, -4, 4])
    plt.colorbar(label='W(x, p)')
    plt.title(f"Wigner Function of a Cat State\nComputed in {torch_total_time:.2f}s with symtorch\nEstimated speedup: {speedup:,.0f}x vs SciPy")
    plt.xlabel('Position (x)')
    plt.ylabel('Momentum (p)')
    # reference time for SciPy (extrapolated)
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "Benchmark_wigner_cat_state.png"), dpi=300)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    # Wrap in torch.no_grad() to save massive amounts of memory
    # since we don't need backpropagation for this benchmark.
    with torch.no_grad():
        main()