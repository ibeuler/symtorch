from __future__ import annotations

import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class UltimateConfig:
    N: int                  # Quadrature nodes (impacts accuracy and GPU work)
    grid_points: int        # Total parameters to evaluate (impacts batching efficiency)
    chunk_params: int       # Batching chunk size for memory control
    k_min: float            # Parameter range lower bound
    k_max: float            # Parameter range upper bound (high k = high oscillation)


@dataclass
class UltimateCase:
    name: str
    expr: Any
    variables: list
    limits: list
    params: list
    ref_fn: Any
    scipy_re: Any
    scipy_im: Any


def _perf_best_of(func, repeats: int = 3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        t = time.perf_counter() - t0
        best = min(best, t)
    return best


def test_ultimate_universal_benchmark(symtorch_instance, sp, torch, device, tmp_path):
    """
    The Ultimate Universal Benchmark Suite for SymTorch.
    
    Reads extremely simple and iterates over multiple challenging mathematical regimes:
      1) Gaussian (Decay, infinite domain)
      2) Fourier-Gaussian (Oscillatory, infinite domain)
      3) Lorentzian / Cosine (High oscillatory, semi-infinite domain)
    
    Sweeps over different batched settings (Grid sizes) and quadrature resolutions (N)
    to find the "best case speed per point".
    """
    dtype = torch.float64
    scipy = pytest.importorskip("scipy")
    import numpy as np
    from scipy import integrate as scipy_integrate

    # -------------------------------------------------------------
    # 1. Define Universal Cases
    # -------------------------------------------------------------
    x = sp.Symbol("x", real=True)
    p = sp.Symbol("p", real=True)

    # Note: lambdify functions for SciPy
    gauss_np = sp.lambdify((x, p), sp.exp(-p * x**2), modules="numpy")
    fourier_np = sp.lambdify((x, p), sp.exp(-x**2) * sp.exp(sp.I * p * x), modules="numpy")
    lorent_np = sp.lambdify((x, p), sp.exp(-x) * sp.cos(p * x), modules="numpy")

    cases = [
        # Case 1: Standard Infinite Domain
        UltimateCase(
            name="1_Gaussian",
            expr=sp.exp(-p * x**2),
            variables=[x],
            limits=[(x, -sp.oo, sp.oo)],
            params=[p],
            ref_fn=lambda p_tensor: torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype) / p_tensor),
            scipy_re=lambda x_val, p_val: gauss_np(x_val, p_val),
            scipy_im=lambda x_val, p_val: 0.0,
        ),
        # Case 2: Oscillatory Infinite Domain
        UltimateCase(
            name="2_Fourier_Gaussian",
            expr=sp.exp(-x**2) * sp.exp(sp.I * p * x),
            variables=[x],
            limits=[(x, -sp.oo, sp.oo)],
            params=[p],
            ref_fn=lambda p_tensor: torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype)) * torch.exp(-(p_tensor**2)/4.0),
            scipy_re=lambda x_val, p_val: np.real(fourier_np(x_val, p_val)),
            scipy_im=lambda x_val, p_val: np.imag(fourier_np(x_val, p_val)),
        ),
        # Case 3: High Oscillatory Semi-Infinite Domain
        UltimateCase(
            name="3_Damped_Cosine",
            expr=sp.exp(-x) * sp.cos(p * x),
            variables=[x],
            limits=[(x, 0, sp.oo)],
            params=[p],
            ref_fn=lambda p_tensor: 1.0 / (1.0 + p_tensor**2),
            scipy_re=lambda x_val, p_val: lorent_np(x_val, p_val),
            scipy_im=lambda x_val, p_val: 0.0,
        ),
    ]

    # -------------------------------------------------------------
    # 2. Define Universal Configurations to Sweep
    # -------------------------------------------------------------
    # We sweep (Grid Points, N) combinations to see scaling behavior.
    configs = [
        UltimateConfig(N=121, grid_points=10000, chunk_params=4096, k_min=0.5, k_max=100.0),
        UltimateConfig(N=301, grid_points=10000, chunk_params=4096, k_min=0.5, k_max=100.0),
        UltimateConfig(N=501, grid_points=10000, chunk_params=4096, k_min=0.5, k_max=100.0),
        UltimateConfig(N=1001, grid_points=10000, chunk_params=4096, k_min=0.5, k_max=100.0),
        UltimateConfig(N=2001, grid_points=10000, chunk_params=4096, k_min=0.5, k_max=100.0),
    ]

    method = "gauss-legendre"
    scipy_limit = 1000

    results = []

    with torch.no_grad():
        for case in cases:
            # 1. Compile Integrand once per case
            integral = sp.Integral(case.expr, *case.limits)
            texpr = symtorch_instance.torchify(integral)
            
            for cfg in configs:
                # 2. Setup grid and analytical answers
                p_grid = torch.linspace(cfg.k_min, cfg.k_max, cfg.grid_points, device=device, dtype=dtype).unsqueeze(-1)
                ref_tensor = case.ref_fn(p_grid.squeeze(-1))

                # 3. Warm-up GPU cache
                _ = texpr.torch_integrate_batched(
                    params_values=p_grid[:2],
                    method=method, N=cfg.N, device=device, dtype=dtype, chunk_size_params=2
                )

                # 4. Measure SymTorch Batched Speed
                def _run_symtorch():
                    return texpr.torch_integrate_batched(
                        params_values=p_grid,
                        method=method,
                        N=cfg.N,
                        device=device,
                        dtype=dtype,
                        chunk_size_params=cfg.chunk_params,
                    )
                
                t_symtorch = _perf_best_of(_run_symtorch, repeats=3)
                re_sym, im_sym = _run_symtorch()
                
                err_sym = float((re_sym - ref_tensor).abs().max().item())
                symtorch_ms_per_point = (t_symtorch / float(cfg.grid_points)) * 1000.0

                # 5. Measure SciPy baseline (sub-sampled for extreme grids to save patience)
                scipy_subset_points = min(cfg.grid_points, 20)
                sub_indices = np.linspace(0, cfg.grid_points - 1, scipy_subset_points, dtype=int)
                p_cpu = p_grid.squeeze(-1).detach().cpu().numpy()[sub_indices]
                ref_cpu = ref_tensor.detach().cpu().numpy()[sub_indices]
                
                scipy_res = []
                import warnings
                from scipy.integrate import IntegrationWarning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", IntegrationWarning)
                    for p_val in p_cpu:
                        re_val, _ = scipy_integrate.quad(case.scipy_re, case.limits[0][1], case.limits[0][2], args=(float(p_val),), limit=scipy_limit)
                        im_val, _ = scipy_integrate.quad(case.scipy_im, case.limits[0][1], case.limits[0][2], args=(float(p_val),), limit=scipy_limit)
                        scipy_res.append(re_val + 1j * im_val if np.abs(im_val) > 1e-15 else re_val)
                err_scipy = float(np.abs(np.array(scipy_res) - ref_cpu).max())
                
                # SciPy integration mapping
                def _run_scipy_subset():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", IntegrationWarning)
                        for p_val in p_cpu:
                            scipy_integrate.quad(case.scipy_re, case.limits[0][1], case.limits[0][2], args=(float(p_val),), limit=scipy_limit)
                            scipy_integrate.quad(case.scipy_im, case.limits[0][1], case.limits[0][2], args=(float(p_val),), limit=scipy_limit)

                t_scipy_subset = _perf_best_of(_run_scipy_subset, repeats=1)
                scipy_ms_per_point = (t_scipy_subset / float(scipy_subset_points)) * 1000.0

                # 6. Verdict Calculation
                speedup = scipy_ms_per_point / symtorch_ms_per_point if symtorch_ms_per_point > 0 else 0.0

                results.append({
                    "case": case.name,
                    "grid": cfg.grid_points,
                    "N": cfg.N,
                    "symtorch_ms": symtorch_ms_per_point,
                    "scipy_ms": scipy_ms_per_point,
                    "speedup": speedup,
                    "err_sym": err_sym,
                    "err_scipy": err_scipy
                })

    # Save format processing
    project_root = Path(__file__).resolve().parents[1]
    bench_dir = project_root / "tests" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Output CSV
    import csv
    with open(bench_dir / "ultimate_universal_benchmark.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Case Name", "Grid", "N", "SymTorch ms/pt", "SciPy ms/pt", "Speedup", "SymTorch Error", "SciPy Error"])
        for r in results:
            writer.writerow([r["case"], r["grid"], r["N"], f"{r['symtorch_ms']:.5f}", f"{r['scipy_ms']:.5f}", f"{r['speedup']:.1f}", f"{r['err_sym']:.2e}", f"{r['err_scipy']:.2e}"])

    # 2. Output LaTeX Table
    case_latex = {
        "1_Gaussian": "\\makecell{1\\_Gaussian: \\\\ $\\bigint_{-\\infty}^{\\infty} e^{-p x^2}\\,dx = \\sqrt{\\frac{\\pi}{p}}$}",
        "2_Fourier_Gaussian": "\\makecell{2\\_Fourier\\_Gaussian: \\\\ $\\bigint_{-\\infty}^{\\infty} e^{-x^2}e^{ipx}\\,dx$ \\\\ $= \\sqrt{\\pi}\\,e^{-p^2/4}$}",
        "3_Damped_Cosine": "\\makecell{3\\_Damped\\_Cosine: \\\\ $\\bigint_{0}^{\\infty} e^{-x}\\cos(px)\\,dx$ \\\\ $= \\frac{1}{1+p^2}$}",
    }

    tex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l c c c c c c c}",
        "\\toprule",
        "\\textbf{Case} & \\textbf{Grid} & \\textbf{N} & \\textbf{SymTorch (ms)} & \\textbf{SciPy (ms)} & \\textbf{Speedup} & \\textbf{SymTorch Err} & \\textbf{SciPy Err} \\\\",
        "\\midrule"
    ]
    # Group results by case for multirow formatting
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for r in results:
        grouped_results[r['case']].append(r)
        
    for i, (cname, rlist) in enumerate(grouped_results.items()):
        safe_case = case_latex.get(cname, cname.replace('_', '\\_'))
        num_rows = len(rlist)
        for j, r in enumerate(rlist):
            case_str = f"\\multirow{{{num_rows}}}{{*}}{{{safe_case}}}" if j == 0 else ""
            tex_lines.append(f"{case_str} & {r['grid']} & {r['N']} & {r['symtorch_ms']:.5f} & {r['scipy_ms']:.5f} & {r['speedup']:.1f}$\\times$ & {r['err_sym']:.2e} & {r['err_scipy']:.2e} \\\\")
        
        if i < len(grouped_results) - 1:
            tex_lines.append("\\hline")

    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{Universal Benchmark Results: SymTorch vs SciPy (ms/point)}",
        "\\label{tab:universal_benchmark}",
        "\\end{table}"
    ])
    (bench_dir / "ultimate_universal_benchmark.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    # 3. Output Markdown (for easy console read and GitHub)
    md_lines = ["| Case Name | Grid | N | SymTorch ms/pt | SciPy ms/pt | Speedup | SymTorch Err | SciPy Err |", "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"]
    for r in results:
        md_lines.append(f"| {r['case']} | {r['grid']} | {r['N']} | `{r['symtorch_ms']:.5f}` | `{r['scipy_ms']:.5f}` | **{r['speedup']:.1f}x** | `{r['err_sym']:.2e}` | `{r['err_scipy']:.2e}` |")
        
    md_text = "\n".join(md_lines)
    (bench_dir / "ultimate_universal_benchmark.md").write_text(md_text, encoding="utf-8")
    
    print("\n# Ultimate Benchmark Complete. Generated .csv, .tex, and .md files.")
    print(md_text)
