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


def test_ultimate_universal_benchmark(torchsympy_instance, sp, torch, device, tmp_path):
    """
    The Ultimate Universal Benchmark Suite for TorchSymPy.
    
    Reads extremely simple and iterates over multiple challenging mathematical regimes:
            1) Gaussian (Decay, infinite domain)
            2) Fourier-Gaussian (Oscillatory, infinite domain)
            3) Damped cosine (High oscillatory, semi-infinite domain)
            4) Notebook-style gallery plots for the public examples
    
        Sweeps over different batched and vectorized settings (Grid sizes) and quadrature
        resolutions (N) to find the "best case speed per point".
    """
    dtype = torch.float64
    scipy = pytest.importorskip("scipy")
    import numpy as np
    from scipy import integrate as scipy_integrate
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        UltimateConfig(N=5001, grid_points=10000, chunk_params=4096, k_min=0.5, k_max=100.0),
    ]

    method = "gauss-legendre"
    scipy_limit = 1000

    from torchquad import GaussLegendre

    vectorized_method = GaussLegendre()

    results = []

    with torch.no_grad():
        for case in cases:
            # 1. Compile Integrand once per case
            integral = sp.Integral(case.expr, *case.limits)
            texpr = torchsympy_instance.torchify(integral)
            
            for cfg in configs:
                # 2. Setup grid and analytical answers
                p_grid = torch.linspace(cfg.k_min, cfg.k_max, cfg.grid_points, device=device, dtype=dtype).unsqueeze(-1)
                ref_tensor = case.ref_fn(p_grid.squeeze(-1))

                # 3. Warm-up GPU cache
                _ = texpr.torchquad_integrate_vectorized(
                    params_values=[p_grid[:2]],
                    method=vectorized_method,
                    N=11,
                )
                _ = texpr.torch_integrate_batched(
                    params_values=p_grid[:2],
                    method=method, N=cfg.N, device=device, dtype=dtype, chunk_size_params=2
                )

                # 4. Measure TorchSymPy Vectorized and Batched Speed
                def _run_vectorized():
                    return texpr.torchquad_integrate_vectorized(
                        params_values=[p_grid],
                        method=vectorized_method,
                        N=cfg.N,
                    )

                def _run_TorchSymPy():
                    return texpr.torch_integrate_batched(
                        params_values=p_grid,
                        method=method,
                        N=cfg.N,
                        device=device,
                        dtype=dtype,
                        chunk_size_params=cfg.chunk_params,
                    )
                
                t_vec = _perf_best_of(_run_vectorized, repeats=3)
                t_TorchSymPy = _perf_best_of(_run_TorchSymPy, repeats=3)
                re_vec = _run_vectorized()
                re_sym, im_sym = _run_TorchSymPy()
                
                err_vec = float((re_vec - ref_tensor).abs().max().item())
                err_sym = float((re_sym - ref_tensor).abs().max().item())
                vectorized_ms_per_point = (t_vec / float(cfg.grid_points)) * 1000.0
                TorchSymPy_ms_per_point = (t_TorchSymPy / float(cfg.grid_points)) * 1000.0

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
                vectorized_speedup = scipy_ms_per_point / vectorized_ms_per_point if vectorized_ms_per_point > 0 else 0.0
                batched_speedup = scipy_ms_per_point / TorchSymPy_ms_per_point if TorchSymPy_ms_per_point > 0 else 0.0

                results.append({
                    "case": case.name,
                    "grid": cfg.grid_points,
                    "N": cfg.N,
                    "vectorized_ms": vectorized_ms_per_point,
                    "TorchSymPy_ms": TorchSymPy_ms_per_point,
                    "scipy_ms": scipy_ms_per_point,
                    "vectorized_speedup": vectorized_speedup,
                    "batched_speedup": batched_speedup,
                    "err_vec": err_vec,
                    "err_sym": err_sym,
                    "err_scipy": err_scipy
                })

    # Save format processing
    project_root = Path(__file__).resolve().parents[1]
    bench_dir = project_root / "tests" / "benchmarks"
    results_dir = bench_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Ultimate Benchmark Results ---")
    print(f"{'Case':<20} | {'Grid':<6} | {'N':<5} | {'Vec ms':<10} | {'Bat ms':<10} | {'SciPy ms':<10} | {'Vec Err':<10} | {'Bat Err':<10} | {'Bat Speedup':<10}")
    for r in results:
        print(f"{r['case']:<20} | {r['grid']:<6} | {r['N']:<5} | {r['vectorized_ms']:10.5f} | {r['TorchSymPy_ms']:10.5f} | {r['scipy_ms']:10.5f} | {r['err_vec']:10.2e} | {r['err_sym']:10.2e} | {r['batched_speedup']:10.1f}x")

    # Output LaTeX Table
    case_latex = {
        "1_Gaussian": "\\makecell{1\\_Gaussian: \\\\ $\\bigint_{-\\infty}^{\\infty} e^{-p x^2}\\,dx = \\sqrt{\\frac{\\pi}{p}}$}",
        "2_Fourier_Gaussian": "\\makecell{2\\_Fourier\\_Gaussian: \\\\ $\\bigint_{-\\infty}^{\\infty} e^{-x^2}e^{ipx}\\,dx$ \\\\ $= \\sqrt{\\pi}\\,e^{-p^2/4}$}",
        "3_Damped_Cosine": "\\makecell{3\\_Damped\\_Cosine: \\\\ $\\bigint_{0}^{\\infty} e^{-x}\\cos(px)\\,dx$ \\\\ $= \\frac{1}{1+p^2}$}",
    }

    from collections import defaultdict
    grouped_results = defaultdict(list)
    for r in results:
        grouped_results[r["case"]].append(r)

    tex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l c c c c c c c c c c}",
        "\\toprule",
    ]
    header_row = (
        "\\textbf{Case} & \\textbf{Grid} & \\textbf{N} & \\textbf{Vectorized (ms)} & "
        "\\textbf{Batched (ms)} & \\textbf{SciPy (ms)} & \\textbf{Vec Speedup} & "
        "\\textbf{Bat Speedup} & \\textbf{Vec Err} & \\textbf{Bat Err} & \\textbf{SciPy Err} "
        + r"\\"
    )
    tex_lines.append(header_row)
    tex_lines.append("\\midrule")

    for index, (cname, rlist) in enumerate(grouped_results.items()):
        safe_case = case_latex.get(cname, cname.replace("_", "\\_"))
        row_count = len(rlist)
        for row_index, r in enumerate(rlist):
            case_cell = f"\\multirow{{{row_count}}}{{*}}{{{safe_case}}}" if row_index == 0 else ""
            tex_lines.append(
                f"{case_cell} & {r['grid']} & {r['N']} & {r['vectorized_ms']:.5f} & {r['TorchSymPy_ms']:.5f} & {r['scipy_ms']:.5f} & {r['vectorized_speedup']:.1f}$\\times$ & {r['batched_speedup']:.1f}$\\times$ & {r['err_vec']:.2e} & {r['err_sym']:.2e} & {r['err_scipy']:.2e} "
                + r"\\"
            )

        if index < len(grouped_results) - 1:
            tex_lines.append("\\hline")

    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{Universal Benchmark Results: Vectorized and Batched TorchSymPy vs SciPy (ms/point)}",
        "\\label{tab:universal_benchmark}",
        "\\end{table}",
    ])
    (results_dir / "ultimate_universal_benchmark.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    # Output Markdown
    md_lines = ["| Case Name | Grid | N | Vectorized ms/pt | Batched ms/pt | SciPy ms/pt | Vectorized Speedup | Batched Speedup | Vectorized Err | Batched Err | SciPy Err |", "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"]
    for r in results:
        md_lines.append(f"| {r['case']} | {r['grid']} | {r['N']} | `{r['vectorized_ms']:.5f}` | `{r['TorchSymPy_ms']:.5f}` | `{r['scipy_ms']:.5f}` | **{r['vectorized_speedup']:.1f}x** | **{r['batched_speedup']:.1f}x** | `{r['err_vec']:.2e}` | `{r['err_sym']:.2e}` | `{r['err_scipy']:.2e}` |")
        
    md_text = "\n".join(md_lines)
    (results_dir / "ultimate_universal_benchmark.md").write_text(md_text, encoding="utf-8")

    # 4. Output graphs for the benchmark suite and notebook-style examples.
    graph_dir = bench_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    grouped_results = defaultdict(list)
    for r in results:
        grouped_results[r["case"]].append(r)

    fig, axes = plt.subplots(len(cases), 1, figsize=(11, 4.2 * len(cases)), sharex=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        case_rows = grouped_results[case.name]
        xs = [r["N"] for r in case_rows]
        ax.plot(xs, [r["vectorized_ms"] for r in case_rows], marker="o", label="Vectorized Gauss-Legendre")
        ax.plot(xs, [r["TorchSymPy_ms"] for r in case_rows], marker="s", label="Batched Gauss-Legendre")
        ax.set_title(case.name.replace("_", " "))
        ax.set_ylabel("ms/point")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("N")
    fig.suptitle("TorchSymPy benchmark runtime comparison")
    fig.tight_layout()
    fig.savefig(graph_dir / "runtime_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(cases), 1, figsize=(11, 4.2 * len(cases)), sharex=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        case_rows = grouped_results[case.name]
        xs = [r["N"] for r in case_rows]
        ax.semilogy(xs, [r["vectorized_err"] if "vectorized_err" in r else r["err_vec"] for r in case_rows], marker="o", label="Vectorized error")
        ax.semilogy(xs, [r["err_sym"] for r in case_rows], marker="s", label="Batched error")
        ax.set_title(case.name.replace("_", " "))
        ax.set_ylabel("max abs error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("N")
    fig.suptitle("TorchSymPy benchmark error comparison")
    fig.tight_layout()
    fig.savefig(graph_dir / "error_comparison.png", dpi=180)
    plt.close(fig)

    x_axis = np.linspace(-4.0, 4.0, 1200)
    gallery_fig, gallery_axes = plt.subplots(2, 2, figsize=(13, 9))
    gallery_axes = gallery_axes.flatten()

    gaussian_curve = np.exp(-x_axis**2)
    gallery_axes[0].plot(x_axis, gaussian_curve, color="#1565c0")
    gallery_axes[0].set_title("Notebook-style Gaussian integrand")
    gallery_axes[0].set_xlabel("x")
    gallery_axes[0].set_ylabel("exp(-x^2)")
    gallery_axes[0].grid(True, alpha=0.2)

    p_val = 1.5
    fourier_numeric = np.exp(-x_axis**2) * np.cos(p_val * x_axis)
    fourier_analytic = np.sqrt(np.pi) * np.exp(-(p_val**2) / 4.0) * np.ones_like(x_axis)
    gallery_axes[1].plot(x_axis, fourier_numeric, label="numeric integrand", color="#ef6c00")
    gallery_axes[1].plot(x_axis, fourier_analytic, label="analytic integral value", linestyle="--", color="#2e7d32")
    gallery_axes[1].set_title("Fourier-Gaussian notebook example")
    gallery_axes[1].set_xlabel("x")
    gallery_axes[1].legend()
    gallery_axes[1].grid(True, alpha=0.2)

    damped_x = np.linspace(0.0, 20.0, 1200)
    damped_curve = np.exp(-damped_x) * np.cos(3.0 * damped_x)
    gallery_axes[2].plot(damped_x, damped_curve, color="#6a1b9a")
    gallery_axes[2].set_title("Damped cosine notebook example")
    gallery_axes[2].set_xlabel("x")
    gallery_axes[2].set_ylabel("e^{-x} cos(3x)")
    gallery_axes[2].grid(True, alpha=0.2)

    tail_x = np.linspace(0.0, 40.0, 1200)
    tail_curve = np.sin(tail_x) / np.sqrt(tail_x**2 + 1.0)
    gallery_axes[3].plot(tail_x, tail_curve, color="#00897b")
    gallery_axes[3].set_title("Struve/Bessel notebook example")
    gallery_axes[3].set_xlabel("x")
    gallery_axes[3].set_ylabel("sin(x)/sqrt(x^2+1)")
    gallery_axes[3].grid(True, alpha=0.2)

    gallery_fig.suptitle("Notebook-style integrals used in the benchmark documentation")
    gallery_fig.tight_layout()
    gallery_fig.savefig(graph_dir / "notebook_examples_gallery.png", dpi=180)
    plt.close(gallery_fig)
    
    print("\n# Ultimate Benchmark Complete. Graphs and tables generated.")
    print(f"# Results written to {results_dir}")


def test_sympy_vs_gauss_legendre_benchmark(torchsympy_instance, sp, torch, device, tmp_path):
    """Benchmark SymPy numeric evaluation against TorchSymPy Gauss-Legendre.

    Uses notebook-style integrals so the README can point to a concrete,
    reproducible comparison between SymPy's numeric evaluation and the
    Gauss-Legendre path supported by both torchify() and torchify_callable().
    """
    dtype = torch.float64
    scipy = pytest.importorskip("scipy")
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from torchquad import GaussLegendre

    x, p = sp.symbols("x p", real=True)
    integrals = [
        ("gaussian", sp.Integral(sp.exp(-p * x**2), (x, -sp.oo, sp.oo)), lambda param: torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype) / param)),
        ("fourier_gaussian", sp.Integral(sp.exp(-x**2) * sp.exp(sp.I * p * x), (x, -sp.oo, sp.oo)), lambda param: torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype)) * torch.exp(-(param**2) / 4.0)),
        ("damped_cosine", sp.Integral(sp.exp(-x) * sp.cos(p * x), (x, 0, sp.oo)), lambda param: 1.0 / (1.0 + param**2)),
    ]

    sample_points = torch.tensor([0.5, 2.0, 10.0], device=device, dtype=dtype)
    results = []

    def _best_time(func, repeats: int = 3):
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            value = func()
            elapsed = time.perf_counter() - t0
            best = min(best, elapsed)
        return best, value

    with torch.no_grad():
        for name, integral, ref_fn in integrals:
            texpr = torchsympy_instance.torchify(integral)

            def _run_TorchSymPy(param_value):
                return texpr.torchquad_integrate_vectorized(
                    params_values=[param_value.unsqueeze(0)],
                    method=GaussLegendre(),
                        N=2001,
                )

            def _run_sympy(param_value):
                numeric_integral = integral.subs(p, float(param_value.item()))
                return numeric_integral.evalf()

            for param_value in sample_points:
                sym_time, sym_result = _best_time(lambda param_value=param_value: _run_sympy(param_value), repeats=3)
                torch_time, torch_result = _best_time(lambda param_value=param_value: _run_TorchSymPy(param_value), repeats=3)

                if ref_fn is not None:
                    reference = ref_fn(param_value)
                else:
                    reference = torch.tensor(float(sym_result), device=device, dtype=dtype)

                results.append({
                    "name": name,
                    "param": float(param_value.item()),
                    "sympy_ms": (sym_time * 1000.0),
                    "gauss_ms": (torch_time * 1000.0),
                    "speedup": (sym_time / torch_time) if torch_time > 0 else 0.0,
                        "sympy_value": float(complex(sym_result).real),
                        "gauss_value": float(complex(torch_result.item()).real),
                    "abs_err": float((torch_result - reference).abs().item()),
                })

    project_root = Path(__file__).resolve().parents[1]
    bench_dir = project_root / "tests" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = bench_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    results_dir = bench_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- SymPy vs Gauss-Legendre Benchmark Results ---")
    for row in results:
        print(f"{row['name']} | Param: {row['param']:.2f} | SymPy ms: {row['sympy_ms']:.5f} | Gauss ms: {row['gauss_ms']:.5f} | Speedup: {row['speedup']:.1f}x | Abs Err: {row['abs_err']:.2e}")

    md_lines = [
        "| Integral | Param | SymPy ms | Gauss-Legendre ms | Speedup | SymPy Value | Gauss-Legendre Value | Abs Error |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in results:
        md_lines.append(
            f"| {row['name']} | {row['param']:.2f} | `{row['sympy_ms']:.5f}` | `{row['gauss_ms']:.5f}` | **{row['speedup']:.1f}x** | `{row['sympy_value']:.8f}` | `{row['gauss_value']:.8f}` | `{row['abs_err']:.2e}` |"
        )
    md_text = "\n".join(md_lines)
    (results_dir / "sympy_gauss_legendre_benchmark.md").write_text(md_text, encoding="utf-8")

    tex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l c c c c c c c}",
        "\\toprule",
    ]
    tex_lines.append(
        "Integral & Param & SymPy (ms) & Gauss-Legendre (ms) & Speedup & SymPy Value & Gauss-Legendre Value & Abs Error " + r"\\"
    )
    tex_lines.append("\\midrule")
    for row in results:
        tex_lines.append(
            f"{row['name']} & {row['param']:.2f} & {row['sympy_ms']:.5f} & {row['gauss_ms']:.5f} & {row['speedup']:.1f}$\\times$ & {row['sympy_value']:.8f} & {row['gauss_value']:.8f} & {row['abs_err']:.2e} " + r"\\"
        )
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{SymPy numeric evaluation vs TorchSymPy Gauss-Legendre on notebook-style integrals}",
        "\\label{tab:sympy_gauss_legendre_benchmark}",
        "\\end{table}",
    ])
    (results_dir / "sympy_gauss_legendre_benchmark.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    grouped = {}
    for row in results:
        grouped.setdefault(row["name"], []).append(row)

    fig, axes = plt.subplots(len(grouped), 1, figsize=(11, 4.0 * len(grouped)), sharex=False)
    if len(grouped) == 1:
        axes = [axes]
    for ax, (name, rows) in zip(axes, grouped.items()):
        xs = [row["param"] for row in rows]
        ax.plot(xs, [row["sympy_ms"] for row in rows], marker="o", label="SymPy evalf")
        ax.plot(xs, [row["gauss_ms"] for row in rows], marker="s", label="Gauss-Legendre")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("ms/evaluation")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("SymPy vs Gauss-Legendre benchmark")
    fig.tight_layout()
    fig.savefig(graph_dir / "sympy_gauss_legendre_runtime.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(grouped), 1, figsize=(11, 4.0 * len(grouped)), sharex=False)
    if len(grouped) == 1:
        axes = [axes]
    for ax, (name, rows) in zip(axes, grouped.items()):
        xs = [row["param"] for row in rows]
        ax.semilogy(xs, [row["abs_err"] for row in rows], marker="o", color="#1565c0")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("absolute error")
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("Gauss-Legendre absolute error against SymPy values")
    fig.tight_layout()
    fig.savefig(graph_dir / "sympy_gauss_legendre_error.png", dpi=180)
    plt.close(fig)

    print("# SymPy vs Gauss-Legendre benchmark complete. Graphs and tables generated.")
