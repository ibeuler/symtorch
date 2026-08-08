from __future__ import annotations

import pytest


def test_gaussian_infinite_torchquad_accuracy(torchsympy_instance, sp, torch, device, dtype, tol):
    torchquad = pytest.importorskip("torchquad")
    assert torchquad is not None

    x = sp.Symbol("x", real=True)
    integral = sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo))

    texpr = torchsympy_instance.torchify(integral)
    re, im = texpr.torchquad_integrate(N=121, dtype=dtype)

    expected = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))
    re_t = torch.as_tensor(re, device=device, dtype=dtype)
    im_t = torch.as_tensor(im, device=device, dtype=dtype)

    assert im_t.abs().item() <= tol["atol"]
    assert torch.allclose(re_t, expected, atol=tol["atol"], rtol=tol["rtol"])


def test_gaussian_infinite_batched_simpson_accuracy(torchsympy_instance, sp, torch, device, dtype, tol):
    x = sp.Symbol("x", real=True)
    integral = sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo))

    texpr = torchsympy_instance.torchify(integral)
    re, im = texpr.torch_integrate_batched(
        params_values=None,
        method="simpson",
        N=401,
        device=device,
        dtype=dtype,
        chunk_size_params=32,
        chunk_size_points=None,
    )

    expected = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))
    assert im.abs().item() <= tol["atol"]
    assert torch.allclose(re, expected, atol=tol["atol"], rtol=tol["rtol"])


def test_fourier_gaussian_parameter_grid(torchsympy_instance, sp, torch, device, dtype, tol):
    x = sp.Symbol("x", real=True)
    k = sp.Symbol("k", real=True)

    integrand = sp.exp(-x**2) * sp.exp(sp.I * k * x)
    integral = sp.Integral(integrand, (x, -sp.oo, sp.oo))

    texpr = torchsympy_instance.torchify(integral)

    k_grid = torch.linspace(-2.0, 2.0, 7, device=device, dtype=dtype).unsqueeze(-1)  # (7, 1)
    re, im = texpr.torch_integrate_batched(
        params_values=k_grid,
        method="gauss-legendre",
        N=80,
        device=device,
        dtype=dtype,
        chunk_size_params=4,
        chunk_size_points=None,
    )

    expected = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype)) * torch.exp(-(k_grid.squeeze(-1) ** 2) / 4.0)

    assert torch.allclose(im, torch.zeros_like(im), atol=tol["atol"], rtol=tol["rtol"])
    assert torch.allclose(re, expected, atol=2 * tol["atol"], rtol=2 * tol["rtol"])

@pytest.mark.parametrize(
    "integrand_expr, limits, expected_val, check_scipy",
    [
        (lambda x, sp: x**(-x), lambda sp: (0, 1), 1.2912859970626, True),
        (lambda x, sp: x**x, lambda sp: (0, 1), 0.7834305107121, True),
        (lambda x, sp: x**(-x), lambda sp: (0, sp.oo), 1.9954559575001, True),
        (lambda x, sp: sp.exp(-x) / (x + 1), lambda sp: (0, sp.oo), 0.5963473623231, True),
        (lambda x, sp: sp.sin(x) / sp.sqrt(x**2 + 1), lambda sp: (0, sp.oo), 3.0499366168233, False),
    ],
)
def test_hard_integrals_from_notebook(
    torchsympy_instance, sp, torch, device, dtype, integrand_expr, limits, expected_val, check_scipy
):
    import scipy.integrate as spi
    import numpy as np
    import warnings
    from scipy.integrate import IntegrationWarning

    x = sp.Symbol("x", real=True)
    integrand = integrand_expr(x, sp)
    lim = limits(sp)
    integral = sp.Integral(integrand, (x, lim[0], lim[1]))

    texpr = torchsympy_instance.torchify(integral)
    
    re, im = texpr.torch_integrate_batched(
        params_values=None,
        method="gauss-legendre",
        N=2001,
        device=device,
        dtype=dtype,
    )

    # 1. Check against the known deterministic regression value from the notebook
    expected_tensor = torch.tensor(expected_val, device=device, dtype=dtype)
    assert im.abs().item() <= 1e-5
    assert torch.allclose(re, expected_tensor, atol=1e-5, rtol=1e-5)

    # 2. For the well-behaved integrals, also cross-check against SciPy
    if check_scipy:
        func = sp.lambdify([x], integrand, modules=["numpy"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            scipy_lim = [(
                float(lim[0]) if lim[0] != sp.oo and lim[0] != -sp.oo else (np.inf if lim[0] == sp.oo else -np.inf),
                float(lim[1]) if lim[1] != sp.oo and lim[1] != -sp.oo else (np.inf if lim[1] == sp.oo else -np.inf),
            )]
            scipy_result, _ = spi.nquad(func, scipy_lim)
            
        scipy_tensor = torch.tensor(scipy_result, device=device, dtype=dtype)
        assert torch.allclose(re, scipy_tensor, atol=1e-5, rtol=1e-5)
