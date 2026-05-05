from __future__ import annotations

import pytest


def test_gaussian_infinite_torchquad_accuracy(symtorch_instance, sp, torch, device, dtype, tol):
    torchquad = pytest.importorskip("torchquad")
    assert torchquad is not None

    x = sp.Symbol("x", real=True)
    integral = sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo))

    texpr = symtorch_instance.torchify(integral)
    re, im = texpr.torchquad_integrate(N=121, dtype=dtype)

    expected = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))
    re_t = torch.as_tensor(re, device=device, dtype=dtype)
    im_t = torch.as_tensor(im, device=device, dtype=dtype)

    assert im_t.abs().item() <= tol["atol"]
    assert torch.allclose(re_t, expected, atol=tol["atol"], rtol=tol["rtol"])


def test_gaussian_infinite_batched_simpson_accuracy(symtorch_instance, sp, torch, device, dtype, tol):
    x = sp.Symbol("x", real=True)
    integral = sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo))

    texpr = symtorch_instance.torchify(integral)
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


def test_fourier_gaussian_parameter_grid(symtorch_instance, sp, torch, device, dtype, tol):
    x = sp.Symbol("x", real=True)
    k = sp.Symbol("k", real=True)

    integrand = sp.exp(-x**2) * sp.exp(sp.I * k * x)
    integral = sp.Integral(integrand, (x, -sp.oo, sp.oo))

    texpr = symtorch_instance.torchify(integral)

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
