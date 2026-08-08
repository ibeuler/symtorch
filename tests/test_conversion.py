from __future__ import annotations

import pytest


def test_torchify_plain_expression_returns_callable(torchsympy_instance, sp, torch, device, dtype):
    x = sp.Symbol("x", real=True)
    expr = sp.sin(x) + x**2

    f = torchsympy_instance.torchify(expr, variables=[x])

    x_val = torch.tensor(1.5, device=device, dtype=dtype)
    out = f(x_val)
    assert torch.is_tensor(out)

    expected = torch.sin(x_val) + x_val**2
    torch.testing.assert_close(out, expected)


def test_torchify_integral_infers_variables_and_limits(torchsympy_instance, sp):
    x = sp.Symbol("x", real=True)
    integral = sp.Integral(sp.exp(-x**2), (x, 0, 1))

    texpr = torchsympy_instance.torchify(integral)

    assert texpr.dim == 1
    assert texpr.n_params == 0
    assert texpr.domain == [[0.0, 1.0]]
    assert texpr.variables[0] == x


def test_torchify_infinite_limits_produces_finite_domain(torchsympy_instance, sp):
    x = sp.Symbol("x", real=True)
    integral = sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo))

    texpr = torchsympy_instance.torchify(integral)

    assert texpr.dim == 1
    assert len(texpr.domain) == 1

    a, b = texpr.domain[0]
    assert a < 0.0 and b > 0.0
    assert (b - a) > 1.0
    # Tangent transform yields bounds inside (-pi/2, pi/2)
    assert abs(a) < 1.58
    assert abs(b) < 1.58


def test_torchify_absorbs_scalar_prefactor(torchsympy_instance, sp, torch, device, dtype, tol):
    x = sp.Symbol("x", real=True)
    # Common pattern: prefactor stored outside the Integral as Mul
    expr = sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo)) / (2 * sp.pi)

    texpr = torchsympy_instance.torchify(expr)
    re, im = texpr.torch_integrate_batched(
        params_values=None,
        N=201,
        device=device,
        dtype=dtype,
        chunk_size_params=32,
        chunk_size_points=None,
    )

    expected = torch.sqrt(torch.tensor(torch.pi, dtype=dtype, device=device)) / (2 * torch.pi)
    assert im.abs().item() <= tol["atol"]
    assert torch.isfinite(re)
    assert torch.allclose(re, expected, atol=tol["atol"], rtol=tol["rtol"])


def test_vectorized_and_batched_gaussian_match(torchsympy_instance, sp, torch, device, dtype, tol):
    from torchquad import GaussLegendre

    x, p = sp.symbols("x p", real=True)
    integral = sp.Integral(sp.exp(-p * x**2), (x, -sp.oo, sp.oo))

    texpr = torchsympy_instance.torchify(integral)
    p_grid = torch.tensor([0.5, 2.0, 10.0], device=device, dtype=dtype).unsqueeze(-1)

    batched_re, batched_im = texpr.torch_integrate_batched(
        params_values=p_grid,
        method="gauss-legendre",
        N=201,
        device=device,
        dtype=dtype,
        chunk_size_params=32,
    )
    vectorized_re = texpr.torchquad_integrate_vectorized(
        params_values=[p_grid],
        method=GaussLegendre(),
        N=201,
    )

    expected = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype) / p_grid.squeeze(-1))
    assert batched_im.abs().max().item() <= tol["atol"]
    assert vectorized_re.shape == expected.shape
    torch.testing.assert_close(batched_re, expected, atol=tol["atol"], rtol=tol["rtol"])
    torch.testing.assert_close(vectorized_re, expected, atol=tol["atol"], rtol=tol["rtol"])
