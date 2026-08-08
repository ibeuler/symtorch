from __future__ import annotations

import pytest


def test_simpson_rule_requires_odd_n(torchsympy_module, torch, device, dtype):
    TorchExpr = torchsympy_module.TorchExpr

    with pytest.raises(ValueError):
        TorchExpr._rule_simpson(0.0, 1.0, 2, device=device, dtype=dtype)
    with pytest.raises(ValueError):
        TorchExpr._rule_simpson(0.0, 1.0, 4, device=device, dtype=dtype)


def test_params_formats_and_batch_shapes(torchsympy_instance, sp, torch, device, dtype, tol):
    x = sp.Symbol("x", real=True)
    a = sp.Symbol("a", real=True, positive=True)

    integral = sp.Integral(sp.exp(-a * x**2), (x, -sp.oo, sp.oo))
    texpr = torchsympy_instance.torchify(integral)
    assert texpr.n_params == 1

    # 1) params as tensor (..., n_params)
    a_vals_1d = torch.linspace(0.5, 2.0, 9, device=device, dtype=dtype)
    params_tensor = a_vals_1d.unsqueeze(-1)
    re1, im1 = texpr.torch_integrate_batched(
        params_values=params_tensor,
        method="gauss-legendre",
        N=80,
        device=device,
        dtype=dtype,
        chunk_size_params=3,
    )

    expected1 = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype) / a_vals_1d)
    assert torch.allclose(im1, torch.zeros_like(im1), atol=tol["atol"], rtol=tol["rtol"])
    assert re1.shape == a_vals_1d.shape
    assert torch.allclose(re1, expected1, atol=2 * tol["atol"], rtol=2 * tol["rtol"])

    # 2) params as list of tensors with batch shape
    a_vals_2d = a_vals_1d[:6].reshape(2, 3)
    re2, im2 = texpr.torch_integrate_batched(
        params_values=[a_vals_2d],
        method="gauss-legendre",
        N=80,
        device=device,
        dtype=dtype,
        chunk_size_params=2,
    )

    expected2 = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype) / a_vals_2d)
    assert re2.shape == a_vals_2d.shape
    assert torch.allclose(im2, torch.zeros_like(im2), atol=tol["atol"], rtol=tol["rtol"])
    assert torch.allclose(re2, expected2, atol=2 * tol["atol"], rtol=2 * tol["rtol"])

    # 3) params as list of scalars
    re3, im3 = texpr.torch_integrate_batched(
        params_values=[1.25],
        method="gauss-legendre",
        N=80,
        device=device,
        dtype=dtype,
        chunk_size_params=1,
    )
    expected3 = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype) / torch.tensor(1.25, device=device, dtype=dtype))
    assert im3.abs().item() <= tol["atol"]
    assert torch.allclose(re3, expected3, atol=2 * tol["atol"], rtol=2 * tol["rtol"])
