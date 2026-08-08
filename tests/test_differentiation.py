from __future__ import annotations
import pytest

def test_torch_differentiate_single(torchsympy_instance, sp, torch, device, dtype, tol):
    x = sp.Symbol("x", real=True)
    p = sp.Symbol("p", real=True)
    
    # Use finite limits to avoid change of variables masking the integrand
    expr = sp.Integral(sp.exp(-p * x**2), (x, 0, 1))
    texpr = torchsympy_instance.torchify(expr)
    
    x_val = torch.tensor([[0.5]], device=device, dtype=dtype)
    p_val = torch.tensor([[1.5]], device=device, dtype=dtype)
    
    jac = texpr.torch_differentiate(
        domain_points=[x_val],
        params_values=[p_val],
        argnums=1
    )
    
    # df/dp = -x**2 * exp(-p * x**2)
    expected_jac = -(x_val**2) * torch.exp(-p_val * x_val**2)
    assert torch.allclose(jac, expected_jac, atol=tol["atol"], rtol=tol["rtol"])

def test_torch_differentiate_batched(torchsympy_instance, sp, torch, device, dtype, tol):
    x = sp.Symbol("x", real=True)
    p = sp.Symbol("p", real=True)
    
    expr = sp.Integral(sp.exp(-p * x**2), (x, 0, 1))
    texpr = torchsympy_instance.torchify(expr)
    
    x_grid = torch.linspace(0.1, 0.9, 10, device=device, dtype=dtype).unsqueeze(-1)
    p_grid = torch.linspace(0.5, 2.0, 10, device=device, dtype=dtype).unsqueeze(-1)
    
    jac = texpr.torch_differentiate(
        domain_points=[x_grid],
        params_values=[p_grid],
        argnums=1
    )
    
    expected_jac = -(x_grid**2) * torch.exp(-p_grid * x_grid**2)
    assert jac.shape == expected_jac.shape
    assert torch.allclose(jac, expected_jac, atol=tol["atol"], rtol=tol["rtol"])

def test_torch_differentiate_callable(torchsympy_instance, torch, device, dtype, tol):
    def my_func(x, p1, p2):
        return p1 * x + p2 * x**2
    
    texpr = torchsympy_instance.torchify_callable(
        my_func,
        domain=[[0.0, 1.0]],
        n_params=2
    )
    
    x_val = torch.tensor([[2.0]], device=device, dtype=dtype)
    p1_val = torch.tensor([[1.0]], device=device, dtype=dtype)
    p2_val = torch.tensor([[2.0]], device=device, dtype=dtype)
    
    # Differentiate wrt p1 and p2 (argnums 1 and 2)
    jac = texpr.torch_differentiate(
        domain_points=[x_val],
        params_values=[p1_val, p2_val],
        argnums=(1, 2)
    )
    
    # jac will be a tuple of two tensors: df/dp1 and df/dp2
    jac_p1, jac_p2 = jac
    expected_jac_p1 = x_val
    expected_jac_p2 = x_val**2
    assert torch.allclose(jac_p1, expected_jac_p1, atol=tol["atol"], rtol=tol["rtol"])
    assert torch.allclose(jac_p2, expected_jac_p2, atol=tol["atol"], rtol=tol["rtol"])
