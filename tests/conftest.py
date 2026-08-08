from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Ensure `import torchsympy` works from a src-layout checkout without installation.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))


@pytest.fixture(scope="session")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="session")
def sp():
    return pytest.importorskip("sympy")


@pytest.fixture(scope="session")
def torchsympy_module():
    import torchsympy

    return torchsympy


@pytest.fixture(scope="session")
def torchsympy_instance(torchsympy_module):
    return torchsympy_module.TorchSymPy()


@pytest.fixture(scope="session")
def device(torch):
    # Tests should be deterministic and not require GPU.
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dtype(torch):
    return torch.float64


@pytest.fixture(scope="session")
def tol():
    # Conservative tolerances for numerical quadrature.
    return {"atol": 5e-3, "rtol": 5e-3}
