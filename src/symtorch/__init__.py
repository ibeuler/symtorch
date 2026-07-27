"""symtorch package.

This package is intended to be used as a standalone library:

	import symtorch
	lt = symtorch.SymTorch()

The implementation lives in :mod:`symtorch.main` and is re-exported here for
convenient imports.
"""

from .main import (
	SymTorch,
	TorchExpr,
	torchify,
	torchify_callable,
	torchquad_integrate,
	torch_integrate_batched,
	torchquad_integrate_vectorized,
)

try:
	from importlib.metadata import version as _pkg_version

	# Distribution name may differ from import package name.
	try:
		__version__ = _pkg_version("libsymtorch")
	except Exception:
		__version__ = _pkg_version("symtorch")
except Exception:
	__version__ = "0.0.0"

__all__ = [
	"SymTorch",
	"TorchExpr",
	"torchify",
	"torchify_callable",
	"torchquad_integrate",
	"torch_integrate_batched",
	"torchquad_integrate_vectorized",
	"__version__",
]

