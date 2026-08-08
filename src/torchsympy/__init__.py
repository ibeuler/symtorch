"""torchsympy package.

This package is intended to be used as a standalone library:

	import torchsympy
	lt = torchsympy.TorchSymPy()

The implementation lives in :mod:`torchsympy.main` and is re-exported here for
convenient imports.
"""

from .main import (
	TorchSymPy,
	TorchExpr,
)

try:
	from importlib.metadata import version as _pkg_version

	# Distribution name may differ from import package name.
	try:
		__version__ = _pkg_version("torchsympy")
	except Exception:
		__version__ = _pkg_version("torchsympy")
except Exception:
	__version__ = "0.3.0"

__all__ = [
	"TorchSymPy",
	"TorchExpr",
	"__version__",
]
