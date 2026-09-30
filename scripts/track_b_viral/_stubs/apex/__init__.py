"""Import-check stub, NOT apex.

LucaVirus's `modeling_lucavirus.py` does:

    try:    from apex.normalization import FusedLayerNorm as _FusedLayerNorm
    except ImportError:  from torch.nn import LayerNorm as LucaVirus1bLayerNorm

The torch fallback is what we want -- FusedLayerNorm is a speed optimisation, numerically
equivalent. But transformers' `check_imports` scans the file STATICALLY and refuses to load it
unless a top-level `apex` is importable.

So this package exists and deliberately provides NO `normalization` submodule: the static check
passes, then `from apex.normalization import ...` raises ModuleNotFoundError (a subclass of
ImportError), the except branch fires, and the model uses torch LayerNorm. Do NOT add a fake
`normalization` here -- that would silently substitute a wrong layer instead of falling back.
"""
__version__ = "0.0.0-stub"
