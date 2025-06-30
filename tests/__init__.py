import pytest
from array_api._2024_12 import ArrayNamespace


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def xp(backend: str) -> ArrayNamespace:
    """
    Get the array namespace for the given backend.
    """
    if backend == "numpy":
        import numpy as xp
    elif backend == "torch":
        import torch as xp
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return xp