import pytest
from array_api._2024_12 import ArrayNamespaceFull


@pytest.fixture(scope="session", params=["numpy", "torch"])
def xp(request: pytest.FixtureRequest) -> ArrayNamespaceFull:
    """
    Get the array namespace for the given backend.
    """
    backend = request.param
    if backend == "numpy":
        import numpy as xp
    elif backend == "torch":
        import torch as xp
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return xp
