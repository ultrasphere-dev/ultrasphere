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
        rng = xp.random.default_rng()
        def random_uniform(low=0, high=1, shape=None):
            return rng.random(shape) * (high - low) + low
        xp.random.random_uniform = random_uniform
    elif backend == "torch":
        import torch as xp
        def random_uniform(low=0, high=1, shape=None):
            return xp.rand(shape) * (high - low) + low
        xp.random.random_uniform = random_uniform
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return xp
