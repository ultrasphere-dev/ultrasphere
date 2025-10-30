from typing import Any

import pytest
from array_api._2024_12 import ArrayNamespaceFull


@pytest.fixture(
    scope="session", params=[("numpy", "cpu"), ("torch", "cpu"), ("torch", "cuda")]
)
def xp_device(request: pytest.FixtureRequest) -> tuple[ArrayNamespaceFull, Any]:
    """
    Get the array namespace and device for the given backend.
    """
    backend, device = request.param
    if backend == "numpy":
        from array_api_compat import numpy as xp

        rng = xp.random.default_rng()

        def random_uniform(low=0, high=1, shape=None):
            return rng.random(shape) * (high - low) + low

        def integers(low, high=None, shape=None):
            return rng.integers(low, high, size=shape)

        xp.random.random_uniform = random_uniform
        xp.random.integers = integers
    elif backend == "torch":
        import torch
        from array_api_compat import torch as xp

        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

        def random_uniform(low=0, high=1, shape=None):
            return xp.rand(shape, device=device) * (high - low) + low

        def integers(low, high=None, shape=None):
            return xp.randint(low, high, size=shape)

        xp.random.random_uniform = random_uniform
        xp.random.integers = integers
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return xp, device


@pytest.fixture(scope="session")
def xp(xp_device: tuple[ArrayNamespaceFull, Any]) -> ArrayNamespaceFull:
    """
    Get the array namespace for the given backend.
    """
    return xp_device[0]


@pytest.fixture(scope="session")
def device(xp_device: tuple[ArrayNamespaceFull, Any]) -> Any:
    """
    Get the device for the given backend.
    """
    return xp_device[1]
