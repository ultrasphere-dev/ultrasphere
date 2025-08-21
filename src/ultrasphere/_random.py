from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np
from array_api._2024_12 import Array, ArrayNamespaceFull
from numpy._typing import NDArray

from ultrasphere._coordinates import (
    BranchingType,
    SphericalCoordinates,
    TEuclidean,
    TSpherical,
)


def _random_sphere(
    shape: Sequence[int],
    dim: int,
    *,
    surface: bool = False,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    r"""
    Generate random points in a unit ball / sphere.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the output array.
    dim : int
        The dimension of the hypersphere.
    surface : bool, optional
        Whether to generate points on the surface of the hypersphere,
        by default False.
    rng : np.random.Generator | None, optional
        The random number generator, by default None.

    Returns
    -------
    NDArray[np.float64]
        The generated points.

    References
    ----------
        Barthe, F., Guedon, O., Mendelson, S., & Naor, A. (2005).
        A probabilistic approach to the geometry of the \ell_p^n-ball.
        arXiv, math/0503650. Retrieved from https://arxiv.org/abs/math/0503650v1

    """
    rng = np.random.default_rng() if rng is None else rng
    g = rng.normal(loc=0, scale=np.sqrt(1 / 2), size=(dim, *shape))
    if surface:
        result = g / np.linalg.vector_norm(g, axis=0)
    else:
        z = rng.exponential(scale=1, size=shape)[None, ...]
        result = g / np.sqrt(np.sum(g**2, axis=0, keepdims=True) + z)
    return np.nan_to_num(result, nan=0.0)  # not sure if nan_to_num is necessary


@overload
def random_ball(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    shape: Sequence[int],
    xp: ArrayNamespaceFull,
    device: Any | None = ...,
    dtype: Any | None = ...,
    type: Literal["uniform"] = ...,
    rng: np.random.Generator | None = ...,
    surface: bool = ...,
) -> Array: ...
@overload
def random_ball(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    shape: Sequence[int],
    xp: ArrayNamespaceFull,
    device: Any | None = ...,
    dtype: Any | None = ...,
    type: Literal["spherical"] = ...,
    rng: np.random.Generator | None = ...,
    surface: Literal[False] = ...,
) -> Mapping[TSpherical | Literal["r"], Array]: ...
@overload
def random_ball(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    shape: Sequence[int],
    xp: ArrayNamespaceFull,
    device: Any | None = ...,
    dtype: Any | None = ...,
    type: Literal["spherical"] = ...,
    rng: np.random.Generator | None = ...,
    surface: Literal[True] = ...,
) -> Mapping[TSpherical, Array]: ...
def random_ball(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    shape: Sequence[int],
    xp: ArrayNamespaceFull,
    device: Any | None = None,
    dtype: Any | None = None,
    type: Literal["uniform", "spherical"] = "uniform",
    rng: np.random.Generator | None = None,
    surface: bool = False,
) -> Array | Mapping[TSpherical | Literal["r"], Array] | Mapping[TSpherical, Array]:
    r"""
    Generate random points in the unit ball / sphere.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the random points.
    device : Any | None, optional
        The device, by default None
    dtype : Any | None, optional
        The dtype, by default None
    type : Literal["uniform", "spherical"], optional
        The type of the random points, by default "uniform"
        If "uniform", the random points are uniformly distributed on the sphere.
        If "spherical", each spherical coordinate (and radius
        if surface=False) is uniformly distributed.
    rng : np.random.Generator | None, optional
        The random number generator, by default None
    surface : bool, optional
        Whether to generate random points on the surface of the sphere
        or inside the sphere, by default False

    Returns
    -------
    Mapping[TSpherical, Array] | Array | Mapping[TSpherical | Literal["r"], Array]
        The random points.

    References
    ----------
        Barthe, F., Guedon, O., Mendelson, S., & Naor, A. (2005).
        A probabilistic approach to the geometry of the \ell_p^n-ball.
        arXiv, math/0503650. Retrieved from https://arxiv.org/abs/math/0503650v1

    """
    rng = np.random.default_rng() if rng is None else rng
    if type == "uniform":
        return xp.asarray(
            _random_sphere(shape, dim=c.e_ndim, surface=surface, rng=rng),
            device=device,
            dtype=dtype,
        )
    elif type == "spherical":
        d = {
            BranchingType.A: (0, 2 * xp.pi),
            BranchingType.B: (0, xp.pi),
            BranchingType.BP: (-xp.pi / 2, xp.pi / 2),
            BranchingType.C: (0, xp.pi / 2),
        }
        result: dict[TSpherical | Literal["r"], Array] = {}
        for node in c.s_nodes:
            low, high = d[c.branching_types[node]]
            result[node] = xp.asarray(
                rng.uniform(low=low, high=high, size=shape), device=device, dtype=dtype
            )
        if not surface:
            result["r"] = xp.asarray(
                rng.uniform(low=0, high=1, size=shape), device=device, dtype=dtype
            )
        return result
    else:
        raise ValueError(f"Invalid type {type}.")
