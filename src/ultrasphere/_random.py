from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np
from array_api._2024_12 import Array, ArrayNamespaceFull
from numpy._typing import NDArray

from ultrasphere._coordinates import (
    BranchingType,
    SphericalCoordinates,
    TCartesian,
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
    Barthe, F., Guédon, O., Mendelson, S., & Naor, A. (2005).
    A probabilistic approach to the geometry of the ? P n -ball.
    The Annals of Probability, 33. https://doi.org/10.1214/009117904000000874

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
    c: SphericalCoordinates[TSpherical, TCartesian],
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
    c: SphericalCoordinates[TSpherical, TCartesian],
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
    c: SphericalCoordinates[TSpherical, TCartesian],
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
    c: SphericalCoordinates[TSpherical, TCartesian],
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
    Barthe, F., Guédon, O., Mendelson, S., & Naor, A. (2005).
    A probabilistic approach to the geometry of the ? P n -ball.
    The Annals of Probability, 33. https://doi.org/10.1214/009117904000000874

    Examples
    --------
    >>> from array_api_compat import numpy as xp
    >>> import ultrasphere as us
    >>> c = us.create_standard(4)
    >>> rng = np.random.default_rng(0)
    >>> points_ball = random_ball(c, shape=(2,), xp=xp, rng=rng)
    >>> points_ball
    array([[ 0.04083225, -0.08099814],
           [ 0.20798418,  0.06431795],
           [-0.17396442,  0.22170665],
           [ 0.4234881 ,  0.58068866],
           [-0.22854562, -0.77587443]])
    >>> xp.linalg.vector_norm(points_ball, axis=0)
    array([0.55386242, 0.99951577])
    >>> points_sphere = random_ball(c, shape=(2,), xp=xp, rng=rng, surface=True)
    >>> points_sphere
    array([[-0.85238384, -0.11470655],
           [-0.45676572, -0.38390797],
           [-0.19953179, -0.16582762],
           [ 0.15090863,  0.54656157],
           [-0.04712233,  0.71639984]])
    >>> xp.linalg.vector_norm(points_sphere, axis=0)
    array([1., 1.])

    """
    rng = np.random.default_rng() if rng is None else rng
    if type == "uniform":
        return xp.asarray(
            _random_sphere(shape, dim=c.c_ndim, surface=surface, rng=rng),
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
