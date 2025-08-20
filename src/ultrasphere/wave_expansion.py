import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace

from .special import binom, lgamma


def homogeneous_ndim(n: int | Array, *, e_ndim: int | Array) -> int | Array:
    """
    The dimension of the homogeneous polynomials of degree n.

    Parameters
    ----------
    n : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.250

    """
    s_ndim = e_ndim - 1
    return binom(n + s_ndim, s_ndim)


def harm_n_ndim(n: int | Array, *, e_ndim: int | Array) -> int | Array:
    """
    The dimension of the spherical harmonics of degree n.

    Parameters
    ----------
    n : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.251

    """
    xp = array_namespace(n, e_ndim)
    if e_ndim == 1:
        return xp.where(n <= 1, 1, 0)
    elif e_ndim == 2:
        return xp.where(n == 0, 1, 2)
    else:
        return (2 * n + e_ndim - 2) / (e_ndim - 2) * binom(n + e_ndim - 3, e_ndim - 3)


def plane_wave_expansion_coef(n: int | Array, *, e_ndim: int | Array) -> Array:
    """
    The coefficients of the plane wave expansion.

    Parameters
    ----------
    n : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    Array
        The coefficients for regular elementary wave solutions
        of degree n.

    """
    xp = array_namespace(n, e_ndim)
    return (
        1j**n
        * (2 * n + e_ndim - 2)
        / (e_ndim - 2)
        * xp.exp(lgamma(e_ndim / 2.0) + np.log(2) * ((e_ndim - 1) / 2))
        / xp.sqrt(xp.pi)
    )
