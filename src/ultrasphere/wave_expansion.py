import ivy
from ivy import Array, NativeArray

from .special import binom


def homogeneous_ndim(
    n: int | Array | NativeArray, *, e_ndim: int | Array | NativeArray
) -> int | Array:
    """
    The dimension of the homogeneous polynomials of degree n.

    Parameters
    ----------
    n : int | Array | NativeArray
        The degree.
    e_ndim : int | Array | NativeArray
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
    n = ivy.array(n)
    s_ndim = e_ndim - 1
    return binom(n + s_ndim, s_ndim)


def harm_n_ndim(
    n: int | Array | NativeArray, *, e_ndim: int | Array | NativeArray
) -> int | Array:
    """
    The dimension of the spherical harmonics of degree n.

    Parameters
    ----------
    n : int | Array | NativeArray
        The degree.
    e_ndim : int | Array | NativeArray
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
    n = ivy.array(n, dtype=float)
    if e_ndim == 1:
        return ivy.where(n <= 1, 1, 0)
    elif e_ndim == 2:
        return ivy.where(n == 0, 1, 2)
    else:
        return (2 * n + e_ndim - 2) / (e_ndim - 2) * binom(n + e_ndim - 3, e_ndim - 3)


def plane_wave_expansion_coef(
    n: int | Array | NativeArray, *, e_ndim: int | Array | NativeArray
) -> Array:
    """
    The coefficients of the plane wave expansion.

    Parameters
    ----------
    n : int | Array | NativeArray
        The degree.
    e_ndim : int | Array | NativeArray
        The dimension of the Euclidean space.

    Returns
    -------
    Array
        The coefficients for regular elementary wave solutions
        of degree n.

    """
    return (
        1j**n
        * (2 * n + e_ndim - 2)
        / (e_ndim - 2)
        * ivy.exp(ivy.lgamma(e_ndim / 2.0) + ivy.log(2) * ((e_ndim - 1) / 2))
        / ivy.sqrt(ivy.pi)
    )
