from typing import Literal

import ivy
from ivy import Array, NativeArray
from scipy.special import jv, jvp, yv, yvp


def sjv(
    v: Array | NativeArray,
    d: Array | NativeArray,
    z: Array | NativeArray,
    derivative: bool = False,
) -> Array:
    """
    Hyperspherical Bessel function of the first kind.

    Parameters
    ----------
    v : Array | NativeArray
        The degree of the hyperspherical Bessel function.
    d : Array | NativeArray
        The dimension of the hypersphere.
    z : Array | NativeArray
        The argument of the hyperspherical Bessel function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Bessel function, by default False

    Returns
    -------
    Array
        The hyperspherical Bessel function of the first kind.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.279

    """
    if ((d > 2) & (v < 0)).any():
        raise ValueError(
            "The hyperspherical Bessel function of "
            "the first kind is not defined for negative degrees."
        )
    if d > 2 and derivative:
        return v / z * sjv(v, d, z) - sjv(v + 1, d, z)
    d_half_minus_1 = d / 2 - 1
    return (
        ivy.sqrt(ivy.pi / 2)
        * ivy.asarray(
            (jvp if derivative else jv)(
                ivy.to_numpy(v + d_half_minus_1), ivy.to_numpy(z)
            )
        )
        / (z**d_half_minus_1)
    )


def syv(
    v: Array | NativeArray,
    d: Array | NativeArray,
    z: Array | NativeArray,
    derivative: bool = False,
) -> Array:
    """
    Hyperspherical Bessel function of the second kind.

    Parameters
    ----------
    v : Array | NativeArray
        The degree of the hyperspherical Bessel function.
    d : Array | NativeArray
        The dimension of the hypersphere.
    z : Array | NativeArray
        The argument of the hyperspherical Bessel function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Bessel function, by default False

    Returns
    -------
    Array
        The hyperspherical Bessel function of the second kind.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.279

    """
    if ((d > 2) & (v < 0)).any():
        raise ValueError(
            "The hyperspherical Bessel function of "
            "the second kind is not defined for negative degrees."
        )
    if d > 2 and derivative:
        return v / z * syv(v, d, z) - syv(v + 1, d, z)
    d_half_minus_1 = d / 2 - 1
    return (
        ivy.sqrt(ivy.pi / 2)
        * ivy.asarray(
            (yvp if derivative else yv)(
                ivy.to_numpy(v + d_half_minus_1), ivy.to_numpy(z)
            )
        )
        / (z**d_half_minus_1)
    )


def shn1(
    v: Array | NativeArray,
    d: Array | NativeArray,
    z: Array | NativeArray,
    derivative: bool = False,
) -> Array:
    """
    Hyperspherical Hankel function of the first kind.

    Parameters
    ----------
    v : Array | NativeArray
        The degree of the hyperspherical Hankel function.
    d : Array | NativeArray
        The dimension of the hypersphere.
    z : Array | NativeArray
        The argument of the hyperspherical Hankel function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Hankel function, by default False

    Returns
    -------
    Array
        The hyperspherical Hankel function of the first kind.

    """
    return sjv(v, d, z, derivative) + 1j * syv(v, d, z, derivative)


def shn2(
    v: Array | NativeArray,
    d: Array | NativeArray,
    z: Array | NativeArray,
    derivative: bool = False,
) -> Array:
    """
    Hyperspherical Hankel function of the second kind.

    Parameters
    ----------
    v : Array | NativeArray
        The degree of the hyperspherical Hankel function.
    d : Array | NativeArray
        The dimension of the hypersphere.
    z : Array | NativeArray
        The argument of the hyperspherical Hankel function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Hankel function, by default False

    Returns
    -------
    Array
        The hyperspherical Hankel function of the second kind.

    """
    return sjv(v, d, z, derivative) - 1j * syv(v, d, z, derivative)


def szv(
    v: Array | NativeArray,
    d: Array | NativeArray,
    z: Array | NativeArray,
    type: Literal["j", "y", "h1", "h2"],
    derivative: bool = False,
) -> Array:
    """
    Utility function to compute hyperspherical functions.

    Parameters
    ----------
    v : Array | NativeArray
        The degree of the hyperspherical Hankel function.
    d : Array | NativeArray
        The dimension of the hypersphere.
    z : Array | NativeArray
        The argument of the hyperspherical Hankel function.
    type : Literal["j", "y", "h1", "h2"]
        The type of the hyperspherical function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Hankel function, by default False

    Returns
    -------
    Array
        The hyperspherical function.

    """
    if type == "j":
        return sjv(v, d, z, derivative)
    if type == "y":
        return syv(v, d, z, derivative)
    if type == "h1":
        return shn1(v, d, z, derivative)
    if type == "h2":
        return shn2(v, d, z, derivative)
    raise ValueError(f"Invalid type {type}.")


def binom(x: Array | NativeArray, y: Array | NativeArray) -> Array:
    """
    Compute the binomial coefficient.

    Parameters
    ----------
    x : Array | NativeArray
        The first argument.
    y : Array | NativeArray
        The second argument.

    Returns
    -------
    Array
        The binomial coefficient.

    """
    return ivy.exp(ivy.lgamma(x + 1.0) - ivy.lgamma(y + 1.0) - ivy.lgamma(x - y + 1.0))
