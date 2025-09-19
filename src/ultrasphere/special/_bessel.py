from typing import Literal, TypeVar

from array_api._2024_12 import Array
from array_api_compat import array_namespace, to_device
from scipy.special import hankel1, hankel2, jv, jvp, yv, yvp

TArray = TypeVar("TArray", bound=Array)


def szv(
    v: TArray,
    d: TArray,
    z: TArray,
    type: Literal["j", "y", "h1", "h2"],
    derivative: bool = False,
) -> TArray:
    r"""
    Utility function to compute hyperspherical functions.

    .. math::
        f_v^{(d)} (z) = \sqrt{\frac{\pi}{2}} \frac{F_{v + d/2 - 1}(z)}{z^{d/2 - 1}}

    Parameters
    ----------
    v : TArray
        The degree of the hyperspherical Hankel function.
    d : TArray
        The dimension of the hypersphere.
    z : TArray
        The argument of the hyperspherical Hankel function.
    type : Literal["j", "y", "h1", "h2"]
        The type of the hyperspherical function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Hankel function, by default False

    Returns
    -------
    TArray
        The hyperspherical function.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.279

    """

    xp = array_namespace(v, d, z)
    if xp.any((d > 2) & (v < 0)):
        raise ValueError(
            "The hyperspherical Bessel function of "
            "the first kind is not defined for negative degrees."
        )
    if (d > 2 or type in ("h1, h2")) and derivative:
        return v / z * szv(v, d, z, type=type) - szv(v + 1, d, z, type=type)
    d_half_minus_1 = d / 2 - 1
    if type == "j":
        if derivative:
            zv = jvp
        else:
            zv = jv
    elif type == "y":
        if derivative:
            zv = yvp
        else:
            zv = yv
    elif type == "h1":
        if derivative:
            raise AssertionError()
        else:
            zv = hankel1
    elif type == "h2":
        if derivative:
            raise AssertionError()
        else:
            zv = hankel2

    dtype = xp.result_type(v, d, z)
    if type in ("h1", "h2"):
        dtype = xp.result_type(dtype, xp.complex64)
    return (
        xp.sqrt(xp.asarray(xp.pi / 2, device=z.device, dtype=dtype))
        * xp.asarray(
            zv(to_device(v + d_half_minus_1, "cpu"), to_device(z, "cpu")),
            device=z.device,
            dtype=dtype,
        )
        / (z**d_half_minus_1)
    )


def sjv(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    r"""
    Hyperspherical Bessel function of the first kind.

    .. math::
        j_v^{(d)} (z) = \sqrt{\frac{\pi}{2}} \frac{J_{v + d/2 - 1}(z)}{z^{d/2 - 1}}

    Parameters
    ----------
    v : TArray
        The degree of the hyperspherical Bessel function.
    d : TArray
        The dimension of the hypersphere.
    z : TArray
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
    return szv(v, d, z, type="j", derivative=derivative)


def syv(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    r"""
    Hyperspherical Bessel function of the second kind.

    .. math::
        y_v^{(d)} (z) = \sqrt{\frac{\pi}{2}} \frac{Y_{v + d/2 - 1}(z)}{z^{d/2 - 1}}

    Parameters
    ----------
    v : TArray
        The degree of the hyperspherical Bessel function.
    d : TArray
        The dimension of the hypersphere.
    z : TArray
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
    return szv(v, d, z, type="y", derivative=derivative)


def shn1(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    r"""
    Hyperspherical Hankel function of the first kind.

    .. math::
        h_v^{(1)(d)} (z) = \sqrt{\frac{\pi}{2}} \frac{H^{(1)}_{v + d/2 - 1}(z)}{z^{d/2 - 1}}

    Parameters
    ----------
    v : TArray
        The degree of the hyperspherical Hankel function.
    d : TArray
        The dimension of the hypersphere.
    z : TArray
        The argument of the hyperspherical Hankel function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Hankel function, by default False

    Returns
    -------
    Array
        The hyperspherical Hankel function of the first kind.

    """
    return szv(v, d, z, type="h1", derivative=derivative)


def shn2(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    r"""
    Hyperspherical Hankel function of the second kind.

    .. math::
        h_v^{(2)(d)} (z) = \sqrt{\frac{\pi}{2}} \frac{H^{(2)}_{v + d/2 - 1}(z)}{z^{d/2 - 1}}

    Parameters
    ----------
    v : TArray
        The degree of the hyperspherical Hankel function.
    d : TArray
        The dimension of the hypersphere.
    z : TArray
        The argument of the hyperspherical Hankel function.
    derivative : bool, optional
        Whether to compute the derivative of the
        hyperspherical Hankel function, by default False

    Returns
    -------
    Array
        The hyperspherical Hankel function of the second kind.

    """
    return szv(v, d, z, type="h2", derivative=derivative)
