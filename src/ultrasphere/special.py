import warnings
from numbers import Number
from typing import Literal, TypeVar

import array_api_extra as xpx
from array_api._2024_12 import Array
from array_api_compat import (
    array_namespace,
    is_jax_array,
    is_numpy_array,
    is_torch_array,
)
from scipy.special import jv, jvp, yv, yvp

TArray = TypeVar("TArray", bound=Array)


def sjv(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    """
    Hyperspherical Bessel function of the first kind.

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
    xp = array_namespace(v, d, z)
    if xp.any((d > 2) & (v < 0)):
        raise ValueError(
            "The hyperspherical Bessel function of "
            "the first kind is not defined for negative degrees."
        )
    if d > 2 and derivative:
        return v / z * sjv(v, d, z) - sjv(v + 1, d, z)
    d_half_minus_1 = d / 2 - 1
    return (
        xp.sqrt(xp.pi / 2)
        * xp.asarray((jvp if derivative else jv)((v + d_half_minus_1), (z)))
        / (z**d_half_minus_1)
    )


def syv(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    """
    Hyperspherical Bessel function of the second kind.

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
    xp = array_namespace(v, d, z)
    if xp.any((d > 2) & (v < 0)):
        raise ValueError(
            "The hyperspherical Bessel function of "
            "the second kind is not defined for negative degrees."
        )
    if d > 2 and derivative:
        return v / z * syv(v, d, z) - syv(v + 1, d, z)
    d_half_minus_1 = d / 2 - 1
    return (
        xp.sqrt(xp.pi / 2)
        * xp.asarray((yvp if derivative else yv)((v + d_half_minus_1), (z)))
        / (z**d_half_minus_1)
    )


def shn1(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    """
    Hyperspherical Hankel function of the first kind.

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
    return sjv(v, d, z, derivative) + 1j * syv(v, d, z, derivative)


def shn2(
    v: TArray,
    d: TArray,
    z: TArray,
    derivative: bool = False,
) -> TArray:
    """
    Hyperspherical Hankel function of the second kind.

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
    return sjv(v, d, z, derivative) - 1j * syv(v, d, z, derivative)


def szv(
    v: TArray,
    d: TArray,
    z: TArray,
    type: Literal["j", "y", "h1", "h2"],
    derivative: bool = False,
) -> Array:
    """
    Utility function to compute hyperspherical functions.

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


def fundamental_solution(
    d: TArray,
    z: TArray,
    k: TArray,
    derivative: bool = False,
) -> TArray:
    """
    Fundamental solution of the Laplace equation in d dimensions.

    Parameters
    ----------
    d : TArray
        The dimension of the space.
    z : TArray
        The argument of the fundamental solution of shape (..., d (coordinates)).
    k : TArray
        The wave number.
    derivative : bool, optional
        Whether to compute the derivative of the fundamental solution, by default False

    Returns
    -------
    TArray
        The fundamental solution of the Helmholtz equation of shape (...,).

    """
    xp = array_namespace(d, z)
    coef = k ** (d - 2) * 1j / (2 * (2 * xp.pi) ** ((d - 1) / 2))
    return coef * shn1(
        xp.asarray(0), d, k * xp.linalg.vector_norm(z, axis=-1), derivative
    )


def potential_coef[TArray: Array](
    n: TArray,
    d: TArray,
    k: TArray,
    /,
    *,
    x_abs: TArray,
    y_abs: TArray,
    x_abs_derivative: bool | None = None,
    y_abs_derivative: bool | None = None,
    derivative: Literal["S", "D", "D*", "N"] | None = None,
    limit: Literal[False, "x_larger", "y_larger", "warn"] = "warn",
    for_func: Literal["harmonics", "solution"] = "harmonics",
) -> Array:
    """
    The coefficients for layer potentials.

    The coefficients for single-layer or double-layer potential
    for hyperspherical harmonics with homogeneous degree
    (maximum quantum number) n.

    y is the integral variable.

    Parameters
    ----------
    n : TArray
        The homogeneous degree of the hyperspherical harmonics.
        (maximum quantum number)
    d : TArray
        The dimension of the hypersphere.
    k : TArray
        The wavenumber.
    x_abs : TArray
        The distance from the origin O to the point x.
    y_abs : TArray
        The radius of the hypersphere.
    x_abs_derivative : bool | None, optional
        Whether the derivative is taken with respect to x_abs,
        by default False.
    y_abs_derivative : bool | None, optional
        Whether the derivative is taken with respect to y_abs,
        by default False.
    derivative : Literal["S", "D", "D*", "N"] | None, optional
        The shorthand for the derivative.
        Note that the integral variable is y.
        "S" <=> x_abs_derivative = False, y_abs_derivative = False
        "D" <=> x_abs_derivative = False, y_abs_derivative = True
        "D*" <=> x_abs_derivative = True, y_abs_derivative = False
        "N" <=> x_abs_derivative = True, y_abs_derivative = True
    limit : Literal[False, "x_larger", "y_larger"], optional
        Whether to return the directional derivative
        of the potential with respect to x
        to the x/|x| direction, by default False.
    for_func : Literal["harmonics", "solution"], optional
        Whether the coefficient is for the harmonics
        or the suitable (singular if outer, regular if inner)
        elementary solution, by default "harmonics".

    Returns
    -------
    Array
        The coefficient for the potential
        integrated by y.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.285

    """
    xp = array_namespace(n, d, k, x_abs, y_abs)
    if x_abs_derivative is None and y_abs_derivative is None:
        x_abs_derivative = derivative in ["D*", "N"]
        y_abs_derivative = derivative in ["D", "N"]
    elif x_abs_derivative is not None and y_abs_derivative is not None:
        pass
    else:
        raise ValueError(
            f"Both {x_abs_derivative=} and {y_abs_derivative=} must be None or not None."
        )
    # p.285
    inner = xp.where(
        x_abs < y_abs,
        shn1(n, d, k * y_abs, derivative=y_abs_derivative)
        * (
            sjv(n, d, k * x_abs, derivative=x_abs_derivative)
            if for_func == "harmonics"
            else 1
        ),
        sjv(n, d, k * y_abs, derivative=y_abs_derivative)
        * (
            shn1(n, d, k * x_abs, derivative=x_abs_derivative)
            if for_func == "harmonics"
            else 1
        ),
    )
    derivative_count = int(x_abs_derivative) + int(y_abs_derivative)
    result = (k) ** derivative_count * 1j * (k ** (d - 2)) * (y_abs ** (d - 1)) * inner
    if derivative_count == 1:
        if for_func == "solution" and limit in ["x_larger", "y_larger"]:
            raise NotImplementedError()
        addition = None
        if limit == "x_larger":
            if x_abs_derivative:
                # outer D^*
                addition = 1 / 2
            if y_abs_derivative:
                # outer D
                addition = -1 / 2
        elif limit == "y_larger":
            if x_abs_derivative:
                # inner D^*
                addition = -1 / 2
            if y_abs_derivative:
                # inner D
                addition = 1 / 2
        elif limit == "warn":
            if xp.any(xpx.isclose(x_abs, y_abs, rtol=1e-4, atol=0)):
                warnings.warn(
                    "As x_abs is close to y_abs "
                    "and requested to calculate D or D*, "
                    "which is not continuous at x_abs=y_abs, "
                    "it might be possible that what you really want "
                    "are the limit values."
                    "To calculate the limit values, set limit="
                    "'x_larger' or 'y_larger', depending on the "
                    "limit direction you want to take."
                    "To suppress this warning, set limit=False.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif limit is False:
            pass
        else:
            raise ValueError(f"Invalid limit: {limit}")
        if addition is not None:
            result += addition
    return result


def lgamma(x: Array) -> Array:
    """
    Compute the logarithm of the absolute value of the gamma function.

    Parameters
    ----------
    x : Array
        The input array.

    Returns
    -------
    Array
        The logarithm of the absolute value of the gamma function.

    """
    if isinstance(x, Number):
        from math import lgamma as lgamma_math

        return lgamma_math(x)  # type: ignore
    elif is_jax_array(x):
        from jax.lax import lgamma as lgamma_jax

        return lgamma_jax(x)
    elif is_numpy_array(x):
        from scipy.special import gammaln as lgamma_scipy

        return lgamma_scipy(x)
    elif is_torch_array(x):
        from torch import lgamma as lgamma_torch

        return lgamma_torch(x)
    else:
        xp = array_namespace(x)
        if hasattr(xp, "lgamma"):
            return xp.lgamma(x)
        elif hasattr(xp, "gammaln"):
            return xp.gammaln(x)
        else:
            raise ValueError(
                "The input array must be a JAX, NumPy, or PyTorch array, "
                "or an array with a lgamma or gammaln method."
            )


def binom(x: Array, y: Array) -> Array:
    """
    Compute the binomial coefficient.

    Parameters
    ----------
    x : Array
        The first argument.
    y : Array
        The second argument.

    Returns
    -------
    Array
        The binomial coefficient.

    """
    xp = array_namespace(x, y)
    return xp.exp(lgamma(x + 1.0) - lgamma(y + 1.0) - lgamma(x - y + 1.0))
