from numbers import Number

from array_api._2024_12 import Array
from array_api_compat import (
    array_namespace,
    is_jax_array,
    is_numpy_array,
    is_torch_array,
)


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
