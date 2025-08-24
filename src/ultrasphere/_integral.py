from collections.abc import Callable, Mapping
from typing import Any, Literal, overload

import array_api_extra as xpx
import numpy as np
from array_api._2024_12 import Array, ArrayNamespaceFull
from scipy.special import roots_jacobi

from ._coordinates import (
    BranchingType,
    SphericalCoordinates,
    TEuclidean,
    TSpherical,
    get_child,
)


def roots(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n: int,
    *,
    expand_dims_x: bool,
    expand_dims_w: bool = False,
    device: Any | None = None,
    dtype: Any | None = None,
    xp: ArrayNamespaceFull,
) -> tuple[Mapping[TSpherical, Array], Mapping[TSpherical, Array]]:
    """
    Gauss-Jacobi quadrature roots and weights.

    Parameters
    ----------
    n : int
        The number of roots.
    expand_dims_x : bool
        Whether to expand dimensions of the roots, by default False
    expand_dims_w : bool, optional
        Whether to expand dimensions of the weights, by default False
    device : Any, optional
        The device, by default None
    dtype : Any, optional
        The data type, by default None

    Returns
    -------
    tuple[Mapping[TSpherical, Array], Mapping[TSpherical, Array]]
        roots and weights

    Raises
    ------
    ValueError
        If the branching type is invalid.

    """
    xs = {}
    ws = {}
    for i, node in enumerate(c.s_nodes):
        branching_type = c.branching_types[node]
        if branching_type == BranchingType.A:
            x = xp.arange(2 * n, device=device, dtype=dtype) * xp.pi / n
            w = xp.ones(2 * n, device=device, dtype=dtype) * xp.pi / n
        elif branching_type == BranchingType.B:
            s_beta = c.S[get_child(c.G, node, "sin")]
            beta = s_beta / 2
            x, w = roots_jacobi(n, beta, beta)
            x = np.acos(x)
        elif branching_type == BranchingType.BP:
            s_alpha = c.S[get_child(c.G, node, "cos")]
            alpha = s_alpha / 2
            x, w = roots_jacobi(n, alpha, alpha)
            x = np.asin(x)
        elif branching_type == BranchingType.C:
            s_alpha = c.S[get_child(c.G, node, "cos")]
            s_beta = c.S[get_child(c.G, node, "sin")]
            alpha = s_alpha / 2
            beta = s_beta / 2
            x, w = roots_jacobi(n, alpha, beta)
            w /= 2 ** (alpha + beta + 2)
            x = np.acos(x) / 2
        else:
            raise ValueError(f"Invalid branching type {branching_type}.")
        x = xp.asarray(x, device=device, dtype=dtype)
        w = xp.asarray(w, device=device, dtype=dtype)
        if expand_dims_x:
            x = x[(None,) * i + (slice(None),) + (None,) * (c.s_ndim - i - 1)]
        if expand_dims_w:
            w = w[(None,) * i + (slice(None),) + (None,) * (c.s_ndim - i - 1)]
        xs[node] = x
        ws[node] = w
    return xs, ws


@overload
def integrate(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    f: (
        Callable[
            [Mapping[TSpherical, Array]],
            Mapping[TSpherical, Array],
        ]
        | Mapping[TSpherical, Array]
    ),
    does_f_support_separation_of_variables: Literal[True],
    n: int,
    *,
    xp: ArrayNamespaceFull,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Mapping[TSpherical, Array]: ...


@overload
def integrate(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    f: (
        Callable[
            [Mapping[TSpherical, Array]],
            Array,
        ]
        | Array
    ),
    does_f_support_separation_of_variables: Literal[False],
    n: int,
    *,
    xp: ArrayNamespaceFull,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Array: ...


def integrate(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    f: (
        Callable[
            [Mapping[TSpherical, Array]],
            Mapping[TSpherical, Array] | Array,
        ]
        | Mapping[TSpherical, Array]
        | Array
    ),
    does_f_support_separation_of_variables: bool,
    n: int,
    *,
    xp: ArrayNamespaceFull,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Array | Mapping[TSpherical, Array]:
    """
    Integrate the function over the hypersphere.

    Parameters
    ----------
    f : Callable[ [Mapping[TSpherical, Array]], Mapping[TSpherical, Array] | Array, ] | Mapping[TSpherical, Array] | Array # noqa: E501
        The function to integrate or the values of the function.

        If mapping, the separated parts of the function for each spherical coordinate.

        If mapping, the shapes do not need to be broadcastable.

        If function, if does_f_support_separation_of_variables is True,
        1D array of integration points are passed,
        and extra axis should be added to the last dimension.

        If function, if does_f_support_separation_of_variables is False,
        ``c.s_ndim``-D array of integration points are passed,
        and extra axis should be added to the last dimension.
    does_f_support_separation_of_variables : bool
        Whether the function supports separation of variables.
        This could significantly reduce the computational cost.
    n : int
        The number of roots.
    device : Any, optional
        The device, by default None
    dtype : Any, optional
        The data type, by default None

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The integrated value.
        Has the same shape as the return values of f or the values of f.

    """
    xs, ws = roots(
        c,
        n,
        device=device,
        dtype=dtype,
        expand_dims_x=not does_f_support_separation_of_variables,
        xp=xp,
    )
    if isinstance(f, Callable):  # type: ignore
        try:
            val = f(xs)  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Error occurred while evaluating {f=}") from e
    else:
        val = f

    # in case f(theta1, ...) = f_1(theta1) * f_2(theta2) * ...
    if isinstance(val, Mapping):
        result = {}
        for node in c.s_nodes:
            value = val[node]
            # supports vectorized function
            # axis=0 because in sph_harm
            # we add axis to the last dimension
            # theta(node),u1,...,uM
            xpx.broadcast_shapes(value.shape[:1], ws[node].shape)
            w = xp.reshape(ws[node], (-1,) + (1,) * (value.ndim - 1))
            if value.shape[0] == 1:
                result[node] = value[0, ...] * xp.sum(w)
            else:
                result[node] = xp.vecdot(value, w, axis=0)
        # we don't know how to einsum the result
        return result
    if val.ndim < c.s_ndim:
        raise ValueError(
            f"The dimension of the return value of f should be at least {c.s_ndim}, got {val.ndim}."
        )
    xpx.broadcast_shapes(
        val.shape[: c.s_ndim],
        xpx.broadcast_shapes(*(xs[node].shape for node in c.s_nodes)),
    )
    # theta1,...,thetaN,u1,...,uM\
    for node in c.s_nodes:
        w = ws[node]
        if val.shape[0] == 1:
            val = val[0, ...] * xp.sum(w)
        else:
            val = xp.vecdot(val, w[(slice(None),) + (None,) * (val.ndim - 1)], axis=0)
    return val
