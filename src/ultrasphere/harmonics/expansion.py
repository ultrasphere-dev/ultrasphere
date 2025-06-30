
from typing import Any, Callable, Literal, Mapping, overload
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ultrasphere.coordinates import BranchingType, SphericalCoordinates, TEuclidean, TSpherical, get_child
from ultrasphere.harmonics.assume import get_n_end_and_include_negative_m_from_expansion
import array_api_extra as xpx
from .eigenfunction import type_a, type_b, type_bdash, type_c

@overload
def expand_evaluate(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array],
    spherical: Mapping[TSpherical, Array],
    *,
    condon_shortley_phase: bool,
) -> Mapping[TSpherical, Array]: ...

@overload
def expand_evaluate(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Array,
    spherical: Mapping[TSpherical, Array],
    *,
    condon_shortley_phase: bool,
) -> Array: ...

def expand_evaluate(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array] | Array,
    spherical: Mapping[TSpherical, Array],
    *,
    condon_shortley_phase: bool,
) -> Array | Mapping[TSpherical, Array]:
    """
    Evaluate the expansion at the spherical coordinates.

    Parameters
    ----------
    expansion : Mapping[TSpherical, Array] | Array
        The expansion coefficients.
        If mapping, assume that the expansion is not expanded.
    spherical : Mapping[TSpherical, Array]
        The spherical coordinates.
    condon_shortley_phase : bool
        Whether to apply the Condon-Shortley phase,
        which just multiplies the result by (-1)^m.

        It seems to be mainly used in quantum mechanics for convenience.

        Note that scipy.special.sph_harm (or scipy.special.lpmv)
        uses the Condon-Shortley phase.

        If False, `Y^{-m}_{l} = Y^{m}_{l}*`.

        If True, `Y^{-m}_{l} = (-1)^m Y^{m}_{l}*`.
        (Simply because `e^{i -m phi} = (e^{i m phi})*`)


    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The evaluated value.

    """
    is_mapping = isinstance(expansion, Mapping)
    xp = array_namespace(*expansion.values()) if is_mapping else array_namespace(expansion)
    n_end, _ = get_n_end_and_include_negative_m_from_expansion(c, expansion)
    harmonics = harmonics(c,   # type: ignore
        spherical,
        n_end,
        condon_shortley_phase=condon_shortley_phase,
        expand_dims=not is_mapping,
        concat=not is_mapping,
    )
    if is_mapping:
        result: dict[TSpherical, Array] = {}
        for node in c.s_nodes:
            expansion_ = expansion[node]
            harmonics_ = harmonics[node]
            # expansion: f1,...,fL,harm1,...,harmN
            # harmonics: u1,...,uM,harm1,...,harmN
            # result: u1,...,uM,f1,...,fL
            ndim_harmonics = ndim_harmonics(c, node)
            ndim_expansion = expansion_.ndim - ndim_harmonics
            ndim_extra_harmonics = harmonics_.ndim - ndim_harmonics
            expansion_ = harmonics_[
                (None,) * (ndim_extra_harmonics)
                + (slice(None),) * (ndim_expansion + ndim_harmonics)
            ]
            harmonics = harmonics_[
                (slice(None),) * ndim_extra_harmonics
                + (None,) * ndim_expansion
                + (slice(None),) * ndim_harmonics
            ]
            result_ = harmonics_ * expansion_
            for _ in range(ndim_harmonics):
                result_ = xp.sum(result_, axis=-1)
            result[node] = result
        return result
    if isinstance(expansion, Mapping):
        raise AssertionError()
    # expansion: f1,...,fL,harm1,...,harmN
    # harmonics: u1,...,uM,harm1,...,harmN
    # result: u1,...,uM,f1,...,fL
    ndim_harmonics = c.s_ndim
    ndim_expansion = expansion.ndim - ndim_harmonics
    ndim_extra_harmonics = harmonics.ndim - ndim_harmonics
    expansion = expansion[
        (None,) * (ndim_extra_harmonics)
        + (slice(None),) * (ndim_expansion + ndim_harmonics)
    ]
    harmonics = harmonics[
        (slice(None),) * ndim_extra_harmonics
        + (None,) * ndim_expansion
        + (slice(None),) * ndim_harmonics
    ]
    result = harmonics * expansion
    for _ in range(ndim_harmonics):
        result = xp.sum(result, axis=-1)
    return result

@overload
def expand(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    f: (
        Callable[
            [Mapping[TSpherical, Array]],
            Mapping[TSpherical, Array] | Array,
        ]
        | Mapping[TSpherical, Array]
        | Array
        
    ),
    does_f_support_separation_of_variables: Literal[True],
    n_end: int,
    n: int,
    *,
    condon_shortley_phase: bool,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Mapping[TSpherical, Array]: ...

@overload
def expand(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    f: (
        Callable[
            [Mapping[TSpherical, Array]],
            Mapping[TSpherical, Array] | Array,
        ]
        | Mapping[TSpherical, Array]
        | Array
        
    ),
    does_f_support_separation_of_variables: Literal[False],
    n_end: int,
    n: int,
    *,
    condon_shortley_phase: bool,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Array: ...

def expand(
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
    n_end: int,
    n: int,
    *,
    condon_shortley_phase: bool,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Array | Mapping[TSpherical, Array]:
    """
    Calculate the expansion coefficients of the function
    over the hypersphere.

    Parameters
    ----------
    f : Callable[ [Mapping[TSpherical, Array]],
        Mapping[TSpherical, Array] | Array, ]
       | Mapping[TSpherical, Array] | Array
        The function to integrate or the values of the function.
        In case of vectorized function, the function should add extra
        axis to the last dimension, not the first dimension.
    does_f_support_separation_of_variables : bool
        Whether the function supports separation of variables.
        This could significantly reduce the computational cost.
    n : int
        The number of integration points.

        Must be equal to or larger than n_end.

        Must be large enough against f, as this method
        does not use adaptive integration. For example,
        consider expanding $f(θ) = e^(2Nθ)$ with $n=N$.

        >>> from ultrasphere import SphericalCoordinates
        >>> from ultrasphere.coordinates import SphericalCoordinates
        >>> import numpy as np
        >>> n = 5
        >>> expansion = SphericalCoordinates.standard(1).expand(
        >>>     lambda x: np.exp(1j * (2 * n) * x["theta0"]) / np.sqrt(2 * np.pi),
        >>>     does_f_support_separation_of_variables=False,
        >>>     n=n,
        >>>     n_end=n,
        >>>     condon_shortley_phase=False
        >>> )
        >>> print(np.round(expansion, 2).real.tolist())
        [1.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0]

        This result claims that f(θ) = 1, which is incorrect.
    n_end : int
        The maximum degree of the harmonic.
    condon_shortley_phase : bool
        Whether to apply the Condon-Shortley phase,
        which just multiplies the result by (-1)^m.

        It seems to be mainly used in quantum mechanics for convenience.

        Note that scipy.special.sph_harm (or scipy.special.lpmv)
        uses the Condon-Shortley phase.

        If False, `Y^{-m}_{l} = Y^{m}_{l}*`.

        If True, `Y^{-m}_{l} = (-1)^m Y^{m}_{l}*`.
        (Simply because `e^{i -m phi} = (e^{i m phi})*`)
    device : Any, optional
        The device, by default None
    dtype : Any, optional
        The data type, by default None

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The expanded value.
        Last `c.s_ndim` axis [-c.s_ndim, -1]
        corresponds to the quantum numbers.

        The dimensions are not expanded if Mapping is returned.
        Use `expand_dims_harmonics(c, )`
        and `concat_harmonics(c, )`
        to expand the dimensions and to concat values.

    """
    if n < n_end:
        raise ValueError(
            f"n={n} < n_end={n_end}, which would lead to incorrect results."
        )

    def inner(
        xs: Mapping[TSpherical, Array],
    ) -> Mapping[TSpherical, Array]:
        # calculate f
        if isinstance(f, Callable):  # type: ignore
            try:
                val = f(xs)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Error occurred while evaluating {f=}") from e
        else:
            val = f

        # calculate harmonics
        harmonics = harmonics(c,   # type: ignore
            xs,
            n_end,
            condon_shortley_phase=condon_shortley_phase,
            expand_dims=not does_f_support_separation_of_variables,
            concat=not does_f_support_separation_of_variables,
        )

        # multiply f and harmonics
        # (C,complex conjugate) is star-algebra
        if isinstance(val, Mapping):
            if not does_f_support_separation_of_variables:
                raise ValueError(
                    "val is Mapping but "
                    "does_f_support_separation_of_variables "
                    "is False."
                )
            result = {}
            for node in c.s_nodes:
                value = val[node]
                # val: theta(node),u1,...,uM
                # harmonics: theta(node),harm1,...,harmN
                # result: theta(node),u1,...,uM,harm1,...,harmN
                xpx.broadcast_shapes(
                    value.shape[:1], harmonics[node].shape[:1]
                )
                ndim_val = value.ndim - 1
                ndim_harm = ndim_harmonics(c, node)
                value = value[(...,) + (None,) * (ndim_harm)]
                harm = harmonics[node][
                    (slice(None),) + (None,) * ndim_val + (slice(None),) * ndim_harm
                ]
                result[node] = value * harm.conj()
        else:
            if does_f_support_separation_of_variables:
                raise ValueError(
                    "val is not Mapping but "
                    "does_f_support_separation_of_variables "
                    "is True."
                )
            # val: theta1,...,thetaN,u1,...,uM
            # harmonics: theta1,...,thetaN,harm1,...,harmN
            # res: theta1,...,thetaN,u1,...,uM,harm1,...,harmN
            xpx.broadcast_shapes(
                val.shape[: c.s_ndim], harmonics.shape[: c.s_ndim]
            )
            ndim_val = val.ndim - c.s_ndim
            val = val[
                (slice(None),) * (c.s_ndim + ndim_val) + (None,) * c.s_ndim
            ]
            harmonics = harmonics[
                (slice(None),) * c.s_ndim
                + (None,) * ndim_val
                + (slice(None),) * c.s_ndim
            ]
            result = val * harmonics.conj()

        return result

    return integrate(c,  # type: ignore
        inner, does_f_support_separation_of_variables, n, device=device, dtype=dtype
    )

def expand_dim_harmoncis(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    node: TSpherical,
    harmonics: Array,
) -> Array:
    """
    Expand the dimension of the harmonics.

    Expand the dimension so that
    all values of the harmonics() result dictionary
    are commomly indexed by the same s_nodes, by default False

    For example, if spherical coordinates,
    if True, the result will be indexed {"phi": [m], "theta": [m, n]}
    if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

    Note that the values will not be repeated
    therefore the computational cost will be the same

    Parameters
    ----------
    node : TSpherical
        The node of the spherical coordinates.
    harmonics : Array
        The harmonics (eigenfunctions).

    Returns
    -------
    Array
        The expanded harmonics.
        The shapes does not need to be either
        same or broadcastable.

    """
    xp = array_namespace(harmonics)
    idx_node = c.s_nodes.index(node)
    branching_type = c.branching_types[node]
    if branching_type == BranchingType.A:
        moveaxis = {0: idx_node}
    elif branching_type == BranchingType.B:
        idx_sin_child = c.s_nodes.index(get_child(c.G, node, "sin"))
        moveaxis = {
            0: idx_sin_child,
            1: idx_node,
        }
    elif branching_type == BranchingType.BP:
        idx_cos_child = c.s_nodes.index(get_child(c.G, node, "cos"))
        moveaxis = {
            0: idx_cos_child,
            1: idx_node,
        }
    elif branching_type == BranchingType.C:
        idx_cos_child = c.s_nodes.index(get_child(c.G, node, "cos"))
        idx_sin_child = c.s_nodes.index(get_child(c.G, node, "sin"))
        moveaxis = {0: idx_cos_child, 1: idx_sin_child, 2: idx_node}
    value_additional_ndim = harmonics.ndim - len(moveaxis)
    moveaxis = {
        k + value_additional_ndim: v + value_additional_ndim
        for k, v in moveaxis.items()
    }
    adding_ndim = c.s_ndim - len(moveaxis)
    harmonics = harmonics[(...,) + (None,) * adding_ndim]
    return xp.moveaxis(harmonics, list(moveaxis.keys()), list(moveaxis.values()))

def expand_dims_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    harmonics: Mapping[TSpherical, Array],
) -> Mapping[TSpherical, Array]:
    """
    Expand dimensions of the harmonics.

    Expand dimensions so that
    all values of the harmonics() result dictionary
    are commomly indexed by the same s_nodes, by default False

    For example, if spherical coordinates,
    if True, the result will be indexed {"phi": [m], "theta": [m, n]}
    if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

    Note that the values will not be repeated
    therefore the computational cost will be the same

    Parameters
    ----------
    harmonics : Mapping[TSpherical, Array]
        The dictionary of harmonics (eigenfunctions).

    Returns
    -------
    Mapping[TSpherical, Array]
        The expanded harmonics.
        The shapes does not need to be either
        same or broadcastable.

    """
    result: dict[TSpherical, Array] = {}
    for node in c.s_nodes:
        result[node] = expand_dim_harmoncis(c, node, harmonics[node])
    return result

def concat_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    harmonics: Mapping[TSpherical, Array],
) -> Array:
    """
    Concatenate the mapping of expanded harmonics.

    Parameters
    ----------
    harmonics : Mapping[TSpherical, Array]
        The expanded harmonics.

    Returns
    -------
    Array
        The concatenated harmonics.

    """
    xp = array_namespace(*harmonics.values())
    try:
        if c.s_ndim == 0:
            return xp.asarray(1)
        return xp.prod(
            xp.stack(xp.broadcast_arrays(*harmonics.values()), axis=0), axis=0
        )
    except Exception as e:
        shapes = {k: v.shape for k, v in harmonics.items()}
        raise RuntimeError(f"Error occurred while concatenating {shapes=}") from e

def ndim_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    node: TSpherical,
) -> int:
    """
    The number of dimensions of the eigenfunction
    corresponding to the node.

    Parameters
    ----------
    node : TSpherical
        The node of the spherical coordinates.

    Returns
    -------
    int
        The number of dimensions.

    """
    return {
        BranchingType.A: 1,
        BranchingType.B: 2,
        BranchingType.BP: 2,
        BranchingType.C: 3,
    }[c.branching_types[node]]

