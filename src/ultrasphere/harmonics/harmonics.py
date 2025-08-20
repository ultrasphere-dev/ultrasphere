from collections.abc import Mapping
from typing import Literal, overload

from array_api._2024_12 import Array
from array_api_compat import array_namespace

from ultrasphere.coordinates import (
    BranchingType,
    SphericalCoordinates,
    TEuclidean,
    TSpherical,
    get_child,
)

from .eigenfunction import type_a, type_b, type_bdash, type_c


@overload
def harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: Mapping[TSpherical, Array],
    n_end: int,
    *,
    condon_shortley_phase: bool,
    include_negative_m: bool = True,
    index_with_surrogate_quantum_number: bool = False,
    expand_dims: Literal[True] = ...,
    concat: Literal[True] = ...,
) -> Array: ...


@overload
def harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: Mapping[TSpherical, Array],
    n_end: int,
    *,
    condon_shortley_phase: bool,
    include_negative_m: bool = True,
    index_with_surrogate_quantum_number: bool = False,
    expand_dims: bool = True,
    concat: Literal[False] = ...,
) -> Mapping[TSpherical, Array]: ...


def harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: Mapping[TSpherical, Array],
    n_end: int,
    *,
    condon_shortley_phase: bool,
    include_negative_m: bool = True,
    index_with_surrogate_quantum_number: bool = False,
    expand_dims: bool = False,
    concat: bool = False,
) -> Mapping[TSpherical, Array] | Array:
    """
    Calculate the spherical harmonics.

    Parameters
    ----------
    spherical : Mapping[TSpherical, Array]
        The spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    condon_shortley_phase : bool, optional
        Whether to apply the Condon-Shortley phase,
        which just multiplies the result by (-1)^m.

        It seems to be mainly used in quantum mechanics for convenience.

        Note that scipy.special.sph_harm (or scipy.special.lpmv)
        uses the Condon-Shortley phase.

        If False, `Y^{-m}_{l} = Y^{m}_{l}*`.

        If True, `Y^{-m}_{l} = (-1)^m Y^{m}_{l}*`.
        (Simply because `e^{i -m phi} = (e^{i m phi})*`)
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
        If True, the m values are [0, 1, ..., n_end-1, -n_end+1, ..., -1],
        and starts from 0, not -n_end+1.
    index_with_surrogate_quantum_number : bool, optional
        Whether to index with surrogate quantum number, by default False
    expand_dims : bool, optional
        Whether to expand dimensions so that
        all values of the result dictionary
        are commomly indexed by the same s_nodes, by default False

        For example, if spherical coordinates,
        if True, the result will be indexed {"phi": [m], "theta": [m, n]}
        if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

        Note that the values will not be repeated
        therefore the computational cost will be the same
    concat : bool, optional
        Whether to concatenate the results, by default True


    Returns
    -------
    Array
        The spherical harmonics.

    """
    if concat and not expand_dims:
        raise ValueError("expand_dims must be True if concat is True.")

    result = {}
    for node in c.s_nodes:
        value = spherical[node]
        if node == "r":
            continue
        if node not in c.s_nodes:
            raise ValueError(f"Key {node} is not in c.s_nodes {c.s_nodes}.")
        if c.branching_types[node] == BranchingType.A:
            result[node] = type_a(
                value,
                n_end=n_end,
                condon_shortley_phase=condon_shortley_phase,
                include_negative_m=include_negative_m,
            )
        elif c.branching_types[node] == BranchingType.B:
            result[node] = type_b(
                value,
                n_end=n_end,
                s_beta=c.S[get_child(c.G, node, "sin")],
                index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                is_beta_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "sin")] == BranchingType.A,
            )
        elif c.branching_types[node] == BranchingType.BP:
            result[node] = type_bdash(
                value,
                n_end=n_end,
                s_alpha=c.S[get_child(c.G, node, "cos")],
                index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                is_alpha_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "cos")] == BranchingType.A,
            )
        elif c.branching_types[node] == BranchingType.C:
            result[node] = type_c(
                value,
                n_end=n_end,
                s_alpha=c.S[get_child(c.G, node, "cos")],
                s_beta=c.S[get_child(c.G, node, "sin")],
                index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                is_alpha_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "cos")] == BranchingType.A,
                is_beta_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "sin")] == BranchingType.A,
            )
        else:
            raise ValueError(f"Invalid branching type {c.branching_types[node]}.")
    if expand_dims:
        result = expand_dims_harmonics(c, result)  # type: ignore
    if concat:
        result = concat_harmonics(c, result)
    return result


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
    xp = array_namespace(*[harmonics[k] for k in c.s_nodes])
    try:
        if c.s_ndim == 0:
            return xp.asarray(1)
        return xp.prod(
            xp.stack(xp.broadcast_arrays(*[harmonics[k] for k in c.s_nodes]), axis=0),
            axis=0,
        )
    except Exception as e:
        shapes = {k: v.shape for k, v in harmonics.items()}
        raise RuntimeError(f"Error occurred while concatenating {shapes=}") from e


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
