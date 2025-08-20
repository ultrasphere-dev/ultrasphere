from collections.abc import Mapping
from typing import overload

from array_api._2024_12 import Array
from array_api_compat import array_namespace

from ..coordinates import BranchingType, SphericalCoordinates, TEuclidean, TSpherical
from .assume import get_n_end_and_include_negative_m_from_expansion


@overload
def expand_cut(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array],
    n_end: int,
) -> Mapping[TSpherical, Array]: ...


@overload
def expand_cut(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Array,
    n_end: int,
) -> Array: ...


def expand_cut(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array] | Array,
    n_end: int,
) -> Mapping[TSpherical, Array] | Array:
    """
    Cut the expansion coefficients to the maximum degree.

    Parameters
    ----------
    expansion : Mapping[TSpherical, Array] | Array
        The expansion coefficients.
        If mapping, assume that the expansion is not expanded.
    n_end : int
        The maximum degree to cut.

    Returns
    -------
    Mapping[TSpherical, Array] | Array
        The cut expansion coefficients.

    """
    xp = (
        array_namespace(*[expansion[k] for k in c.s_nodes])
        if isinstance(expansion, Mapping)
        else array_namespace(expansion)
    )
    from shift_nth_row_n_steps import take_slice

    is_mapping = isinstance(expansion, Mapping)
    n_end_prev, include_negative_m = get_n_end_and_include_negative_m_from_expansion(
        c, expansion
    )
    if n_end > n_end_prev:
        raise ValueError(f"n_end={n_end} > n_end_prev={n_end_prev}.")
    if is_mapping:
        raise NotImplementedError()
        # for node, branching_type in c.branching_types.items():
        #     for axis in ndim_harmonics(c, node):
    for i, node in enumerate(c.s_nodes):
        axis = -c.s_ndim + i
        branching_type = c.branching_types[node]
        if branching_type == BranchingType.A and include_negative_m:
            expansion = xp.concat(
                [
                    take_slice(expansion, 0, n_end, axis=axis),
                ]
                + (
                    []
                    if n_end == 1
                    else [
                        take_slice(expansion, -n_end + 1, None, axis=axis),
                    ]
                ),
                axis=axis,
            )
        else:
            expansion = take_slice(expansion, 0, n_end, axis=axis)
    return expansion
