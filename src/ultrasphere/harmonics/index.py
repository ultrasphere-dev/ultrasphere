
from typing import Literal, Mapping, overload
from array_api._2024_12 import Array, ArrayNamespace
from shift_nth_row_n_steps._torch_like import create_slice
from ultrasphere.coordinates import BranchingType, SphericalCoordinates, TEuclidean, TSpherical
from ultrasphere.harmonics.flatten import flatten_mask_harmonics
from ultrasphere.symmetry import to_symmetric


def index_array_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    node: TSpherical,
    *,
    n_end: int,
    xp: ArrayNamespace,
    expand_dims: bool = False,
    include_negative_m: bool = True,
) -> Array:
    """
    The index of the eigenfunction
    corresponding to the node.

    Parameters
    ----------
    node : TSpherical
        The node of the spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    expand_dims : bool, optional
        Whether to expand dimensions, by default False
    include_negative_m : bool, optional
        Whether to include negative m values, by default True

    Returns
    -------
    Array
        The index.

    """
    branching_type = c.branching_types[node]
    if branching_type == BranchingType.A and include_negative_m:
        result = to_symmetric(xp.arange(0, n_end), asymmetric=True)
    elif (
        branching_type == BranchingType.B
        or branching_type == BranchingType.BP
        or (branching_type == BranchingType.A and not include_negative_m)
    ):
        result = xp.arange(0, n_end)
    elif branching_type == BranchingType.C:
        # result = xp.arange(0, (n_end + 1) // 2)
        result = xp.arange(0, n_end)
    if expand_dims:
        idx = c.s_nodes.index(node)
        result = result[
            create_slice(c.s_ndim, [(idx, slice(None))], default=None)
        ]
    return result

@overload
def index_array_harmonics_all(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    n_end: int,
    xp: ArrayNamespace,
    include_negative_m: bool = ...,
    expand_dims: bool,
    as_array: Literal[False],
    mask: Literal[False] = ...,
) -> Mapping[TSpherical, Array]: ...
@overload
def index_array_harmonics_all(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    n_end: int,
    xp: ArrayNamespace,
    include_negative_m: bool = ...,
    expand_dims: Literal[True],
    as_array: Literal[True],
    mask: bool = ...,
) -> Array: ...

def index_array_harmonics_all(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    n_end: int,
    xp: ArrayNamespace,
    include_negative_m: bool = True,
    expand_dims: bool,
    as_array: bool,
    mask: bool = False,
) -> Array | Mapping[TSpherical, Array]:
    """
    The all indices of the eigenfunction
    corresponding to the spherical coordinates.

    Parameters
    ----------
    n_end : int
        The maximum degree of the harmonic.
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
    expand_dims : bool, optional
        Whether to expand dimensions, by default True
        Must be True if as_array is True.
    as_array : bool, optional
        Whether to return as an array, by default False
    mask : bool, optional
        Whether to fill invalid quantum numbers with NaN, by default False
        Must be False if as_array is False.

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        If as_array is True, the indices of shape
        [c.s_ndim,
        len(index_array_harmonics(c, node1)),
        ...,
        len(index_array_harmonics(c, node(c.s_ndim)))].
        If as_array is False, the dictionary of indices.

    Notes
    -----
        To check the indices where all quantum numbers match,
        `(numbers1 == numbers2).all(axis=0)`
        can be used.

    Raises
    ------
    ValueError
        If expand_dims is False and as_array is True.
        If mask is True and as_array is False.

    """
    if not expand_dims and as_array:
        raise ValueError("expand_dims must be True if as_array is True.")
    if mask and not as_array:
        raise ValueError("mask must be False if as_array is False.")
    index_arrays = {
        node: index_array_harmonics(c, 
            node,
            xp=xp,
            n_end=n_end,
            expand_dims=expand_dims,
            include_negative_m=include_negative_m,
        )
        for node in c.s_nodes
    }
    if as_array:
        result = xp.stack(
            xp.broadcast_arrays(*[index_arrays[node] for node in c.s_nodes]),
            axis=0,
        )
        if mask:
            result[:, ~flatten_mask_harmonics(c, n_end, xp=xp, include_negative_m=include_negative_m)] = (
                xp.nan
            )
        return result
    return index_arrays