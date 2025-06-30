from array_api_compat import array_namespace
from ultrasphere.coordinates import BranchingType, SphericalCoordinates, TEuclidean, TSpherical, get_child
from ultrasphere.harmonics.assume import get_n_end_and_include_negative_m_from_expansion
from ultrasphere.harmonics.index import index_array_harmonics_all
import array_api_extra as xpx
from array_api._2024_12 import Array, ArrayNamespace

def flatten_mask_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    xp: ArrayNamespace,
    include_negative_m: bool = True,
) -> Array:
    """
    Create a mask representing the
    valid combinations of the quantum numbers
    which can be used to flatten the harmonics.

    Parameters
    ----------
    n_end : int
        The maximum degree of the harmonic.
    include_negative_m : bool, optional
        Whether to include negative m values, by default True

    Returns
    -------
    Array
        The mask.

    """
    index_arrays = index_array_harmonics_all(c, 
        n_end=n_end,
        include_negative_m=include_negative_m,
        as_array=False,
        expand_dims=True,
        xp=xp
    )
    shape = xpx.broadcast_shapes(
        *[index_array.shape for index_array in index_arrays.values()]
    )
    mask = xp.ones(shape, dtype=bool)
    for node, branching_type in c.branching_types.items():
        if branching_type == BranchingType.B:
            mask = mask & (
                xp.abs(index_arrays[get_child(c.G, node, "sin")])
                <= index_arrays[node]
            )
        if branching_type == BranchingType.BP:
            mask = mask & (
                xp.abs(index_arrays[get_child(c.G, node, "cos")])
                <= index_arrays[node]
            )
        if branching_type == BranchingType.C:
            value = (
                index_arrays[node]
                - xp.abs(index_arrays[get_child(c.G, node, "sin")])
                - xp.abs(index_arrays[get_child(c.G, node, "cos")])
            )
            mask = mask & (value % 2 == 0) & (value >= 0)
    return mask


def flatten_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    harmonics: Array,
) -> Array:
    """
    Flatten the harmonics.

    Parameters
    ----------
    harmonics : Array
        The (unflattend) harmonics.

    Returns
    -------
    Array
        The flattened harmonics of shape (..., n_harmonics).

    """
    xp = array_namespace(harmonics)
    n_end, include_negative_m = (
        get_n_end_and_include_negative_m_from_expansion(c, harmonics)
    )
    mask = flatten_mask_harmonics(c, n_end, xp, include_negative_m)
    return harmonics[..., mask]


def unflatten_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    harmonics: Array,
    *,
    n_end: int,
    include_negative_m: bool = True,
) -> Array:
    """
    Unflatten the harmonics.

    Parameters
    ----------
    harmonics : Array
        The flattened harmonics.
    n_end : int
        The maximum degree of the harmonic.
    include_negative_m : bool, optional
        Whether to include negative m values, by default True

    Returns
    -------
    Array
        The unflattened harmonics of shape (..., n_1, n_2, ..., n_(c.s_ndim)).

    """
    xp = array_namespace(harmonics)
    mask = flatten_mask_harmonics(c, n_end, include_negative_m)
    shape = (*harmonics.shape[:-1], *mask.shape)
    result = xp.zeros(shape, dtype=harmonics.dtype, device=harmonics.device)
    result[..., mask] = harmonics
    return result