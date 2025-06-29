from ultrasphere.coordinates import BranchingType, SphericalCoordinates, get_child


def flatten_mask_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
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
    index_arrays = c.index_array_harmonics_all(
        n_end=n_end,
        include_negative_m=include_negative_m,
        as_array=False,
        expand_dims=True,
    )
    shape = ivy.broadcast_shapes(
        *[index_array.shape for index_array in index_arrays.values()]
    )
    mask = ivy.ones(shape, dtype=bool)
    for node, branching_type in c.branching_types.items():
        if branching_type == BranchingType.B:
            mask = mask & (
                ivy.abs(index_arrays[get_child(c.G, node, "sin")])
                <= index_arrays[node]
            )
        if branching_type == BranchingType.BP:
            mask = mask & (
                ivy.abs(index_arrays[get_child(c.G, node, "cos")])
                <= index_arrays[node]
            )
        if branching_type == BranchingType.C:
            value = (
                index_arrays[node]
                - ivy.abs(index_arrays[get_child(c.G, node, "sin")])
                - ivy.abs(index_arrays[get_child(c.G, node, "cos")])
            )
            mask = mask & (value % 2 == 0) & (value >= 0)
    return mask


def flatten_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    harmonics: Array | NativeArray,
) -> Array:
    """
    Flatten the harmonics.

    Parameters
    ----------
    harmonics : Array | NativeArray
        The (unflattend) harmonics.

    Returns
    -------
    Array
        The flattened harmonics of shape (..., n_harmonics).

    """
    n_end, include_negative_m = (
        c.get_n_end_and_include_negative_m_from_expansion(harmonics)
    )
    mask = c.flatten_mask_harmonics(n_end, include_negative_m)
    return harmonics[..., mask]


def unflatten_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    harmonics: Array | NativeArray,
    *,
    n_end: int,
    include_negative_m: bool = True,
) -> Array:
    """
    Unflatten the harmonics.

    Parameters
    ----------
    harmonics : Array | NativeArray
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
    mask = c.flatten_mask_harmonics(n_end, include_negative_m)
    shape = (*harmonics.shape[:-1], *mask.shape)
    result = ivy.zeros(shape, dtype=harmonics.dtype, device=harmonics.device)
    result[..., mask] = harmonics
    return result