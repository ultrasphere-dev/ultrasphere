
def get_n_end_and_include_negative_m_from_expansion(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array | NativeArray] | Array | NativeArray,
) -> tuple[int, bool]:
    """
    Assume `n_end` and `include_negative_m` from the expansion coefficients.

    Parameters
    ----------
    expansion : Mapping[TSpherical, Array  |  NativeArray] | Array | NativeArray
        The expansion coefficients.
        If mapping, assume that the expansion is not expanded.

    Returns
    -------
    tuple[int, bool]
        n_end, include_negative_m

    """
    if c.s_ndim == 0:
        return 0, False
    is_mapping = isinstance(expansion, Mapping)
    if is_mapping:
        sizes = [ivy.shape(expansion[k])[-1] for k in c.s_nodes]
    else:
        sizes = ivy.shape(expansion)[-c.s_ndim :]
    n_end = (max(sizes) + 1) // 2
    include_negative_m = not all(size == n_end for size in sizes)
    return n_end, include_negative_m
