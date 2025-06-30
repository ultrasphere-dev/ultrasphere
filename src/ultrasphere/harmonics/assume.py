from collections.abc import Mapping

from array_api._2024_12 import Array

from ..coordinates import SphericalCoordinates, TEuclidean, TSpherical


def get_n_end_and_include_negative_m_from_expansion(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array] | Array,
) -> tuple[int, bool]:
    """
    Assume `n_end` and `include_negative_m` from the expansion coefficients.

    Parameters
    ----------
    expansion : Mapping[TSpherical, Array] | Array
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
        sizes = [expansion[k].shape[-1] for k in c.s_nodes]
    else:
        sizes = expansion.shape[-c.s_ndim :]  # type: ignore
    n_end = (max(sizes) + 1) // 2
    include_negative_m = not all(size == n_end for size in sizes)
    return n_end, include_negative_m
