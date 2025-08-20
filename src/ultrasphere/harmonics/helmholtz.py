from collections.abc import Mapping
from typing import Literal, overload

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from shift_nth_row_n_steps._torch_like import create_slice

from ultrasphere.coordinates import SphericalCoordinates, TEuclidean, TSpherical
from ultrasphere.harmonics.assume import get_n_end_and_include_negative_m_from_expansion
from ultrasphere.harmonics.flatten import index_array_harmonics
from ultrasphere.special import szv


@overload
def harmonics_regular_singular(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = ...,
    harmonics: Mapping[TSpherical, Array],
    multiply: bool = True,
) -> Mapping[TSpherical, Array]: ...


@overload
def harmonics_regular_singular(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = ...,
    harmonics: Array,
    multiply: bool = True,
) -> Array: ...


def harmonics_regular_singular(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    harmonics: Array | Mapping[TSpherical, Array],
    multiply: bool = True,
) -> Array | Mapping[TSpherical, Array]:
    """
    Regular or singular harmonics.

    Parameters
    ----------
    spherical : Mapping[TSpherical | Literal['r'],
        Array] | Mapping[Literal['r'],
        Array]
        The spherical coordinates.
    k : Array
        The wavenumber. Must be positive.
    type : Literal['regular', 'singular', 'j', 'y', 'h1', 'h2']
        The type of the spherical Bessel/Hankel function.
    harmonics : Array | Mapping[TSpherical, Array]
        The harmonics.
    derivative : bool, optional
        Whether to return the directional derivative to r,
        in other words whether to return the derivative with respect to r,
        by default False
    multiply : bool, optional
        Whether to multiply the harmonics by the result,
        by default True

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The regular or singular harmonics.

    Raises
    ------
    ValueError
        If the wavenumber is not positive.

    """
    xp = array_namespace(
        *(
            *[spherical[k] for k in c.s_nodes],  # type: ignore
            k,
            *(harmonics.values() if isinstance(harmonics, Mapping) else (harmonics,)),
        )
    )
    is_mapping = isinstance(harmonics, Mapping)
    if multiply and is_mapping:
        raise ValueError("multiply must be False if harmonics is Mapping.")
    n_end, include_negative_m = get_n_end_and_include_negative_m_from_expansion(
        c, harmonics
    )
    n = index_array_harmonics(
        c, c.root, n_end=n_end, include_negative_m=include_negative_m, xp=xp
    )[(None,) * spherical["r"].ndim + (slice(None),)]
    kr = k * spherical["r"]
    kr = kr[..., None]

    if type == "regular":
        type = "j"
    elif type == "singular":
        type = "h1"
    val = szv(n, c.e_ndim, kr, type=type, derivative=derivative)
    val = xp.nan_to_num(val, nan=0)
    expand_dims = not (is_mapping and len({h.ndim for h in harmonics.values()}) > 1)
    if expand_dims:
        idx = c.s_nodes.index(c.root)
        val = val[(..., *create_slice(c.s_ndim, [(idx, slice(None))], default=None))]
    if is_mapping:
        res = {"r": val}
        if harmonics is not None:
            res.update(harmonics)  # type: ignore
        return res
    if multiply:
        return val * harmonics
    return val
