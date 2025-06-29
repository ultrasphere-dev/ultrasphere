
@overload
def harmonics_regular_singular(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array | NativeArray]
        | Mapping[Literal["r"], Array | NativeArray]
    ),
    *,
    k: Array | NativeArray,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = ...,
    harmonics: Mapping[TSpherical, Array | NativeArray],
    multiply: bool = True,
) -> Mapping[TSpherical, Array]: ...

@overload
def harmonics_regular_singular(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array | NativeArray]
        | Mapping[Literal["r"], Array | NativeArray]
    ),
    *,
    k: Array | NativeArray,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = ...,
    harmonics: Array | NativeArray,
    multiply: bool = True,
) -> Array: ...

def harmonics_regular_singular(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array | NativeArray]
        | Mapping[Literal["r"], Array | NativeArray]
    ),
    *,
    k: Array | NativeArray,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    harmonics: Array | NativeArray | Mapping[TSpherical, Array | NativeArray],
    multiply: bool = True,
) -> Array | Mapping[TSpherical, Array]:
    """
    Regular or singular harmonics.

    Parameters
    ----------
    spherical : Mapping[TSpherical | Literal['r'],
        Array | NativeArray] | Mapping[Literal['r'],
        Array | NativeArray]
        The spherical coordinates.
    k : Array | NativeArray
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
    is_mapping = isinstance(harmonics, Mapping)
    if multiply and is_mapping:
        raise ValueError("multiply must be False if harmonics is Mapping.")
    n_end, include_negative_m = (
        c.get_n_end_and_include_negative_m_from_expansion(harmonics)
    )
    n = c.index_array_harmonics(
        c.root, n_end=n_end, include_negative_m=include_negative_m
    )[(None,) * spherical["r"].ndim + (slice(None),)]
    kr = k * spherical["r"]
    kr = kr[..., None]

    if type == "regular":
        type = "j"
    elif type == "singular":
        type = "h1"
    val = szv(n, c.e_ndim, kr, type=type, derivative=derivative)
    val = ivy.nan_to_num(val, nan=0)
    expand_dims = not (is_mapping and len({h.ndim for h in harmonics.values()}) > 1)
    if expand_dims:
        idx = c.s_nodes.index(c.root)
        val = val[
            (..., *create_slice(c.s_ndim, [(idx, slice(None))], default=None))
        ]
    if is_mapping:
        res = {"r": val}
        if harmonics is not None:
            res.update(harmonics)  # type: ignore
        return res
    if multiply:
        return val * harmonics
    return val
