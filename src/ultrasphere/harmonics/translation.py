
from typing import Literal, Mapping

from array_api._2024_12 import Array
from ultrasphere.coordinates import SphericalCoordinates, TEuclidean, TSpherical
from array_api_compat import array_namespace

from ultrasphere.harmonics.helmholtz import harmonics_regular_singular
from ultrasphere.harmonics.index import index_array_harmonics
from .harmonics import harmonics as harmonics_
def harmonics_translation_coef(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    euclidean: Mapping[TEuclidean, Array],
    *,
    n_end: int,
    n_end_add: int,
    condon_shortley_phase: bool,
    k: Array,
) -> Array:
    r"""
    Translation coefficients between same type of elementary solutions.

    Returns (R|R) = (S|S), where

    .. math::
        R(x + t) = \sum_n (R|R)_n(t) R(x)
        S(x + t) = \sum_n (S|S)_n(t) S(x)

    Parameters
    ----------
    euclidean : Mapping[TEuclidean, Array]
        The translation vector in euclidean coordinates.
    n_end : int
        The maximum degree of the harmonic.
    n_end_add : int
        The maximum degree of the harmonic to be summed over.
    condon_shortley_phase : bool
        Whether to apply the Condon-Shortley phase.
    k : Array
        The wavenumber.

    Returns
    -------
    Array
        The translation coefficients of `2 * c.s_ndim` dimensions.
        [-c.s_ndim,-1] dimensions are to be
        summed over with the elementary solutions
        to get translated elementary solution
        which quantum number is [-2*c.s_ndim,-c.s_ndim-1] indices.

    """
    xp = array_namespace(*euclidean.values())
    _, k = xp.broadcast_arrays(euclidean[c.e_nodes[0]], k)
    n = index_array_harmonics(c, c.root, n_end=n_end, xp=xp, expand_dims=True)[
        (...,) + (None,) * c.s_ndim
    ]
    ns = index_array_harmonics(c, c.root, n_end=n_end_add,  xp=xp, expand_dims=True)[
        (None,) * c.s_ndim + (...,)
    ]

    def to_expand(spherical: Mapping[TSpherical, Array]) -> Array:
        # returns [spherical1,...,sphericalN,user1,...,userM,n1,...,nN]
        # [spherical1,...,sphericalN,n1,...,nN]
        harmonics = harmonics_(
            c,
            spherical,
            n_end,
            condon_shortley_phase=condon_shortley_phase,
            expand_dims=True,
            concat=True,
        )
        x = c.to_euclidean(spherical)
        ndim_user = euclidean[c.e_nodes[0]].ndim
        ndim_spherical = c.s_ndim
        ip = xp.sum(
            xp.stack(
                xp.broadcast_arrays(
                    *[
                        euclidean[i][
                            (None,) * ndim_spherical + (slice(None),) * ndim_user
                        ]
                        * x[i][
                            (slice(None),) * ndim_spherical + (None,) * ndim_user
                        ]
                        for i in c.e_nodes
                    ]
                ),
                axis=0,
            ),
            axis=0,
        )
        # [spherical1,...,sphericalN,user1,...,userM]
        e = xp.exp(
            1j * k[(None,) * ndim_spherical + (slice(None),) * ndim_user] * ip
        )
        result = (
            harmonics[
                (slice(None),) * ndim_spherical
                + (None,) * ndim_user
                + (slice(None),) * ndim_spherical
            ]
            * e[
                (slice(None),) * (ndim_spherical + ndim_user)
                + (None,) * ndim_spherical
            ]
        )
        return result

    # returns [user1,...,userM,n1,...,nN,np1,...,npN]
    return (-1j) ** (n - ns) * expand(c, 
        to_expand,
        does_f_support_separation_of_variables=False,
        n=n_end + n_end_add - 1,
        n_end=n_end_add,
        condon_shortley_phase=condon_shortley_phase,
    )

def harmonics_twins_expansion(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    *,
    n_end_1: int,
    n_end_2: int,
    condon_shortley_phase: bool,
    conj_1: bool = False,
    conj_2: bool = False,
    analytic: bool = False,
) -> Array:
    """
    Expansion coefficients of the twins of the harmonics.

    .. math:: Y_{n_1} Y_{n_2}

    Parameters
    ----------
    n_end_1 : int
        The maximum degree of the harmonic
        for the first harmonics.
    n_end_2 : int
        The maximum degree of the harmonic
        for the second harmonics.
    condon_shortley_phase : bool
        Whether to apply the Condon-Shortley phase.
    conj_1 : bool
        Whether to conjugate the first harmonics.
        by default False
    conj_2 : bool
        Whether to conjugate the second harmonics.
        by default False
    analytic : bool
        Whether to use the analytic formula,
        instead of the numerical integration,
        by default False

    Returns
    -------
    Array
        The expansion coefficients of the twins of shape
        [*1st quantum number, *2nd quantum number, *3rd quantum number]
        and dim `3 * c.s_ndim` and of dtype float, not complex.
        The n_end for 1st quantum number is `n_end_1`,
        The n_end for 2nd quantum number is `n_end_2`,
        The n_end for 3rd quantum number is `n_end_1 + n_end_2 - 1`.
        (not `n_end_1` or `n_end_2`)

    Notes
    -----
    To get ∫Y_{n1}(x)Y_{n2}(x)Y_{n3}(x)dx
    (integral involving three harmonics),
    one may use
    `harmonics_twins_expansion(conj_1=True, conj_2=True)`

    """
    if analytic:
        n1 = index_array_harmonics(c, c.root, n_end=n_end_1, expand_dims=True)
        n2 = index_array_harmonics(c, c.root, n_end=n_end_2, expand_dims=True)
        n3 = index_array_harmonics(c, 
            c.root, n_end=n_end_1 + n_end_2 - 1, expand_dims=True
        )
        if c.e_ndim == 2:
            n1 = n1[:, None, None]
            n2 = n2[None, :, None]
            n3 = n3[None, None, :]
            if conj_1:
                n1 = -n1
            if conj_2:
                n2 = -n2
            n3 = -n3
            result = (n1 + n2 + n3 == 0) / xp.sqrt(2 * xp.pi)
            if condon_shortley_phase:
                result *= (-1) ** (n1 + n2 + n3)
            return result
        elif c.e_ndim == 3:
            from py3nj import wigner3j

            another_node = (set(c.s_nodes) - {c.root}).pop()
            m1 = index_array_harmonics(c, 
                another_node, n_end=n_end_1, expand_dims=True
            )
            m2 = index_array_harmonics(c, 
                another_node, n_end=n_end_2, expand_dims=True
            )
            m3 = index_array_harmonics(c, 
                another_node, n_end=n_end_1 + n_end_2 - 1, expand_dims=True
            )
            n1 = n1[(...,) + (None,) * 4]
            m1 = m1[(...,) + (None,) * 4]
            n2 = n2[(None,) * 2 + (...,) + (None,) * 2]
            m2 = m2[(None,) * 2 + (...,) + (None,) * 2]
            n3 = n3[(None,) * 4 + (...,)]
            m3 = m3[(None,) * 4 + (...,)]
            if conj_1:
                m1 = -m1
            if conj_2:
                m2 = -m2
            m3 = -m3
            result = (
                xp.where(m1 <= 0, 1, (-1) ** xp.abs(m1))
                * xp.where(m2 <= 0, 1, (-1) ** xp.abs(m2))
                * xp.where(m3 <= 0, 1, (-1) ** xp.abs(m3))
                # * (n1 >= xp.abs(m1))
                # * (n2 >= xp.abs(m2))
                # * (n3 >= xp.abs(m3))
                * xp.sqrt(
                    (2 * n1 + 1) * (2 * n2 + 1) * (2 * n3 + 1) / (4 * xp.pi)
                )
                * wigner3j(
                    2 * n1,
                    2 * n2,
                    2 * n3,
                    2 * m1,
                    2 * m2,
                    2 * m3,
                    ignore_invalid=True,
                )
                * wigner3j(
                    2 * n1,
                    2 * n2,
                    2 * n3,
                    xp.zeros_like(n1, dtype=int),
                    xp.zeros_like(n2, dtype=int),
                    xp.zeros_like(n3, dtype=int),
                    ignore_invalid=True,
                )
            )
            if condon_shortley_phase:
                result *= (-1) ** (m1 + m2 + m3)
            return result

    def to_expand(spherical: Mapping[TSpherical, Array]) -> Array:
        # returns [theta,n1,...,nN,nsummed1,...,nsummedN]
        # Y(n)Y*(nsummed)
        Y1 = harmonics(c, 
            spherical,
            n_end_1,
            condon_shortley_phase=condon_shortley_phase,
            expand_dims=True,
            concat=False,
        )
        Y1 = {
            k: v[
                (slice(None),) * 1
                + (slice(None),) * c.s_ndim
                + (None,) * c.s_ndim
            ]
            for k, v in Y1.items()
        }
        if conj_1:
            Y1 = {k: v.conj() for k, v in Y1.items()}
        Y2 = harmonics(c, 
            spherical,
            n_end_2,
            condon_shortley_phase=condon_shortley_phase,
            expand_dims=True,
            concat=False,
        )
        Y2 = {
            k: v[
                (slice(None),) * 1
                + (None,) * c.s_ndim
                + (slice(None),) * c.s_ndim
            ]
            for k, v in Y2.items()
        }
        if conj_2:
            Y2 = {k: v.conj() for k, v in Y2.items()}
        return {k: Y1[k] * Y2[k] for k in c.s_nodes}

    # returns [user1,...,userM,n1,...,nN,np1,...,npN]
    return concat_harmonics(c, 
        expand_dims_harmonics(c, 
            expand(c, 
                to_expand,
                does_f_support_separation_of_variables=True,
                n=n_end_1 + n_end_2 - 1,  # at least n_end + 2
                n_end=n_end_1 + n_end_2 - 1,
                condon_shortley_phase=condon_shortley_phase,
            )
        )
    ).real

def harmonics_translation_coef_using_triplet(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: Mapping[TSpherical | Literal["r"], Array],
    *,
    n_end: int,
    n_end_add: int,
    condon_shortley_phase: bool,
    k: Array,
    is_type_same: bool,
) -> Array:
    r"""
    Translation coefficients between same or different type of elementary solutions.

    If is_type_same is True, returns (R|R) = (S|S).
    If is_type_same is False, returns (S|R).

    .. math::
        R(x + t) = \sum_n (R|R)_n(t) R(x)
        S(x + t) = \sum_n (S|S)_n(t) S(x)
        S(x + t) = \sum_n (S|R)_n(t) R(x)

    Parameters
    ----------
    spherical : Mapping[TSpherical, Array]
        The translation vector in spherical coordinates.
    n_end : int
        The maximum degree of the harmonic.
    n_end_add : int
        The maximum degree of the harmonic to be summed over.
    condon_shortley_phase : bool
        Whether to apply the Condon-Shortley phase.
    k : Array
        The wavenumber.
    is_type_same : bool
        Whether the type of the elementary solutions is same.

    Returns
    -------
    Array
        The translation coefficients of `2 * c.s_ndim` dimensions.
        [-c.s_ndim,-1] dimensions are to be
        summed over with the elementary solutions
        to get translated elementary solution
        which quantum number is [-2*c.s_ndim,-c.s_ndim-1] indices.

    """
    # [user1,...,userM,n1,...,nN,nsummed1,...,nsummedN,ntemp1,...,ntempN]
    n = index_array_harmonics(c, c.root, n_end=n_end, expand_dims=True)[
        (...,) + (None,) * (2 * c.s_ndim)
    ]
    ns = index_array_harmonics(c, c.root, n_end=n_end_add, expand_dims=True)[
        (None,) * c.s_ndim + (...,) + (None,) * c.s_ndim
    ]
    ntemp = index_array_harmonics(c, 
        c.root, n_end=n_end + n_end_add - 1, expand_dims=True
    )[(None,) * (2 * c.s_ndim) + (...,)]

    # returns [user1,...,userM,n1,...,nN,np1,...,npN]
    coef = (2 * xp.pi) ** (c.e_ndim / 2) * xp.sqrt(2 / xp.pi)
    t_Y = harmonics(c,   # type: ignore
        spherical,
        n_end=n_end + n_end_add - 1,
        condon_shortley_phase=condon_shortley_phase,
        expand_dims=True,
        concat=True,
    )
    t_RS = harmonics_regular_singular(c,
        spherical,
        harmonics=t_Y,
        k=k,
        type="regular" if is_type_same else "singular",
    )
    return coef * xp.sum(
        (-1j) ** (n - ns - ntemp)
        * t_RS[(...,) + (None,) * (2 * c.s_ndim) + (slice(None),) * c.s_ndim]
        * c.harmonics_twins_expansion(
            n_end_1=n_end,
            n_end_2=n_end_add,
            condon_shortley_phase=condon_shortley_phase,
            conj_1=False,
            conj_2=True,
            analytic=True,
        ),
        axis=list(range(-c.s_ndim, 0)),
    )
