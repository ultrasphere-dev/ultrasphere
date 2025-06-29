
from typing import Any, Callable, Literal, Mapping, overload
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from ultrasphere.coordinates import BranchingType, SphericalCoordinates, TEuclidean, TSpherical, get_child
from ultrasphere.harmonics.assume import get_n_end_and_include_negative_m_from_expansion
import array_api_extra as xpx

from ultrasphere.harmonics.expansion import concat_harmonics
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
                and c.branching_types[get_child(c.G, node, "sin")]
                == BranchingType.A,
            )
        elif c.branching_types[node] == BranchingType.BP:
            result[node] = type_bdash(
                value,
                n_end=n_end,
                s_alpha=c.S[get_child(c.G, node, "cos")],
                index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                is_alpha_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "cos")]
                == BranchingType.A,
            )
        elif c.branching_types[node] == BranchingType.C:
            result[node] = type_c(
                value,
                n_end=n_end,
                s_alpha=c.S[get_child(c.G, node, "cos")],
                s_beta=c.S[get_child(c.G, node, "sin")],
                index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                is_alpha_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "cos")]
                == BranchingType.A,
                is_beta_type_a_and_include_negative_m=include_negative_m
                and c.branching_types[get_child(c.G, node, "sin")]
                == BranchingType.A,
            )
        else:
            raise ValueError(
                f"Invalid branching type {c.branching_types[node]}."
            )
    if expand_dims:
        result = expand_dims_harmonics(c, result)  # type: ignore
    if concat:
        result = concat_harmonics(c, result)
    return result
