from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal

import networkx as nx
import numpy as np

from ultrasphere._coordinates import BranchingType, SphericalCoordinates
from ultrasphere._coordinates import SphericalCoordinates as cls


def _get_digraph_from_branching_type(
    branching_types: str | Sequence[BranchingType],
) -> nx.DiGraph:
    """
    Get a rooted tree from the branching types.

    Parameters
    ----------
    branching_types : str | Sequence[BranchingType]
        The branching types. e.g. "ba" for standard spherical coordinates.

    Returns
    -------
    nx.DiGraph
        The rooted tree representing the coordinates.

    Raises
    ------
    ValueError
        If the branching types are invalid.

    """
    if isinstance(branching_types, str):
        branching_types_str = branching_types.replace("bp", "b'")
        branching_types_: list[BranchingType] = []
        while branching_types_str:
            if branching_types_str.startswith("b'"):
                branching_types_.append(BranchingType.BP)
                branching_types_str = branching_types_str[2:]
            elif branching_types_str[0] in ["a", "b", "c"]:
                branching_types_.append(BranchingType(branching_types_str[0]))
                branching_types_str = branching_types_str[1:]
            else:
                raise ValueError(f"Invalid branching type: {branching_types_str}")
    else:
        branching_types_ = list(branching_types)

    G = nx.DiGraph()
    type_c_stack: list[Any] = []
    next_e_idx = 0
    next_s_idx = 0
    current_node = _s_node_name_default(next_s_idx)
    G.add_node(current_node)
    next_s_idx += 1
    for i, branching_type in enumerate(branching_types_):
        if branching_type == BranchingType.A:
            G.add_node(next_e_idx)
            G.add_edge(current_node, next_e_idx, type="cos")
            next_e_idx += 1
            G.add_node(next_e_idx)
            G.add_edge(current_node, next_e_idx, type="sin")
            next_e_idx += 1
            if i == len(branching_types_) - 1:
                break
            else:
                try:
                    next_node = type_c_stack.pop()
                except IndexError as e:
                    raise ValueError("Invalid branching types.") from e
        elif branching_type == BranchingType.B:
            G.add_node(next_e_idx)
            G.add_edge(current_node, next_e_idx, type="cos")
            next_e_idx += 1
            G.add_node(_s_node_name_default(next_s_idx))
            G.add_edge(current_node, _s_node_name_default(next_s_idx), type="sin")
            next_node = _s_node_name_default(next_s_idx)
            next_s_idx += 1
        elif branching_type == BranchingType.BP:
            G.add_node(_s_node_name_default(next_s_idx))
            G.add_edge(current_node, _s_node_name_default(next_s_idx), type="cos")
            next_node = _s_node_name_default(next_s_idx)
            next_s_idx += 1
            G.add_node(next_e_idx)
            G.add_edge(current_node, next_e_idx, type="sin")
            next_e_idx += 1
        elif branching_type == BranchingType.C:
            G.add_node(_s_node_name_default(next_s_idx))
            G.add_edge(current_node, _s_node_name_default(next_s_idx), type="cos")
            next_node = _s_node_name_default(next_s_idx)
            next_s_idx += 1
            G.add_node(_s_node_name_default(next_s_idx))
            G.add_edge(current_node, _s_node_name_default(next_s_idx), type="sin")
            type_c_stack.append(_s_node_name_default(next_s_idx))
            next_s_idx += 1
        current_node = next_node
    return G


def create_polar() -> 'SphericalCoordinates[Literal["phi"], Literal[0, 1]]':
    r"""
    Polar coordinates.

    .. math::
        x_0 &= r \cos(\phi) \\
        x_1 &= r \sin(\phi)

    Returns
    -------
    SphericalCoordinates
        The polar coordinates.

    """
    G = _get_digraph_from_branching_type("a")
    G = nx.relabel_nodes(G, {"theta0": "phi"})
    return cls(G)


def create_spherical() -> (
    'SphericalCoordinates[Literal["theta", "phi"], Literal[0, 1, 2]]'
):
    r"""
    Spherical coordinates.

    .. math::
        x_0 &= r \sin(\theta) \cos(\phi) \\
        x_1 &= r \sin(\theta) \sin(\phi) \\
        x_2 &= r \cos(\theta)

    Returns
    -------
    SphericalCoordinates
        The spherical coordinates.

    """
    G = _get_digraph_from_branching_type("ba")
    # swap x0 and x2
    G = nx.relabel_nodes(G, {0: 2, 2: 1, 1: 0, "theta0": "theta", "theta1": "phi"})
    return cls(G)


def create_standard(s_ndim: int) -> "SphericalCoordinates[Any, Any]":
    r"""
    Standard spherical coordinates.

    .. math::
        x_0 &= \cos(\theta_0) \\
        x_1 &= \sin(\theta_0) \cos(\theta_1) \\
        x_2 &= \sin(\theta_0) \sin(\theta_1) \cos(\theta_2) \\
        x_3 &= \sin(\theta_0) \sin(\theta_1) \sin(\theta_2) \cos(\theta_3) \\
        &\vdots \\

    Parameters
    ----------
    s_ndim : int
        The number of spherical dimensions.

    Returns
    -------
    SphericalCoordinates
        The standard coordinates.

    """
    if s_ndim == 0:
        return create_from_branching_types("")
    return cls(_get_digraph_from_branching_type("b" * (s_ndim - 1) + "a"))


def create_standard_prime(s_ndim: int) -> "SphericalCoordinates[Any, Any]":
    r"""
    Standard prime spherical coordinates.

    .. math::
        x_0 &= \sin(\theta_0) \\
        x_1 &= \cos(\theta_0) \sin(\theta_1) \\
        x_2 &= \cos(\theta_0) \cos(\theta_1) \sin(\theta_2) \\
        x_3 &= \cos(\theta_0) \cos(\theta_1) \cos(\theta_2) \sin(\theta_3) \\
        &\vdots \\

    Parameters
    ----------
    s_ndim : int
        The number of spherical dimensions.

    Returns
    -------
    SphericalCoordinates
        The standard prime coordinates.

    """
    if s_ndim == 0:
        return create_from_branching_types("")
    return cls(_get_digraph_from_branching_type("bp" * (s_ndim - 1) + "a"))


def create_hopf(q: int) -> "SphericalCoordinates[Any, Any]":
    """
    Hopf coordinates.

    Parameters
    ----------
    q : int
        Where 2^q = c.c_ndim.

    Returns
    -------
    SphericalCoordinates
        The Hopf coordinates.

    """

    @lru_cache
    def _hoph(q: int) -> str:
        if q < 0:
            raise ValueError("q should be non-negative.")
        elif q == 0:
            return ""
        elif q == 1:
            return "a"
        return f"c{_hoph(q - 1)}{_hoph(q - 1)}"

    return cls(_get_digraph_from_branching_type(_hoph(q)))


def create_from_branching_types(
    branching_types: str | Sequence[BranchingType],
) -> "SphericalCoordinates[Any, Any]":
    """
    Spherical coordinates from branching types.

    Parameters
    ----------
    branching_types : str | Sequence[BranchingType]
        The branching types. e.g. "ba" for standard spherical coordinates.

    Returns
    -------
    SphericalCoordinates
        The spherical coordinates.

    """
    return cls(_get_digraph_from_branching_type(branching_types))


def _s_node_name_default(idx: int) -> Any:
    """
    The naming convention for the spherical node.

    Parameters
    ----------
    idx : int
        The index of the spherical node.

    Returns
    -------
    str
        The name of the spherical node.

    """
    return f"theta{idx}"


def _e_node_name_default(idx: int) -> Any:
    """
    The naming convention for the Cartesian node.

    Parameters
    ----------
    idx : int
        The index of the Cartesian node.

    Returns
    -------
    str
        The name of the Cartesian node.

    """
    return idx


def _get_random_digraph(
    s_ndim: int, *, rng: np.random.Generator | None = None
) -> nx.DiGraph:
    """
    Get a random rooted tree representing the coordinates.

    Parameters
    ----------
    s_ndim : int
        The number of spherical dimensions.
    rng : np.random.Generator | None, optional
        The random number generator, by default None

    Returns
    -------
    nx.DiGraph
        The rooted tree representing the coordinates.

    """
    rng = np.random.default_rng() if rng is None else rng
    G = nx.DiGraph()
    leaf_nodes = [0]
    G.add_node(0)
    for _ in range(s_ndim):
        node_parent = rng.choice(leaf_nodes)
        leaf_nodes.remove(node_parent)
        node_cos = len(G)
        node_sin = len(G) + 1

        for type, node in [("cos", node_cos), ("sin", node_sin)]:
            G.add_node(node)
            G.add_edge(node_parent, node, type=type)
            leaf_nodes.append(node)
    non_leaf_nodes = set(G.nodes) - set(leaf_nodes)
    G = nx.relabel_nodes(
        G,
        {node: _e_node_name_default(i) for i, node in enumerate(leaf_nodes)}
        | {node: _s_node_name_default(i) for i, node in enumerate(non_leaf_nodes)},
    )
    return G


def create_random(
    s_ndim: int, *, rng: np.random.Generator | None = None
) -> "SphericalCoordinates[Any, Any]":
    """
    Get a random spherical coordinates.

    Parameters
    ----------
    s_ndim : int
        The number of spherical dimensions.
    rng : np.random.Generator | None, optional
        The random number generator, by default None

    Returns
    -------
    SphericalCoordinates
        The random spherical coordinates.

    """
    return cls(_get_random_digraph(s_ndim, rng=rng))
