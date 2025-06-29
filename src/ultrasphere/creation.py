from collections.abc import Sequence
from functools import lru_cache
from ultrasphere.coordinates import BranchingType, SphericalCoordinates, get_digraph_from_branching_type
import numpy as np

import networkx as nx


from typing import Any, Literal


@classmethod
def polar(cls) -> 'SphericalCoordinates[Literal["phi"], Literal[0, 1]]':
    """
    Polar coordinates.

    Returns
    -------
    SphericalCoordinates
        The polar coordinates.

    """
    G = get_digraph_from_branching_type("b")
    G = nx.relabel_nodes(G, {"theta0", "phi"})
    return cls(G)  # type: ignore


@classmethod
def spherical(
    cls,
) -> 'SphericalCoordinates[Literal["theta", "phi"], Literal[0, 1, 2]]':
    """
    Spherical coordinates.

    Returns
    -------
    SphericalCoordinates
        The spherical coordinates.

    """
    G = get_digraph_from_branching_type("ba")
    # swap x0 and x2
    G = nx.relabel_nodes(G, {0: 2, 2: 1, 1: 0, "theta0": "theta", "theta1": "phi"})
    return cls(G)  # type: ignore


@classmethod
def standard(cls, s_ndim: int) -> "SphericalCoordinates[Any, Any]":
    """
    Standard spherical coordinates.

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
        return cls.from_branching_types("")
    return cls(get_digraph_from_branching_type("b" * (s_ndim - 1) + "a"))


@classmethod
def standard_prime(cls, s_ndim: int) -> "SphericalCoordinates[Any, Any]":
    """
    Standard prime spherical coordinates.

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
        return cls.from_branching_types("")
    return cls(get_digraph_from_branching_type("bp" * (s_ndim - 1) + "a"))


@classmethod
def hopf(cls, q: int) -> "SphericalCoordinates[Any, Any]":
    """
    Hopf coordinates.

    Parameters
    ----------
    q : int
        Where 2^q = c.e_ndim.

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

    return cls(get_digraph_from_branching_type(_hoph(q)))


@classmethod
def from_branching_types(
    cls, branching_types: str | Sequence[BranchingType]
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
    return cls(get_digraph_from_branching_type(branching_types))


def s_node_name(idx: int) -> Any:
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


def e_node_name(idx: int) -> Any:
    """
    The naming convention for the Euclidean node.

    Parameters
    ----------
    idx : int
        The index of the Euclidean node.

    Returns
    -------
    str
        The name of the Euclidean node.

    """
    return idx


def get_random_digraph(
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
        {node: e_node_name(i) for i, node in enumerate(leaf_nodes)}
        | {node: s_node_name(i) for i, node in enumerate(non_leaf_nodes)},
    )
    return G


@classmethod
def random(
    cls, s_ndim: int, *, rng: np.random.Generator | None = None
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
    return cls(get_random_digraph(s_ndim, rng=rng))