from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypeVar, overload

import networkx as nx
import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from strenum import StrEnum

from .special import lgamma

TEuclidean = TypeVar("TEuclidean")
TSpherical = TypeVar("TSpherical")
COSSIN = Literal["cos", "sin"]


class BranchingType(StrEnum):
    """
    The branching types of the nodes in a rooted tree representing the coordinates.

    (Vilenkin's method of trees)
    """

    A = "a"
    B = "b"
    BP = "b'"
    C = "c"


def check_tree(graph: nx.DiGraph, /) -> None:
    """
    Check if the graph is a valid tree representing the coordinates.
    Definition of "a tree representing the coordinates":
    - The graph is a tree.
    - The graph is rooted.
    - Each node has 0 or 2 successors.
      (full binary tree, not necessarily perfect binary tree)
    - If the node has 2 successors, one of the outgoing edges has
      attribute "type" of "cos" and
      the other has attribute "type" of "sin".

    Parameters
    ----------
    graph : nx.DiGraph
        The graph representing the coordinates.

    Raises
    ------
    ValueError
        If the graph is not a valid tree.

    """
    # check if the graph is a tree
    if not nx.is_tree(graph):
        raise ValueError("The graph is not a tree.")
    # check if the graph is rooted
    if not nx.is_arborescence(graph):
        raise ValueError("The graph is not rooted.")
    # check every node has 0 or 2 successors
    for node in graph.nodes:
        if (degree := graph.out_degree(node)) not in {0, 2}:
            raise ValueError(
                f"Each node should have 0 or 2 successors, "
                f"but {node} has {degree} successors."
            )
    # check every node except the root has attribute
    # "direction" of type Literal["cos", "sin"]
    for node in graph.nodes:
        if graph.out_degree(node) == 0:
            continue
        if {e[2] for e in graph.out_edges(nbunch=node, data="type")} != {
            "cos",
            "sin",
        }:
            raise ValueError(
                f"Node {node} should have 0 or 2 successors "
                "with types 'cos' and 'sin', "
                f"got {set(graph.out_edges.data(nbunch=node, data='type'))}."
            )


def get_non_leaf_descendants(graph: nx.DiGraph, /) -> dict[Any, int]:
    """
    Calculate the number of non-leaf descendants for each node in a rooted tree.

    Parameters
    ----------
    graph : nx.DiGraph
        A rooted tree.

    Returns
    -------
    dict[Any, int]
        NaN for leaf nodes, otherwise the number of non-leaf descendants.

    """
    # calculate number of non-leaf descendants for each node
    leaf_nodes = {node for node in graph.nodes if graph.out_degree(node) == 0}
    non_leaf_descendants = {}
    for node in reversed(list(nx.topological_sort(graph))):
        if node in leaf_nodes:
            non_leaf_descendants[node] = -1
        else:
            non_leaf_descendants[node] = sum(
                non_leaf_descendants[successor] + 1
                for successor in graph.successors(node)
            )
    for node in leaf_nodes:
        non_leaf_descendants[node] = np.nan
    return non_leaf_descendants


def get_child(G: nx.DiGraph, node: Any, type: COSSIN, /) -> Any:
    """
    Get the child node of the given type in a rooted tree representing the coordinates.

    Parameters
    ----------
    G : nx.DiGraph
        A rooted tree representing the coordinates.
    node : Any
        The node to get the child.
    type : COSSIN
        The type of the child._description_

    Raises
    ------
    ValueError
        If the node has no child of the given type.
        If the node has multiple children of the given type.

    """
    child_candidates = [
        child for child in G.successors(node) if G.edges[node, child]["type"] == type
    ]
    if len(child_candidates) == 0:
        raise ValueError(f"No {type} child for node {node}.")
    elif len(child_candidates) > 1:
        raise ValueError(f"Multiple {type} children for node {node}.")
    return child_candidates[0]


def get_parent(G: nx.DiGraph, node: Any, /) -> Any | None:
    """
    Get the parent node in a rooted tree representing the coordinates.

    Parameters
    ----------
    G : nx.DiGraph
        A rooted tree representing the coordinates.
    node : Any
        The node to get the parent.

    Returns
    -------
    Any
        The parent node if exists, otherwise None.

    Raises
    ------
    ValueError
        If the node has multiple parents.

    """
    predecessors = list(G.predecessors(node))
    if len(predecessors) == 0:
        return None
    elif len(predecessors) > 1:
        raise ValueError("The node has multiple parents.")
    return predecessors[0]


def get_branching_types(G: nx.DiGraph, /) -> dict[Any, BranchingType]:
    """
    Get the branching types of each node in a rooted tree.

    Parameters
    ----------
    G : nx.DiGraph
        A rooted tree.

    Returns
    -------
    dict[Any, BranchingType]
        The branching types of each node.

    """
    branching_types = {}
    for node in G.nodes:
        if G.out_degree(node) == 0:
            continue
        sin_child = get_child(G, node, "sin")
        cos_child = get_child(G, node, "cos")
        sin_child_is_leaf = G.out_degree(sin_child) == 0
        cos_child_is_leaf = G.out_degree(cos_child) == 0
        if cos_child_is_leaf and sin_child_is_leaf:
            branching_types[node] = BranchingType.A
        elif cos_child_is_leaf and not sin_child_is_leaf:
            branching_types[node] = BranchingType.B
        elif not cos_child_is_leaf and sin_child_is_leaf:
            branching_types[node] = BranchingType.BP
        else:
            branching_types[node] = BranchingType.C
    return branching_types


def get_branching_type_from_digraph(G: nx.DiGraph, /) -> Iterable[BranchingType]:
    """
    Get the branching types of each node in a rooted tree.

    Parameters
    ----------
    G : nx.DiGraph
        A rooted tree.

    Returns
    -------
    Sequence[BranchingType]
        The branching types of each node.

    """
    root = next(nx.topological_sort(G))
    stack = [root]
    while stack:
        node = stack.pop()
        if G.out_degree(node) == 0:
            raise AssertionError()
        cos_child = get_child(G, node, "cos")
        sin_child = get_child(G, node, "sin")
        cos_child_is_leaf = G.out_degree(cos_child) == 0
        sin_child_is_leaf = G.out_degree(sin_child) == 0
        if cos_child_is_leaf and sin_child_is_leaf:
            branching_type = BranchingType.A
        elif cos_child_is_leaf and not sin_child_is_leaf:
            branching_type = BranchingType.B
            stack.append(sin_child)
        elif not cos_child_is_leaf and sin_child_is_leaf:
            branching_type = BranchingType.BP
            stack.append(cos_child)
        else:
            branching_type = BranchingType.C
            stack.extend([sin_child, cos_child])
        yield branching_type


class SphericalCoordinates[TSpherical, TEuclidean]:
    """Stores the spherical coordinates using the method of trees by Vilenkin."""

    G: nx.DiGraph
    """The rooted tree representing the coordinates."""
    S: Mapping[TSpherical, int]
    """The number of non-leaf descendants for each spherical node."""
    branching_types: Mapping[TSpherical, BranchingType]
    """The branching types of each node."""
    s_nodes: Sequence[TSpherical]
    """The spherical nodes."""
    e_nodes: Sequence[TEuclidean]
    """The Euclidean nodes."""
    cos_edges: Sequence[tuple[TSpherical, TEuclidean | TSpherical]]
    """The edges with type 'cos'."""
    sin_edges: Sequence[tuple[TSpherical, TEuclidean | TSpherical]]
    """The edges with type 'sin'."""

    def __hash__(self) -> int:
        """__hash__."""
        return hash(self.G)

    def __getstate__(self) -> dict[Any, Any]:
        """__getstate__."""
        return {"G": nx.to_dict_of_dicts(self.G)}

    def __setstate__(self, state: dict[Any, Any]) -> None:
        """__setstate__."""
        return self.__init__(nx.from_dict_of_dicts(state["G"]))  # type: ignore

    def __eq__(self, other: Any) -> bool:
        """Check if the spherical coordinates are equal."""
        if not isinstance(other, SphericalCoordinates):
            return False
        return self.G == other.G

    def __repr__(self) -> str:
        """Print the spherical coordinates."""
        return f"SphericalCoordinates({self.branching_types_expression_str})"

    def __str__(self) -> str:
        """Print the spherical coordinates."""
        return f"SphericalCoordinates({self.branching_types_expression_str})"

    @property
    def root(self) -> Any:
        """The root node."""
        return next(nx.topological_sort(self.G))

    @property
    def root_index(self) -> int:
        """The index of the root node."""
        return self.s_nodes.index(self.root)

    @property
    def branching_types_expression(self) -> Sequence[BranchingType]:
        """The branching types."""
        return list(get_branching_type_from_digraph(self.G))

    @property
    def branching_types_expression_str(self) -> str:
        """The branching types as a string."""
        return "".join(
            branching_type.value for branching_type in self.branching_types_expression
        )

    @property
    def s_ndim(self) -> int:
        """The number of spherical dimensions."""
        return len(self.s_nodes)

    @property
    def e_ndim(self) -> int:
        """The number of Euclidean dimensions."""
        return len(self.e_nodes)

    def __init__(self, tree: nx.DiGraph, cache: bool | None = None) -> None:
        """
        Initialize the spherical coordinates.

        Parameters
        ----------
        tree : nx.DiGraph
            The rooted tree representing the coordinates.

            Definition of "a tree representing the coordinates":
            - The graph is a tree.
            - The graph is rooted.
            - Each node has 0 or 2 successors.
            (full binary tree, not necessarily perfect binary tree)
            - If the node has 2 successors, one of the outgoing edges has
            attribute "type" of "cos" and
            the other has attribute "type" of "sin".
        cache : bool | None, optional
            Whether to cache the harmonics twins expansion, by default None

        """
        check_tree(tree)
        self.G = tree
        self.S = get_non_leaf_descendants(tree)
        self.branching_types = get_branching_types(tree)
        nodes = list(nx.topological_sort(tree))
        # even if sorted by name, e_nodes are still topologically sorted
        # because they do not have any successors
        self.e_nodes = sorted(node for node in nodes if tree.out_degree(node) == 0)
        self.s_nodes = [node for node in nodes if tree.out_degree(node) == 2]
        self.cos_edges = [e for e in self.G.edges if self.G.edges[e]["type"] == "cos"]
        self.sin_edges = [e for e in self.G.edges if self.G.edges[e]["type"] == "sin"]

    def surface_area(self, r: float = 1) -> float:
        """
        The surface area of the unit sphere.

        Returns
        -------
        float
            The surface area.

        """
        return (
            2
            * (np.pi ** (self.e_ndim / 2))
            / np.exp(lgamma(self.e_ndim / 2.0))
            * r**self.s_ndim
        )

    def volume(self, r: float = 1) -> float:
        """
        The volume of the unit sphere.

        Returns
        -------
        float
            The volume.

        References
        ----------
        McLean, W. (2000). Strongly Elliptic Systems and
        Boundary Integral Equations. p.247 (Upsilon_n)

        """
        return (
            (np.pi ** (self.e_ndim / 2))
            / np.exp(lgamma(self.e_ndim / 2.0 + 1.0))
            * r**self.e_ndim
        )

    def from_euclidean(
        self, euclidean: Mapping[TEuclidean, Array]
    ) -> Mapping[TSpherical | Literal["r"], Array]:
        """
        Convert the Euclidean coordinates to the spherical coordinates.

        Parameters
        ----------
        euclidean : Mapping[TEuclidean, Array]
            The Euclidean coordinates.

        Returns
        -------
        Array
            The spherical coordinates.

        """
        xp = array_namespace(*[euclidean[k] for k in self.e_nodes])
        r = (
            xp.linalg.vector_norm(
                xp.stack(xp.broadcast_arrays(*[euclidean[k] for k in self.e_nodes])),
                axis=0,
            )
            if self.s_ndim > 0
            else euclidean[self.e_nodes[0]]
        )
        result: dict[TSpherical | Literal["r"], Array] = {"r": r}
        tmp = {k: euclidean[k] for k in self.e_nodes}
        for node in reversed(list(nx.topological_sort(self.G))):
            if node in self.e_nodes:
                continue
            elif node not in self.s_nodes:
                raise AssertionError()
            cos_child = get_child(self.G, node, "cos")
            sin_child = get_child(self.G, node, "sin")
            cos = tmp[cos_child]
            sin = tmp[sin_child]
            result[node] = xp.atan2(sin, cos)
            tmp[node] = xp.linalg.vector_norm(
                xp.stack(xp.broadcast_arrays(sin, cos)), axis=0
            )
        return result

    @overload
    def to_euclidean(
        self,
        spherical: (
            Mapping[TSpherical | Literal["r"], Array] | Mapping[TSpherical, Array]
        ),
        as_array: Literal[False] = ...,
    ) -> Mapping[TEuclidean, Array]: ...

    @overload
    def to_euclidean(
        self,
        spherical: (
            Mapping[TSpherical | Literal["r"], Array] | Mapping[TSpherical, Array]
        ),
        as_array: Literal[True] = ...,
    ) -> Array: ...

    def to_euclidean(
        self,
        spherical: (
            Mapping[TSpherical | Literal["r"], Array] | Mapping[TSpherical, Array]
        ),
        as_array: bool = False,
    ) -> Mapping[TEuclidean, Array] | Array:
        """
        Convert the spherical coordinates to the Euclidean coordinates.

        Parameters
        ----------
        spherical : Mapping[TSpherical, Array]
            The spherical coordinates.
        as_array : bool, optional
            Whether to return as an array, by default False


        Returns
        -------
        Array
            The Euclidean coordinates.

        """
        xp = array_namespace(*[spherical[k] for k in self.s_nodes])
        result = {self.root: spherical.get("r", 1)}  # type: ignore
        for node in nx.topological_sort(self.G):
            if node in self.e_nodes:
                continue
            elif node not in self.s_nodes:
                raise AssertionError()
            cos_child = get_child(self.G, node, "cos")
            sin_child = get_child(self.G, node, "sin")
            cos = xp.cos(spherical[node])
            sin = xp.sin(spherical[node])
            result[cos_child] = result[node] * cos
            result[sin_child] = result[node] * sin
        result = {k: result[k] for k in self.e_nodes}
        if as_array:
            return xp.stack(
                xp.broadcast_arrays(*[result[k] for k in self.e_nodes]),
                axis=0,
            )

        return result

    @property
    def is_e_keys_range(self) -> bool:
        """Whether the Euclidean keys are 0, 1, ..., e_ndim - 1."""
        return set(self.e_nodes) == set(range(self.e_ndim))

    @property
    def is_s_keys_range(self) -> bool:
        """Whether the spherical keys are 0, 1, ..., s_ndim - 1."""
        return set(self.s_nodes) == set(range(self.s_ndim))

    def jacobian(
        self,
        spherical: (
            Mapping[TSpherical | Literal["r"], Array] | Mapping[TSpherical, Array]
        ),
    ) -> Array:
        """
        Calculate the Jacobian of the spherical coordinates.

        Parameters
        ----------
        spherical : Mapping[TSpherical, Array]
            The spherical coordinates.

        Returns
        -------
        Array
            The Jacobian of the spherical coordinates.

        """
        xp = array_namespace(*[spherical[k] for k in self.s_nodes])
        if "r" in spherical:
            jacobian = spherical["r"]  # type: ignore
        else:
            jacobian = 1
        for node in self.s_nodes:
            cos = xp.cos(spherical[node])
            sin = xp.sin(spherical[node])
            cos_child = get_child(self.G, node, "cos")
            sin_child = get_child(self.G, node, "sin")
            jacobian *= cos ** (self.S.get(cos_child, -1) + 1) * sin ** (
                self.S.get(sin_child, -1) + 1
            )
        return jacobian
