import sys
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any, Generic, Literal, TypeVar, overload

import ivy
import networkx as nx
import numpy as np
from ivy import Array, NativeArray
from joblib.memory import Memory
from networkx.drawing.nx_pydot import graphviz_layout
from numpy.typing import NDArray
from shift_nth_row_n_steps import create_slice
from strenum import StrEnum

from .special import szv
from .symmetry import to_symmetric
from .wave_expansion import harm_n_ndim, homogeneous_ndim


def random_sphere(
    shape: Sequence[int],
    dim: int,
    *,
    surface: bool = False,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    r"""
    Generate random points on / in a unit hypersphere.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the output array.
    dim : int
        The dimension of the hypersphere.
    surface : bool, optional
        Whether to generate points on the surface of the hypersphere,
        by default False.
    rng : np.random.Generator | None, optional
        The random number generator, by default None.

    Returns
    -------
    NDArray[np.float64]
        The generated points.

    References
    ----------
        Barthe, F., Guedon, O., Mendelson, S., & Naor, A. (2005).
        A probabilistic approach to the geometry of the \ell_p^n-ball.
        arXiv, math/0503650. Retrieved from https://arxiv.org/abs/math/0503650v1

    """
    rng = np.random.default_rng() if rng is None else rng
    g = rng.normal(loc=0, scale=np.sqrt(1 / 2), size=(dim, *shape))
    if surface:
        result = g / np.linalg.vector_norm(g, axis=0)
    else:
        z = rng.exponential(scale=1, size=shape)[None, ...]
        result = g / np.sqrt(np.sum(g**2, axis=0, keepdims=True) + z)
    return np.nan_to_num(result, nan=0.0)  # not sure if nan_to_num is necessary


class BranchingType(StrEnum):
    """
    The branching types of the nodes in a rooted tree representing the coordinates.

    (Vilenkin's method of trees)
    """

    A = "a"
    B = "b"
    BP = "b'"
    C = "c"


TEuclidean = TypeVar("TEuclidean")
TSpherical = TypeVar("TSpherical")
COSSIN = Literal["cos", "sin"]


def check_tree(graph: nx.DiGraph) -> None:
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


def get_non_leaf_descendants(graph: nx.DiGraph) -> dict[Any, int]:
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
        non_leaf_descendants[node] = ivy.nan
    return non_leaf_descendants


def get_child(G: nx.DiGraph, node: Any, type: COSSIN) -> Any:
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


def get_parent(G: nx.DiGraph, node: Any) -> Any | None:
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


def get_branching_types(graph: nx.DiGraph) -> dict[Any, BranchingType]:
    """
    Get the branching types of each node in a rooted tree.

    Parameters
    ----------
    graph : nx.DiGraph
        A rooted tree.

    Returns
    -------
    dict[Any, BranchingType]
        The branching types of each node.

    """
    branching_types = {}
    for node in graph.nodes:
        if graph.out_degree(node) == 0:
            continue
        sin_child = get_child(graph, node, "sin")
        cos_child = get_child(graph, node, "cos")
        sin_child_is_leaf = graph.out_degree(sin_child) == 0
        cos_child_is_leaf = graph.out_degree(cos_child) == 0
        if cos_child_is_leaf and sin_child_is_leaf:
            branching_types[node] = BranchingType.A
        elif cos_child_is_leaf and not sin_child_is_leaf:
            branching_types[node] = BranchingType.B
        elif not cos_child_is_leaf and sin_child_is_leaf:
            branching_types[node] = BranchingType.BP
        else:
            branching_types[node] = BranchingType.C
    return branching_types


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


def get_digraph_from_branching_type(
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
    current_node = s_node_name(next_s_idx)
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
            G.add_node(s_node_name(next_s_idx))
            G.add_edge(current_node, s_node_name(next_s_idx), type="sin")
            next_node = s_node_name(next_s_idx)
            next_s_idx += 1
        elif branching_type == BranchingType.BP:
            G.add_node(s_node_name(next_s_idx))
            G.add_edge(current_node, s_node_name(next_s_idx), type="cos")
            next_node = s_node_name(next_s_idx)
            next_s_idx += 1
            G.add_node(next_e_idx)
            G.add_edge(current_node, next_e_idx, type="sin")
            next_e_idx += 1
        elif branching_type == BranchingType.C:
            G.add_node(s_node_name(next_s_idx))
            G.add_edge(current_node, s_node_name(next_s_idx), type="cos")
            next_node = s_node_name(next_s_idx)
            next_s_idx += 1
            G.add_node(s_node_name(next_s_idx))
            G.add_edge(current_node, s_node_name(next_s_idx), type="sin")
            type_c_stack.append(s_node_name(next_s_idx))
            next_s_idx += 1
        current_node = next_node
    return G


def get_branching_type_from_digraph(G: nx.DiGraph) -> Iterable[BranchingType]:
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


class SphericalCoordinates(Generic[TSpherical, TEuclidean]):
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
            Where 2^q = self.e_ndim.

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
        if cache is True or (cache is None and "pytest" not in sys.modules):
            self.memory = Memory(
                location=f".cache/{self.__class__.__name__}", verbose=1
            )
            self.harmonics = self.memory.cache(  # type: ignore
                self.harmonics, verbose=0
            )
            # self.harmonics_regular_singular = self.memory.cache(  # type: ignore
            #     self.harmonics_regular_singular, verbose=0
            # )
            self.harmonics_twins_expansion = self.memory.cache(  # type: ignore
                self.harmonics_twins_expansion, verbose=0
            )
            self.harmonics_translation_coef = self.memory.cache(  # type: ignore
                self.harmonics_translation_coef, verbose=0
            )
            self.harmonics_translation_coef_using_triplet = self.memory.cache(  # type: ignore
                self.harmonics_translation_coef_using_triplet, verbose=0
            )
        else:
            warnings.warn(
                "The harmonics twins expansion is not cached.",
                RuntimeWarning,
                stacklevel=2,
            )

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
            * (ivy.pi ** (self.e_ndim / 2))
            / ivy.exp(ivy.lgamma(self.e_ndim / 2.0))
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
            (ivy.pi ** (self.e_ndim / 2))
            / ivy.exp(ivy.lgamma(self.e_ndim / 2.0 + 1.0))
            * r**self.e_ndim
        )

    def homogeneous_ndim(self, n: int | Array | NativeArray) -> int | Array:
        """
        The dimension of the homogeneous polynomials of degree n.

        Parameters
        ----------
        n : int | Array | NativeArray
            The degree.

        Returns
        -------
        int | Array
            The dimension.

        References
        ----------
        McLean, W. (2000). Strongly Elliptic Systems and
        Boundary Integral Equations. p.250

        """
        return homogeneous_ndim(n, e_ndim=self.e_ndim)

    def harm_n_ndim(self, n: int | Array | NativeArray) -> int | Array:
        """
        The dimension of the spherical harmonics of degree n.

        Parameters
        ----------
        n : int | Array | NativeArray
            The degree.

        Returns
        -------
        int | Array
            The dimension.

        References
        ----------
        McLean, W. (2000). Strongly Elliptic Systems and
        Boundary Integral Equations. p.251

        """
        return harm_n_ndim(n, e_ndim=self.e_ndim)

    def draw(self, root_bottom: bool = True) -> tuple[float, float]:
        """
        Nicely draw the rooted tree representing the coordinates.

        Parameters
        ----------
        root_bottom : bool, optional
            Whether to draw the root at the bottom, by default True

        Returns
        -------
        tuple[float, float]
            The recommended width and height of the figure (in inches).

        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
        from networkx.algorithms.dag import dag_longest_path

        ASCII_TO_GREEK = {
            "alpha": "α",  # noqa
            "beta": "β",
            "gamma": "γ",  # noqa
            "delta": "δ",
            "epsilon": "ε",
            "zeta": "ζ",
            "theta": "θ",
            "iota": "ι",  # noqa
            "kappa": "κ",
            "lambda": "λ",
            "mu": "μ",
            "nu": "ν",  # noqa
            "xi": "ξ",
            "pi": "π",
            "rho": "ρ",  # noqa
            "sigma": "σ",  # noqa
            "tau": "τ",
            "upsilon": "υ",  # noqa
            "phi": "φ",
            "chi": "χ",
            "psi": "ψ",
            "omega": "ω",
            "eta": "η",  # last
        }

        def ascii_to_greek(s: str) -> str:
            for k, v in ASCII_TO_GREEK.items():
                s = s.replace(k, v)
            return s

        # plt.rcParams["text.usetex"] = True
        # remove spines
        plt.box(False)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        width = max(self.e_ndim * 0.5 + 1, 3.5)
        height = max((len(dag_longest_path(self.G)) + 1) * 0.5, 3.5)
        plt.gcf().set_size_inches(width, height)

        # layout
        try:
            pos = graphviz_layout(self.G, prog="dot")
            if root_bottom:
                # invert y-axis
                y_center = np.mean([y for x, y in pos.values()])
                pos = {k: (x, 2 * y_center - y) for k, (x, y) in pos.items()}
        except FileNotFoundError as e:
            warnings.warn(
                "Graphviz is not installed. "
                "The layout will be calculated by spring layout.",
                RuntimeWarning,
                stacklevel=2,
                source=e,
            )
            pos = nx.spring_layout(self.G)

        # Spherical
        nx.draw_networkx_nodes(
            self.G,
            pos,
            nodelist=self.s_nodes,
            node_color="darkgray",
            node_size=800,
            label="Spherical",
        )
        nx.draw_networkx_labels(
            self.G,
            pos,
            labels={
                n: f"{ascii_to_greek(str(n))}\n"
                f"{self.branching_types[n].value}/{self.S[n]}"
                for n in self.s_nodes
            },
        )

        # Euclidean
        nx.draw_networkx_nodes(
            self.G,
            pos,
            nodelist=self.e_nodes,
            node_color="lightgray",
            node_shape="s",
            label="Euclidean",
        )
        nx.draw_networkx_labels(self.G, pos, labels={n: f"{n}" for n in self.e_nodes})

        # edges
        cos_color = "orange"
        sin_color = "blue"
        nx.draw_networkx_edges(
            self.G,
            pos,
            edgelist=self.cos_edges,
            edge_color=cos_color,
            label="cos",
            style="dashed",
        )
        nx.draw_networkx_edges(
            self.G, pos, edgelist=self.sin_edges, edge_color=sin_color, label="sin"
        )

        # legend
        handles = [
            Circle(
                (0, 0),
                0.25,
                facecolor="darkgray",
                label="Spherical(Name\nBranching type\n/Descendants)",
            ),
            Circle((0, 0), 0.12, facecolor="lightgray", label="Euclidean(Name)"),
            Line2D([0], [0], color=cos_color, lw=2, label="cos", linestyle="dashed"),
            Line2D([0], [0], color=sin_color, lw=2, label="sin"),
        ]
        plt.legend(handles=handles, loc="lower right" if root_bottom else "upper right")
        plt.title(f"Type {self.branching_types_expression_str} coordinates")

        return width, height

    def from_euclidean(
        self, euclidean: Mapping[TEuclidean, Array | NativeArray]
    ) -> Mapping[TSpherical | Literal["r"], Array]:
        """
        Convert the Euclidean coordinates to the spherical coordinates.

        Parameters
        ----------
        euclidean : Mapping[TEuclidean, Array | NativeArray]
            The Euclidean coordinates.

        Returns
        -------
        Array
            The spherical coordinates.

        """
        r = (
            ivy.vector_norm(
                ivy.broadcast_arrays(*[euclidean[k] for k in self.e_nodes]), axis=0
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
            result[node] = ivy.atan2(sin, cos)
            tmp[node] = ivy.vector_norm(ivy.broadcast_arrays(sin, cos), axis=0)
        return result

    @overload
    def to_euclidean(
        self,
        spherical: (
            Mapping[TSpherical | Literal["r"], Array | NativeArray]
            | Mapping[TSpherical, Array | NativeArray]
        ),
        as_array: Literal[False] = ...,
    ) -> Mapping[TEuclidean, Array]: ...

    @overload
    def to_euclidean(
        self,
        spherical: (
            Mapping[TSpherical | Literal["r"], Array | NativeArray]
            | Mapping[TSpherical, Array | NativeArray]
        ),
        as_array: Literal[True] = ...,
    ) -> Array: ...

    def to_euclidean(
        self,
        spherical: (
            Mapping[TSpherical | Literal["r"], Array | NativeArray]
            | Mapping[TSpherical, Array | NativeArray]
        ),
        as_array: bool = False,
    ) -> Mapping[TEuclidean, Array] | Array:
        """
        Convert the spherical coordinates to the Euclidean coordinates.

        Parameters
        ----------
        spherical : Mapping[TSpherical, Array | NativeArray]
            The spherical coordinates.
        as_array : bool, optional
            Whether to return as an array, by default False


        Returns
        -------
        Array
            The Euclidean coordinates.

        """
        result = {self.root: spherical.get("r", 1)}  # type: ignore
        for node in nx.topological_sort(self.G):
            if node in self.e_nodes:
                continue
            elif node not in self.s_nodes:
                raise AssertionError()
            cos_child = get_child(self.G, node, "cos")
            sin_child = get_child(self.G, node, "sin")
            cos = ivy.cos(spherical[node])
            sin = ivy.sin(spherical[node])
            result[cos_child] = result[node] * cos
            result[sin_child] = result[node] * sin
        result = {k: result[k] for k in self.e_nodes}
        if as_array:
            return ivy.stack(
                ivy.broadcast_arrays(*[result[k] for k in self.e_nodes]),
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
            Mapping[TSpherical | Literal["r"], Array | NativeArray]
            | Mapping[TSpherical, Array | NativeArray]
        ),
    ) -> Array:
        """
        Calculate the Jacobian of the spherical coordinates.

        Parameters
        ----------
        spherical : Mapping[TSpherical, Array | NativeArray]
            The spherical coordinates.

        Returns
        -------
        Array
            The Jacobian of the spherical coordinates.

        """
        jacobian = ivy.get(spherical, "r", 1)
        for node in self.s_nodes:
            cos = ivy.cos(spherical[node])
            sin = ivy.sin(spherical[node])
            cos_child = get_child(self.G, node, "cos")
            sin_child = get_child(self.G, node, "sin")
            jacobian *= cos ** (self.S.get(cos_child, -1) + 1) * sin ** (
                self.S.get(sin_child, -1) + 1
            )
        return jacobian

    @overload
    def random_points(
        self,
        *,
        shape: Sequence[int],
        device: ivy.Device | ivy.NativeDevice | None = ...,
        dtype: ivy.Dtype | ivy.NativeDtype | None = ...,
        type: Literal["uniform"] = ...,
        rng: np.random.Generator | None = ...,
        surface: bool = ...,
    ) -> Array: ...
    @overload
    def random_points(  # type: ignore
        self,
        *,
        shape: Sequence[int],
        device: ivy.Device | ivy.NativeDevice | None = ...,
        dtype: ivy.Dtype | ivy.NativeDtype | None = ...,
        type: Literal["spherical"] = ...,
        rng: np.random.Generator | None = ...,
        surface: Literal[False] = ...,
    ) -> Mapping[TSpherical | Literal["r"], Array]: ...
    @overload
    def random_points(
        self,
        *,
        shape: Sequence[int],
        device: ivy.Device | ivy.NativeDevice | None = ...,
        dtype: ivy.Dtype | ivy.NativeDtype | None = ...,
        type: Literal["spherical"] = ...,
        rng: np.random.Generator | None = ...,
        surface: Literal[True] = ...,
    ) -> Mapping[TSpherical, Array]: ...
    def random_points(
        self,
        *,
        shape: Sequence[int],
        device: ivy.Device | ivy.NativeDevice | None = None,
        dtype: ivy.Dtype | ivy.NativeDtype | None = None,
        type: Literal["uniform", "spherical"] = "uniform",
        rng: np.random.Generator | None = None,
        surface: bool = False,
    ) -> Array | Mapping[TSpherical | Literal["r"], Array] | Mapping[TSpherical, Array]:
        r"""
        Generate random points in/on the unit sphere.

        Parameters
        ----------
        shape : Sequence[int]
            The shape of the random points.
        device : ivy.Device | ivy.NativeDevice | None, optional
            The device, by default None
        dtype : ivy.Dtype | ivy.NativeDtype | None, optional
            The dtype, by default None
        type : Literal["uniform", "spherical"], optional
            The type of the random points, by default "uniform"
            If "uniform", the random points are uniformly distributed on the sphere.
            If "spherical", each spherical coordinate (and radius
            if surface=False) is uniformly distributed.
        rng : np.random.Generator | None, optional
            The random number generator, by default None
        surface : bool, optional
            Whether to generate random points on the surface of the sphere
            or inside the sphere, by default False

        Returns
        -------
        Mapping[TSpherical, Array] | Array | Mapping[TSpherical | Literal["r"], Array]
            The random points.

        References
        ----------
            Barthe, F., Guedon, O., Mendelson, S., & Naor, A. (2005).
            A probabilistic approach to the geometry of the \ell_p^n-ball.
            arXiv, math/0503650. Retrieved from https://arxiv.org/abs/math/0503650v1

        """
        if type == "uniform":
            return ivy.asarray(
                random_sphere(shape, dim=self.e_ndim, surface=surface, rng=rng),
                device=device,
                dtype=dtype,
            )
        elif type == "spherical":
            d = {
                BranchingType.A: (0, 2 * ivy.pi),
                BranchingType.B: (0, ivy.pi),
                BranchingType.BP: (-ivy.pi / 2, ivy.pi / 2),
                BranchingType.C: (0, ivy.pi / 2),
            }
            result: dict[TSpherical | Literal["r"], Array] = {}
            for node in self.s_nodes:
                low, high = d[self.branching_types[node]]
                result[node] = ivy.random_uniform(
                    low=low, high=high, shape=shape, device=device, dtype=dtype
                )
            if not surface:
                result["r"] = ivy.random_uniform(
                    low=0, high=1, shape=shape, device=device, dtype=dtype
                )
            return result
        else:
            raise ValueError(f"Invalid type {type}.")

    def roots(
        self,
        n: int,
        *,
        expand_dims_x: bool,
        expand_dims_w: bool = False,
        device: ivy.Device | ivy.NativeDevice | None = None,
        dtype: ivy.Dtype | ivy.NativeDtype | None = None,
    ) -> tuple[Mapping[TSpherical, Array], Mapping[TSpherical, Array]]:
        """
        Gauss-Jacobi quadrature roots and weights.

        Parameters
        ----------
        n : int
            The number of roots.
        expand_dims_x : bool
            Whether to expand dimensions of the roots, by default False
        expand_dims_w : bool, optional
            Whether to expand dimensions of the weights, by default False
        device : ivy.Device | ivy.NativeDevice, optional
            The device, by default None
        dtype : ivy.Dtype | ivy.NativeDtype, optional
            The data type, by default None

        Returns
        -------
        tuple[Mapping[TSpherical, Array], Mapping[TSpherical, Array]]
            roots and weights

        Raises
        ------
        ValueError
            If the branching type is invalid.

        """
        from scipy.special import roots_jacobi

        xs = {}
        ws = {}
        for i, node in enumerate(self.s_nodes):
            branching_type = self.branching_types[node]
            if branching_type == BranchingType.A:
                x = ivy.arange(2 * n, device=device, dtype=dtype) * ivy.pi / n
                w = ivy.ones(2 * n, device=device, dtype=dtype) * ivy.pi / n
            elif branching_type == BranchingType.B:
                s_beta = self.S[get_child(self.G, node, "sin")]
                beta = s_beta / 2
                x, w = roots_jacobi(n, beta, beta)
                x = ivy.acos(x)
            elif branching_type == BranchingType.BP:
                s_alpha = self.S[get_child(self.G, node, "cos")]
                alpha = s_alpha / 2
                x, w = roots_jacobi(n, alpha, alpha)
                x = ivy.asin(x)
            elif branching_type == BranchingType.C:
                s_alpha = self.S[get_child(self.G, node, "cos")]
                s_beta = self.S[get_child(self.G, node, "sin")]
                alpha = s_alpha / 2
                beta = s_beta / 2
                x, w = roots_jacobi(n, alpha, beta)
                w /= 2 ** (alpha + beta + 2)
                x = ivy.acos(x) / 2
            else:
                raise ValueError(f"Invalid branching type {branching_type}.")
            x = ivy.array(x, device=device, dtype=dtype)
            w = ivy.array(w, device=device, dtype=dtype)
            if expand_dims_x:
                x = x[(None,) * i + (slice(None),) + (None,) * (self.s_ndim - i - 1)]
            if expand_dims_w:
                w = w[(None,) * i + (slice(None),) + (None,) * (self.s_ndim - i - 1)]
            xs[node] = x
            ws[node] = w
        return xs, ws

    @overload
    def integrate(
        self,
        f: (
            Callable[
                [Mapping[TSpherical, Array | NativeArray]],
                Mapping[TSpherical, Array | NativeArray],
            ]
            | Mapping[TSpherical, Array | NativeArray]
        ),
        does_f_support_separation_of_variables: Literal[True],
        n: int,
        *,
        device: ivy.Device | ivy.NativeDevice | None = None,
        dtype: ivy.Dtype | ivy.NativeDtype | None = None,
    ) -> Mapping[TSpherical, Array]: ...

    @overload
    def integrate(
        self,
        f: (
            Callable[
                [Mapping[TSpherical, Array | NativeArray]],
                Array | NativeArray,
            ]
            | Array
            | NativeArray
        ),
        does_f_support_separation_of_variables: Literal[False],
        n: int,
        *,
        device: ivy.Device | ivy.NativeDevice | None = None,
        dtype: ivy.Dtype | ivy.NativeDtype | None = None,
    ) -> Array: ...

    def integrate(
        self,
        f: (
            Callable[
                [Mapping[TSpherical, Array | NativeArray]],
                Mapping[TSpherical, Array | NativeArray] | Array | NativeArray,
            ]
            | Mapping[TSpherical, Array | NativeArray]
            | Array
            | NativeArray
        ),
        does_f_support_separation_of_variables: bool,
        n: int,
        *,
        device: ivy.Device | ivy.NativeDevice | None = None,
        dtype: ivy.Dtype | ivy.NativeDtype | None = None,
    ) -> Array | Mapping[TSpherical, Array]:
        """
        Integrate the function over the hypersphere.

        Parameters
        ----------
        f : Callable[ [Mapping[TSpherical, Array  |  NativeArray]],
            Mapping[TSpherical, Array  |  NativeArray]  |  Array  |  NativeArray, ]
            |  Mapping[TSpherical, Array  |  NativeArray]  |  Array  |  NativeArray
                The function to integrate or the values of the function.
            In case of vectorized function, the function should add extra
            axis to the last dimension, not the first dimension.
        does_f_support_separation_of_variables : bool
            Whether the function supports separation of variables.
            This could significantly reduce the computational cost.
        n : int
            The number of roots.
        device : ivy.Device | ivy.NativeDevice, optional
            The device, by default None
        dtype : ivy.Dtype | ivy.NativeDtype, optional
            The data type, by default None

        Returns
        -------
        Array | Mapping[TSpherical, Array]
            The integrated value.

        """
        xs, ws = self.roots(
            n,
            device=device,
            dtype=dtype,
            expand_dims_x=not does_f_support_separation_of_variables,
        )
        if isinstance(f, Callable):  # type: ignore
            try:
                val = f(xs)  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Error occurred while evaluating {f=}") from e
        else:
            val = f

        # in case f(theta1, ...) = f_1(theta1) * f_2(theta2) * ...
        if isinstance(val, Mapping):
            result = {}
            for node in self.s_nodes:
                value = val[node]
                # supports vectorized function
                # axis=0 because in sph_harm
                # we add axis to the last dimension
                # theta(node),u1,...,uM
                ivy.broadcast_shapes(ivy.shape(value)[:1], ivy.shape(ws[node]))
                w = ws[node].reshape([-1] + [1] * (value.ndim - 1))
                result[node] = ivy.sum(value * w, axis=0)
            # we don't know how to einsum the result
            return result
        # theta1,...,thetaN,u1,...,uM\
        for node in self.s_nodes:
            w = ws[node]
            ivy.broadcast_shapes(ivy.shape(val)[:1], ivy.shape(w))
            # val = ivy.einsum("i...,i->...", val, w.astype(val.dtype))
            val = ivy.sum(val * w[(slice(None),) + (None,) * (val.ndim - 1)], axis=0)
        return val

    @overload
    def expand_cut(
        self,
        expansion: Mapping[TSpherical, Array | NativeArray],
        n_end: int,
    ) -> Mapping[TSpherical, Array]: ...

    @overload
    def expand_cut(
        self,
        expansion: Array | NativeArray,
        n_end: int,
    ) -> Array: ...

    def expand_cut(
        self,
        expansion: Mapping[TSpherical, Array | NativeArray] | Array | NativeArray,
        n_end: int,
    ) -> Mapping[TSpherical, Array] | Array:
        """
        Cut the expansion coefficients to the maximum degree.

        Parameters
        ----------
        expansion : Mapping[TSpherical, Array  |  NativeArray] | Array | NativeArray
            The expansion coefficients.
            If mapping, assume that the expansion is not expanded.
        n_end : int
            The maximum degree to cut.

        Returns
        -------
        Mapping[TSpherical, Array] | Array
            The cut expansion coefficients.

        """
        from shift_nth_row_n_steps import take_slice

        is_mapping = isinstance(expansion, Mapping)
        n_end_prev, include_negative_m = (
            self.get_n_end_and_include_negative_m_from_expansion(expansion)
        )
        if n_end > n_end_prev:
            raise ValueError(f"n_end={n_end} > n_end_prev={n_end_prev}.")
        if is_mapping:
            raise NotImplementedError()
            # for node, branching_type in self.branching_types.items():
            #     for axis in self.ndim_harmonics(node):
        for i, node in enumerate(self.s_nodes):
            axis = -self.s_ndim + i
            branching_type = self.branching_types[node]
            if branching_type == BranchingType.A and include_negative_m:
                expansion = ivy.concat(
                    [
                        take_slice(expansion, 0, n_end, axis=axis),
                    ]
                    + (
                        []
                        if n_end == 1
                        else [
                            take_slice(expansion, -n_end + 1, None, axis=axis),
                        ]
                    ),
                    axis=axis,
                )
            else:
                expansion = take_slice(expansion, 0, n_end, axis=axis)
        return expansion

    def get_n_end_and_include_negative_m_from_expansion(
        self,
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
        if self.s_ndim == 0:
            return 0, False
        is_mapping = isinstance(expansion, Mapping)
        if is_mapping:
            sizes = [ivy.shape(expansion[k])[-1] for k in self.s_nodes]
        else:
            sizes = ivy.shape(expansion)[-self.s_ndim :]
        n_end = (max(sizes) + 1) // 2
        include_negative_m = not all(size == n_end for size in sizes)
        return n_end, include_negative_m

    @overload
    def expand_evaluate(
        self,
        expansion: Mapping[TSpherical, Array | NativeArray],
        spherical: Mapping[TSpherical, Array | NativeArray],
        *,
        condon_shortley_phase: bool,
    ) -> Mapping[TSpherical, Array]: ...

    @overload
    def expand_evaluate(
        self,
        expansion: Array | NativeArray,
        spherical: Mapping[TSpherical, Array | NativeArray],
        *,
        condon_shortley_phase: bool,
    ) -> Array: ...

    def expand_evaluate(
        self,
        expansion: Mapping[TSpherical, Array | NativeArray] | Array | NativeArray,
        spherical: Mapping[TSpherical, Array | NativeArray],
        *,
        condon_shortley_phase: bool,
    ) -> Array | Mapping[TSpherical, Array]:
        """
        Evaluate the expansion at the spherical coordinates.

        Parameters
        ----------
        expansion : Mapping[TSpherical, Array  |  NativeArray] | Array | NativeArray
            The expansion coefficients.
            If mapping, assume that the expansion is not expanded.
        spherical : Mapping[TSpherical, Array  |  NativeArray]
            The spherical coordinates.
        condon_shortley_phase : bool
            Whether to apply the Condon-Shortley phase,
            which just multiplies the result by (-1)^m.

            It seems to be mainly used in quantum mechanics for convenience.

            Note that scipy.special.sph_harm (or scipy.special.lpmv)
            uses the Condon-Shortley phase.

            If False, `Y^{-m}_{l} = Y^{m}_{l}*`.

            If True, `Y^{-m}_{l} = (-1)^m Y^{m}_{l}*`.
            (Simply because `e^{i -m phi} = (e^{i m phi})*`)


        Returns
        -------
        Array | Mapping[TSpherical, Array]
            The evaluated value.

        """
        is_mapping = isinstance(expansion, Mapping)
        n_end, _ = self.get_n_end_and_include_negative_m_from_expansion(expansion)
        harmonics = self.harmonics(  # type: ignore
            spherical,
            n_end,
            condon_shortley_phase=condon_shortley_phase,
            expand_dims=not is_mapping,
            concat=not is_mapping,
        )
        if is_mapping:
            result: dict[TSpherical, Array] = {}
            for node in self.s_nodes:
                expansion_ = expansion[node]
                harmonics_ = harmonics[node]
                # expansion: f1,...,fL,harm1,...,harmN
                # harmonics: u1,...,uM,harm1,...,harmN
                # result: u1,...,uM,f1,...,fL
                ndim_harmonics = self.ndim_harmonics(node)
                ndim_expansion = expansion_.ndim - ndim_harmonics
                ndim_extra_harmonics = harmonics_.ndim - ndim_harmonics
                expansion_ = harmonics_[
                    (None,) * (ndim_extra_harmonics)
                    + (slice(None),) * (ndim_expansion + ndim_harmonics)
                ]
                harmonics = harmonics_[
                    (slice(None),) * ndim_extra_harmonics
                    + (None,) * ndim_expansion
                    + (slice(None),) * ndim_harmonics
                ]
                result_ = harmonics_ * expansion_
                for _ in range(ndim_harmonics):
                    result_ = ivy.sum(result_, axis=-1)
                result[node] = result
            return result
        if isinstance(expansion, Mapping):
            raise AssertionError()
        # expansion: f1,...,fL,harm1,...,harmN
        # harmonics: u1,...,uM,harm1,...,harmN
        # result: u1,...,uM,f1,...,fL
        ndim_harmonics = self.s_ndim
        ndim_expansion = expansion.ndim - ndim_harmonics
        ndim_extra_harmonics = harmonics.ndim - ndim_harmonics
        expansion = expansion[
            (None,) * (ndim_extra_harmonics)
            + (slice(None),) * (ndim_expansion + ndim_harmonics)
        ]
        harmonics = harmonics[
            (slice(None),) * ndim_extra_harmonics
            + (None,) * ndim_expansion
            + (slice(None),) * ndim_harmonics
        ]
        result = harmonics * expansion
        for _ in range(ndim_harmonics):
            result = ivy.sum(result, axis=-1)
        return result

    @overload
    def expand(
        self,
        f: (
            Callable[
                [Mapping[TSpherical, Array | NativeArray]],
                Mapping[TSpherical, Array | NativeArray] | Array | NativeArray,
            ]
            | Mapping[TSpherical, Array | NativeArray]
            | Array
            | NativeArray
        ),
        does_f_support_separation_of_variables: Literal[True],
        n_end: int,
        n: int,
        *,
        condon_shortley_phase: bool,
        device: ivy.Device | ivy.NativeDevice | None = None,
        dtype: ivy.Dtype | ivy.NativeDtype | None = None,
    ) -> Mapping[TSpherical, Array]: ...

    @overload
    def expand(
        self,
        f: (
            Callable[
                [Mapping[TSpherical, Array | NativeArray]],
                Mapping[TSpherical, Array | NativeArray] | Array | NativeArray,
            ]
            | Mapping[TSpherical, Array | NativeArray]
            | Array
            | NativeArray
        ),
        does_f_support_separation_of_variables: Literal[False],
        n_end: int,
        n: int,
        *,
        condon_shortley_phase: bool,
        device: ivy.Device | ivy.NativeDevice | None = None,
        dtype: ivy.Dtype | ivy.NativeDtype | None = None,
    ) -> Array: ...

    def expand(
        self,
        f: (
            Callable[
                [Mapping[TSpherical, Array | NativeArray]],
                Mapping[TSpherical, Array | NativeArray] | Array | NativeArray,
            ]
            | Mapping[TSpherical, Array | NativeArray]
            | Array
            | NativeArray
        ),
        does_f_support_separation_of_variables: bool,
        n_end: int,
        n: int,
        *,
        condon_shortley_phase: bool,
        device: ivy.Device | ivy.NativeDevice | None = None,
        dtype: ivy.Dtype | ivy.NativeDtype | None = None,
    ) -> Array | Mapping[TSpherical, Array]:
        """
        Calculate the expansion coefficients of the function
        over the hypersphere.

        Parameters
        ----------
        f : Callable[ [Mapping[TSpherical, Array  |  NativeArray]],
            Mapping[TSpherical, Array  |  NativeArray]  |  Array  |  NativeArray, ]
            |  Mapping[TSpherical, Array  |  NativeArray]  |  Array  |  NativeArray
            The function to integrate or the values of the function.
            In case of vectorized function, the function should add extra
            axis to the last dimension, not the first dimension.
        does_f_support_separation_of_variables : bool
            Whether the function supports separation of variables.
            This could significantly reduce the computational cost.
        n : int
            The number of integration points.

            Must be equal to or larger than n_end.

            Must be large enough against f, as this method
            does not use adaptive integration. For example,
            consider expanding $f(θ) = e^(2Nθ)$ with $n=N$.

            >>> from ultrasphere import SphericalCoordinates
            >>> from ultrasphere.coordinates import SphericalCoordinates
            >>> import numpy as np
            >>> n = 5
            >>> expansion = SphericalCoordinates.standard(1).expand(
            >>>     lambda x: np.exp(1j * (2 * n) * x["theta0"]) / np.sqrt(2 * np.pi),
            >>>     does_f_support_separation_of_variables=False,
            >>>     n=n,
            >>>     n_end=n,
            >>>     condon_shortley_phase=False
            >>> )
            >>> print(np.round(expansion, 2).real.tolist())
            [1.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0]

            This result claims that f(θ) = 1, which is incorrect.
        n_end : int
            The maximum degree of the harmonic.
        condon_shortley_phase : bool
            Whether to apply the Condon-Shortley phase,
            which just multiplies the result by (-1)^m.

            It seems to be mainly used in quantum mechanics for convenience.

            Note that scipy.special.sph_harm (or scipy.special.lpmv)
            uses the Condon-Shortley phase.

            If False, `Y^{-m}_{l} = Y^{m}_{l}*`.

            If True, `Y^{-m}_{l} = (-1)^m Y^{m}_{l}*`.
            (Simply because `e^{i -m phi} = (e^{i m phi})*`)
        device : ivy.Device | ivy.NativeDevice, optional
            The device, by default None
        dtype : ivy.Dtype | ivy.NativeDtype, optional
            The data type, by default None

        Returns
        -------
        Array | Mapping[TSpherical, Array]
            The expanded value.
            Last `self.s_ndim` axis [-self.s_ndim, -1]
            corresponds to the quantum numbers.

            The dimensions are not expanded if Mapping is returned.
            Use `self.expand_dims_harmonics()`
            and `self.concat_harmonics()`
            to expand the dimensions and to concat values.

        """
        if n < n_end:
            raise ValueError(
                f"n={n} < n_end={n_end}, which would lead to incorrect results."
            )

        def inner(
            xs: Mapping[TSpherical, Array | NativeArray],
        ) -> Mapping[TSpherical, Array | NativeArray]:
            # calculate f
            if isinstance(f, Callable):  # type: ignore
                try:
                    val = f(xs)  # type: ignore
                except Exception as e:
                    raise RuntimeError(f"Error occurred while evaluating {f=}") from e
            else:
                val = f

            # calculate harmonics
            harmonics = self.harmonics(  # type: ignore
                xs,
                n_end,
                condon_shortley_phase=condon_shortley_phase,
                expand_dims=not does_f_support_separation_of_variables,
                concat=not does_f_support_separation_of_variables,
            )

            # multiply f and harmonics
            # (C,complex conjugate) is star-algebra
            if isinstance(val, Mapping):
                if not does_f_support_separation_of_variables:
                    raise ValueError(
                        "val is Mapping but "
                        "does_f_support_separation_of_variables "
                        "is False."
                    )
                result = {}
                for node in self.s_nodes:
                    value = val[node]
                    # val: theta(node),u1,...,uM
                    # harmonics: theta(node),harm1,...,harmN
                    # result: theta(node),u1,...,uM,harm1,...,harmN
                    ivy.broadcast_shapes(
                        ivy.shape(value)[:1], ivy.shape(harmonics[node])[:1]
                    )
                    ndim_val = value.ndim - 1
                    ndim_harm = self.ndim_harmonics(node)
                    value = value[(...,) + (None,) * (ndim_harm)]
                    harm = harmonics[node][
                        (slice(None),) + (None,) * ndim_val + (slice(None),) * ndim_harm
                    ]
                    result[node] = value * harm.conj()
            else:
                if does_f_support_separation_of_variables:
                    raise ValueError(
                        "val is not Mapping but "
                        "does_f_support_separation_of_variables "
                        "is True."
                    )
                # val: theta1,...,thetaN,u1,...,uM
                # harmonics: theta1,...,thetaN,harm1,...,harmN
                # res: theta1,...,thetaN,u1,...,uM,harm1,...,harmN
                ivy.broadcast_shapes(
                    ivy.shape(val)[: self.s_ndim], ivy.shape(harmonics)[: self.s_ndim]
                )
                ndim_val = val.ndim - self.s_ndim
                val = val[
                    (slice(None),) * (self.s_ndim + ndim_val) + (None,) * self.s_ndim
                ]
                harmonics = harmonics[
                    (slice(None),) * self.s_ndim
                    + (None,) * ndim_val
                    + (slice(None),) * self.s_ndim
                ]
                result = val * harmonics.conj()

            return result

        return self.integrate(  # type: ignore
            inner, does_f_support_separation_of_variables, n, device=device, dtype=dtype
        )

    def expand_dim_harmoncis(
        self,
        node: TSpherical,
        harmonics: Array | NativeArray,
    ) -> Array:
        """
        Expand the dimension of the harmonics.

        Expand the dimension so that
        all values of the harmonics() result dictionary
        are commomly indexed by the same s_nodes, by default False

        For example, if spherical coordinates,
        if True, the result will be indexed {"phi": [m], "theta": [m, n]}
        if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

        Note that the values will not be repeated
        therefore the computational cost will be the same

        Parameters
        ----------
        node : TSpherical
            The node of the spherical coordinates.
        harmonics : Array | NativeArray
            The harmonics (eigenfunctions).

        Returns
        -------
        Array
            The expanded harmonics.
            The shapes does not need to be either
            same or broadcastable.

        """
        idx_node = self.s_nodes.index(node)
        branching_type = self.branching_types[node]
        if branching_type == BranchingType.A:
            moveaxis = {0: idx_node}
        elif branching_type == BranchingType.B:
            idx_sin_child = self.s_nodes.index(get_child(self.G, node, "sin"))
            moveaxis = {
                0: idx_sin_child,
                1: idx_node,
            }
        elif branching_type == BranchingType.BP:
            idx_cos_child = self.s_nodes.index(get_child(self.G, node, "cos"))
            moveaxis = {
                0: idx_cos_child,
                1: idx_node,
            }
        elif branching_type == BranchingType.C:
            idx_cos_child = self.s_nodes.index(get_child(self.G, node, "cos"))
            idx_sin_child = self.s_nodes.index(get_child(self.G, node, "sin"))
            moveaxis = {0: idx_cos_child, 1: idx_sin_child, 2: idx_node}
        value_additional_ndim = harmonics.ndim - len(moveaxis)
        moveaxis = {
            k + value_additional_ndim: v + value_additional_ndim
            for k, v in moveaxis.items()
        }
        adding_ndim = self.s_ndim - len(moveaxis)
        harmonics = harmonics[(...,) + (None,) * adding_ndim]
        return ivy.moveaxis(harmonics, list(moveaxis.keys()), list(moveaxis.values()))

    def expand_dims_harmonics(
        self,
        harmonics: Mapping[TSpherical, Array | NativeArray],
    ) -> Mapping[TSpherical, Array]:
        """
        Expand dimensions of the harmonics.

        Expand dimensions so that
        all values of the harmonics() result dictionary
        are commomly indexed by the same s_nodes, by default False

        For example, if spherical coordinates,
        if True, the result will be indexed {"phi": [m], "theta": [m, n]}
        if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

        Note that the values will not be repeated
        therefore the computational cost will be the same

        Parameters
        ----------
        harmonics : Mapping[TSpherical, Array  |  NativeArray]
            The dictionary of harmonics (eigenfunctions).

        Returns
        -------
        Mapping[TSpherical, Array]
            The expanded harmonics.
            The shapes does not need to be either
            same or broadcastable.

        """
        result: dict[TSpherical, Array] = {}
        for node in self.s_nodes:
            result[node] = self.expand_dim_harmoncis(node, harmonics[node])
        return result

    def concat_harmonics(
        self,
        harmonics: Mapping[TSpherical, Array | NativeArray],
    ) -> Array:
        """
        Concatenate the mapping of expanded harmonics.

        Parameters
        ----------
        harmonics : Mapping[TSpherical, Array  |  NativeArray]
            The expanded harmonics.

        Returns
        -------
        Array
            The concatenated harmonics.

        """
        try:
            if self.s_ndim == 0:
                return ivy.array(1)
            return ivy.prod(
                ivy.stack(ivy.broadcast_arrays(*harmonics.values()), axis=0), axis=0
            )
        except Exception as e:
            shapes = {k: v.shape for k, v in harmonics.items()}
            raise RuntimeError(f"Error occurred while concatenating {shapes=}") from e

    def flatten_mask_harmonics(
        self,
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
        index_arrays = self.index_array_harmonics_all(
            n_end=n_end,
            include_negative_m=include_negative_m,
            as_array=False,
            expand_dims=True,
        )
        shape = ivy.broadcast_shapes(
            *[index_array.shape for index_array in index_arrays.values()]
        )
        mask = ivy.ones(shape, dtype=bool)
        for node, branching_type in self.branching_types.items():
            if branching_type == BranchingType.B:
                mask = mask & (
                    ivy.abs(index_arrays[get_child(self.G, node, "sin")])
                    <= index_arrays[node]
                )
            if branching_type == BranchingType.BP:
                mask = mask & (
                    ivy.abs(index_arrays[get_child(self.G, node, "cos")])
                    <= index_arrays[node]
                )
            if branching_type == BranchingType.C:
                value = (
                    index_arrays[node]
                    - ivy.abs(index_arrays[get_child(self.G, node, "sin")])
                    - ivy.abs(index_arrays[get_child(self.G, node, "cos")])
                )
                mask = mask & (value % 2 == 0) & (value >= 0)
        return mask

    def flatten_harmonics(
        self,
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
            self.get_n_end_and_include_negative_m_from_expansion(harmonics)
        )
        mask = self.flatten_mask_harmonics(n_end, include_negative_m)
        return harmonics[..., mask]

    def unflatten_harmonics(
        self,
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
            The unflattened harmonics of shape (..., n_1, n_2, ..., n_(self.s_ndim)).

        """
        mask = self.flatten_mask_harmonics(n_end, include_negative_m)
        shape = (*harmonics.shape[:-1], *mask.shape)
        result = ivy.zeros(shape, dtype=harmonics.dtype, device=harmonics.device)
        result[..., mask] = harmonics
        return result

    def ndim_harmonics(
        self,
        node: TSpherical,
    ) -> int:
        """
        The number of dimensions of the eigenfunction
        corresponding to the node.

        Parameters
        ----------
        node : TSpherical
            The node of the spherical coordinates.

        Returns
        -------
        int
            The number of dimensions.

        """
        return {
            BranchingType.A: 1,
            BranchingType.B: 2,
            BranchingType.BP: 2,
            BranchingType.C: 3,
        }[self.branching_types[node]]

    def index_array_harmonics(
        self,
        node: TSpherical,
        *,
        n_end: int,
        expand_dims: bool = False,
        include_negative_m: bool = True,
    ) -> Array:
        """
        The index of the eigenfunction
        corresponding to the node.

        Parameters
        ----------
        node : TSpherical
            The node of the spherical coordinates.
        n_end : int
            The maximum degree of the harmonic.
        expand_dims : bool, optional
            Whether to expand dimensions, by default False
        include_negative_m : bool, optional
            Whether to include negative m values, by default True

        Returns
        -------
        Array
            The index.

        """
        branching_type = self.branching_types[node]
        if branching_type == BranchingType.A and include_negative_m:
            result = to_symmetric(ivy.arange(0, n_end), asymmetric=True)
        elif (
            branching_type == BranchingType.B
            or branching_type == BranchingType.BP
            or (branching_type == BranchingType.A and not include_negative_m)
        ):
            result = ivy.arange(0, n_end)
        elif branching_type == BranchingType.C:
            # result = ivy.arange(0, (n_end + 1) // 2)
            result = ivy.arange(0, n_end)
        if expand_dims:
            idx = self.s_nodes.index(node)
            result = result[
                create_slice(self.s_ndim, [(idx, slice(None))], default=None)
            ]
        return result

    @overload
    def index_array_harmonics_all(
        self,
        *,
        n_end: int,
        include_negative_m: bool = ...,
        expand_dims: bool,
        as_array: Literal[False],
        mask: Literal[False] = ...,
    ) -> Mapping[TSpherical, Array]: ...
    @overload
    def index_array_harmonics_all(
        self,
        *,
        n_end: int,
        include_negative_m: bool = ...,
        expand_dims: Literal[True],
        as_array: Literal[True],
        mask: bool = ...,
    ) -> Array: ...

    def index_array_harmonics_all(
        self,
        *,
        n_end: int,
        include_negative_m: bool = True,
        expand_dims: bool,
        as_array: bool,
        mask: bool = False,
    ) -> Array | Mapping[TSpherical, Array]:
        """
        The all indices of the eigenfunction
        corresponding to the spherical coordinates.

        Parameters
        ----------
        n_end : int
            The maximum degree of the harmonic.
        include_negative_m : bool, optional
            Whether to include negative m values, by default True
        expand_dims : bool, optional
            Whether to expand dimensions, by default True
            Must be True if as_array is True.
        as_array : bool, optional
            Whether to return as an array, by default False
        mask : bool, optional
            Whether to fill invalid quantum numbers with NaN, by default False
            Must be False if as_array is False.

        Returns
        -------
        Array | Mapping[TSpherical, Array]
            If as_array is True, the indices of shape
            [self.s_ndim,
            len(self.index_array_harmonics(node1)),
            ...,
            len(self.index_array_harmonics(node(self.s_ndim)))].
            If as_array is False, the dictionary of indices.

        Notes
        -----
            To check the indices where all quantum numbers match,
            `(numbers1 == numbers2).all(axis=0)`
            can be used.

        Raises
        ------
        ValueError
            If expand_dims is False and as_array is True.
            If mask is True and as_array is False.

        """
        if not expand_dims and as_array:
            raise ValueError("expand_dims must be True if as_array is True.")
        if mask and not as_array:
            raise ValueError("mask must be False if as_array is False.")
        index_arrays = {
            node: self.index_array_harmonics(
                node,
                n_end=n_end,
                expand_dims=expand_dims,
                include_negative_m=include_negative_m,
            )
            for node in self.s_nodes
        }
        if as_array:
            result = ivy.stack(
                ivy.broadcast_arrays(*[index_arrays[node] for node in self.s_nodes]),
                axis=0,
            )
            if mask:
                result[:, ~self.flatten_mask_harmonics(n_end, include_negative_m)] = (
                    ivy.nan
                )
            return result
        return index_arrays

    @overload
    def harmonics(
        self,
        spherical: Mapping[TSpherical, Array | NativeArray],
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
        self,
        spherical: Mapping[TSpherical, Array | NativeArray],
        n_end: int,
        *,
        condon_shortley_phase: bool,
        include_negative_m: bool = True,
        index_with_surrogate_quantum_number: bool = False,
        expand_dims: bool = True,
        concat: Literal[False] = ...,
    ) -> Mapping[TSpherical, Array]: ...

    def harmonics(
        self,
        spherical: Mapping[TSpherical, Array | NativeArray],
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
        spherical : Mapping[TSpherical, Array | NativeArray]
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
        from .eigenfunction import type_a, type_b, type_bdash, type_c

        result = {}
        for node in self.s_nodes:
            value = spherical[node]
            if node == "r":
                continue
            if node not in self.s_nodes:
                raise ValueError(f"Key {node} is not in self.s_nodes {self.s_nodes}.")
            if self.branching_types[node] == BranchingType.A:
                result[node] = type_a(
                    value,
                    n_end=n_end,
                    condon_shortley_phase=condon_shortley_phase,
                    include_negative_m=include_negative_m,
                )
            elif self.branching_types[node] == BranchingType.B:
                result[node] = type_b(
                    value,
                    n_end=n_end,
                    s_beta=self.S[get_child(self.G, node, "sin")],
                    index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                    is_beta_type_a_and_include_negative_m=include_negative_m
                    and self.branching_types[get_child(self.G, node, "sin")]
                    == BranchingType.A,
                )
            elif self.branching_types[node] == BranchingType.BP:
                result[node] = type_bdash(
                    value,
                    n_end=n_end,
                    s_alpha=self.S[get_child(self.G, node, "cos")],
                    index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                    is_alpha_type_a_and_include_negative_m=include_negative_m
                    and self.branching_types[get_child(self.G, node, "cos")]
                    == BranchingType.A,
                )
            elif self.branching_types[node] == BranchingType.C:
                result[node] = type_c(
                    value,
                    n_end=n_end,
                    s_alpha=self.S[get_child(self.G, node, "cos")],
                    s_beta=self.S[get_child(self.G, node, "sin")],
                    index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
                    is_alpha_type_a_and_include_negative_m=include_negative_m
                    and self.branching_types[get_child(self.G, node, "cos")]
                    == BranchingType.A,
                    is_beta_type_a_and_include_negative_m=include_negative_m
                    and self.branching_types[get_child(self.G, node, "sin")]
                    == BranchingType.A,
                )
            else:
                raise ValueError(
                    f"Invalid branching type {self.branching_types[node]}."
                )
        if expand_dims:
            result = self.expand_dims_harmonics(result)  # type: ignore
        if concat:
            result = self.concat_harmonics(result)
        return result

    @overload
    def harmonics_regular_singular(
        self,
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
        self,
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
        self,
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
            self.get_n_end_and_include_negative_m_from_expansion(harmonics)
        )
        n = self.index_array_harmonics(
            self.root, n_end=n_end, include_negative_m=include_negative_m
        )[(None,) * spherical["r"].ndim + (slice(None),)]
        kr = k * spherical["r"]
        kr = kr[..., None]

        if type == "regular":
            type = "j"
        elif type == "singular":
            type = "h1"
        val = szv(n, self.e_ndim, kr, type=type, derivative=derivative)
        val = ivy.nan_to_num(val, nan=0)
        expand_dims = not (is_mapping and len({h.ndim for h in harmonics.values()}) > 1)
        if expand_dims:
            idx = self.s_nodes.index(self.root)
            val = val[
                (..., *create_slice(self.s_ndim, [(idx, slice(None))], default=None))
            ]
        if is_mapping:
            res = {"r": val}
            if harmonics is not None:
                res.update(harmonics)  # type: ignore
            return res
        if multiply:
            return val * harmonics
        return val

    def harmonics_translation_coef(
        self,
        euclidean: Mapping[TEuclidean, Array | NativeArray],
        *,
        n_end: int,
        n_end_add: int,
        condon_shortley_phase: bool,
        k: Array | NativeArray,
    ) -> Array:
        r"""
        Translation coefficients between same type of elementary solutions.

        Returns (R|R) = (S|S), where

        .. math::
            R(x + t) = \sum_n (R|R)_n(t) R(x)
            S(x + t) = \sum_n (S|S)_n(t) S(x)

        Parameters
        ----------
        euclidean : Mapping[TEuclidean, Array | NativeArray]
            The translation vector in euclidean coordinates.
        n_end : int
            The maximum degree of the harmonic.
        n_end_add : int
            The maximum degree of the harmonic to be summed over.
        condon_shortley_phase : bool
            Whether to apply the Condon-Shortley phase.
        k : Array | NativeArray
            The wavenumber.

        Returns
        -------
        Array
            The translation coefficients of `2 * self.s_ndim` dimensions.
            [-self.s_ndim,-1] dimensions are to be
            summed over with the elementary solutions
            to get translated elementary solution
            which quantum number is [-2*self.s_ndim,-self.s_ndim-1] indices.

        """
        _, k = ivy.broadcast_arrays(euclidean[self.e_nodes[0]], k)
        n = self.index_array_harmonics(self.root, n_end=n_end, expand_dims=True)[
            (...,) + (None,) * self.s_ndim
        ]
        ns = self.index_array_harmonics(self.root, n_end=n_end_add, expand_dims=True)[
            (None,) * self.s_ndim + (...,)
        ]

        def to_expand(spherical: Mapping[TSpherical, Array | NativeArray]) -> Array:
            # returns [spherical1,...,sphericalN,user1,...,userM,n1,...,nN]
            # [spherical1,...,sphericalN,n1,...,nN]
            harmonics = self.harmonics(
                spherical,
                n_end,
                condon_shortley_phase=condon_shortley_phase,
                expand_dims=True,
                concat=True,
            )
            x = self.to_euclidean(spherical)
            ndim_user = euclidean[self.e_nodes[0]].ndim
            ndim_spherical = self.s_ndim
            ip = ivy.sum(
                ivy.stack(
                    ivy.broadcast_arrays(
                        *[
                            euclidean[i][
                                (None,) * ndim_spherical + (slice(None),) * ndim_user
                            ]
                            * x[i][
                                (slice(None),) * ndim_spherical + (None,) * ndim_user
                            ]
                            for i in self.e_nodes
                        ]
                    ),
                    axis=0,
                ),
                axis=0,
            )
            # [spherical1,...,sphericalN,user1,...,userM]
            e = ivy.exp(
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
        return (-1j) ** (n - ns) * self.expand(
            to_expand,
            does_f_support_separation_of_variables=False,
            n=n_end + n_end_add - 1,
            n_end=n_end_add,
            condon_shortley_phase=condon_shortley_phase,
        )

    def harmonics_twins_expansion(
        self,
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
            and dim `3 * self.s_ndim` and of dtype float, not complex.
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
            n1 = self.index_array_harmonics(self.root, n_end=n_end_1, expand_dims=True)
            n2 = self.index_array_harmonics(self.root, n_end=n_end_2, expand_dims=True)
            n3 = self.index_array_harmonics(
                self.root, n_end=n_end_1 + n_end_2 - 1, expand_dims=True
            )
            if self.e_ndim == 2:
                n1 = n1[:, None, None]
                n2 = n2[None, :, None]
                n3 = n3[None, None, :]
                if conj_1:
                    n1 = -n1
                if conj_2:
                    n2 = -n2
                n3 = -n3
                result = (n1 + n2 + n3 == 0) / ivy.sqrt(2 * ivy.pi)
                if condon_shortley_phase:
                    result *= (-1) ** (n1 + n2 + n3)
                return result
            elif self.e_ndim == 3:
                from py3nj import wigner3j

                another_node = (set(self.s_nodes) - {self.root}).pop()
                m1 = self.index_array_harmonics(
                    another_node, n_end=n_end_1, expand_dims=True
                )
                m2 = self.index_array_harmonics(
                    another_node, n_end=n_end_2, expand_dims=True
                )
                m3 = self.index_array_harmonics(
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
                    ivy.where(m1 <= 0, 1, (-1) ** ivy.abs(m1))
                    * ivy.where(m2 <= 0, 1, (-1) ** ivy.abs(m2))
                    * ivy.where(m3 <= 0, 1, (-1) ** ivy.abs(m3))
                    # * (n1 >= ivy.abs(m1))
                    # * (n2 >= ivy.abs(m2))
                    # * (n3 >= ivy.abs(m3))
                    * ivy.sqrt(
                        (2 * n1 + 1) * (2 * n2 + 1) * (2 * n3 + 1) / (4 * ivy.pi)
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
                        ivy.zeros_like(n1, dtype=int),
                        ivy.zeros_like(n2, dtype=int),
                        ivy.zeros_like(n3, dtype=int),
                        ignore_invalid=True,
                    )
                )
                if condon_shortley_phase:
                    result *= (-1) ** (m1 + m2 + m3)
                return result

        def to_expand(spherical: Mapping[TSpherical, Array | NativeArray]) -> Array:
            # returns [theta,n1,...,nN,nsummed1,...,nsummedN]
            # Y(n)Y*(nsummed)
            Y1 = self.harmonics(
                spherical,
                n_end_1,
                condon_shortley_phase=condon_shortley_phase,
                expand_dims=True,
                concat=False,
            )
            Y1 = {
                k: v[
                    (slice(None),) * 1
                    + (slice(None),) * self.s_ndim
                    + (None,) * self.s_ndim
                ]
                for k, v in Y1.items()
            }
            if conj_1:
                Y1 = {k: v.conj() for k, v in Y1.items()}
            Y2 = self.harmonics(
                spherical,
                n_end_2,
                condon_shortley_phase=condon_shortley_phase,
                expand_dims=True,
                concat=False,
            )
            Y2 = {
                k: v[
                    (slice(None),) * 1
                    + (None,) * self.s_ndim
                    + (slice(None),) * self.s_ndim
                ]
                for k, v in Y2.items()
            }
            if conj_2:
                Y2 = {k: v.conj() for k, v in Y2.items()}
            return {k: Y1[k] * Y2[k] for k in self.s_nodes}

        # returns [user1,...,userM,n1,...,nN,np1,...,npN]
        return self.concat_harmonics(
            self.expand_dims_harmonics(
                self.expand(
                    to_expand,
                    does_f_support_separation_of_variables=True,
                    n=n_end_1 + n_end_2 - 1,  # at least n_end + 2
                    n_end=n_end_1 + n_end_2 - 1,
                    condon_shortley_phase=condon_shortley_phase,
                )
            )
        ).real

    def harmonics_translation_coef_using_triplet(
        self,
        spherical: Mapping[TSpherical | Literal["r"], Array | NativeArray],
        *,
        n_end: int,
        n_end_add: int,
        condon_shortley_phase: bool,
        k: Array | NativeArray,
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
        spherical : Mapping[TSpherical, Array | NativeArray]
            The translation vector in spherical coordinates.
        n_end : int
            The maximum degree of the harmonic.
        n_end_add : int
            The maximum degree of the harmonic to be summed over.
        condon_shortley_phase : bool
            Whether to apply the Condon-Shortley phase.
        k : Array | NativeArray
            The wavenumber.
        is_type_same : bool
            Whether the type of the elementary solutions is same.

        Returns
        -------
        Array
            The translation coefficients of `2 * self.s_ndim` dimensions.
            [-self.s_ndim,-1] dimensions are to be
            summed over with the elementary solutions
            to get translated elementary solution
            which quantum number is [-2*self.s_ndim,-self.s_ndim-1] indices.

        """
        # [user1,...,userM,n1,...,nN,nsummed1,...,nsummedN,ntemp1,...,ntempN]
        n = self.index_array_harmonics(self.root, n_end=n_end, expand_dims=True)[
            (...,) + (None,) * (2 * self.s_ndim)
        ]
        ns = self.index_array_harmonics(self.root, n_end=n_end_add, expand_dims=True)[
            (None,) * self.s_ndim + (...,) + (None,) * self.s_ndim
        ]
        ntemp = self.index_array_harmonics(
            self.root, n_end=n_end + n_end_add - 1, expand_dims=True
        )[(None,) * (2 * self.s_ndim) + (...,)]

        # returns [user1,...,userM,n1,...,nN,np1,...,npN]
        coef = (2 * ivy.pi) ** (self.e_ndim / 2) * ivy.sqrt(2 / ivy.pi)
        t_Y = self.harmonics(  # type: ignore
            spherical,
            n_end=n_end + n_end_add - 1,
            condon_shortley_phase=condon_shortley_phase,
            expand_dims=True,
            concat=True,
        )
        t_RS = self.harmonics_regular_singular(
            spherical,
            harmonics=t_Y,
            k=k,
            type="regular" if is_type_same else "singular",
        )
        return coef * ivy.sum(
            (-1j) ** (n - ns - ntemp)
            * t_RS[(...,) + (None,) * (2 * self.s_ndim) + (slice(None),) * self.s_ndim]
            * self.harmonics_twins_expansion(
                n_end_1=n_end,
                n_end_2=n_end_add,
                condon_shortley_phase=condon_shortley_phase,
                conj_1=False,
                conj_2=True,
                analytic=True,
            ),
            axis=list(range(-self.s_ndim, 0)),
        )
