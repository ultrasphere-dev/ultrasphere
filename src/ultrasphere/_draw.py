import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from networkx.algorithms.dag import dag_longest_path
from networkx.drawing.nx_agraph import graphviz_layout

from ultrasphere._coordinates import SphericalCoordinates, TEuclidean, TSpherical

_ASCII_TO_GREEK = {
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


def _ascii_to_greek(s: str) -> str:
    for k, v in _ASCII_TO_GREEK.items():
        s = s.replace(k, v)
    return s


def draw(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    root_bottom: bool = True,
    ax: Axes | None = None,
) -> tuple[float, float]:
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

    # plt.rcParams["text.usetex"] = True
    # remove spines
    ax = ax or plt.gca()
    fig = ax.figure
    ax.set_frame_on(False)
    ax.grid(False)
    width = max(c.e_ndim * 0.5 + 1, 3.5)
    height = max((len(dag_longest_path(c.G)) + 1) * 0.5, 3.5)
    additional_width = 0.8
    fig.set_size_inches(width + additional_width, height)
    fig.subplots_adjust(
        right=0.9 - additional_width / (width + additional_width),
        left=0,
        top=0.9,
        bottom=0,
    )

    # layout
    try:
        pos = graphviz_layout(c.G, prog="dot", args='-GTBbalance="max"')
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
        pos = nx.spring_layout(c.G)

    # Spherical
    nx.draw_networkx_nodes(
        c.G,
        pos,
        nodelist=c.s_nodes,
        node_color="darkgray",
        node_size=850,
        label="Spherical",
        ax=ax,
        margins=0.1,
    )
    nx.draw_networkx_labels(
        c.G,
        pos,
        labels={
            n: f"{_ascii_to_greek(str(n))}\n{c.branching_types[n].value}/{c.S[n]}"
            for n in c.s_nodes
        },
        ax=ax,
    )

    # Euclidean
    nx.draw_networkx_nodes(
        c.G,
        pos,
        nodelist=c.e_nodes,
        node_color="lightgray",
        node_shape="s",
        label="Euclidean",
        ax=ax,
    )
    nx.draw_networkx_labels(c.G, pos, labels={n: f"{n}" for n in c.e_nodes}, ax=ax)

    # edges
    cos_color = "orange"
    sin_color = "blue"
    nx.draw_networkx_edges(
        c.G,
        pos,
        edgelist=c.cos_edges,
        edge_color=cos_color,
        label="cos",
        style="dashed",
        ax=ax,
    )
    nx.draw_networkx_edges(
        c.G, pos, edgelist=c.sin_edges, edge_color=sin_color, label="sin", ax=ax
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
    fig.legend(
        handles=handles,
        loc="lower right" if root_bottom else "upper right",
    )
    ax.set_title(f"Type {c.branching_types_expression_str} coordinates")

    return width, height
