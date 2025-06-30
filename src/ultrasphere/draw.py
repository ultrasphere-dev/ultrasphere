import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from networkx.algorithms.dag import dag_longest_path
from networkx.drawing.nx_pydot import graphviz_layout

from ultrasphere.coordinates import SphericalCoordinates, TEuclidean, TSpherical


def draw(
    c: SphericalCoordinates[TSpherical, TEuclidean], root_bottom: bool = True
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
    width = max(c.e_ndim * 0.5 + 1, 3.5)
    height = max((len(dag_longest_path(c.G)) + 1) * 0.5, 3.5)
    plt.gcf().set_size_inches(width, height)

    # layout
    try:
        pos = graphviz_layout(c.G, prog="dot")
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
        node_size=800,
        label="Spherical",
    )
    nx.draw_networkx_labels(
        c.G,
        pos,
        labels={
            n: f"{ascii_to_greek(str(n))}{c.branching_types[n].value}/{c.S[n]}"
            for n in c.s_nodes
        },
    )

    # Euclidean
    nx.draw_networkx_nodes(
        c.G,
        pos,
        nodelist=c.e_nodes,
        node_color="lightgray",
        node_shape="s",
        label="Euclidean",
    )
    nx.draw_networkx_labels(c.G, pos, labels={n: f"{n}" for n in c.e_nodes})

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
    )
    nx.draw_networkx_edges(
        c.G, pos, edgelist=c.sin_edges, edge_color=sin_color, label="sin"
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
    plt.title(f"Type {c.branching_types_expression_str} coordinates")

    return width, height
