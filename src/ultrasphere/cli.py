import matplotlib.pyplot as plt
import typer
from aquarel import load_theme

from ._creation import from_branching_types
from ._draw import draw

app = typer.Typer()


@app.command()
def main(branching_types: str, format: str = "jpg", theme: str = "boxy_dark") -> None:
    """
    Create a spherical coordinate system from branching types.

    Parameters
    ----------
    branching_types : str
        String representation of the branching types, e.g. "aabcc".
    format : str, optional
        The format to save the figure, by default "jpg".
    theme : str, optional
        The theme to apply to the plot, by default "boxy_dark".
        Set to "none" to disable theming.

    """
    if theme != "none":
        theme_ = load_theme(theme)
        theme_.apply()
    fig, ax = plt.subplots()
    c = from_branching_types(branching_types)
    draw(c, ax=ax)
    fig.savefig(f"{branching_types}.{format}")
