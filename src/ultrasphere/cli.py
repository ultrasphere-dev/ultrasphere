import matplotlib.pyplot as plt
import typer
from aquarel import load_theme

from ._creation import from_branching_types
from ._draw import draw

app = typer.Typer()


@app.command()
def main(branching_types: str, format: str = "jpg") -> None:
    """
    Create a spherical coordinate system from branching types.

    Parameters
    ----------
    branching_types : str
        String representation of the branching types, e.g. "aabcc".

    """
    theme = load_theme("boxy_dark")
    theme.apply()
    fig, ax = plt.subplots()
    c = from_branching_types(branching_types)
    draw(c, ax=ax)
    fig.savefig(f"{branching_types}.{format}")
