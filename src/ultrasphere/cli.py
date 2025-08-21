import matplotlib.pyplot as plt
import typer

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

    c = from_branching_types(branching_types)
    draw(c)
    plt.savefig(f"{branching_types}.{format}")
