from pathlib import Path

import pytest
from matplotlib import pyplot as plt

from ultrasphere._coordinates import SphericalCoordinates
from ultrasphere._creation import c_spherical, hopf, random
from ultrasphere._draw import draw

PATH = Path("tests/.cache/")
Path.mkdir(PATH, exist_ok=True)


@pytest.mark.parametrize(
    "name, c",
    [
        ("spherical", c_spherical()),
        ("hoph", hopf(3)),
        ("random-1", random(1)),
        ("random-10", random(10)),
    ],
)
def test_draw(name: str, c: SphericalCoordinates[str, int]) -> None:
    try:
        import pydot  # noqa
        import pygraphviz  # noqa
    except ImportError:
        pytest.skip("pydot is not installed, skipping draw test")
    draw(c)
    plt.savefig(PATH / f"{name}.jpg")
    plt.close()
