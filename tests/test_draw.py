from pathlib import Path

import pytest
from matplotlib import pyplot as plt

from ultrasphere._coordinates import SphericalCoordinates
from ultrasphere._creation import (
    create_hopf,
    create_random,
    create_spherical,
    create_standard,
    create_standard_prime,
)
from ultrasphere._draw import draw

PATH = Path("tests/.cache/")
Path.mkdir(PATH, exist_ok=True)


@pytest.mark.parametrize(
    "name, c",
    [
        ("spherical", create_spherical()),
        ("hoph", create_hopf(3)),
        ("standard-4", create_standard(4)),
        ("standard-prime-4", create_standard_prime(4)),
        ("random-1", create_random(1)),
        ("random-10", create_random(10)),
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
