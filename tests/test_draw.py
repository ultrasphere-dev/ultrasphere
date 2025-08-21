from pathlib import Path

import pytest
from matplotlib import pyplot as plt

from ultrasphere.coordinates import SphericalCoordinates
from ultrasphere.creation import c_spherical, hopf, random
from ultrasphere.draw import draw

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
    fig, _ = plt.subplots(layout="constrained")
    w, h = draw(c)
    plt.savefig(PATH / f"{name}.svg")
    plt.savefig(PATH / f"{name}.png", dpi=600)
    plt.close()
