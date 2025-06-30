from tests.test_cooridnates import PATH
from ultrasphere.coordinates import SphericalCoordinates
from ultrasphere.creation import hopf, random, spherical
from ultrasphere.draw import draw


import pytest
from matplotlib import pyplot as plt


@pytest.mark.parametrize(
    "name, c",
    [
        ("spherical", spherical()),
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