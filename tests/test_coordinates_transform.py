from tests import xp


import pytest
from array_api._2024_12 import Array


from typing import Literal

from ultrasphere.coordinates import SphericalCoordinates
from ultrasphere.creation import hopf, random, spherical


def to_cartesian(
    *, r: Array, theta: Array, phi: Array
) -> tuple[Array, Array, Array]:
    rsin = r * xp.sin(theta)
    x = rsin * xp.cos(phi)
    y = rsin * xp.sin(phi)
    z = r * xp.cos(theta)
    return x, y, z


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 3),
        (4, 5, 6),
    ],
)
def test_spherical_to_3d(shape: tuple[int, ...]) -> None:
    r = xp.random.random_uniform(low=0, high=1, shape=shape)
    theta = xp.random.random_uniform(low=0, high=xp.pi, shape=shape)
    phi = xp.random.random_uniform(low=0, high=2 * xp.pi, shape=shape)
    spherical: dict[Literal["r", "theta", "phi"], Array] = {
        "r": r,
        "theta": theta,
        "phi": phi,
    }
    excepted = to_cartesian(r=r, theta=theta, phi=phi)
    actual = spherical().to_euclidean(spherical)
    for i, value in enumerate(excepted):
        assert xp.all(xpx.isclose(value, actual[i], rtol=1e-3, atol=1e-3))  # type: ignore
    assert set(actual.keys()) == {0, 1, 2}


def to_spherical(
    *, x: Array, y: Array, z: Array
) -> tuple[Array, Array, Array]:
    r = xp.linalg.vector_norm([x, y, z], axis=0)
    theta = xp.acos(z / r)
    phi = xp.atan2(y, x)
    return r, theta, phi


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 3),
        (4, 5, 6),
    ],
)
def test_cartesian_to_spherical_3d(shape: tuple[int, ...]) -> None:
    x = xp.random.random_uniform(low=-1, high=1, shape=shape)
    y = xp.random.random_uniform(low=-1, high=1, shape=shape)
    z = xp.random.random_uniform(low=-1, high=1, shape=shape)
    r_expected, theta_expected, phi_expected = to_spherical(x=x, y=y, z=z)
    excepted = {"r": r_expected, "theta": theta_expected, "phi": phi_expected}
    actual = spherical().from_euclidean({0: x, 1: y, 2: z})
    for key, value in excepted.items():
        assert xp.all(xpx.isclose(value, actual[key], rtol=1e-3, atol=1e-3))  # type: ignore
    assert set(actual.keys()) == {"r", "theta", "phi"}


@pytest.mark.parametrize(
    "c",
    [
        spherical(),
        hopf(3),
        random(1),
        random(5),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 3),
        (4, 5, 6),
    ],
)
def test_cartesian_to_spherical(
    c: SphericalCoordinates[TSpherical, TEuclidean], shape: tuple[int, ...]
) -> None:
    print(c.e_nodes)
    x = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    spherical = c.from_euclidean(x)
    x_reconstructed = c.to_euclidean(spherical)
    x_reconstructed = xp.stack([x_reconstructed[key] for key in range(c.e_ndim)])  # type: ignore
    assert xp.all(xpx.isclose(x, x_reconstructed, rtol=1e-6, atol=1e-6))