import ivy
import pytest

from ultrasphere.coordinates import SphericalCoordinates, TEuclidean, TSpherical
from ultrasphere.polynomial import gegenbauer
from ultrasphere.special import sjv
from ultrasphere.wave_expansion import plane_wave_expansion_coef


@pytest.fixture(autouse=True, scope="session", params=["numpy", "torch"])
def setup(request: pytest.FixtureRequest) -> None:
    ivy.set_backend(request.param)


@pytest.mark.parametrize(
    "c",
    [
        (SphericalCoordinates.spherical()),
        (SphericalCoordinates.from_branching_types("bba")),
    ],
)
@pytest.mark.parametrize("n_end", [30])
def test_plane_wave_decomposition(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
) -> None:
    shape = (5,)
    r = ivy.random.random_uniform(low=0, high=2, shape=shape)
    gamma = ivy.random.random_uniform(low=0, high=ivy.pi, shape=shape)
    k = ivy.ones_like(r)
    expected = ivy.exp(1j * k * r * ivy.cos(gamma))
    n = ivy.arange(n_end)[(None,) * len(shape) + (slice(None),)]
    coef = plane_wave_expansion_coef(n, e_ndim=c.e_ndim)
    actual = ivy.sum(
        coef
        * sjv(
            n,
            ivy.asarray(c.e_ndim),
            k[..., None] * r[..., None],
        )
        # * legendre(ivy.cos(gamma), ndim=c.e_ndim, n_end=n_end)
        * gegenbauer(
            ivy.cos(gamma), alpha=ivy.asarray((c.e_ndim - 2) / 2), n_end=n_end
        ),
        axis=-1,
    )
    assert ivy.allclose(actual, expected, rtol=1e-3, atol=1e-3)
