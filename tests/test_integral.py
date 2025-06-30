from collections import defaultdict
from tests import xp
from ultrasphere.coordinates import SphericalCoordinates
from ultrasphere.creation import hopf, random, spherical, standard


import pytest
from array_api._2024_12 import Array, ArrayNamespace


from collections.abc import Callable, Mapping
from typing import Any, Literal


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("f, expected", [(lambda s: xp.asarray(1), 4 * xp.pi)])
def test_sphere_surface_integrate(
    f: Callable[[Mapping[Literal["theta", "phi"], Array]], Array],
    n: int,
    expected: float,
) -> None:
    c = spherical()
    assert c.integrate(
        f, does_f_support_separation_of_variables=False, n=n
    ).item() == pytest.approx(expected, rel=1e-2)


@pytest.mark.parametrize("r", [1, 3])
@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize(
    "c",
    [
        (spherical()),
        (standard(3)),
        (hopf(2)),
    ],
)
@pytest.mark.parametrize("concat", [True, False])
def test_integrate(
    c: SphericalCoordinates[TSpherical, TEuclidean], n: int, concat: bool, r: float, xp: ArrayNamespace
) -> None:
    # surface integral (area) of the sphere
    def f(s: Mapping[TSpherical, Array]) -> Array:
        if concat:
            return xp.asarray(r**c.s_ndim)
        else:
            return defaultdict(lambda: xp.asarray(r))

    actual = c.integrate(f, does_f_support_separation_of_variables=not concat, n=n)  # type: ignore
    if not concat:
        actual = xp.prod(list(actual.values()))
    expected = c.surface_area(r)
    assert actual.item() == pytest.approx(expected, rel=1e-2)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_integrate_match(n: int, xp: ArrayNamespace) -> None:
    cs: list[SphericalCoordinates[Any, Any]] = [
        random(n - 1) for _ in range(4)
    ]
    k = xp.random.random_uniform(low=0, high=1, shape=(n,))

    def create_f(
        c: SphericalCoordinates[TSpherical, TEuclidean],
    ) -> Callable[[Mapping[TSpherical, Array]], Array]:
        def f(s: Mapping[TSpherical, Array]) -> Array:
            x = c.to_euclidean(s, as_array=True)
            return xp.einsum("v,v...->...", k.astype(x.dtype), x)

        return f

    actual = [
        c.integrate(create_f(c), does_f_support_separation_of_variables=False, n=8)
        for c in cs
    ]
    assert xp.all(xpx.isclose(xp.asarray(actual)), actual[0], rtol=1e-3, atol=1e-3)