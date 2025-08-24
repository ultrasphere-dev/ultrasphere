import math
from collections.abc import Callable, Mapping
from typing import Any, Literal

import array_api_extra as xpx
import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull

from ultrasphere._coordinates import SphericalCoordinates, TEuclidean, TSpherical
from ultrasphere._creation import c_spherical, hopf, random, standard
from ultrasphere._integral import integrate


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("f, expected", [(lambda s: 1, 4 * math.pi)])
def test_sphere_surface_integrate(
    f: Callable[[Mapping[Literal["theta", "phi"], Array]], Array],
    n: int,
    expected: float,
    xp: ArrayNamespaceFull,
) -> None:
    def f2(s):
        return xp.asarray(f(s)) * xp.ones_like(s["theta"])

    c = c_spherical()
    assert integrate(
        c, f2, does_f_support_separation_of_variables=False, n=n, xp=xp
    ).item() == pytest.approx(expected, rel=1e-2)


@pytest.mark.parametrize("r", [1, 3])
@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
        (hopf(2)),
    ],
)
@pytest.mark.parametrize("concat", [True, False])
def test_integrate(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n: int,
    concat: bool,
    r: float,
    xp: ArrayNamespaceFull,
) -> None:
    # surface integral (area) of the sphere
    def f(s: Mapping[TSpherical, Array]) -> Array:
        if concat:
            return xp.asarray(r**c.s_ndim) * xp.ones_like(next(iter(s.values())))
        else:
            return {k: xp.asarray(r) * xp.ones_like(s[k]) for k in c.s_nodes}

    actual = integrate(
        c,
        f,
        does_f_support_separation_of_variables=not concat,  # type: ignore
        n=n,
        xp=xp,
    )
    if not concat:
        actual = xp.prod(xp.stack(list(actual.values())))
    expected = c.surface_area(r)
    assert actual.item() == pytest.approx(expected, rel=1e-2)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_integrate_match(n: int, xp: ArrayNamespaceFull) -> None:
    cs: list[SphericalCoordinates[Any, Any]] = [random(n - 1) for _ in range(4)]
    k = xp.random.random_uniform(low=0, high=1, shape=(n,))

    def create_f(
        c: SphericalCoordinates[TSpherical, TEuclidean],
    ) -> Callable[[Mapping[TSpherical, Array]], Array]:
        def f(s: Mapping[TSpherical, Array]) -> Array:
            x = c.to_euclidean(s, as_array=True)
            return xp.einsum("v,v...->...", xp.astype(k, x.dtype), x)

        return f

    actual = [
        integrate(
            c, create_f(c), does_f_support_separation_of_variables=False, n=8, xp=xp
        )
        for c in cs
    ]
    assert xp.all(xpx.isclose(xp.asarray(actual), actual[0], rtol=1e-3, atol=1e-3))
