from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from array_api_compat import array_namespace
import array_api_extra as xpx
from array_api._2024_12 import Array
import pytest

from matplotlib import pyplot as plt

from ultrasphere.coordinates import SphericalCoordinates, TEuclidean, TSpherical
from ultrasphere.polynomial import gegenbauer, legendre
from ultrasphere.special import szv
from ultrasphere.symmetry import to_symmetric

PATH = Path("tests/.cache/")
Path.mkdir(PATH, exist_ok=True)


@pytest.fixture(autouse=True, scope="module", params=["numpy", "torch"])
def setup(request: pytest.FixtureRequest) -> None:
    xp.set_backend(request.param)
    xp.set_default_dtype(xp.float64)


@pytest.mark.parametrize(
    "name, c",
    [
        ("spherical", SphericalCoordinates.spherical()),
        ("hoph", SphericalCoordinates.hopf(3)),
        ("random-1", SphericalCoordinates.random(1)),
        ("random-10", SphericalCoordinates.random(10)),
    ],
)
def test_draw(name: str, c: SphericalCoordinates[str, int]) -> None:
    fig, _ = plt.subplots(layout="constrained")
    w, h = c.draw()
    plt.savefig(PATH / f"{name}.svg")
    plt.savefig(PATH / f"{name}.png", dpi=600)
    plt.close()


@pytest.mark.parametrize("s_ndim", [0, 1, 2, 3, 6, 103])
def test_generate(s_ndim: int) -> None:
    assert SphericalCoordinates.random(s_ndim).s_ndim == s_ndim
    assert SphericalCoordinates.standard(s_ndim).s_ndim == s_ndim
    assert SphericalCoordinates.standard_prime(s_ndim).s_ndim == s_ndim


@pytest.mark.parametrize("q", [0, 1, 2, 5])
def test_generate_hopf(q: int) -> None:
    assert SphericalCoordinates.hopf(q).e_ndim == 2**q


def to_cartesian(
    *, r: Array, theta: Array, phi: Array
) -> tuple[Array, Array, Array]:
    rsin = r * xp.sin(theta)
    x = rsin * xp.cos(phi)
    y = rsin * xp.sin(phi)
    z = r * xp.cos(theta)
    return x, y, z


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
    actual = SphericalCoordinates.spherical().to_euclidean(spherical)
    for i, value in enumerate(excepted):
        assert xp.allclose(value, actual[i], rtol=1e-3, atol=1e-3)  # type: ignore
    assert set(actual.keys()) == {0, 1, 2}


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
    actual = SphericalCoordinates.spherical().from_euclidean({0: x, 1: y, 2: z})
    for key, value in excepted.items():
        assert xp.allclose(value, actual[key], rtol=1e-3, atol=1e-3)  # type: ignore
    assert set(actual.keys()) == {"r", "theta", "phi"}


@pytest.mark.parametrize(
    "c",
    [
        SphericalCoordinates.spherical(),
        SphericalCoordinates.hopf(3),
        SphericalCoordinates.random(1),
        SphericalCoordinates.random(5),
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
    assert xp.allclose(x, x_reconstructed, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("f, expected", [(lambda s: xp.asarray(1), 4 * xp.pi)])
def test_sphere_surface_integrate(
    f: Callable[[Mapping[Literal["theta", "phi"], Array]], Array],
    n: int,
    expected: float,
) -> None:
    c = SphericalCoordinates.spherical()
    assert c.integrate(
        f, does_f_support_separation_of_variables=False, n=n
    ).item() == pytest.approx(expected, rel=1e-2)


@pytest.mark.parametrize("r", [1, 3])
@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize(
    "c",
    [
        (SphericalCoordinates.spherical()),
        (SphericalCoordinates.from_branching_types("bba")),
        (SphericalCoordinates.hopf(2)),
    ],
)
@pytest.mark.parametrize("concat", [True, False])
def test_integrate(
    c: SphericalCoordinates[TSpherical, TEuclidean], n: int, concat: bool, r: float
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
def test_integrate_match(n: int) -> None:
    cs: list[SphericalCoordinates[Any, Any]] = [
        SphericalCoordinates.random(n - 1) for _ in range(4)
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
    assert xp.allclose(xp.asarray(actual), actual[0], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "c",
    [
        (SphericalCoordinates.spherical()),
        (SphericalCoordinates.from_branching_types("bba")),
        (SphericalCoordinates.hopf(2)),
    ],
)
@pytest.mark.parametrize("n", [4])
@pytest.mark.parametrize("condon_shortley_phase", [True, False])
def test_harmonics_orthogonal(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n: int,
    condon_shortley_phase: bool,
) -> None:
    s, ws = c.roots(n=n, expand_dims_x=True)
    Y = harmonics(c, 
        s,
        n_end=n,
        condon_shortley_phase=condon_shortley_phase,
        concat=True,
        expand_dims=True,
    )
    Yl = Y[
        (slice(None),) * c.s_ndim
        + (
            slice(
                None,
            ),
        )
        * c.s_ndim
        + (None,) * c.s_ndim
    ]
    Yr = Y[(slice(None),) * c.s_ndim + (None,) * c.s_ndim + (slice(None),) * c.s_ndim]
    result = Yl * Yr.conj()
    for w in ws.values():
        result = xp.einsum("w,w...->...", w.astype(result.dtype), result)

    # assert quantum numbers are the same for non-zero values
    expansion_nonzero = (result.abs() > 1e-3).nonzero(as_tuple=False)
    l, r = expansion_nonzero[:, : c.s_ndim], expansion_nonzero[:, c.s_ndim :]
    assert xp.all_equal(l, r), expansion_nonzero

    # assert non-zero values are all 1
    expansion_nonzero_values = result[(result.abs() > 1e-3).nonzero()]
    assert xp.allclose(
        expansion_nonzero_values,
        xp.ones_like(expansion_nonzero_values),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "c",
    [
        (SphericalCoordinates.spherical()),
        (SphericalCoordinates.from_branching_types("bba")),
        (SphericalCoordinates.from_branching_types("bbba")),
        (SphericalCoordinates.hopf(2)),
    ],
)
@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("condon_shortley_phase", [True, False])
@pytest.mark.parametrize("concat", [True, False])
def test_orthogonal_expand(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n: int,
    condon_shortley_phase: bool,
    concat: bool,
) -> None:
    def f(s: Mapping[TSpherical, Array]) -> Array:
        return harmonics(c,   # type: ignore
            s,
            n_end=n,
            condon_shortley_phase=condon_shortley_phase,
            concat=concat,
            expand_dims=concat,
        )

    actual = expand(c,   # type: ignore
        f,
        n=n,
        n_end=n,
        does_f_support_separation_of_variables=not concat,
        condon_shortley_phase=condon_shortley_phase,
    )
    if not concat:
        for key, value in actual.items():
            # assert quantum numbers are the same for non-zero values
            expansion_nonzero = (value.abs() > 1e-3).nonzero(as_tuple=False)
            l, r = (
                expansion_nonzero[:, : ndim_harmonics(c, key)],
                expansion_nonzero[:, ndim_harmonics(c, key) :],
            )
            idx = (l[:-1, :] == r[:-1, :]).all(axis=-1).nonzero(as_tuple=False)
            assert xp.all_equal(l[idx, :], r[idx, :])

            # assert non-zero values are all 1
    else:
        # assert quantum numbers are the same for non-zero values
        expansion_nonzero = (actual.abs() > 1e-3).nonzero(as_tuple=False)
        l, r = expansion_nonzero[:, : c.s_ndim], expansion_nonzero[:, c.s_ndim :]
        assert xp.all_equal(l, r), expansion_nonzero

        # assert non-zero values are all 1
        expansion_nonzero_values = actual[(actual.abs() > 1e-3).nonzero()]
        assert xp.allclose(
            expansion_nonzero_values,
            xp.ones_like(expansion_nonzero_values),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "name, c, n_end",
    [
        ("spherical", SphericalCoordinates.spherical(), 25),
        ("standard-3'", SphericalCoordinates.from_branching_types("bpa"), 10),
        ("standard-4", SphericalCoordinates.from_branching_types("bba"), 7),
        ("hoph-2", SphericalCoordinates.hopf(2), 6),
        # ("hoph-3", SphericalCoordinates.hopf(3), 3),
        # ("random-1", SphericalCoordinates.random(1), 30),
        # ("random-10", SphericalCoordinates.random(6), 5),
    ],
)
@pytest.mark.parametrize("condon_shortley_phase", [True, False])
def test_approximate(
    name: str,
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    condon_shortley_phase: bool,
) -> None:
    k = xp.random.random_uniform(low=0, high=1, shape=(c.e_ndim,))

    def f(s: Mapping[TSpherical, Array]) -> Array:
        x = c.to_euclidean(s, as_array=True)
        return xp.exp(1j * xp.einsum("v,v...->...", k.astype(x.dtype), x))

    spherical, _ = c.roots(n=n_end, expand_dims_x=True)
    expected = f(spherical)
    error = {}
    expansion = expand(c, 
        f,
        n=n_end,
        n_end=n_end,
        does_f_support_separation_of_variables=False,
        condon_shortley_phase=condon_shortley_phase,
    )
    for n_end_c in xp.linspace(1, n_end, 5).to_numpy():
        n_end_c = int(n_end_c)
        expansion_cut = c.expand_cut(expansion, n_end_c)
        approx = c.expand_evaluate(
            expansion_cut,
            spherical,
            condon_shortley_phase=condon_shortley_phase,
        )
        error[n_end_c] = xp.mean(xp.abs(approx - expected))
    fig, ax = plt.subplots()
    ax.plot(list(error.keys()), list(error.values()))
    ax.set_xlabel("Degree")
    ax.set_ylabel("MAE")
    ax.set_title(f"Spherical Harmonics Expansion Error for {c}")
    ax.set_yscale("log")
    fig.savefig(PATH / f"{name}-approximate.png")
    assert error[max(error.keys())] < 2e-3


@pytest.mark.skip(reason="test_translation_coef covers this")
@pytest.mark.parametrize(
    "c",
    [
        (SphericalCoordinates.spherical()),
        (SphericalCoordinates.from_branching_types("bba")),
        (SphericalCoordinates.hopf(2)),
    ],
)
@pytest.mark.parametrize("n", [5])
@pytest.mark.parametrize(
    "concat, expand_dims", [(True, True), (False, False), (False, True)]
)
@pytest.mark.parametrize("type", ["j"])  # , "y", "h1", "h2"])
def test_harmonics_regular_singular_j_expansion(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n: int,
    concat: bool,
    expand_dims: bool,
    type: Literal["j", "y", "h1", "h2"],
) -> None:
    shape = (5,)
    x = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    y = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    k = xp.random.random_uniform(low=0, high=1, shape=shape)

    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    expected = szv(0, c.e_ndim, k * xp.vector_norm(x - y, axis=0), type=type)
    x_Y = harmonics(c,   # type: ignore
        x_spherical,
        n_end=n,
        condon_shortley_phase=False,
        concat=concat,
        expand_dims=expand_dims,
    )
    x_Z = harmonics_regular_singular(c,
        x_spherical, k=k, harmonics=x_Y, type=type, multiply=concat
    )
    x_R = harmonics_regular_singular(c,
        x_spherical,
        k=k,
        harmonics=x_Y,
        type="regular",
        multiply=concat,
    )
    y_Y = harmonics(c,   # type: ignore
        y_spherical,
        n_end=n,
        condon_shortley_phase=False,
        concat=concat,
        expand_dims=expand_dims,
    )
    y_Z = harmonics_regular_singular(c,
        y_spherical, k=k, harmonics=y_Y, type=type, multiply=concat
    )
    y_R = harmonics_regular_singular(c,
        y_spherical,
        k=k,
        harmonics=y_Y,
        type="regular",
        multiply=concat,
    )
    if concat:
        coef = 2 * (2 * xp.pi) ** ((c.e_ndim - 1) / 2)
        # smaller one (in terms of l2 norm) -> j, larger one -> z
        actual = coef * xp.where(
            x_spherical["r"] < y_spherical["r"],
            xp.sum(x_R * y_Z * y_Y.conj(), axis=tuple(range(-c.s_ndim, 0))),
            xp.sum(x_Z * y_R * y_Y.conj(), axis=tuple(range(-c.s_ndim, 0))),
        )
        assert xp.allclose(actual, expected.real, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "c",
    [
        (SphericalCoordinates.spherical()),
        (SphericalCoordinates.from_branching_types("bba")),
    ],
)
@pytest.mark.parametrize("n_end", [5])
def test_addition_theorem_same_x(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
) -> None:
    """
    Test the addition theorem for spherical harmonics.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.335

    """
    shape = (5,)
    x = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    x_spherical = c.from_euclidean(x)
    n = xp.arange(n_end)[(None,) * len(shape) + (slice(None),)]
    expected = (
        c.harm_n_ndim(n) / c.surface_area() * xp.ones_like(x_spherical["r"])[:, None]
    )
    x_Y = harmonics(c,   # type: ignore
        x_spherical,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    axis = set(range(0, c.s_ndim)) - {c.s_nodes.index(c.root)}
    actual = xp.sum(
        x_Y * x_Y.conj(), axis=tuple(a + x_spherical["r"].ndim for a in axis)
    ).real
    assert xp.allclose(actual, expected)


@pytest.mark.parametrize(
    "c",
    [
        (SphericalCoordinates.spherical()),
        (SphericalCoordinates.from_branching_types("bba")),
    ],
)
@pytest.mark.parametrize("n_end", [12])
@pytest.mark.parametrize("type", ["legendre", "gegenbauer", "gegenbauer-cohl"])
def test_addition_theorem(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    type: Literal["legendre", "gegenbauer", "gegenbauer-cohl"],
) -> None:
    """
    Test the addition theorem for spherical harmonics.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.335

    """
    shape = (5,)
    x = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    y = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))

    # [...]
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    ip = xp.sum(x * y, axis=0)
    ip_normalized = ip / x_spherical["r"] / y_spherical["r"]
    # expected [..., n]
    n = xp.arange(n_end)[(None,) * c.s_ndim + (slice(None),)]
    d = c.e_ndim
    if type == "legendre":
        expected = (
            legendre(
                ip_normalized,
                ndim=xp.asarray(d),
                n_end=n_end,
            )
            * c.harm_n_ndim(n)
            / c.surface_area()
        )
    elif type == "gegenbauer":
        alpha = xp.asarray((d - 2) / 2)[(None,) * ip_normalized.ndim]
        expected = (
            gegenbauer(ip_normalized, alpha=alpha, n_end=n_end)
            / gegenbauer(xp.ones_like(ip_normalized), alpha=alpha, n_end=n_end)
            * c.harm_n_ndim(n)
            / c.surface_area()
        )
    elif type == "gegenbauer-cohl":
        alpha = xp.asarray((d - 2) / 2)[(None,) * ip_normalized.ndim]
        expected = (
            gegenbauer(ip_normalized, alpha=alpha, n_end=n_end)
            * (2 * n + d - 2)
            / (d - 2)
            / c.surface_area()
        )
    else:
        raise ValueError("type must be 'legendre' or 'gegenbauer")

    x_Y = harmonics(c,   # type: ignore
        x_spherical,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    y_Y = harmonics(c,   # type: ignore
        y_spherical,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    # [..., n]
    axis = set(range(0, c.s_ndim)) - {c.s_nodes.index(c.root)}
    actual = xp.sum(
        x_Y * y_Y.conj(), axis=tuple(a + x_spherical["r"].ndim for a in axis)
    ).real
    assert xp.allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "c",
    [
        (SphericalCoordinates.spherical()),
        (SphericalCoordinates.from_branching_types("a")),
    ],
)
@pytest.mark.parametrize("n_end, n_end_add", [(4, 14)])
@pytest.mark.parametrize("condon_shortley_phase", [False])
@pytest.mark.parametrize("type", ["regular", "singular"])
def test_harmonics_translation_coef(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    n_end_add: int,
    condon_shortley_phase: bool,
    type: Literal["regular", "singular"],
) -> None:
    shape = (20,)
    # get x, t, y := x + t
    x = c.random_points(shape=shape)
    t = c.random_points(shape=shape)
    k = xp.random.random_uniform(low=0.8, high=1.2, shape=shape)
    if type == "singular":
        # |t| < |x|
        t *= xp.random.random_uniform(low=0.05, high=0.1, shape=shape)
        assert (xp.vector_norm(t, axis=0) < xp.vector_norm(x, axis=0)).all()
    # t = xp.zeros_like(t)
    y = x + t
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    y_RS = harmonics_regular_singular(c,
        y_spherical,
        k=k,
        harmonics=harmonics(c,   # type: ignore
            y_spherical,
            n_end=n_end,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=type,
    )
    x_RS = harmonics_regular_singular(c,
        x_spherical,
        k=k,
        harmonics=harmonics(c,   # type: ignore
            x_spherical,
            n_end=n_end_add,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=type,
    )
    # expected (y)
    expected = y_RS

    # actual
    coef = c.harmonics_translation_coef(
        t,
        n_end=n_end,
        n_end_add=n_end_add,
        k=k,
        condon_shortley_phase=condon_shortley_phase,
    )
    actual = xp.sum(
        x_RS[(...,) + (None,) * c.s_ndim + (slice(None),) * c.s_ndim] * coef,
        axis=tuple(range(-c.s_ndim, 0)),
    )
    if type == "regular":
        assert xp.allclose(actual, expected, rtol=1e-4, atol=1e-4)
    else:
        pytest.skip("singular case does not converge in real world computation")


def test_harmonics_translation_coef_gumerov_table() -> None:
    if xp.current_backend_str() == "torch":
        pytest.skip("round_cpu not implemented in torch")
    # Gumerov, N.A., & Duraiswami, R. (2001). Fast, Exact,
    # and Stable Computation of Multipole Translation and
    # Rotation Coefficients for the 3-D Helmholtz Equation.
    # got completely same results as the table in 12.3 Example
    c = SphericalCoordinates.spherical()
    x = xp.asarray([-1.0, 1.0, 0.0])
    t = xp.asarray([2.0, -7.0, 1.0])
    y = xp.add(x, t)
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)
    t_spherical = c.from_euclidean(t)
    k = xp.asarray(1)

    n_end = 6
    for n_end_add in [1, 3, 5, 7, 9]:
        y_RS = harmonics_regular_singular(c,
            y_spherical,
            k=k,
            harmonics=harmonics(c,   # type: ignore
                y_spherical,
                n_end=n_end,
                condon_shortley_phase=False,
                concat=True,
                expand_dims=True,
            ),
            type="singular",
        )
        x_RS = harmonics_regular_singular(c,
            x_spherical,
            k=k,
            harmonics=harmonics(c,   # type: ignore
                x_spherical,
                n_end=n_end_add,
                condon_shortley_phase=False,
                concat=True,
                expand_dims=True,
            ),
            type="regular",
        )
        # expected (y)
        expected = y_RS

        # actual
        coef = c.harmonics_translation_coef_using_triplet(
            t_spherical,
            n_end=n_end,
            n_end_add=n_end_add,
            k=k,
            condon_shortley_phase=False,
            is_type_same=False,
        )
        actual = xp.sum(
            x_RS[(...,) + (None,) * c.s_ndim + (slice(None),) * c.s_ndim] * coef,
            axis=tuple(range(-c.s_ndim, 0)),
        )
        print(
            xp.round(expected[5, 2], decimals=6), xp.round(actual[5, 2], decimals=6)
        )


@pytest.mark.parametrize(
    "c",
    [
        (SphericalCoordinates.from_branching_types("a")),
        (SphericalCoordinates.spherical()),
    ],
)
@pytest.mark.parametrize("n_end", [6])
@pytest.mark.parametrize("condon_shortley_phase", [False])
@pytest.mark.parametrize("conj_1", [True, False])
@pytest.mark.parametrize("conj_2", [True, False])
def test_harmonics_twins_expansion(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    condon_shortley_phase: bool,
    n_end: int,
    conj_1: bool,
    conj_2: bool,
) -> None:
    actual = c.harmonics_twins_expansion(
        n_end_1=n_end,
        n_end_2=n_end,
        condon_shortley_phase=condon_shortley_phase,
        conj_1=conj_1,
        conj_2=conj_2,
        analytic=False,
    )
    expected = c.harmonics_twins_expansion(
        n_end_1=n_end,
        n_end_2=n_end,
        condon_shortley_phase=condon_shortley_phase,
        conj_1=conj_1,
        conj_2=conj_2,
        analytic=True,
    )
    # unmatched = ~xp.isclose(actual, expected, atol=1e-5, rtol=1e-5)
    # print(unmatched.nonzero(as_tuple=False), actual[unmatched], expected[unmatched])
    assert xp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "c",
    [
        (SphericalCoordinates.from_branching_types("a")),
        (SphericalCoordinates.spherical()),
    ],
)
@pytest.mark.parametrize("n_end, n_end_add", [(4, 4)])
@pytest.mark.parametrize("condon_shortley_phase", [False])
@pytest.mark.parametrize(
    "from_,to_",
    [("regular", "regular"), ("singular", "singular"), ("regular", "singular")],
)
def test_harmonics_translation_coef_using_triplet(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    n_end_add: int,
    condon_shortley_phase: bool,
    from_: Literal["regular", "singular"],
    to_: Literal["regular", "singular"],
) -> None:
    shape = ()
    # get x, t, y := x + t
    x = c.random_points(shape=shape)
    t = c.random_points(shape=shape)
    k = xp.random.random_uniform(low=0.8, high=1.2, shape=shape)
    if (from_, to_) == ("singular", "singular"):
        # |t| < |x| (if too close, the result would be inaccurate)
        t = t * xp.random.random_uniform(low=0.05, high=0.1, shape=shape)
        assert (xp.vector_norm(t, axis=0) < xp.vector_norm(x, axis=0)).all()
    elif (from_, to_) == ("regular", "singular"):
        # |t| > |x| (if too close, the result would be inaccurate)
        t = t * xp.random.random_uniform(low=10, high=20, shape=shape)
        assert (xp.vector_norm(t, axis=0) > xp.vector_norm(x, axis=0)).all()

    # t = xp.zeros_like(t)
    y = x + t
    t_spherical = c.from_euclidean(t)
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    y_RS = harmonics_regular_singular(c,
        y_spherical,
        k=k,
        harmonics=harmonics(c,   # type: ignore
            y_spherical,
            n_end=n_end,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=to_,
    )
    x_RS = harmonics_regular_singular(c,
        x_spherical,
        k=k,
        harmonics=harmonics(c,   # type: ignore
            x_spherical,
            n_end=n_end_add,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=from_,
    )
    # expected (y)
    expected = y_RS

    # actual
    coef = c.harmonics_translation_coef_using_triplet(
        t_spherical,
        n_end=n_end,
        n_end_add=n_end_add,
        k=k,
        condon_shortley_phase=condon_shortley_phase,
        is_type_same=from_ == to_,
    )
    if c.e_ndim == 2:
        n = to_symmetric(xp.arange(n_end), asymmetric=True)
        n_add = to_symmetric(xp.arange(n_end_add), asymmetric=True)
        idx = n[:, None] - n_add[None, :]
        expected2 = (
            2
            * harmonics_regular_singular(c,
                t_spherical,
                k=k,
                harmonics=harmonics(c,   # type: ignore
                    t_spherical,
                    n_end=n_end + n_end_add - 1,
                    condon_shortley_phase=condon_shortley_phase,
                    concat=True,
                    expand_dims=True,
                ),
                type="regular" if from_ == to_ else "singular",
            )[..., idx]
        )
        assert xp.allclose(
            coef,
            expected2,
            rtol=1e-5,
            atol=1e-5,
        )
    actual = xp.sum(
        x_RS[(...,) + (None,) * c.s_ndim + (slice(None),) * c.s_ndim] * coef,
        axis=tuple(range(-c.s_ndim, 0)),
    )
    wrong_idx = xp.abs(actual - expected) > 1e-3
    if wrong_idx.any():
        print(actual[wrong_idx], expected[wrong_idx], wrong_idx.nonzero(as_tuple=False))
    if (from_, to_) == ("singular", "singular"):
        pytest.skip("singular case does not converge in real world computation")
    assert xp.allclose(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "c",
    [
        SphericalCoordinates.random(1),
        SphericalCoordinates.random(2),
        SphericalCoordinates.spherical(),
        SphericalCoordinates.hopf(2),
    ],
)
@pytest.mark.parametrize("n_end", [4, 7])
def test_flatten_mask_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
) -> None:
    points = c.roots(n=n_end, expand_dims_x=True)[0]
    harmonics = harmonics(c, 
        # c.random_points(shape=shape, type="spherical"),
        points,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    expected = (harmonics.abs() > 1e-3).any(
        axis=tuple(range(harmonics.ndim - c.s_ndim)), keepdims=False
    )
    actual = flatten_mask_harmonics(c, n_end)
    assert actual.shape == expected.shape
    try:
        assert xp.all_equal(actual, expected)
    except AssertionError:
        wrong_index = actual != expected
        print(
            actual[wrong_index],
            expected[wrong_index],
            wrong_index.nonzero(as_tuple=False),
        )
        raise


@pytest.mark.parametrize(
    "c",
    [
        SphericalCoordinates.random(1),
        SphericalCoordinates.random(2),
        SphericalCoordinates.spherical(),
        SphericalCoordinates.hopf(2),
    ],
)
def test_flatten_unflatten_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean],
) -> None:
    n_end = 4
    harmonics = harmonics(c, 
        c.roots(n=n_end, expand_dims_x=True)[0],
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    flattened = c.flatten_harmonics(harmonics)
    unflattened = c.unflatten_harmonics(flattened, n_end=n_end)
    assert xp.all_equal(harmonics, unflattened)


@pytest.mark.parametrize(
    "c",
    [
        SphericalCoordinates.random(1),
        SphericalCoordinates.random(2),
        SphericalCoordinates.spherical(),
        SphericalCoordinates.hopf(2),
    ],
)
@pytest.mark.parametrize("n_end", [4, 7])
def test_index_array_harmonics_all(
    c: SphericalCoordinates[TSpherical, TEuclidean], n_end: int
) -> None:
    iall_concat = index_array_harmonics_all(c, 
        n_end=n_end, include_negative_m=False, expand_dims=True, as_array=True
    )
    iall = index_array_harmonics_all(c, 
        n_end=n_end, include_negative_m=False, expand_dims=True, as_array=False
    )
    assert iall_concat.shape == (
        c.s_ndim,
        *xpx.broadcast_shapes(*[v.shape for v in iall.values()]),
    )
    for i, s_node in enumerate(c.s_nodes):
        # the shapes not necessarily match, so all_equal cannot be used
        assert (iall_concat[i] == iall[s_node]).all()
