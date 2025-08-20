from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import array_api_extra as xpx
import numpy as np
import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull
from matplotlib import pyplot as plt

from ultrasphere.coordinates import SphericalCoordinates, TEuclidean, TSpherical
from ultrasphere.creation import (
    c_spherical,
    from_branching_types,
    hopf,
    random,
    standard,
)
from ultrasphere.harmonics.assume import ndim_harmonics
from ultrasphere.harmonics.cut import expand_cut
from ultrasphere.harmonics.expansion import expand, expand_evaluate
from ultrasphere.harmonics.flatten import (
    flatten_harmonics,
    flatten_mask_harmonics,
    unflatten_harmonics,
)
from ultrasphere.harmonics.harmonics import harmonics
from ultrasphere.harmonics.harmonics import harmonics as harmonics_
from ultrasphere.harmonics.helmholtz import harmonics_regular_singular
from ultrasphere.harmonics.translation import (
    harmonics_translation_coef,
    harmonics_translation_coef_using_triplet,
    harmonics_twins_expansion,
)
from ultrasphere.integral import roots
from ultrasphere.polynomial import gegenbauer, legendre
from ultrasphere.random import random_points
from ultrasphere.special import szv
from ultrasphere.symmetry import to_symmetric

PATH = Path("tests/.cache/")
Path.mkdir(PATH, exist_ok=True)


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
        (hopf(2)),
    ],
)
@pytest.mark.parametrize("n", [4])
@pytest.mark.parametrize("condon_shortley_phase", [True, False])
def test_harmonics_orthogonal(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n: int,
    condon_shortley_phase: bool,
    xp: ArrayNamespaceFull,
) -> None:
    s, ws = roots(c, n=n, expand_dims_x=True, xp=xp)
    Y = harmonics(
        c,
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
    result = Yl * xp.conj(Yr)
    for w in ws.values():
        result = xp.einsum("w,w...->...", xp.astype(w, result.dtype), result)

    # assert quantum numbers are the same for non-zero values
    expansion_nonzero = (xp.abs(result) > 1e-3).nonzero(as_tuple=False)
    l, r = expansion_nonzero[:, : c.s_ndim], expansion_nonzero[:, c.s_ndim :]
    assert xp.all(l == r), expansion_nonzero

    # assert non-zero values are all 1
    expansion_nonzero_values = result[(xp.abs(result) > 1e-3).nonzero()]
    assert xp.all(
        xpx.isclose(
            expansion_nonzero_values,
            xp.ones_like(expansion_nonzero_values),
            rtol=1e-3,
            atol=1e-3,
        )
    )


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
        (standard(4)),
        (hopf(2)),
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
    xp: ArrayNamespaceFull,
) -> None:
    def f(s: Mapping[TSpherical, Array]) -> Array:
        return harmonics(
            c,  # type: ignore
            s,
            n_end=n,
            condon_shortley_phase=condon_shortley_phase,
            concat=concat,
            expand_dims=concat,
        )

    actual = expand(
        c,  # type: ignore
        f,
        n=n,
        n_end=n,
        does_f_support_separation_of_variables=not concat,
        condon_shortley_phase=condon_shortley_phase,
    )
    if not concat:
        for key, value in actual.items():
            # assert quantum numbers are the same for non-zero values
            expansion_nonzero = (xp.abs(value) > 1e-3).nonzero(as_tuple=False)
            l, r = (
                expansion_nonzero[:, : ndim_harmonics(c, key)],
                expansion_nonzero[:, ndim_harmonics(c, key) :],
            )
            idx = (l[:-1, :] == r[:-1, :]).all(axis=-1).nonzero(as_tuple=False)
            assert xp.all(l[idx, :] == r[idx, :])

            # assert non-zero values are all 1
    else:
        # assert quantum numbers are the same for non-zero values
        expansion_nonzero = (xp.abs(actual) > 1e-3).nonzero(as_tuple=False)
        l, r = expansion_nonzero[:, : c.s_ndim], expansion_nonzero[:, c.s_ndim :]
        assert xp.all(l == r), expansion_nonzero

        # assert non-zero values are all 1
        expansion_nonzero_values = actual[(xp.abs(actual) > 1e-3).nonzero()]
        assert xp.all(
            xpx.isclose(
                expansion_nonzero_values,
                xp.ones_like(expansion_nonzero_values),
                rtol=1e-3,
                atol=1e-3,
            )
        )


@pytest.mark.parametrize(
    "name, c, n_end",
    [
        ("spherical", c_spherical(), 25),
        ("standard-3'", from_branching_types("bpa"), 10),
        ("standard-4", standard(3), 7),
        ("hoph-2", hopf(2), 6),
        # ("hoph-3", hopf(3), 3),
        # ("random-1", random(1), 30),
        # ("random-10", random(6), 5),
    ],
)
@pytest.mark.parametrize("condon_shortley_phase", [True, False])
def test_approximate(
    name: str,
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    condon_shortley_phase: bool,
    xp: ArrayNamespaceFull,
) -> None:
    k = xp.random.random_uniform(low=0, high=1, shape=(c.e_ndim,))

    def f(s: Mapping[TSpherical, Array]) -> Array:
        x = c.to_euclidean(s, as_array=True)
        return xp.exp(1j * xp.einsum("v,v...->...", k.astype(x.dtype), x))

    spherical, _ = roots(c, n=n_end, expand_dims_x=True, xp=xp)
    expected = f(spherical)
    error = {}
    expansion = expand(
        c,
        f,
        n=n_end,
        n_end=n_end,
        does_f_support_separation_of_variables=False,
        condon_shortley_phase=condon_shortley_phase,
        xp=xp,
    )
    for n_end_c in np.linspace(1, n_end, 5):
        n_end_c = int(n_end_c)
        expansion_cut = expand_cut(c, expansion, n_end_c)
        approx = expand_evaluate(
            c,
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
        (c_spherical()),
        (standard(3)),
        (hopf(2)),
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
    xp: ArrayNamespaceFull,
) -> None:
    shape = (5,)
    x = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    y = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    k = xp.random.random_uniform(low=0, high=1, shape=shape)

    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    expected = szv(0, c.e_ndim, k * xp.linalg.vector_norm(x - y, axis=0), type=type)
    x_Y = harmonics(
        c,  # type: ignore
        x_spherical,
        n_end=n,
        condon_shortley_phase=False,
        concat=concat,
        expand_dims=expand_dims,
    )
    x_Z = harmonics_regular_singular(
        c, x_spherical, k=k, harmonics=x_Y, type=type, multiply=concat
    )
    x_R = harmonics_regular_singular(
        c,
        x_spherical,
        k=k,
        harmonics=x_Y,
        type="regular",
        multiply=concat,
    )
    y_Y = harmonics(
        c,  # type: ignore
        y_spherical,
        n_end=n,
        condon_shortley_phase=False,
        concat=concat,
        expand_dims=expand_dims,
    )
    y_Z = harmonics_regular_singular(
        c, y_spherical, k=k, harmonics=y_Y, type=type, multiply=concat
    )
    y_R = harmonics_regular_singular(
        c,
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
        assert xp.all(xpx.isclose(actual, xp.real(expected), rtol=1e-3, atol=1e-3))


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
    ],
)
@pytest.mark.parametrize("n_end", [5])
def test_addition_theorem_same_x(
    c: SphericalCoordinates[TSpherical, TEuclidean], n_end: int, xp: ArrayNamespaceFull
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
    x_Y = harmonics(
        c,  # type: ignore
        x_spherical,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    axis = set(range(0, c.s_ndim)) - {c.s_nodes.index(c.root)}
    actual = xp.sum(
        xp.real(x_Y * x_Y.conj()), axis=tuple(a + x_spherical["r"].ndim for a in axis)
    )
    assert xp.all(xpx.isclose(actual, expected))


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
    ],
)
@pytest.mark.parametrize("n_end", [12])
@pytest.mark.parametrize("type", ["legendre", "gegenbauer", "gegenbauer-cohl"])
def test_addition_theorem(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    type: Literal["legendre", "gegenbauer", "gegenbauer-cohl"],
    xp: ArrayNamespaceFull,
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

    x_Y = harmonics(
        c,  # type: ignore
        x_spherical,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    y_Y = harmonics(
        c,  # type: ignore
        y_spherical,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    # [..., n]
    axis = set(range(0, c.s_ndim)) - {c.s_nodes.index(c.root)}
    actual = xp.sum(
        xp.real(x_Y * y_Y.conj()), axis=tuple(a + x_spherical["r"].ndim for a in axis)
    )
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-4, atol=1e-4))


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (from_branching_types("a")),
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
    xp: ArrayNamespaceFull,
) -> None:
    shape = (20,)
    # get x, t, y := x + t
    x = random_points(c, shape=shape, xp=xp)
    t = random_points(c, shape=shape, xp=xp)
    k = xp.random.random_uniform(low=0.8, high=1.2, shape=shape)
    if type == "singular":
        # |t| < |x|
        t *= xp.random.random_uniform(low=0.05, high=0.1, shape=shape)
        assert (
            xp.linalg.vector_norm(t, axis=0) < xp.linalg.vector_norm(x, axis=0)
        ).all()
    # t = xp.zeros_like(t)
    y = x + t
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    y_RS = harmonics_regular_singular(
        c,
        y_spherical,
        k=k,
        harmonics=harmonics(
            c,  # type: ignore
            y_spherical,
            n_end=n_end,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=type,
    )
    x_RS = harmonics_regular_singular(
        c,
        x_spherical,
        k=k,
        harmonics=harmonics(
            c,  # type: ignore
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
    coef = harmonics_translation_coef(
        c,
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
        assert xp.all(xpx.isclose(actual, expected, rtol=1e-4, atol=1e-4))
    else:
        pytest.skip("singular case does not converge in real world computation")


def test_harmonics_translation_coef_gumerov_table(xp: ArrayNamespaceFull) -> None:
    if "torch" in xp.__name__:
        pytest.skip("round_cpu not implemented in torch")
    # Gumerov, N.A., & Duraiswami, R. (2001). Fast, Exact,
    # and Stable Computation of Multipole Translation and
    # Rotation Coefficients for the 3-D Helmholtz Equation.
    # got completely same results as the table in 12.3 Example
    c = c_spherical()
    x = xp.asarray([-1.0, 1.0, 0.0])
    t = xp.asarray([2.0, -7.0, 1.0])
    y = xp.add(x, t)
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)
    t_spherical = c.from_euclidean(t)
    k = xp.asarray(1)

    n_end = 6
    for n_end_add in [1, 3, 5, 7, 9]:
        y_RS = harmonics_regular_singular(
            c,
            y_spherical,
            k=k,
            harmonics=harmonics(
                c,  # type: ignore
                y_spherical,
                n_end=n_end,
                condon_shortley_phase=False,
                concat=True,
                expand_dims=True,
            ),
            type="singular",
        )
        x_RS = harmonics_regular_singular(
            c,
            x_spherical,
            k=k,
            harmonics=harmonics(
                c,  # type: ignore
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
        coef = harmonics_translation_coef_using_triplet(
            c,
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
        print(xp.round(expected[5, 2], decimals=6), xp.round(actual[5, 2], decimals=6))


@pytest.mark.parametrize(
    "c",
    [
        (from_branching_types("a")),
        (c_spherical()),
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
    xp: ArrayNamespaceFull,
) -> None:
    actual = harmonics_twins_expansion(
        c,
        n_end_1=n_end,
        n_end_2=n_end,
        condon_shortley_phase=condon_shortley_phase,
        conj_1=conj_1,
        conj_2=conj_2,
        analytic=False,
        xp=xp,
    )
    expected = harmonics_twins_expansion(
        c,
        n_end_1=n_end,
        n_end_2=n_end,
        condon_shortley_phase=condon_shortley_phase,
        conj_1=conj_1,
        conj_2=conj_2,
        analytic=True,
        xp=xp,
    )
    # unmatched = ~xp.isclose(actual, expected, atol=1e-5, rtol=1e-5)
    # print(unmatched.nonzero(as_tuple=False), actual[unmatched], expected[unmatched])
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-5, atol=1e-5))


@pytest.mark.parametrize(
    "c",
    [
        (from_branching_types("a")),
        (c_spherical()),
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
    xp: ArrayNamespaceFull,
) -> None:
    shape = ()
    # get x, t, y := x + t
    x = random_points(c, shape=shape, xp=xp)
    t = random_points(c, shape=shape, xp=xp)
    k = xp.random.random_uniform(low=0.8, high=1.2, shape=shape)
    if (from_, to_) == ("singular", "singular"):
        # |t| < |x| (if too close, the result would be inaccurate)
        t = t * xp.random.random_uniform(low=0.05, high=0.1, shape=shape)
        assert (
            xp.linalg.vector_norm(t, axis=0) < xp.linalg.vector_norm(x, axis=0)
        ).all()
    elif (from_, to_) == ("regular", "singular"):
        # |t| > |x| (if too close, the result would be inaccurate)
        t = t * xp.random.random_uniform(low=10, high=20, shape=shape)
        assert (
            xp.linalg.vector_norm(t, axis=0) > xp.linalg.vector_norm(x, axis=0)
        ).all()

    # t = xp.zeros_like(t)
    y = x + t
    t_spherical = c.from_euclidean(t)
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    y_RS = harmonics_regular_singular(
        c,
        y_spherical,
        k=k,
        harmonics=harmonics(
            c,  # type: ignore
            y_spherical,
            n_end=n_end,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=to_,
    )
    x_RS = harmonics_regular_singular(
        c,
        x_spherical,
        k=k,
        harmonics=harmonics(
            c,  # type: ignore
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
    coef = harmonics_translation_coef_using_triplet(
        c,
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
            * harmonics_regular_singular(
                c,
                t_spherical,
                k=k,
                harmonics=harmonics(
                    c,  # type: ignore
                    t_spherical,
                    n_end=n_end + n_end_add - 1,
                    condon_shortley_phase=condon_shortley_phase,
                    concat=True,
                    expand_dims=True,
                ),
                type="regular" if from_ == to_ else "singular",
            )[..., idx]
        )
        assert xp.all(
            xpx.isclose(
                coef,
                expected2,
                rtol=1e-5,
                atol=1e-5,
            )
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
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-3, atol=1e-3))


@pytest.mark.parametrize(
    "c",
    [
        random(1),
        random(2),
        c_spherical(),
        hopf(2),
    ],
)
@pytest.mark.parametrize("n_end", [4, 7])
def test_flatten_mask_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean], n_end: int, xp: ArrayNamespaceFull
) -> None:
    points = roots(c, n=n_end, expand_dims_x=True, xp=xp)[0]
    harmonics = harmonics_(
        c,
        # random_points(c, shape=shape, type="spherical"),
        points,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    expected = (xp.abs(harmonics) > 1e-3).any(
        axis=tuple(range(harmonics.ndim - c.s_ndim)), keepdims=False
    )
    actual = flatten_mask_harmonics(c, n_end, xp=xp)
    assert actual.shape == expected.shape
    try:
        assert xp.all(actual == expected)
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
        random(1),
        random(2),
        c_spherical(),
        hopf(2),
    ],
)
def test_flatten_unflatten_harmonics(
    c: SphericalCoordinates[TSpherical, TEuclidean], xp: ArrayNamespaceFull
) -> None:
    n_end = 4
    harmonics = harmonics_(
        c,
        roots(c, n=n_end, expand_dims_x=True, xp=xp)[0],
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    flattened = flatten_harmonics(c, harmonics)
    unflattened = unflatten_harmonics(c, flattened, n_end=n_end)
    assert xp.all(harmonics == unflattened)
