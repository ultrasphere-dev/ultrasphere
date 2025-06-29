from typing import Literal

import ivy
import pytest
from scipy.special import eval_gegenbauer, eval_jacobi, roots_jacobi

from ultrasphere.polynomial import (
    gegenbauer,
    jacobi,
    jacobi_triplet_integral,
    legendre,
)
from ultrasphere.special import binom


@pytest.fixture(autouse=True, scope="session", params=["numpy"])
def setup(request: pytest.FixtureRequest) -> None:
    ivy.set_backend(request.param)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 3),
        (3, 3, 4),
    ],
)
@pytest.mark.parametrize("n_end", [8])
def test_jacobi(shape: tuple[int, ...], n_end: int) -> None:
    alpha = ivy.random.random_uniform(low=0, high=5, shape=shape)
    beta = ivy.random.random_uniform(low=0, high=5, shape=shape)
    x = ivy.random.random_uniform(low=0, high=1, shape=shape)
    n = ivy.arange(n_end)[
        (None,) * len(shape)
        + (
            slice(
                None,
            ),
        )
    ]
    expected = eval_jacobi(n, alpha[..., None], beta[..., None], x[..., None])
    actual = jacobi(x, alpha=alpha, beta=beta, n_end=n_end)
    assert ivy.allclose(expected, actual, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 3),
        (3, 3, 4),
    ],
)
@pytest.mark.parametrize("n_end", [8])
def test_gegenbauer(shape: tuple[int, ...], n_end: int) -> None:
    alpha = ivy.random.random_uniform(low=0, high=5, shape=shape)
    x = ivy.random.random_uniform(low=0, high=1, shape=shape)
    n = ivy.arange(n_end)[
        (None,) * len(shape)
        + (
            slice(
                None,
            ),
        )
    ]
    expected = eval_gegenbauer(n, alpha[..., None], x[..., None])
    actual = gegenbauer(x, alpha=alpha, n_end=n_end)
    assert ivy.allclose(expected, actual, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 3),
        (3, 3, 4),
    ],
)
@pytest.mark.parametrize("n_end", [8])
@pytest.mark.parametrize("type", ["gegenbauer", "jacobi"])
def test_legendre(
    shape: tuple[int, ...], n_end: int, type: Literal["gegenbauer", "jacobi"]
) -> None:
    d = ivy.random.randint(3, 8, shape=shape)
    alpha = (d - 3) / 2
    x = ivy.random.random_uniform(low=0, high=1, shape=shape)
    n = ivy.arange(n_end)[
        (None,) * len(shape)
        + (
            slice(
                None,
            ),
        )
    ]
    if type == "jacobi":
        expected = jacobi(x, alpha=alpha, beta=alpha, n_end=n_end) / binom(
            n + alpha[..., None], n
        )
    elif type == "gegenbauer":
        expected = gegenbauer(x, alpha=alpha + 1 / 2, n_end=n_end) / binom(
            n + 2 * alpha[..., None],
            n,
            # n + d[..., None] - 3, d[..., None] - 3
        )
    else:
        raise ValueError(f"Invalid type {type}")
    actual = legendre(x, ndim=ivy.asarray(d), n_end=n_end)
    assert ivy.allclose(expected, actual, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("alpha_eq_beta", [True, False])
def test_jacobi_triplet_integral(alpha_eq_beta: bool) -> None:
    n_samples = 20
    alphas = ivy.random.randint(0, 5, shape=(3, n_samples))
    alphas[2, ...] = alphas[0, ...] + alphas[1, ...]
    betas = ivy.random.randint(0, 5, shape=(3, n_samples))
    betas[2, ...] = betas[0, ...] + betas[1, ...]
    if alpha_eq_beta:
        betas = alphas

    ns = ivy.random.randint(0, 5, shape=(3, n_samples))

    expected = []
    for sample in range(n_samples):
        alpha, beta, n = alphas[..., sample], betas[..., sample], ns[..., sample]

        # expected
        x, w = roots_jacobi(
            24, ivy.sum(alpha).to_numpy() / 2, ivy.sum(beta).to_numpy() / 2
        )
        js = [eval_jacobi(n[i], alpha[i], beta[i], x) for i in range(3)]
        expected.append(ivy.sum(js[0] * js[1] * js[2] * w, axis=-1))
    expected = ivy.stack(expected, axis=-1)

    # actual
    actual = jacobi_triplet_integral(
        alpha1=alphas[0, ...],
        alpha2=alphas[1, ...],
        alpha3=alphas[2, ...],
        beta1=betas[0, ...],
        beta2=betas[1, ...],
        beta3=betas[2, ...],
        n1=ns[0, ...],
        n2=ns[1, ...],
        n3=ns[2, ...],
        normalized=False,
    )
    assert ivy.allclose(expected, actual, rtol=1e-3, atol=1e-3)
