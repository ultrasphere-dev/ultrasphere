from array_api_compat import array_namespace
import array_api_extra as xpx
from array_api._2024_12 import Array, ArrayNamespace


from .special import binom, lgamma


def jacobi(
    x: Array,
    *,
    alpha: Array,
    beta: Array,
    n_end: int,
) -> Array:
    """
    Computes the Jacobi polynomials of order {1,...,n_max} at the points x.
    The shape of x, alpha, and beta should be broadcastable to a common shape.

    (...) -> (..., n_max)

    Parameters
    ----------
    x : Array
        X
    alpha : Array
        Alpha
    beta : Array
        Beta
    n_end : int
        The maximum order of the polynomials.

    Returns
    -------
    Array
        The values of the Jacobi polynomials at the points x of order {1,...,n_end-1}.

    """
    xp = array_namespace(x, alpha, beta)
    x, alpha, beta = xp.broadcast_arrays(x, alpha, beta)

    ps = []

    # Compute the first two polynomials
    # https://en.wikipedia.org/wiki/Jacobi_polynomials#Special_cases
    p = xp.ones(
        xpx.broadcast_shapes(x.shape, alpha.shape, beta.shape),
        dtype=x.dtype,
        device=x.device,
    )
    ps.append(p)
    if n_end > 1:
        p = (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2
        ps.append(p)

    # Use recurrence relation to compute the rest
    # https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations
    for n in range(2, n_end):
        a = n + alpha
        b = n + beta
        c = a + b
        d = (c - 1) * (c * (c - 2) * x + (a - b) * (c - 2 * n)) * ps[-1] - 2 * (
            a - 1
        ) * (b - 1) * c * ps[-2]
        p = d / (2 * n * (c - n) * (c - 2))
        ps.append(p)
    return xp.stack(ps, axis=-1)


def log_jacobi_normalization_constant(
    *, alpha: Array, beta: Array, n: Array
) -> Array:
    """
    Computes the log of normalization constant of
    the Jacobi polynomials of order n with parameters alpha and beta.

    Parameters
    ----------
    alpha : Array
        Alpha
    beta : Array
        Beta
    n : Array
        The order of the Jacobi polynomial.

    Returns
    -------
    Array
        The log of the normalization constant.

    """
    xp = array_namespace(alpha, beta, n)
    logupper = (
        xp.log(2 * n + alpha + beta + 1)
        + lgamma(n + alpha + beta + 1.0)
        + lgamma(n + 1.0)
    )
    loglower = (
        xp.log(2) * (alpha + beta + 1)
        + lgamma(n + alpha + 1.0)
        + lgamma(n + beta + 1.0)
    )
    return 0.5 * (logupper - loglower)


def jacobi_normalization_constant(
    *, alpha: Array, beta: Array, n: Array, xp: ArrayNamespace
) -> Array:
    """
    Computes the normalization constant of
    the Jacobi polynomials of order n with parameters alpha and beta.

    Parameters
    ----------
    alpha : Array
        Alpha
    beta : Array
        Beta
    n : Array
        The order of the Jacobi polynomial.

    Returns
    -------
    Array
        The normalization constant.

    """
    return xp.exp(log_jacobi_normalization_constant(alpha=alpha, beta=beta, n=n))


def gegenbauer(
    x: Array, *, alpha: Array, n_end: int
) -> Array:
    """
    Computes the Gegenbauer polynomials of
    order {1,...,n_max} at the points x.
    The shape of x and alpha should be broadcastable to a common shape.

    (...) -> (..., n_max)

    Parameters
    ----------
    x : Array
        X
    alpha : Array
        Alpha
    n_end : int
        The maximum order of the polynomials.

    Returns
    -------
    Array
        The values of the Gegenbauer polynomials
        at the points x of order {1,...,n_end-1}.

    """
    xp = array_namespace(x, alpha)
    x, alpha = xp.broadcast_arrays(x, alpha)
    n = xp.arange(0, n_end, dtype=x.dtype, device=x.device)[
        (None,) * x.ndim + (slice(None),)
    ]
    alpha = xp.asarray(alpha - 1 / 2, dtype=x.dtype, device=x.device)
    log_coef = xp.astype(
        lgamma(2.0 * alpha[..., None] + 1.0 + n)
        - lgamma(2.0 * alpha[..., None] + 1.0)
        - (lgamma(alpha[..., None] + 1.0 + n) - lgamma(alpha[..., None] + 1.0))
    , x.dtype)
    return xp.exp(log_coef) * jacobi(x, alpha=alpha, beta=alpha, n_end=n_end)


def legendre(x: Array, *, ndim: Array, n_end: int) -> Array:
    """
    Computes the generalized Legendre polynomials of
    order {1,...,n_max} at the points x.
    The shape of x should be broadcastable to a common shape.

    (...) -> (..., n_max)

    Parameters
    ----------
    x : Array
        X
    ndim : int
        The dimension of the space.
    n_end : int
        The maximum order of the polynomials.

    Returns
    -------
    Array
        The values of the generalized Legendre polynomials
        at the points x of order {1,...,n_end-1}.

    """
    # return jacobi(x, alpha=xp.asarray((ndim-3)/2),
    # beta=xp.asarray((ndim-3)/2), n_end=n_end)
    xp = array_namespace(x, ndim)
    x, ndim = xp.broadcast_arrays(x, ndim)
    n = xp.arange(0, n_end, dtype=x.dtype, device=x.device)[
        (None,) * x.ndim + (slice(None),)
    ]
    return xp.where(
        ndim[..., None] == 2,
        # Chebyshev polynomials of the first kind
        xp.cos(n * xp.acos(x)[..., None]),
        gegenbauer(x, alpha=(ndim - 2) / 2, n_end=n_end)
        / binom(n + ndim[..., None] - 3, ndim[..., None] - 3),
    )


def jacobi_triplet_integral(
    alpha1: Array,
    alpha2: Array,
    alpha3: Array | None,
    beta1: Array,
    beta2: Array,
    beta3: Array | None,
    n1: Array,
    n2: Array,
    n3: Array,
    *,
    normalized: bool,
) -> Array:
    r"""
    Integral of three Jacobi polynomials.

    .. math::
        \int_{-1}^{1} P_n^{(\alpha_1, \beta_1)}(x) P_n^{(\alpha_2, \beta_2)}(x)
        P_n^{(\alpha_3, \beta_3)}(x) dx

    The special case (alpha_a = beta_a)
    would be Gaunt coefficients (for associated Legendre polynomials).

    Parameters
    ----------
    alpha1 : Array
        The alpha parameter of the first Jacobi polynomial.
    alpha2 : Array
        The alpha parameter of the second Jacobi polynomial.
    alpha3 : Array | None
        The alpha parameter of the third Jacobi polynomial.
        Must be alpha_1 + alpha_2.
    beta1 : Array
        The beta parameter of the first Jacobi polynomial.
    beta2 : Array
        The beta parameter of the second Jacobi polynomial.
    beta3 : Array | None
        The beta parameter of the third Jacobi polynomial.
        Must be beta_1 + beta_2.
    n1 : Array
        The order of the first Jacobi polynomial.
    n2 : Array
        The order of the second Jacobi polynomial.
    n3 : Array
        The order of the third Jacobi polynomial.
    normalized : bool
        Whether all Jacobi polynomials are normalized.
        The computation is faster when True.

    Returns
    -------
    Array
        The integral of the three Jacobi polynomials.

    Raises
    ------
    ValueError
        If the sum of the orders is not an integer.
        If alpha3 is not None and alpha3 is not alpha1 + alpha2.
        If beta3 is not None and beta3 is not beta1 + beta2.

    """
    xp = array_namespace(
        alpha1, alpha2, alpha3, beta1, beta2, beta3, n1, n2, n3
    )
    from py3nj import wigner3j

    if alpha3 is None:
        alpha3 = alpha1 + alpha2
    elif xp.any(alpha3 != alpha1 + alpha2):
        raise ValueError(f"{alpha3=} should be {(alpha1 + alpha2)=}")
    if beta3 is None:
        beta3 = beta1 + beta2
    elif xp.any(beta3 != beta1 + beta2):
        raise ValueError(f"{beta3=} should be {(beta1 + beta2)=}")

    alphas = xp.stack(xp.broadcast_arrays(alpha1, alpha2, alpha3), axis=0)
    betas = xp.stack(xp.broadcast_arrays(beta1, beta2, beta3), axis=0)
    ns = xp.stack(xp.broadcast_arrays(n1, n2, n3), axis=0)
    del alpha1, alpha2, alpha3, beta1, beta2, beta3, n1, n2, n3

    # wigner arguments
    Ls2 = 2 * ns + alphas + betas
    Ms2 = alphas + betas
    Ns2 = betas - alphas
    # check if Ls2, Ms2, Ns2 are integers
    if (
        xp.any(Ls2 != xp.round(Ls2))
        or xp.any(Ms2 != xp.round(Ms2))
        or xp.any(Ns2 != xp.round(Ns2))
    ):
        raise ValueError(
            f"The sum of the orders should be an integer. {Ls2=}, {Ms2=}, {Ns2=}"
        )
    # round and cast to int
    Ls2, Ms2, Ns2 = xp.round(Ls2), xp.round(Ms2), xp.round(Ns2)
    Ls2, Ms2, Ns2 = xp.astype(Ls2, int), xp.astype(Ms2, int), xp.astype(Ns2, int)

    # coefficients
    # note that there is no need to sqrt the normalization constant
    logcoefs = [
        (
            0
            if normalized
            else -log_jacobi_normalization_constant(
                alpha=alphas[i, ...], beta=betas[i, ...], n=ns[i, ...]
            )
        )
        + 0.5 * xp.log(2 * ns[i, ...] + alphas[i, ...] + betas[i, ...] + 1)
        for i in range(3)
    ]
    phase = (-1) ** (-Ls2[0] + Ls2[1] - betas[2, ...])
    coef = phase / xp.sqrt(2) * xp.exp(xp.sum(logcoefs, axis=0))
    return (
        coef
        * wigner3j(
            Ls2[0, ...],
            Ls2[1, ...],
            Ls2[2, ...],
            Ms2[0, ...],
            Ms2[1, ...],
            -Ms2[2, ...],
            ignore_invalid=True,
        )
        * wigner3j(
            Ls2[0, ...],
            Ls2[1, ...],
            Ls2[2, ...],
            Ns2[0, ...],
            Ns2[1, ...],
            -Ns2[2, ...],
            ignore_invalid=True,
        )
    )
