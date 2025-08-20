import numpy as np
import numpy as xp
import pytest

from ultrasphere.harmonics.eigenfunction import type_b, type_bdash, type_c


def type_b_scalar(theta: float, s_beta: float, l_beta: int, l: int) -> float:
    """
    Scalar version of type_b mainly for testing.

    Parameters
    ----------
    theta : float
        [0, π]
    s_beta : float
        The number of non-leaf child nodes of the node beta.
    l_beta : int
        The quantum number of the node beta.
    l : int
        The quantum number of the node.

    Returns
    -------
    float
        The value of the eigenfunction.

    """
    array = type_b(xp.asarray(theta), n_end=l + l_beta + 1, s_beta=xp.asarray(s_beta))
    return array[l_beta, l].item()


def type_bdash_scalar(theta: float, s_alpha: float, l_alpha: int, l: int) -> float:
    """
    Scalar version of type_bdash mainly for testing.

    Parameters
    ----------
    theta : float
        [-π/2, π/2]
    s_alpha : float
        The number of non-leaf child nodes of the node alpha.
    l_alpha : int
        The quantum number of the node alpha.
    l : int
        The quantum number of the node.

    Returns
    -------
    float
        The value of the eigenfunction.

    """
    array = type_bdash(
        xp.asarray(theta), n_end=l + l_alpha + 1, s_alpha=xp.asarray(s_alpha)
    )
    return array[l_alpha, l].item()


def type_c_scalar(
    theta: float,
    s_alpha: float,
    s_beta: float,
    l_alpha: int,
    l_beta: int,
    l: int,
    index_with_surrogate_quantum_number: bool = False,
) -> float:
    """
    Scalar version of type_c mainly for testing.

    Parameters
    ----------
    theta : float
        [0, π/2]
    s_alpha : float
        The number of non-leaf child nodes of the node alpha.
    s_beta : float
        The number of non-leaf child nodes of the node beta.
    l_alpha : int
        The quantum number of the node alpha.
    l_beta : int
        The quantum number of the node beta.
    l : int
        The quantum number of the node.
    index_with_surrogate_quantum_number : bool, optional
        Whether to index with surrogate quantum number, by default False

    Returns
    -------
    float
        The value of the eigenfunction.

    """
    even = l - l_alpha - l_beta
    if even % 2 != 0:
        raise ValueError("l - l_alpha - l_beta must be even")

    if index_with_surrogate_quantum_number:
        array = type_c(
            xp.asarray(theta),
            n_end=2 * (l + l_alpha + l_beta) + 1,
            s_alpha=xp.asarray(s_alpha),
            s_beta=xp.asarray(s_beta),
            index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
        )
        return array[l_alpha, l_beta, even // 2].item()
    array = type_c(
        xp.asarray(theta),
        n_end=2 * (l + l_alpha + l_beta) + 1,
        s_alpha=xp.asarray(s_alpha),
        s_beta=xp.asarray(s_beta),
        index_with_surrogate_quantum_number=index_with_surrogate_quantum_number,
    )
    return array[l_alpha, l_beta, l].item()


def test_type_b() -> None:
    for theta in xp.random.random_uniform(low=0, high=xp.pi, shape=(3)):
        # we refer to 3d spherical harmonics table where s_beta = 0
        # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
        s_beta = 0
        # l = 0
        assert type_b_scalar(theta, s_beta, 0, 0) == pytest.approx(np.sqrt(1 / 2))
        # l = 1
        assert type_b_scalar(theta, s_beta, 1, 1) == pytest.approx(
            np.sqrt(3) / 2 * xp.sin(theta)
        )
        assert type_b_scalar(theta, s_beta, 0, 1) == pytest.approx(
            np.sqrt(3 / 2) * xp.cos(theta)
        )


# @pytest.mark.parametrize("shape", [(1,), (2, 3), (4, 5, 6)])
# def test_type_b(shape: tuple[int, ...]) -> None:
#     # we refer to 3d spherical harmonics table where s_beta = 0
#     # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
#     theta = xp.random.random_uniform(low=0, high=xp.pi, shape=shape)
#     expected = {
#         (0, 0, 0): np.sqrt(1 / 2),
#         (0, 1, 1): np.sqrt(3) / 2 * xp.sin(theta),
#         (1, 0, 1): np.sqrt(3 / 2) * xp.cos(theta),
#     }
#     s_beta, l_beta, l = xp.asarray(list(expected.keys())).T
#     s_beta_ = xp.reshape(s_beta,s_beta.shape + (1,) * theta.ndim)
#     l_beta_ = xp.reshape(l_beta,l_beta.shape + (1,) * theta.ndim)
#     l_ = xp.reshape(l,l.shape + (1,) * theta.ndim)
#     theta_ = xp.reshape(theta,(1,) * s_beta.ndim + theta.shape)
#     actual = type_b(theta=theta, s_beta=s_beta_, l_beta_, l_)
#     expected = xp.asarray(list(expected.values()), dtype=theta.dtype)
#     assert xp.all(xpx.isclose(
#         actual,
#         expected,
#         rtol=1e-3,
#         atol=1e-3,
#     )


@pytest.mark.parametrize("index_with_surrogate_quantum_number", [True, False])
def test_type_c(
    index_with_surrogate_quantum_number: bool,
) -> None:
    # we consider O(2) \otimes O(2) 4d spherical harmonics where s_beta = 1
    # http://kuiperbelt.la.coocan.jp/sf/egan/Diaspora/atomic-orbital/laplacian/4D-2.html
    for theta in xp.random.random_uniform(low=0, high=xp.pi / 2, shape=(3)):
        for l, l_alpha, l_beta in [
            (0, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (2, 0, 0),
            (2, 2, 0),
            (3, 1, 0),
        ]:
            s_alpha = 0
            s_beta = 0
            res = type_c_scalar(
                theta,
                s_alpha,
                s_beta,
                l_alpha,
                l_beta,
                l,
                index_with_surrogate_quantum_number,
            )
            if (l, l_alpha, l_beta) == (0, 0, 0):
                assert res == pytest.approx(np.sqrt(2))
            elif (l, l_alpha, l_beta) == (1, 1, 0):
                assert res == pytest.approx(xp.cos(theta) * 2)
            elif (l, l_alpha, l_beta) == (1, 0, 1):
                assert res == pytest.approx(xp.sin(theta) * 2)
            elif (l, l_alpha, l_beta) == (2, 0, 0):
                # alpha = beta = 0, n = 1, phi = 2 * N_1^00 * P_1^00 (cos 2 theta)
                # P_1^00(x) = x, N_1^00 = sqrt(3/2)
                assert res == pytest.approx((xp.cos(2 * theta)) * np.sqrt(3 / 2) * 2)
