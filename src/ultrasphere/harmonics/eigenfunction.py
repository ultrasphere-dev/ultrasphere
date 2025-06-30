r"""
For all types of eigenfunctions,
the maximum value of the absolute value of
the quantum number of the child nodes will be
the same as that of the parent,
and the converse does not hold.

Therefore, the argument takes
the maximum value of the quantum number of the most parent node.

We do not consider negative m for type a nodes because
$$
type_a(\theta, m) = type_a(\theta, -m)^* * (-1)^m
$$

Reference
---------
DOI:10.3842/SIGMA.2013.042 p.17-18
"""

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from shift_nth_row_n_steps import shift_nth_row_n_steps

from ..polynomial import jacobi, jacobi_normalization_constant
from ..symmetry import to_symmetric


def type_a(
    theta: Array,
    n_end: int,
    *,
    condon_shortley_phase: bool,
    include_negative_m: bool = True,
) -> Array:
    """
    Eigenfunction for type a node.

    Parameters
    ----------
    theta : Array
        [0, 2π)
    n_end : int
        The maximum degree of the harmonic.
    condon_shortley_phase : bool, optional
        Whether to apply the Condon-Shortley phase,
        which just multiplies the result by (-1)^m.

        It seems to be mainly used in quantum mechanics for convenience.

        scipy.special.sph_harm (or scipy.special.lpmv) uses the Condon-Shortley phase.

        If False, `Y^{-m}_{l} = Y^{m}_{l}*`. If True, `Y^{-m}_{l} = (-1)^m Y^{m}_{l}*`.
        (Simply because `e^{i -m phi} = (e^{i m phi})*`)
    include_negative_m : bool, optional
        Whether to include negative m values, by default True
        If True, the m values are [0, 1, ..., n_end-1, -n_end+1, ..., -1],
        and starts from 0, not -n_end+1.

    Returns
    -------
    Array
        The result of the eigenfunction.

    """
    xp = array_namespace(theta)
    m = xp.reshape(
        xp.arange(0, n_end, dtype=theta.dtype, device=theta.device),
        [1] * (theta.ndim) + [-1],
    )
    if include_negative_m:
        m = to_symmetric(m, axis=-1, asymmetric=True, conjugate=False)
    res = xp.exp(
        xp.asarray(
            1j,
            dtype=(
                xp.complex64
                if theta.dtype in [xp.complex64, xp.float32]
                else xp.complex128
            ),
            device=theta.device,
        )
        * m
        * theta[..., None]
    ) / xp.sqrt(2 * xp.pi)
    if condon_shortley_phase:
        res *= (-1) ** ((m + xp.abs(m)) // 2)
    return res


def type_b(
    theta: Array,
    *,
    n_end: int,
    s_beta: Array | int,
    index_with_surrogate_quantum_number: bool = False,
    is_beta_type_a_and_include_negative_m: bool = False,
    fill_value: float = 0,
) -> Array:
    """
    Eigenfunction for type b node.

    Parameters
    ----------
    theta : Array
        [0, π]
    n_end : int
        Positive integer, l - l_beta, where l is the quantum number of this node.
    s_beta : Array
        The number of non-leaf child nodes of the node beta.
    index_with_surrogate_quantum_number : bool, optional
        Whether to index with surrogate quantum number, by default False
    is_beta_type_a_and_include_negative_m : bool, optional
        Whether the node beta is type a and include negative m, by default False
    fill_value : float, optional
        The value to fill for the indices that are not possible, by default 0

    Returns
    -------
    Array
        If index_with_surrogate_quantum_number is True,
        [..., l_beta, n] of size (..., n_end, n_end)
        Otherwise,
        [..., l_beta, l] of size (..., n_end, n_end), if l < l_beta value is 0.

    """
    xp = array_namespace(theta)
    if isinstance(s_beta, int):
        s_beta = xp.asarray(s_beta)
    # using broadcasting may cause problems, we have to be very careful here
    l_beta = xp.reshape(
        xp.arange(0, n_end, dtype=theta.dtype, device=theta.device),
        [1] * (theta.ndim) + [-1],
    )
    n = xp.reshape(
        xp.arange(0, n_end, dtype=theta.dtype, device=theta.device),
        [1] * (theta.ndim) + [1, -1],
    )
    alpha = l_beta + s_beta[..., None] / 2
    res = (
        jacobi_normalization_constant(
            alpha=alpha[..., None], beta=alpha[..., None], n=n, xp=xp
        )
        * (xp.sin(theta[..., None, None]) ** l_beta[..., None])
        * jacobi(n_end=n_end, alpha=alpha, beta=alpha, x=xp.cos(theta[..., None]))
    )
    if not index_with_surrogate_quantum_number:
        # [l_beta, n] -> [l_beta, l = n + l_beta]
        res = shift_nth_row_n_steps(
            res,
            axis_row=-2,
            axis_shift=-1,
            cut_padding=True,
            fill_values=fill_value,
        )
    if is_beta_type_a_and_include_negative_m:
        res = to_symmetric(res, axis=-2, asymmetric=False, conjugate=False)
    return res


def type_bdash(
    theta: Array,
    *,
    n_end: int,
    s_alpha: Array | int,
    index_with_surrogate_quantum_number: bool = False,
    is_alpha_type_a_and_include_negative_m: bool = False,
    fill_value: float = 0,
) -> Array:
    """
    Eigenfunction for type b node.

    Parameters
    ----------
    theta : Array
        [-π/2, π/2]
    n_end : int
        Positive integer, l - l_alpha, where l is the quantum number of this node.
    s_alpha : Array
        The number of non-leaf child nodes of the node alpha.
    index_with_surrogate_quantum_number : bool, optional
        Whether to index with surrogate quantum number, by default False
    is_alpha_type_a_and_include_negative_m : bool, optional
        Whether the node alpha is type a and include negative m, by default False
    fill_value : float, optional
        The value to fill for the indices that are not possible, by default 0

    Returns
    -------
    Array
        If index_with_surrogate_quantum_number is True,
        [..., l_alpha, n] of size (..., n_end, n_end)
        Otherwise,
        [..., l_alpha, l] of size (..., n_end, n_end), if l < l_alpha value is 0.

    """
    xp = array_namespace(theta)
    if isinstance(s_alpha, int):
        s_alpha = xp.asarray(s_alpha)
    l_alpha = xp.reshape(
        xp.arange(0, n_end, dtype=theta.dtype, device=theta.device),
        [1] * (theta.ndim) + [-1],
    )
    n = xp.reshape(
        xp.arange(0, n_end, dtype=theta.dtype, device=theta.device),
        [1] * (theta.ndim) + [1, -1],
    )
    beta = l_alpha + s_alpha[..., None] / 2
    res = (
        jacobi_normalization_constant(
            alpha=beta[..., None], beta=beta[..., None], n=n, xp=xp
        )
        * (xp.cos(theta[..., None, None]) ** l_alpha[..., None])
        * jacobi(n_end=n_end, alpha=beta, beta=beta, x=xp.sin(theta[..., None]))
    )
    if not index_with_surrogate_quantum_number:
        res = shift_nth_row_n_steps(
            res,
            axis_row=-2,
            axis_shift=-1,
            cut_padding=True,
            fill_values=fill_value,
        )
    # [l_alpha, n] -> [l_alpha, l = n + l_alpha]
    if is_alpha_type_a_and_include_negative_m:
        res = to_symmetric(res, axis=-2, asymmetric=False, conjugate=False)
    return res


def type_c(
    theta: Array,
    *,
    n_end: int,
    s_alpha: Array | int,
    s_beta: Array | int,
    index_with_surrogate_quantum_number: bool = False,
    is_alpha_type_a_and_include_negative_m: bool = False,
    is_beta_type_a_and_include_negative_m: bool = False,
    fill_value: float = 0,
) -> Array:
    """
    Eigenfunction for type c node.

    Parameters
    ----------
    theta : Array
        [0, π/2]
    n_end : int
        Positive integer, (l - l_alpha - l_beta) / 2,
        where l is the quantum number of this node.
    s_alpha : Array
        The number of non-leaf child nodes of the node alpha.
    s_beta : Array
        The number of non-leaf child nodes of the node beta.
    index_with_surrogate_quantum_number : bool, optional
        Whether to index with surrogate quantum number, by default False
    is_alpha_type_a_and_include_negative_m : bool, optional
        Whether the node alpha is type a and include negative m, by default False
    is_beta_type_a_and_include_negative_m : bool, optional
        Whether the node beta is type a and include negative m, by default False
    fill_value : float, optional
        The value to fill for the indices that are not possible, by default 0

    Returns
    -------
    Array
        [..., l_alpha, l_beta, l] of size (..., n_end, n_end),
        if l < l_alpha + l_beta value is 0.

    """
    xp = array_namespace(theta)
    if isinstance(s_alpha, int):
        s_alpha = xp.asarray(s_alpha)
    if isinstance(s_beta, int):
        s_beta = xp.asarray(s_beta)
    l_alpha = xp.reshape(
        xp.arange(0, n_end, dtype=theta.dtype, device=theta.device),
        [1] * (theta.ndim) + [-1, 1],
    )  # 2d
    l_beta = xp.reshape(
        xp.arange(0, n_end, dtype=theta.dtype, device=theta.device),
        [1] * (theta.ndim) + [1, -1],
    )  # 2d
    n = xp.reshape(
        xp.arange(0, (n_end + 1) // 2, dtype=theta.dtype, device=theta.device),
        [1] * (theta.ndim) + [1, 1, -1],
    )  # 3d
    alpha = l_alpha + s_alpha[..., None, None] / 2  # 2d
    beta = l_beta + s_beta[..., None, None] / 2  # 2d
    res = (
        2 ** ((alpha + beta) / 2 + 1)[..., None]
        * jacobi_normalization_constant(
            alpha=alpha[..., None], beta=beta[..., None], n=n, xp=xp
        )
        * (xp.sin(theta[..., None, None, None]) ** l_beta[..., None])
        * (xp.cos(theta[..., None, None, None]) ** l_alpha[..., None])
        * jacobi(
            n_end=(n_end + 1) // 2,
            alpha=beta,
            beta=alpha,  # this is weird but correct
            x=xp.cos(2 * theta[..., None, None]),
        )
    )
    # n_end = 3 -> max l = 2 -> max jacobi order = 1 -> jacobi n_end = 2
    # n_end = 4 -> max l = 3 -> max jacobi order = 1 -> jacobi n_end = 2
    # http://kuiperbelt.la.coocan.jp/sf/egan/Diaspora/atomic-orbital/
    # laplacian/4D-2.html
    if not index_with_surrogate_quantum_number:
        # complicated reshaping
        # [l_alpha, l_beta, n] -> [l_alpha, l_beta, l = 2n + l_alpha + l_beta]
        # 1. [l_alpha, l_beta, n] -> [l_alpha, l_beta, 2n]
        # add zeros to the left for each row, i.e. [1, 2, 3] -> [1, 0, 2, 0, 3, 0]
        res_expaneded = xp.zeros(res.shape[:-1] + (n_end,))
        res_expaneded[..., ::2] = res
        # 2. [l_alpha, l_beta, 2n] -> [l_alpha, l_beta, 2n + l_alpha]
        res_expaneded = shift_nth_row_n_steps(
            res_expaneded,
            axis_row=-3,
            axis_shift=-1,
            cut_padding=True,
            fill_values=fill_value,
        )
        # 3. [l_alpha, l_beta, 2n + l_alpha] ->
        # [l_alpha, l_beta, 2n + l_alpha + l_beta]
        res = shift_nth_row_n_steps(
            res_expaneded,
            axis_row=-2,
            axis_shift=-1,
            cut_padding=True,
            fill_values=fill_value,
        )
    if is_alpha_type_a_and_include_negative_m:
        res = to_symmetric(res, axis=-3, asymmetric=False, conjugate=False)
    if is_beta_type_a_and_include_negative_m:
        res = to_symmetric(res, axis=-2, asymmetric=False, conjugate=False)
    return res
