from array_api_compat import array_namespace
import array_api_extra as xpx
from array_api._2024_12 import Array

from shift_nth_row_n_steps import narrow


def to_symmetric(
    input: Array,
    *,
    axis: int = -1,
    asymmetric: bool = False,
    conjugate: bool = False,
    include_zero_twice: bool = False,
) -> Array:
    """
    Extend a tensor to its opposite symmetric form.

    Parameters
    ----------
    input : Array
        The input tensor.
    axis : int, optional
        The axis to extend, by default -1
    asymmetric : bool, optional
        If True, the input tensor is multiplied by -1, by default False
    conjugate : bool, optional
        If True, the input tensor is conjugated, by default False
    include_zero_twice : bool, optional
        If True, the zeroth element is included twice, by default False

    Returns
    -------
    Array
        The symmetric tensor.
        If not include_zero_twice,
        forall a < input.shape[axis] result[-a] = result[a]
        Else,
        forall a < input.shape[axis] result[-a-1] = result[a]

    """
    xp = array_namespace(input)
    input_to_symmetric = input
    if asymmetric:
        input_to_symmetric = -input_to_symmetric
    if conjugate:
        input_to_symmetric = xp.conj(input_to_symmetric)
    if not include_zero_twice:
        input_to_symmetric = narrow(
            input_to_symmetric, 1, input_to_symmetric.shape[axis] - 1, axis=axis
        )
    input_to_symmetric = xp.flip(input_to_symmetric, axis=axis)
    return xp.concat([input, input_to_symmetric], axis=axis)


def flip_symmetric_tensor(
    input: Array, *, axis: int = -1, include_zero_twice: bool = False
) -> Array:
    """
    Flip a symmetric tensor.

    Parameters
    ----------
    input : Array
        The input tensor.
    axis : int, optional
        The axis to flip, by default -1
    include_zero_twice : bool, optional
        If True, the zeroth element is included twice, by default False

    Returns
    -------
    Array
        The flipped tensor.
        If not include_zero_twice,
        forall a < input.shape[axis] result[-a] = result[a] = input[a] = input[-a]
        Else,
        forall a < input.shape[axis] result[-a-1] = result[a] = input[a] = input[-a-1]

    """
    xp = array_namespace(input)
    if include_zero_twice:
        return xp.flip(input, axis=axis)
    zero = narrow(input, 0, 1, axis=axis)
    nonzero = narrow(input, 1, input.shape[axis] - 1, axis=axis)
    return xp.concat([zero, xp.flip(nonzero, axis=axis)], axis=axis)
