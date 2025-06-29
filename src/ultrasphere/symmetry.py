import ivy
from ivy import Array, NativeArray
from shift_nth_row_n_steps import narrow


def to_symmetric(
    input: Array | NativeArray,
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
    input : Array | NativeArray
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
    input_to_symmetric = input
    if asymmetric:
        input_to_symmetric = -input_to_symmetric
    if conjugate:
        input_to_symmetric = ivy.conj(input_to_symmetric)
    if not include_zero_twice:
        input_to_symmetric = narrow(
            input_to_symmetric, 1, ivy.shape(input_to_symmetric)[axis] - 1, axis=axis
        )
    input_to_symmetric = ivy.flip(input_to_symmetric, axis=axis)
    return ivy.concat([input, input_to_symmetric], axis=axis)


def flip_symmetric_tensor(
    input: Array | NativeArray, *, axis: int = -1, include_zero_twice: bool = False
) -> Array:
    """
    Flip a symmetric tensor.

    Parameters
    ----------
    input : Array | NativeArray
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
    if include_zero_twice:
        return ivy.flip(input, axis=axis)
    zero = narrow(input, 0, 1, axis=axis)
    nonzero = narrow(input, 1, ivy.shape(input)[axis] - 1, axis=axis)
    return ivy.concat([zero, ivy.flip(nonzero, axis=axis)], axis=axis)
