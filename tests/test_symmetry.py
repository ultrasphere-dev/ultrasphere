import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from shift_nth_row_n_steps import select

from ultrasphere.symmetry import flip_symmetric_tensor, to_symmetric


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("asymmetric", [True, False])
@pytest.mark.parametrize("conjugate", [True, False])
def test_to_symmetric(
    axis: int, asymmetric: bool, conjugate: bool, xp: ArrayNamespaceFull
) -> None:
    array = xp.arange(1, 10).reshape((3, 3))
    result = to_symmetric(array, axis=axis, asymmetric=asymmetric, conjugate=conjugate)
    result_ = result
    axis = axis % len(array.shape)
    if asymmetric:
        result_ = -result_
        result_[
            (slice(None),) * axis
            + (0,)
            + (slice(None),) * (len(array.shape) - axis - 1)
        ] = -select(result_, 0, axis=axis)
    if conjugate:
        result_ = xp.conj(result_)
        result_[
            (slice(None),) * axis
            + (0,)
            + (slice(None),) * (len(array.shape) - axis - 1)
        ] = xp.conj(select(result_, 0, axis=axis))

    # test if flipped result is equal to result if index 0 is removed
    assert xp.all(xpx.isclose(result, flip_symmetric_tensor(result_, axis=axis)))

    # test manually
    random_index = xp.randint(1, array.shape[axis], shape=(1,))
    assert xp.all(
        xpx.isclose(
            select(result, random_index, axis=axis),
            select(result_, -random_index, axis=axis),
        )
    )


@pytest.mark.parametrize("asymmetric", [True, False])
@pytest.mark.parametrize("conjugate", [True, False])
def test_to_symmetric_manual(
    asymmetric: bool, conjugate: bool, xp: ArrayNamespaceFull
) -> None:
    array = [0, 1 + 1j]
    result = to_symmetric(xp.asarray(array), asymmetric=asymmetric, conjugate=conjugate)
    if asymmetric and conjugate:
        expected = [0, 1 + 1j, -1 + 1j]
    elif asymmetric and not conjugate:
        expected = [0, 1 + 1j, -1 - 1j]
    elif not asymmetric and conjugate:
        expected = [0, 1 + 1j, 1 - 1j]
    else:
        expected = [0, 1 + 1j, 1 + 1j]
    assert xp.all(xpx.isclose(result, xp.asarray(expected)))
