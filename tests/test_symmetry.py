from array_api_compat import array_namespace
import array_api_extra as xpx
from array_api._2024_12 import Array
import pytest
from xp import NativeArray
from shift_nth_row_n_steps import select

from ultrasphere.symmetry import flip_symmetric_tensor, to_symmetric


@pytest.fixture(autouse=True, scope="session", params=["numpy"])
def setup(request: pytest.FixtureRequest) -> None:
    xp.set_backend(request.param)


@pytest.mark.parametrize("array", [xp.arange(1, 10).reshape((3, 3))])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("asymmetric", [True, False])
@pytest.mark.parametrize("conjugate", [True, False])
def test_to_symmetric(
    array: xp.asarray, axis: int, asymmetric: bool, conjugate: bool
) -> None:
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
        result_ = result_.conj()
        result_[
            (slice(None),) * axis
            + (0,)
            + (slice(None),) * (len(array.shape) - axis - 1)
        ] = select(result_, 0, axis=axis).conj()

    # test if flipped result is equal to result if index 0 is removed
    assert xp.allclose(result, flip_symmetric_tensor(result_, axis=axis))

    # test manually
    random_index = xp.randint(1, array.shape[axis], shape=(1,))
    assert xp.allclose(
        select(result, random_index, axis=axis),
        select(result_, -random_index, axis=axis),
    )


@pytest.mark.parametrize("asymmetric", [True, False])
@pytest.mark.parametrize("conjugate", [True, False])
def test_to_symmetric_manual(asymmetric: bool, conjugate: bool) -> None:
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
    assert xp.allclose(result, xp.asarray(expected))
