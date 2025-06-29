import ivy
import pytest
from ivy import NativeArray
from shift_nth_row_n_steps import select

from ultrasphere.symmetry import flip_symmetric_tensor, to_symmetric


@pytest.fixture(autouse=True, scope="session", params=["numpy"])
def setup(request: pytest.FixtureRequest) -> None:
    ivy.set_backend(request.param)


@pytest.mark.parametrize("array", [ivy.arange(1, 10).reshape((3, 3))])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("asymmetric", [True, False])
@pytest.mark.parametrize("conjugate", [True, False])
def test_to_symmetric(
    array: ivy.Array | NativeArray, axis: int, asymmetric: bool, conjugate: bool
) -> None:
    result = to_symmetric(array, axis=axis, asymmetric=asymmetric, conjugate=conjugate)
    result_ = result
    axis = axis % len(ivy.shape(array))
    if asymmetric:
        result_ = -result_
        result_[
            (slice(None),) * axis
            + (0,)
            + (slice(None),) * (len(ivy.shape(array)) - axis - 1)
        ] = -select(result_, 0, axis=axis)
    if conjugate:
        result_ = result_.conj()
        result_[
            (slice(None),) * axis
            + (0,)
            + (slice(None),) * (len(ivy.shape(array)) - axis - 1)
        ] = select(result_, 0, axis=axis).conj()

    # test if flipped result is equal to result if index 0 is removed
    assert ivy.allclose(result, flip_symmetric_tensor(result_, axis=axis))

    # test manually
    random_index = ivy.randint(1, ivy.shape(array)[axis], shape=(1,))
    assert ivy.allclose(
        select(result, random_index, axis=axis),
        select(result_, -random_index, axis=axis),
    )


@pytest.mark.parametrize("asymmetric", [True, False])
@pytest.mark.parametrize("conjugate", [True, False])
def test_to_symmetric_manual(asymmetric: bool, conjugate: bool) -> None:
    array = [0, 1 + 1j]
    result = to_symmetric(ivy.array(array), asymmetric=asymmetric, conjugate=conjugate)
    if asymmetric and conjugate:
        expected = [0, 1 + 1j, -1 + 1j]
    elif asymmetric and not conjugate:
        expected = [0, 1 + 1j, -1 - 1j]
    elif not asymmetric and conjugate:
        expected = [0, 1 + 1j, 1 - 1j]
    else:
        expected = [0, 1 + 1j, 1 + 1j]
    assert ivy.allclose(result, ivy.array(expected))
