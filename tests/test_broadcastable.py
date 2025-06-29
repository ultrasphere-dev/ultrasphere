import pytest

from ultrasphere.check import is_same_shape


@pytest.mark.parametrize(
    "shapes, expected",
    [([(1, 2, 3), (1, 1, 3), (3, 2, 1)], True), ([(4,), (5,)], False)],
)
def test_is_broadcastable(shapes, expected):
    assert is_same_shape(*shapes) == expected
