import pytest

from ultrasphere._creation import (
    create_hopf,
    create_random,
    create_standard,
    create_standard_prime,
)


@pytest.mark.parametrize("s_ndim", [0, 1, 2, 3, 6, 103])
def test_generate(s_ndim: int) -> None:
    assert create_random(s_ndim).s_ndim == s_ndim
    assert create_standard(s_ndim).s_ndim == s_ndim
    assert create_standard_prime(s_ndim).s_ndim == s_ndim


@pytest.mark.parametrize("q", [0, 1, 2, 5])
def test_generate_hopf(q: int) -> None:
    assert create_hopf(q).c_ndim == 2**q
