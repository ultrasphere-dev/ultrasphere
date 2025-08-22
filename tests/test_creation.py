import pytest

from ultrasphere._creation import hopf, random, standard, standard_prime


@pytest.mark.parametrize("s_ndim", [0, 1, 2, 3, 6, 103])
def test_generate(s_ndim: int) -> None:
    assert random(s_ndim).s_ndim == s_ndim
    assert standard(s_ndim).s_ndim == s_ndim
    assert standard_prime(s_ndim).s_ndim == s_ndim


@pytest.mark.parametrize("q", [0, 1, 2, 5])
def test_generate_hopf(q: int) -> None:
    assert hopf(q).e_ndim == 2**q
