from typing import Literal

import array_api_extra as xpx
import numpy as np
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from scipy.special import hankel1, hankel2, jv, jvp, spherical_jn, spherical_yn, yv, yvp

from ultrasphere.special._bessel import shn1, shn2, sjv, syv


@pytest.mark.parametrize("derivative", [True, False])
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("type", ["j", "y", "h1", "h2"])
def test_sjyn(
    d: Literal[2, 3],
    type: Literal["j", "y", "h1", "h2"],
    derivative: bool,
    xp: ArrayNamespaceFull,
) -> None:
    n = np.random.randint(5, size=(10, 10))
    z = np.random.random((10, 10))
    if type == "j":
        if d == 2:
            expected = np.sqrt(np.pi / 2) * (jvp(n, z) if derivative else jv(n, z))
        elif d == 3:
            expected = spherical_jn(n, z, derivative=derivative)
        else:
            raise ValueError("d must be 2 or 3")
        expected = xp.asarray(expected)
        actual = sjv(xp.asarray(n), xp.asarray(d), xp.asarray(z), derivative=derivative)
    elif type == "y":
        if d == 2:
            expected = np.sqrt(np.pi / 2) * (yvp(n, z) if derivative else yv(n, z))
        elif d == 3:
            expected = spherical_yn(n, z, derivative=derivative)
        else:
            raise ValueError("d must be 2 or 3")
        expected = xp.asarray(expected)
        actual = syv(xp.asarray(n), xp.asarray(d), xp.asarray(z), derivative=derivative)
    elif type == "h1":
        if derivative:
            pytest.skip("derivative of hankel1 not implemented")
        if d == 2:
            expected = np.sqrt(np.pi / 2) * hankel1(n, z)
        elif d == 3:
            expected = spherical_jn(n, z) + 1j * spherical_yn(n, z)
        expected = xp.asarray(expected)
        actual = shn1(xp.asarray(n), xp.asarray(d), xp.asarray(z))
    elif type == "h2":
        if derivative:
            pytest.skip("derivative of hankel2 not implemented")
        if d == 2:
            expected = np.sqrt(np.pi / 2) * hankel2(n, z)
        elif d == 3:
            expected = spherical_jn(n, z) - 1j * spherical_yn(n, z)
        expected = xp.asarray(expected)
        actual = shn2(xp.asarray(n), xp.asarray(d), xp.asarray(z))
    else:
        raise ValueError("type must be 'j' or 'y'")
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-6, atol=1e-6))
