from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import jv, jvp, spherical_jn, spherical_yn, yv, yvp

from ultrasphere.special._bessel import sjv, syv


@pytest.mark.parametrize("derivative", [True, False])
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("type", ["j", "y"])
def test_sjyn(d: Literal[2, 3], type: Literal["j", "y"], derivative: bool) -> None:
    n = np.random.randint(5, size=(10, 10))
    z = np.random.random((10, 10))
    if type == "j":
        actual = sjv(n, np.array(d), z, derivative=derivative)
        if d == 2:
            expected = np.sqrt(np.pi / 2) * (jvp(n, z) if derivative else jv(n, z))
        elif d == 3:
            expected = spherical_jn(n, z, derivative=derivative)
        else:
            raise ValueError("d must be 2 or 3")
    elif type == "y":
        actual = syv(n, np.array(d), z, derivative=derivative)
        if d == 2:
            expected = np.sqrt(np.pi / 2) * (yvp(n, z) if derivative else yv(n, z))
        elif d == 3:
            expected = spherical_yn(n, z, derivative=derivative)
        else:
            raise ValueError("d must be 2 or 3")
    else:
        raise ValueError("type must be 'j' or 'y'")
    assert_allclose(actual, expected)
