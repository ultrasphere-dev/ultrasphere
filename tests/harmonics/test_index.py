import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull

from ultrasphere.coordinates import SphericalCoordinates, TEuclidean, TSpherical
from ultrasphere.creation import c_spherical, hopf, random
from ultrasphere.harmonics.flatten import index_array_harmonics_all


@pytest.mark.parametrize(
    "c",
    [
        random(1),
        random(2),
        c_spherical(),
        hopf(2),
    ],
)
@pytest.mark.parametrize("n_end", [4, 7])
def test_index_array_harmonics_all(
    c: SphericalCoordinates[TSpherical, TEuclidean], n_end: int, xp: ArrayNamespaceFull
) -> None:
    iall_concat = index_array_harmonics_all(
        c, n_end=n_end, include_negative_m=False, expand_dims=True, as_array=True, xp=xp
    )
    iall = index_array_harmonics_all(
        c,
        n_end=n_end,
        include_negative_m=False,
        expand_dims=True,
        as_array=False,
        xp=xp,
    )
    assert iall_concat.shape == (
        c.s_ndim,
        *xpx.broadcast_shapes(*[v.shape for v in iall.values()]),
    )
    for i, s_node in enumerate(c.s_nodes):
        # the shapes not necessarily match, so all_equal cannot be used
        assert xp.all(iall_concat[i] == iall[s_node])
