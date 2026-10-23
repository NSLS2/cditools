from __future__ import annotations

import numpy as np

from cditools.diffractometer import CDIDiffractometer

theta = 34.55
tth = 2 * theta
REFLECTION1 = {
    "h": 4,
    "k": 0,
    "l": 0,
    "mu": 0,
    "chi": 0,
    "phi": 0,
    "omega2": theta,
    "chi2": 0,
    "gamma1": tth,
    "delta1": 0,
}
REFLECTION2 = {
    "h": 0,
    "k": 4,
    "l": 0,
    "mu": 0,
    "chi": 0,
    "phi": 90,
    "omega2": theta,
    "chi2": 0,
    "gamma1": tth,
    "delta1": 0,
}


class TestDiffractometer:
    def test_registered_geometry(self):
        diff = CDIDiffractometer()
        assert diff.geometry.name == "cdi-geometry"

    # If this test fails, the geometry is invalid
    def test_forward_and_inverse(self):
        geo = CDIDiffractometer().geometry
        hkl_init = (1, 1, 1)
        solutions = geo.forward(*hkl_init)
        for sol in solutions:
            hkl = geo.inverse(sol)
            np.testing.assert_approx_equal(hkl_init[0], hkl[0])
            np.testing.assert_approx_equal(hkl_init[1], hkl[1])
            np.testing.assert_approx_equal(hkl_init[2], hkl[2])

    def test_solver(self):
        pass
