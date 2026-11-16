from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import ad_hoc_diffractometer as ahd
import hklpy2
import numpy as np
import pytest
from ophyd import SoftPositioner

from cditools.diffractometer import CDIDiffractometer

omega = 34.55
gamma = 2 * omega
REFLECTION1 = {
    "h": 4,
    "k": 0,
    "l": 0,
    "mu": 0,
    "omega1": 0,
    "chi1": 0,
    "omega2": omega,
    "chi2": 0,
    "gamma": gamma,
    "delta": 0,
}
REFLECTION2 = {
    "h": 0,
    "k": 4,
    "l": 0,
    "mu": 0,
    "omega1": 0,
    "chi1": 90,
    "omega2": omega,
    "chi2": 0,
    "gamma": gamma,
    "delta": 0,
}


@pytest.fixture
def all_angles():
    real_gon_angles = {
        "mu": None,
        "omega1": None,
        "chi1": None,
        "omega2": None,
        "chi2": None,
    }

    real_tdms_angles = {
        "gamma": {
            "class": SoftPositioner,
            "init_pos": 0,
            "limits": (-180, 180),
        },
        "delta": {
            "class": SoftPositioner,
            "init_pos": 0,
            "limits": (-180, 180),
        },
    }

    return copy.deepcopy(real_gon_angles) | copy.deepcopy(real_tdms_angles)


@pytest.fixture(scope="session")
def register_geometry():
    ahd.register_geometry_file(
        str(Path(__file__).parent.parent / "notebooks/diffractometer-geometry.yml"),
        name="cdi-geometry",
    )


class TestDiffractometer:
    @pytest.mark.usefixtures("register_geometry")
    def test_registered_geometry(self, all_angles: dict[str, Any]):
        # diff = CDIDiffractometer(name="test-diff", pseudos=["h", "k", "l"],
        #  reals=["mu", "omega1", "chi1", "omega2", "chi2", "gamma", "delta"])
        diffr = hklpy2.creator(
            prefix="",
            name="diffr",
            solver="ad_hoc",
            geometry="cdi-geometry",
            reals=all_angles,
            class_bases=[CDIDiffractometer],
        )
        assert diffr.core.geometry == "cdi-geometry"

    # If this test fails, the geometry is invalid
    @pytest.mark.usefixtures("register_geometry")
    def test_forward_and_inverse(self, all_angles: dict[str, Any]):
        diffr = hklpy2.creator(
            prefix="",
            name="diffr",
            solver="ad_hoc",
            geometry="cdi-geometry",
            reals=all_angles,
            class_bases=[CDIDiffractometer],
        )
        hkl_init = (1, 1, 1)
        solutions = diffr.core.forward(hkl_init)
        for sol in solutions:
            hkl = diffr.core.inverse(sol)
            np.testing.assert_allclose(
                [hkl["h"], hkl["k"], hkl["l"]], hkl_init, atol=1e-5
            )

    def test_solver(self):
        pass
