from __future__ import annotations

import copy
from pathlib import Path

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
def _register_geometry():
    ahd.register_geometry_file(
        str(Path(__file__).parent.parent / "notebooks/diffractometer-geometry.yml"),
        name="cdi-geometry",
    )


class TestDiffractometer:
    @pytest.fixture
    def diffr(self, all_angles, _register_geometry):
        return hklpy2.creator(
            prefix="",
            name="diffr",
            solver="ad_hoc",
            geometry="cdi-geometry",
            reals=all_angles,
            class_bases=[CDIDiffractometer],
        )

    def test_registered_geometry(self, diffr):
        assert diffr.core.geometry == "cdi-geometry"

    # If this test fails, the geometry is invalid
    def test_forward_and_inverse(self, diffr):
        hkl_init = (1, 1, 1)
        solutions = diffr.core.forward(hkl_init)
        assert solutions
        for sol in solutions:
            hkl = diffr.core.inverse(sol)
            np.testing.assert_allclose(
                [hkl["h"], hkl["k"], hkl["l"]], hkl_init, atol=1e-5
            )

    def test_solver(self):
        pass

    def test_get_families_of_reflections(self, diffr):
        actual_family_100 = sorted(
            [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
        )
        calc_family_100 = diffr._get_families_of_reflections([(1, 0, 0)])

        actual_family_110 = sorted(
            [
                (1, 1, 0),
                (1, -1, 0),
                (-1, 1, 0),
                (-1, -1, 0),
                (1, 0, 1),
                (1, 0, -1),
                (-1, 0, 1),
                (-1, 0, -1),
                (0, 1, 1),
                (0, 1, -1),
                (0, -1, 1),
                (0, -1, -1),
            ]
        )
        calc_family_110 = diffr._get_families_of_reflections([(1, 1, 0)])

        assert actual_family_110 == calc_family_110
        assert actual_family_100 == calc_family_100

    def test_calc_UB_matrix(self, diffr):
        diffr.add_reflection(
            pseudos=(1, 0, 0),
            reals={
                "mu": 30,
                "omega1": 0,
                "chi1": 0,
                "omega2": 0,
                "chi2": 0,
                "gamma": 0,
                "delta": 60,
            },
        )
        diffr.add_reflection(
            pseudos=(0, 0, 1),
            reals={
                "mu": 120,
                "omega1": 0,
                "chi1": 0,
                "omega2": 0,
                "chi2": 0,
                "gamma": 0,
                "delta": 60,
            },
        )

        ub = diffr.calc_UB()
        np.testing.assert_allclose(ub, 2 * np.pi * np.eye(3), atol=1e-10)
