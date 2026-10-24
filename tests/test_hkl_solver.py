from __future__ import annotations

from typing import Any

import ad_hoc_diffractometer as ahd
import hklpy2
import numpy as np
import pytest
from hklpy2.exceptions import SolverError

from cditools.hkl_solver import CDIAdHocSolver

GEOMETRY = "cdi-solver-test"

# The CDI stack from diffractometer-geometry.yml, kept here so these tests
# don't depend on the in-progress geometry file.
GEOMETRY_YAML = """
ad_hoc_diffractometer_geometry:
  schema_revision: 1
name: cdi-solver-test
basis:
  vertical: [0.0, 1.0, 0.0]
  longitudinal: [0.0, 0.0, 1.0]
  transverse: [1.0, 0.0, 0.0]
stages:
  - { name: mu, axis: +vertical, parent: null, role: sample }
  - { name: omega1, axis: +transverse, parent: mu, role: sample }
  - { name: chi1, axis: +longitudinal, parent: omega1, role: sample }
  - { name: omega2, axis: +transverse, parent: chi1, role: sample }
  - { name: chi2, axis: +longitudinal, parent: omega2, role: sample }
  - { name: gamma, axis: +transverse, parent: null, role: detector }
  - { name: delta, axis: +vertical, parent: gamma, role: detector }
modes:
  fixed_incidence:
    default: true
    constraints:
      - { type: sample, stage: chi2, value: 0.0 }
      - { type: sample, stage: omega2, value: 0.0 }
      - { type: sample, stage: omega1, value: 0.0 }
      - { type: reference, name: incidence, value: 3 }
    computed: [mu, gamma, delta, chi1]
    extras:
      n_hat: REQUIRED
      incidence: REQUIRED
  fixed_chi1_chi2_omega2:
    constraints:
      - { type: sample, stage: chi2, value: 0.0 }
      - { type: sample, stage: omega2, value: 0.0 }
      - { type: sample, stage: chi1, value: 0.0 }
    computed: [mu, omega1, delta, gamma]
  fixed_gamma_chi2_omega2:
    constraints:
      - { type: sample, stage: chi2, value: 0.0 }
      - { type: sample, stage: omega2, value: 0.0 }
      - { type: detector, stage: gamma, value: 0.0 }
      - { type: sample, stage: omega1, value: 0.0 }
    computed: [mu, chi1, delta]
  fixed_incidence_gamma:
    constraints:
      - { type: sample, stage: chi2, value: 0.0 }
      - { type: sample, stage: omega2, value: 0.0 }
      - { type: detector, stage: gamma, value: 0.47 }
      - { type: reference, name: incidence, value: 3 }
    computed: [mu, omega1, chi1, delta]
    extras:
      n_hat: REQUIRED
      incidence: REQUIRED
  too_many_constraints:
    constraints:
      - { type: sample, stage: chi2, value: 0.0 }
      - { type: sample, stage: omega2, value: 0.0 }
      - { type: sample, stage: omega1, value: 0.0 }
      - { type: sample, stage: gamma, value: 0.47 }
      - { type: reference, name: incidence, value: 3 }
    computed: [mu, delta, chi1]
    extras:
      n_hat: REQUIRED
      incidence: REQUIRED
"""

INCIDENCE = 3.0
FIXED = {
    "fixed_incidence": {"omega1": 0.0, "omega2": 0.0, "chi2": 0.0},
    "fixed_incidence_gamma": {"omega2": 0.0, "chi2": 0.0, "gamma": 0.47},
    "fixed_chi1_chi2_omega2": {"chi1": 0.0, "omega2": 0.0, "chi2": 0.0},
    "fixed_gamma_chi2_omega2": {
        "omega1": 0.0,
        "omega2": 0.0,
        "chi2": 0.0,
        "gamma": 0.0,
    },
}
INCIDENCE_MODES = {"fixed_incidence", "fixed_incidence_gamma", "too_many_constraints"}


@pytest.fixture(scope="module", autouse=True)
def register_geometry(tmp_path_factory: pytest.TempPathFactory) -> None:
    if GEOMETRY not in ahd.list_geometries():
        path = tmp_path_factory.mktemp("geometry") / f"{GEOMETRY}.yml"
        path.write_text(GEOMETRY_YAML)
        ahd.register_geometry_file(str(path), name=GEOMETRY)


def make_diffractometer(
    solver: str = "cdi_ad_hoc", mode: str = "fixed_gamma_chi2_omega2"
):
    """Gold sample oriented from the (111) and (200) reflections measured at CDI."""
    diff = hklpy2.creator(name="cdi", geometry=GEOMETRY, solver=solver)
    diff.beam.energy.put(10.002)  # keV
    diff.add_sample("gold", a=4.08)
    r1 = diff.add_reflection(
        (1, 1, 1),
        {
            "mu": 18.2,
            "omega1": 0,
            "chi1": 1.4,
            "omega2": 0,
            "chi2": 0,
            "gamma": 0.47,
            "delta": 31.84,
        },
        name="r111",
    )
    r2 = diff.add_reflection(
        (2, 0, 0),
        {
            "mu": 3,
            "omega1": 0,
            "chi1": 0,
            "omega2": 0,
            "chi2": 0,
            "gamma": 2.39,
            "delta": 38.5,
        },
        name="r200",
    )
    diff.core.calc_UB(r1, r2)
    diff.core.mode = mode
    if mode in INCIDENCE_MODES:
        diff.core.extras = {"n_hat": (1, 0, 0), "incidence": INCIDENCE}  # pyright: ignore[reportAttributeAccessIssue]
    return diff


def assert_valid(
    diff: Any, solution: Any, hkl: tuple[int, int, int], mode: str
) -> None:
    np.testing.assert_allclose(tuple(diff.inverse(solution)), hkl, rtol=0, atol=1e-6)
    for axis, value in FIXED[mode].items():
        assert getattr(solution, axis) == pytest.approx(value, abs=1e-9)
    if mode in INCIDENCE_MODES:
        incidence = diff.core.solver.incidence_angle(solution._asdict())
        assert incidence == pytest.approx(INCIDENCE, abs=1e-6)


def test_solver_is_registered() -> None:
    assert "cdi_ad_hoc" in hklpy2.solvers()
    assert isinstance(make_diffractometer().core.solver, CDIAdHocSolver)


# Counts come from an independent brute-force search
# (debug/ad_hoc_solver/09_cdi_solver_vs_brute_force.py).
@pytest.mark.parametrize(
    ("mode", "hkl", "count"),
    [
        ("fixed_gamma_chi2_omega2", (1, 1, 1), 4),
        ("fixed_gamma_chi2_omega2", (2, 0, 0), 4),
        ("fixed_gamma_chi2_omega2", (1, -1, 1), 4),
        ("fixed_gamma_chi2_omega2", (0, 2, 2), 4),
        ("fixed_chi1_chi2_omega2", (1, 1, 1), 4),
        ("fixed_chi1_chi2_omega2", (0, 2, 2), 4),
        ("fixed_incidence", (1, 1, 1), 8),
        ("fixed_incidence", (1, -1, 1), 8),
        ("fixed_incidence", (0, 2, 2), 8),
        # Q is parallel to the surface normal, so incidence must equal theta (17.7 deg).
        ("fixed_incidence", (2, 0, 0), 0),
        ("fixed_incidence_gamma", (1, 1, 1), 8),
        ("fixed_incidence_gamma", (2, 0, 0), 0),
    ],
)
def test_returns_every_valid_solution(
    mode: str, hkl: tuple[int, int, int], count: int
) -> None:
    diff = make_diffractometer(mode=mode)
    solutions = diff.core.forward(hkl)
    assert len(solutions) == count
    for solution in solutions:
        assert_valid(diff, solution, hkl, mode)


def test_fixed_gamma_solutions_match_brute_force() -> None:
    diff = make_diffractometer(mode="fixed_gamma_chi2_omega2")
    found = sorted((s.mu, s.chi1, s.delta) for s in diff.core.forward((1, 1, 1)))
    expected = sorted(
        [
            (17.532, 2.128, 30.510),
            (167.022, 2.128, -30.510),
            (-17.532, -177.872, -30.510),
            (-167.022, -177.872, 30.510),
        ]
    )
    np.testing.assert_allclose(found, expected, atol=1e-3)


@pytest.mark.parametrize("mode", list(FIXED))
@pytest.mark.parametrize("hkl", [(1, 1, 1), (2, 0, 0), (1, -1, 1)])
def test_includes_every_valid_library_solution(
    mode: str, hkl: tuple[int, int, int]
) -> None:
    library = make_diffractometer(solver="ad_hoc", mode=mode)
    ours = make_diffractometer(mode=mode).core.forward(hkl)
    try:
        library_solutions = library.core.forward(hkl)
    except SolverError:
        library_solutions = []  # e.g. a ConstraintViolation in fixed_incidence_gamma
    for solution in library_solutions:
        if not np.allclose(tuple(library.inverse(solution)), hkl, atol=1e-6):
            continue  # the library returns invalid solutions in fixed_incidence
        assert any(np.allclose(tuple(solution), tuple(s), atol=1e-4) for s in ours)


@pytest.mark.parametrize("gamma", [0.47, 2.39])
def test_underdetermined_mode_holds_gamma_at_current_position(gamma: float) -> None:
    diff = make_diffractometer(mode="fixed_chi1_chi2_omega2")
    diff.gamma.move(gamma)
    solutions = diff.core.forward((2, 0, 0))
    assert len(solutions) == 4
    for solution in solutions:
        assert solution.gamma == pytest.approx(gamma)
        assert_valid(diff, solution, (2, 0, 0), "fixed_chi1_chi2_omega2")


def test_solutions_are_ordered_least_rotated_first() -> None:
    diff = make_diffractometer(mode="fixed_incidence")
    totals = [sum(abs(v) for v in s) for s in diff.core.forward((1, 1, 1))]
    assert totals == sorted(totals)


def test_unhandled_mode_uses_library_solver() -> None:
    results = {}
    for solver in ("ad_hoc", "cdi_ad_hoc"):
        diff = hklpy2.creator(name="probe", geometry="fivec", solver=solver)
        diff.add_sample("cubic", a=1.0)
        diff.beam.wavelength.put(1.0)
        diff.core.mode = "bissector"
        results[solver] = [tuple(s) for s in diff.core.forward((1, 1, 0))]
    assert results["cdi_ad_hoc"] == results["ad_hoc"]


def test_unreachable_reflection_raises() -> None:
    diff = make_diffractometer()
    with pytest.raises(SolverError):
        diff.core.solver.forward({"h": 20, "k": 0, "l": 0})


def test_mode_with_too_many_constraints_raises() -> None:
    diff = make_diffractometer(mode="too_many_constraints")
    with pytest.raises(SolverError, match="fixes too many axes"):
        diff.core.forward((1, 1, 1))
