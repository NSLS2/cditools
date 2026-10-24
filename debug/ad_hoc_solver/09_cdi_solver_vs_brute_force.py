"""Check cditools' CDIAdHocSolver against brute force in every CDI mode.

For each case: the solutions from ``solver="ad_hoc"`` (library) and from
``solver="cdi_ad_hoc"`` (cditools.hkl_solver), both checked with the library's
inverse, against an independent brute-force search (common.brute_force).  A
case passes when the new solver's solutions match brute force exactly.

fixed_chi1_chi2_omega2 has one more free axis than it has constraints, so
both solvers hold gamma at its current motor position.  It is run here with
gamma at 0, 0.47 and 2.39, and brute force uses the same gamma.

fixed_incidence_gamma (added at runtime below) is the current
diffractometer-geometry.yml fixed_incidence (gamma fixed at 0.47) with omega1
freed, since that YAML mode fixes one axis too many to solve.

The incidence cases each take a minute or two of brute force.
"""

from __future__ import annotations

import time

import numpy as np
from ad_hoc_diffractometer.mode import (
    REQUIRED,
    ConstraintSet,
    DetectorConstraint,
    ReferenceConstraint,
    SampleConstraint,
)
from ad_hoc_diffractometer.reference import incidence_angle
from common import brute_force, geom, hkl_of, make_diffractometer

INCIDENCE = 3.0
HKLS = [(1, 1, 1), (2, 0, 0), (1, -1, 1), (0, 2, 2)]

CASES = [
    # mode, extras, current gamma motor, fixed axes, free axes
    ("fixed_gamma_chi2_omega2", None, 0.0,
     {"omega1": 0, "omega2": 0, "chi2": 0, "gamma": 0}, ["mu", "chi1", "delta"]),
    ("fixed_chi1_chi2_omega2", None, 0.0,
     {"chi1": 0, "omega2": 0, "chi2": 0, "gamma": 0.0}, ["mu", "omega1", "delta"]),
    ("fixed_chi1_chi2_omega2", None, 0.47,
     {"chi1": 0, "omega2": 0, "chi2": 0, "gamma": 0.47}, ["mu", "omega1", "delta"]),
    ("fixed_chi1_chi2_omega2", None, 2.39,
     {"chi1": 0, "omega2": 0, "chi2": 0, "gamma": 2.39}, ["mu", "omega1", "delta"]),
    ("fixed_incidence", {"n_hat": (1, 0, 0), "incidence": INCIDENCE}, 0.0,
     {"omega1": 0, "omega2": 0, "chi2": 0}, ["mu", "chi1", "gamma", "delta"]),
    ("fixed_incidence_gamma", {"n_hat": (1, 0, 0), "incidence": INCIDENCE}, 0.0,
     {"omega2": 0, "chi2": 0, "gamma": 0.47}, ["mu", "omega1", "chi1", "delta"]),
]

# Same as diffractometer-geometry.yml's fixed_incidence (gamma fixed at 0.47),
# but with omega1 free so Bragg + incidence has the 4 free axes it needs.
FIXED_INCIDENCE_GAMMA = ConstraintSet(
    constraints=[
        SampleConstraint("chi2", 0.0),
        SampleConstraint("omega2", 0.0),
        DetectorConstraint("gamma", 0.47),
        ReferenceConstraint("incidence", INCIDENCE),
    ],
    computed=["mu", "omega1", "chi1", "delta"],
    extras={"n_hat": REQUIRED, "incidence": REQUIRED},
)

library = make_diffractometer(solver="ad_hoc")
new = make_diffractometer(solver="cdi_ad_hoc")
for diff in (library, new):
    geom(diff).modes["fixed_incidence_gamma"] = FIXED_INCIDENCE_GAMMA
g = geom(new)


def valid(diff, solutions, hkl):
    return [s for s in solutions if np.allclose(hkl_of(diff, s), hkl, atol=1e-6)]


def same_set(found, truth, free):
    """Same solutions on the free axes, to 1e-4 degrees."""
    if len(found) != len(truth):
        return False
    remaining = list(truth)
    for sol in found:
        match = next(
            (t for t in remaining if all(abs((sol[a] - t[a] + 180) % 360 - 180) < 1e-4 for a in free)),
            None,
        )
        if match is None:
            return False
        remaining.remove(match)
    return True


all_ok = True
for mode, extras, gamma, fixed, free in CASES:
    for diff in (library, new):
        diff.core.mode = mode
        if extras:
            diff.core.extras = extras
        diff.gamma.move(gamma)
    extra_residual = None
    if extras:
        extra_residual = lambda angles: incidence_angle(g, angles=angles) - INCIDENCE  # noqa: E731

    for hkl in HKLS:
        try:
            old_solutions = library.core.forward(hkl)
        except Exception:  # e.g. the library's ConstraintViolation in fixed_incidence_gamma
            old_solutions = []
        t = time.perf_counter()
        new_solutions = new.core.forward(hkl)
        ms = (time.perf_counter() - t) * 1000
        truth = brute_force(new, hkl, fixed=fixed, free=free, extra_residual=extra_residual)
        new_dicts = [s._asdict() for s in new_solutions]
        ok = len(valid(new, new_solutions, hkl)) == len(new_solutions) and same_set(new_dicts, truth, free)
        all_ok &= ok
        print(
            f"{'OK  ' if ok else 'FAIL'} {mode:24s} gamma={gamma:<5} {hkl!s:11s}"
            f" brute force={len(truth)}  cdi_ad_hoc={len(new_solutions)} ({ms:.0f} ms)"
            f"  ad_hoc={len(old_solutions)} ({len(valid(library, old_solutions, hkl))} valid)"
        )

print("\nALL MATCH" if all_ok else "\nMISMATCHES FOUND")
