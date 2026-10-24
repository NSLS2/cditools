"""fixed_incidence returns solutions that do not reach the requested hkl.

This mode is routed to forward.py::_solve_surface.  Its legacy branch (lines
3399-3449) adjusts only the first free sample stage (mu) until the incidence
target is met.  chi1 and gamma stay at their current values and delta is set
to +2theta, so the Bragg condition is never solved.  The upstream fix for this
(issue #307, line 3385) only applies to stages named "chi" and "phi".
_validate_solutions doesn't check hkl, so these solutions are returned anyway.

Brute force over (mu, chi1, gamma, delta) finds 8 real solutions, all with
gamma near +/-30.5 or +/-149.5.  The brute force takes a minute or two.
"""

from __future__ import annotations

import numpy as np
from ad_hoc_diffractometer.reference import incidence_angle
from common import (
    brute_force,
    dispatched_solver,
    fmt,
    geom,
    hkl_of,
    make_diffractometer,
    to_solver,
)

INCIDENCE = 3.0
hkl = (1, 1, 1)
free = ["mu", "chi1", "gamma", "delta"]

diff = make_diffractometer()
diff.core.mode = "fixed_incidence"
diff.core.extras = {"n_hat": (1, 0, 0), "incidence": INCIDENCE}
g = geom(diff)

solutions = diff.core.forward(hkl)
print(f"fixed_incidence {hkl} dispatched to: {dispatched_solver(diff)}")
print(f"solver: {len(solutions)} solution(s)")
for s in solutions:
    incidence = incidence_angle(g, angles=to_solver(diff, s))
    print(f"   {fmt(s, free)}  -> hkl {np.round(hkl_of(diff, s), 4)}  incidence={incidence:.3f}")

truth = brute_force(
    diff,
    hkl,
    fixed={"omega1": 0, "omega2": 0, "chi2": 0},
    free=free,
    extra_residual=lambda angles: incidence_angle(g, angles=angles) - INCIDENCE,
)
print(f"brute force: {len(truth)} solution(s)")
for t in truth:
    print("   ", fmt(t, free))
