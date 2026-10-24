"""Workaround: rename the solver stages omega1 -> chi and chi1 -> phi.

Uses geometries/cdi-renamed.yml.  The diffractometer still exposes
omega1/chi1 because hklpy2 maps axis names by position (see axes_xref).

With the names the library expects, fixed_incidence is routed to
_solve_free_detectors, which solves gamma and delta too, and returns 4
correct solutions.  Brute force (05) finds 4 more; they reach the same
detector direction from the far side (gamma near +/-150, delta near +/-180).
fixed_gamma_chi2_omega2 is not helped (still 1 solution, see 08).
"""

from __future__ import annotations

import numpy as np
from ad_hoc_diffractometer.reference import incidence_angle
from common import dispatched_solver, fmt, geom, hkl_of, make_diffractometer, to_solver

diff = make_diffractometer("cdi-renamed.yml")
g = geom(diff)
print("diffractometer axes:", diff.real_axis_names)
print("solver axes:        ", diff.core.solver_real_axis_names)
print("axes_xref:          ", diff.core.axes_xref)

diff.core.mode = "fixed_incidence"
diff.core.extras = {"n_hat": (1, 0, 0), "incidence": 3}
solutions = diff.core.forward((1, 1, 1))
print(f"\nfixed_incidence (1, 1, 1): {len(solutions)} solution(s) via {dispatched_solver(diff)}")
for s in solutions:
    incidence = incidence_angle(g, angles=to_solver(diff, s))
    print(
        f"   {fmt(s, ['mu', 'chi1', 'gamma', 'delta'])}"
        f"  -> hkl {np.round(hkl_of(diff, s), 4)}  incidence={incidence:.3f}"
    )

diff.core.mode = "fixed_gamma_chi2_omega2"
solutions = diff.core.forward((1, 1, 1))
print(f"\nfixed_gamma_chi2_omega2 (1, 1, 1): {len(solutions)} solution(s) via {dispatched_solver(diff)}")
