"""fixed_gamma_chi2_omega2: the solver finds 1 of 4 solutions.

With omega1 = omega2 = chi2 = gamma = 0, the free axes are mu, chi1 and delta.
Brute force finds 4 solutions: two with delta = +2theta and two with
delta = -2theta.  The solver returns one, because:
- it never tries delta = -2theta (_solve_fixed_sample pins the detector to
  +2theta, line 4371), and
- it throws away the second +2theta branch (see 04).
"""

from __future__ import annotations

import numpy as np
from common import brute_force, fmt, hkl_of, make_diffractometer

diff = make_diffractometer()
diff.core.mode = "fixed_gamma_chi2_omega2"
free = ["mu", "chi1", "delta"]

for hkl in [(1, 1, 1), (2, 0, 0)]:
    solutions = diff.core.forward(hkl)
    truth = brute_force(
        diff, hkl, fixed={"omega1": 0, "omega2": 0, "chi2": 0, "gamma": 0}, free=free
    )
    print(f"\n{hkl}: solver={len(solutions)}  brute force={len(truth)}")
    print("  brute force:")
    for t in truth:
        print("    ", fmt(t, free))
    print("  solver:")
    for s in solutions:
        print("    ", fmt(s, free), " -> hkl", np.round(hkl_of(diff, s), 4))
