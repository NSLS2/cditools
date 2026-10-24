"""Workaround: mode-level cut_points recover the dropped +2theta branch.

geometries/cdi-cut.yml adds `cut_points: { mu: -180, chi1: -180 }` to
fixed_gamma_chi2_omega2.  _apply_cut_points then wraps the Newton results
before _check_limits sees them (see 04).  Result: 2 correct solutions instead
of 1.  The delta = -2theta pair from 03 is still missing; getting it needs a
change upstream.
"""

from __future__ import annotations

import numpy as np
from common import fmt, hkl_of, make_diffractometer

diff = make_diffractometer("cdi-cut.yml")
diff.core.mode = "fixed_gamma_chi2_omega2"

for hkl in [(1, 1, 1), (2, 0, 0)]:
    solutions = diff.core.forward(hkl)
    print(f"\n{hkl}: {len(solutions)} solution(s)")
    for s in solutions:
        print(f"   {fmt(s, ['mu', 'chi1', 'delta'])}  -> hkl {np.round(hkl_of(diff, s), 4)}")
