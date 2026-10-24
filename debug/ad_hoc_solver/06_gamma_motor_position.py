"""fixed_chi1_chi2_omega2 depends on where the gamma motor currently is.

_solve_fixed_sample sets delta = +2theta and leaves gamma wherever the motor
is (forward.py lines 4369-4371).  At gamma = 0 that's correct.  At the gamma
values from the orienting reflections (0.47, 2.39) the detector angle is
wrong, Newton can't converge, and no solutions come back.

Also note this mode has only 3 constraints, and the YAML comment says 7 axes
need 4.
"""

from __future__ import annotations

import numpy as np
from common import fmt, hkl_of, make_diffractometer

diff = make_diffractometer()
diff.core.mode = "fixed_chi1_chi2_omega2"

for gamma in (0.0, 0.47, 2.39):
    diff.gamma.move(gamma)
    solutions = diff.core.forward((1, 1, 1))
    print(f"\ngamma motor at {gamma}: {len(solutions)} solution(s)")
    for s in solutions:
        print(f"   {fmt(s, diff.real_axis_names)}  -> hkl {np.round(hkl_of(diff, s), 4)}")
