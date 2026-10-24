"""Run every mode of the CDI geometry and check each returned solution.

For each mode: which forward.py solver handles it, how many solutions come
back, and whether each one really reaches the requested hkl (checked with the
solver's own inverse).

Things to notice:
- gamma = 0 and delta = +2theta in every result, even in modes that list
  gamma/delta as computed.
- fixed_incidence solutions do not reproduce the requested hkl (see 05).
"""

from __future__ import annotations

import numpy as np
from common import dispatched_solver, fmt, hkl_of, make_diffractometer

diff = make_diffractometer()

for mode in diff.core.modes:
    diff.core.mode = mode
    if mode == "fixed_incidence":
        diff.core.extras = {"n_hat": (1, 0, 0), "incidence": 3}
    for hkl in [(1, 1, 1), (2, 0, 0), (1, -1, 1)]:
        solutions = diff.core.forward(hkl)
        print(f"\n{mode} {hkl}: {len(solutions)} solution(s) via {dispatched_solver(diff)}")
        for s in solutions:
            print(f"   {fmt(s, diff.real_axis_names)}  -> hkl {np.round(hkl_of(diff, s), 4)}")
