"""Packaged fivec geometry in bisecting mode: 2 solutions at every layer.

This is by design in ad_hoc_diffractometer 1.0.0.  compute_forward() sets
ttheta = 2*asin(...), which is always positive (line 439), and
_solve_bisecting() solves only the chi/phi pair for that one detector angle
(docstring, lines 1005-1007).  The mirror solutions with ttheta = -90,
omega = -45 are never generated.

The count is the same from hklpy2 Core, the hklpy2 solver adapter, and the
raw AdHocDiffractometer, so hklpy2 is not dropping anything.
"""

from __future__ import annotations

import hklpy2

probe = hklpy2.creator(name="probe", geometry="fivec", solver="ad_hoc")
probe.add_sample("cubic", a=1.0)
probe.beam.wavelength.put(1.0)
probe.core.mode = "bissector"

core = probe.core.forward((1, 1, 0))
adapter = probe.core.solver.forward({"h": 1, "k": 1, "l": 0})
raw = probe.core.solver._geom.forward(1, 1, 0)

print(f"hklpy2 core.forward:         {len(core)}")
print(f"AdHocSolver.forward:         {len(adapter)}")
print(f"AdHocDiffractometer.forward: {len(raw)}")
for solution in core:
    print("   ", solution)
