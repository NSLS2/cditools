"""Why fixed_gamma_chi2_omega2 loses its second +2theta solution.

Replays the seed loop in forward.py::_solve_fixed_sample (lines 4389-4433)
for (1, 1, 1).  The Newton solver (_solve_two_angles) returns angles without
wrapping them.  Seeds that converge to the second branch land on
mu = 192.978, chi1 = 182.128 (the same as -167.022, -177.872).  _check_limits
then rejects that as outside +/-180, because the geometry has no cut points
to wrap it first (kept=False below).
"""

from __future__ import annotations

import math

import numpy as np
from ad_hoc_diffractometer import forward as F
from ad_hoc_diffractometer.orientation import angles_to_phi_vector
from common import geom, make_diffractometer

diff = make_diffractometer()
diff.core.mode = "fixed_gamma_chi2_omega2"
diff.core.forward((1, 1, 1))  # pushes mode, UB and wavelength into the solver
g = geom(diff)
mode = g.mode
print("geometry cut_points:", g.cut_points, "  mode cut_points:", mode.cut_points)
print("stage limits:", {s.name: s.limits for s in g.sample_stages + g.detector_stages})

Q_phi = g.sample.UB @ np.array([1.0, 1.0, 1.0])
ttheta = math.degrees(2 * math.asin(np.linalg.norm(Q_phi) * g.wavelength / (4 * math.pi)))

# Same baseline _solve_fixed_sample builds: constraints applied, delta = +2theta.
angles = {s.name: s.angle for s in g._stages.values()}
angles.update(omega1=0.0, omega2=0.0, chi2=0.0, gamma=0.0, delta=ttheta)
ctx = F.ForwardContext(g)
ctx.prepare_caching(angles, {"mu", "chi1"})

# Same seed grid as _solve_fixed_sample (mu is its "chi_stage", chi1 its "phi_stage").
for mu_seed in (0.0, 45.0, 90.0, -90.0, 135.0, 180.0):
    for chi1_seed in (0.0, 90.0, 180.0, 270.0):
        result = F._solve_two_angles(
            g, angles, "mu", "chi1", mu_seed, chi1_seed, Q_phi, angles_to_phi_vector, _ctx=ctx
        )
        if result is None:
            print(f"seed mu={mu_seed:6.1f} chi1={chi1_seed:6.1f} -> no convergence")
            continue
        sol = dict(angles, mu=result[0], chi1=result[1])
        F._apply_cut_points(sol, mode, g)
        print(
            f"seed mu={mu_seed:6.1f} chi1={chi1_seed:6.1f} -> "
            f"mu={sol['mu']:9.3f} chi1={sol['chi1']:9.3f}  kept={F._check_limits(g, sol)}"
        )
