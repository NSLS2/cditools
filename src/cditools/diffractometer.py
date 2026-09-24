from __future__ import annotations

import contextlib
import math

import ad_hoc_diffractometer as ahd
import bluesky.plans as bp
import hklpy2
import hklpy2.blocks.reflection
import hklpy2.user
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from hklpy2.user import cahkl_table
from hklpy2.utils import pick_closest_solution

"""
1. Export diffractometer ophyd object to profile collection
2. Create two ophyd objects, one for each arm
3. Connect ophyd objects to PVs for arm
4. Verify simulated movement
5. Verify real movement

Extras
- figure out how to adjust tolerance of incidence
- set energy/wavelength with ophyd object
"""

# Register geometry
try:
    ahd.register_geometry_file("./diffractometer-geometry.yml", name="cdi-geometry")
except ValueError:  # geometry already registered
    contextlib.suppress(ValueError)

# Create diffractometer
diffr = hklpy2.creator(name="cdi-geometry", geometry="cdi-geometry", solver="ad_hoc")

# Add Sample
hklpy2.user.set_diffractometer(diffr)
hklpy2.user.add_sample("silicon", a=hklpy2.SI_LATTICE_PARAMETER)

# Add beam
# TODO - add conversion between our energy and beam energy
diffr.beam.wavelength.put(1.54)  # Angstroms

# Add orientation reflections
theta = math.degrees(math.asin(1.54 / (2 * 5.431 / 4)))  # ≈ 34.55° for (400)
tth = 2 * theta

# Have to specify all seven angles for an orientation reflection
try:
    r1 = hklpy2.user.setor(
        4,
        0,
        0,
        mu=0,
        chi=0,
        phi=0,
        omega2=theta,
        chi2=0,
        gamma1=tth,
        delta1=0,
    )
    r2 = hklpy2.user.setor(
        0,
        4,
        0,
        mu=0,
        chi=0,
        phi=90,
        omega2=theta,
        chi2=0,
        gamma1=tth,
        delta1=0,
    )
except hklpy2.blocks.reflection.ReflectionError:
    pass

# To check on status of things, run pa() and wh()
# pa()
# wh()

# Calculate UB matrix
hklpy2.user.calc_UB(r1, r2)

# Add constraints
diffr.core.constraints["chi"].limits = (0, 180)
diffr.core.constraints["omega2"].limits = (180, 0)

# Set surface normal
diffr.core.extras = {"n_hat": (1, 1, 1)}

# Get solutions
hkl_or = (4, 0, 0)
# diffr.core.forward gives a list of solutions
# diffr.forward gives the first solution
solutions = diffr.core.forward(hkl_or)
# print table of solutions:
cahkl_table(hkl_or)

# optionally change solution picker before moving:

diffr._forward_solution = pick_closest_solution
diffr.move((4, 0, 0))

# see full state of diffractometer:
diffr.wh(full=True)


# Scan in reciprocal space
bec = BestEffortCallback()
bec.disable_plots()

RE = RunEngine({})
RE.subscribe(bec)
RE(bp.scan([diffr], diffr.h, 3.9, 4.1, 5))
