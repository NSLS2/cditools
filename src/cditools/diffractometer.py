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
from ophyd import Component as Cpt
from ophyd import EpicsMotor
from ophyd import FormattedComponent as FCpt

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

# Define Epics PVs for angles
# mu = Cpt(EpicsMotor, "Gon:1-Ax:Ry}Mtr")
# chi = Cpt(EpicsMotor, "Gon:1-Ax:Rx2}Mtr")
# phi = Cpt(EpicsMotor, "Gon:1-Ax:Rz2}Mtr")
# omega2 = Cpt(EpicsMotor, "Gon:1-Ax:Rx1}Mtr")
# chi2 = Cpt(EpicsMotor, "Gon:1-Ax:Rz1}Mtr")
# prefix = "XF:09IDC-"
# prefix_gon = "OP:1{Gon:1-Ax:"
# prefix_tdms = "ES:1{TDMS:"
mu = Cpt(EpicsMotor, "OP:1{Gon:1-Ax:Ry}Mtr")
chi = Cpt(EpicsMotor, "OP:1{Gon:1-Ax:Rx2}Mtr")
phi = Cpt(EpicsMotor, "OP:1{Gon:1-Ax:Rz2}Mtr")
omega2 = Cpt(EpicsMotor, "OP:1{Gon:1-Ax:Rx1}Mtr")
chi2 = Cpt(EpicsMotor, "OP:1{Gon:1-Ax:Rz1}Mtr")
gamma1 = FCpt(EpicsMotor, "ES:1{{TDMS:T{self._num}-Ax:AX}}MTR:RBV-RB0")
delta1 = FCpt(EpicsMotor, "ES:1{{TDMS:A{self._num}-Ax:AY}}MTR:RBV-RB0")
real_gon_angles = {
    "mu": "OP:1{Gon:1-Ax:Ry}Mtr",
    "chi": "OP:1{Gon:1-Ax:Rx2}Mtr",
    "phi": "OP:1{Gon:1-Ax:Rz2}Mtr",
    "omega2": "OP:1{Gon:1-Ax:Rx1}Mtr",
    "chi2": "OP:1{Gon:1-Ax:Rz1}Mtr",
}
real_tdms1_angles = {
    "gamma1": "ES:1{TDMS:T1-Ax:AX}MTR:RBV-RB0",
    "delta1": "ES:1{TDMS:A1-Ax:AY}MTR:RBV-RB0",
}
real_tdms2_angles = {
    "gamma1": "ES:1{TDMS:T2-Ax:AX}MTR:RBV-RB0",
    "delta1": "ES:1{TDMS:A2-Ax:AY}MTR:RBV-RB0",
}
_real_seq = ["mu", "chi", "phi", "omega2", "chi2", "gamma1", "delta1"]

#
# Create diffractometer
diffr1 = hklpy2.creator(
    # name="cdi-geometry", geometry="cdi-geometry", solver="ad_hoc", prefix="Gon:1-Ax:"
    name="cdi-geometry",
    geometry="cdi-geometry",
    solver="ad_hoc",
    prefix="XF:09IDC-",
    reals=real_gon_angles | real_tdms1_angles,
    _real=_real_seq,
)

diffr2 = hklpy2.creator(
    # name="cdi-geometry", geometry="cdi-geometry", solver="ad_hoc", prefix="Gon:1-Ax:"
    name="cdi-geometry",
    geometry="cdi-geometry",
    solver="ad_hoc",
    prefix="XF:09IDC-",
    reals=real_gon_angles | real_tdms2_angles,
    _real=_real_seq,
)

# Add Sample
hklpy2.user.set_diffractometer(diffr1)
hklpy2.user.add_sample("silicon", a=hklpy2.SI_LATTICE_PARAMETER)

# Add beam
# TODO - add conversion between our energy and beam energy
diffr1.beam.wavelength.put(1.54)  # Angstroms
diffr2.beam.wavelength.put(1.54)

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
# TODO - add constraints based on TDMS location
diffr1.core.constraints["chi"].limits = (0, 180)
diffr1.core.constraints["omega2"].limits = (180, 0)

# Set surface normal
diffr1.core.extras = {"n_hat": (1, 1, 1)}  # pyright: ignore[reportAttributeAccessIssue]

# Get solutions
hkl_or = (4, 0, 0)
# diffr1.core.forward gives a list of solutions
# diffr1.forward gives the first solution
solutions = diffr1.core.forward(hkl_or)
# print table of solutions:
cahkl_table(hkl_or)

# optionally change solution picker before moving:

diffr1._forward_solution = pick_closest_solution
diffr1.move((4, 0, 0))

# see full state of diffr1actometer:
diffr1.wh(full=True)


# Scan in reciprocal space
bec = BestEffortCallback()
bec.disable_plots()

RE = RunEngine({})
RE.subscribe(bec)
RE(bp.scan([diffr1], diffr1.h, 3.9, 4.1, 5))
