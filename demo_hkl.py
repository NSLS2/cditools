import math
import numpy as np
import ad_hoc_diffractometer as ahd
import hklpy2
import hklpy2.user as user

from ad_hoc_diffractometer.mode import (
    ConstraintSet,
    SampleConstraint,
    DetectorConstraint,
)

if "cdi-geometry" not in ahd.list_geometries():
    ahd.register_geometry_file(
        "./diffractometer-geometry.yml", name="cdi-geometry"
    )

# Fresh instance; omitting motor PVs gives simulated motors.
diff = hklpy2.creator(
    name="si10_test",
    geometry="cdi-geometry",
    solver="ad_hoc",
)

# Private adapter access, used to install a custom test mode.
geometry = diff.core.solver._geom
geometry.modes["roundtrip_test"] = ConstraintSet(
    constraints=[
        SampleConstraint("omega1", 0),
        SampleConstraint("omega2", 0),
        SampleConstraint("chi2", 0),
        DetectorConstraint("gamma", 0),
    ],
    computed=["mu", "chi1", "delta"],
)
diff.core.mode = "roundtrip_test"

diff.beam.energy.put(10.0)  # keV
wavelength = diff.beam.wavelength.get()
a = hklpy2.SI_LATTICE_PARAMETER

user.set_diffractometer(diff)
user.add_sample("silicon", a=a)

theta = math.degrees(math.asin(2 * wavelength / a))
tth = 2 * theta

p400 = dict(
    mu=theta, omega1=0, chi1=0,
    omega2=0, chi2=0, gamma=0, delta=tth,
)
p040 = dict(p400, chi1=-90)

r1 = user.setor(4, 0, 0, name="r400", **p400)
r2 = user.setor(0, 4, 0, name="r040", **p040)
user.calc_UB(r1, r2)

# Check that the supplied reference positions invert correctly.
for target, positions in [((4, 0, 0), p400), ((0, 4, 0), p040)]:
    back = diff.inverse(diff.RealPosition(**positions))
    np.testing.assert_allclose(tuple(back), target, rtol=0, atol=1e-6)

# Check forward/inverse, including a reflection not used to determine UB.
for target in [(4, 0, 0), (0, 4, 0), (4, 4, 0)]:
    positions = diff.forward(target)
    back = diff.inverse(positions)

    print(f"{target} -> {positions} -> {back}")
    np.testing.assert_allclose(tuple(back), target, rtol=0, atol=1e-6)