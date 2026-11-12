from __future__ import annotations

import logging

import ad_hoc_diffractometer as ahd
import hklpy2
import hklpy2.user
from hklpy2.diffract import DiffractometerBase, Sample

PSEUDOS = ["h", "k", "l"]
REALS = ["mu", "omega1", "chi1", "omega2", "chi2", "gamma1", "delta1"]

logger = logging.getLogger(__name__)


class CDIDiffractometer(DiffractometerBase):
    """This class provides a simplified API for working with hklpy2 diffractometer ophyd objects
    at the CDI beamline using the ad_hoc_diffractometer solver. For more detailed information about
    the permitted calculations and operations of hklpy2 and ad_hoc_diffractometer, consult the
    documentation for those libraries:
    hklpy2: https://blueskyproject.io/hklpy2/1.0.0/quickstart.html
    ad_hoc_diffractometer: https://bcda-aps.github.io/ad_hoc_diffractometer/v1.0.0/quick_start.html

    At a high level, hklpy2 provides the ophyd layer abstractions that allow the python objects to
    connect to the actual hardware. ad_hoc_diffractometer is strictly a calculation library that
    allows custom geometries to be defined and passed through to the underlying C-level hkl
    calcuation engine.

    To use this diffractometer, the cdi-geometry should already
    be registered, and the diffractometer ophyd object created,
    and connected to the TDMS detector arm and goniometer.

    To check the ophyd object detection:
    >>> diffr.connected # should return True

    To get a report of the diffractometer positions, run:
    >>> diffr.wh()

    or, for a fuller report:
    >>> diffr.wh(True)

    To use the diffractometer object to find the Bragg peak of a sample,
    take the following the steps:

    1. Add beam energy:
    This should be done automatically on first startup. If the beam energy changes
    during the experiment or after the diffractometer is initially set up, it should
    be updated manually.
    >>> diffr.beam.energy.put(energy.energy.readback.get())

    2. Add sample:
    >>> diffr.add_sample(name="gold", a=4.08)

    `name` and `a` are required parameters. Other optional parameters can be seen
    by inspecting the docstring of add_sample()

    3. Add orientation reflections.
    This step is necessary to calculate the UB matrix:

    >>> diffr.add_reflection(
            pseudos=(1,1,1),
            reals={"mu": 18.2, "omega1": 0, "chi1": 1.4, "omega2": 0, "chi2": 0, "gamma": 0.47, "delta": 31.84},
        )

    4. You can calculate the UB matrix with either 1 or 2 reflections. With one reflection, a referenece hkl, and a reference stage:

    >>> diffr.ub_from_one_reflection(reflection, reference_hkl, reference_stage, set_ub=False)

    To calculate UB with two reflections, set two reflections as outlined above, and call calc_UB():
    >>> diffr.calc_UB()
    or
    >>> diffr.core.calc_UB(*diffr.sample.reflections)

    5. Set the mode. Modes can be found in the geometry defintion yaml file,
    or listed through
    >>> diffr.core.modes

    To set the mode:
    >>> diffr.core.mode = "mode_name"

    6. Set the extras.

    Some modes required extra parameters to be set for the solver. Check which extras are
    required for the selected mode with:
    >>> diffr.core.extras

    And set their values with:
    >>> diffr.core.extras['key'] = value

    For example:
    >>> diffr.core.mode = "fixed_incidence"
    "fixed_incidence" mode requires two extras:
    >>> diffr.core.extras
        > {'n_hat': 0, 'incidence': 0}
    >>> diffr.core.extras['n_hat'] = (1,0,0)
    >>> diffr.core.extras['incidence'] = 5
    >>> diffr.core.extras
        > {'n_hat': (1, 0, 0), 'incidence': 5}

    7. Calculate the Bragg peak locations

    For the closest peak:
    >>> diffr.forward(h, k, l)

    For a table of solutions:
    >>> diffr.cahkl_table((h, k, l))

    cahkl_table() accepts an arbitrary number of tuples of
    the form (h, k, l).
    """

    @property
    def mode(self) -> str:
        return self.core.mode

    @mode.setter
    def mode(self, value: str) -> None:
        self.core.mode = value

    @property
    def modes(self) -> list[str]:
        return self.core.modes

    def cahkl_table(
        self,
        *reflections: list[hklpy2.user.AxesTuple],
        digits: int = 4,
        all_perms: bool = False,
    ):
        hklpy2.user.set_diffractometer(self)
        all_reflections = reflections
        if all_perms:
            all_reflections = []
            negs = [-1, -1, -1]
            permutations = []
            for i in range(1 << len(negs)):
                perm = [
                    d ** int(xp) for d, xp in zip(negs, format(i, "03b"), strict=False)
                ]
                permutations.append(perm)
            for ref in reflections:
                for perm in permutations:
                    # TODO - change ruff formatter to not require strict param
                    ref_perm = [r * n for r, n in zip(ref, perm, strict=False)]
                    if ref_perm not in all_reflections:
                        all_reflections.append(ref_perm)
        return hklpy2.user.cahkl_table(*all_reflections, digits=digits)

    def calc_UB(self):
        """A wrapper around core.calc_UB() to simplify the syntax."""
        return self.core.calc_UB(*self.sample.reflections)  # type: ignore[reportOptionalMemberAccess]

    def ub_from_one_reflection(self, reference_hkl, reference_stage, set_ub=False):  # type: ignore[reportUnknownParameterType]
        """Alias over ad_hoc_diffractometer.ub_from_one_reflection with an extra option to set the UB matrix on the sample."""
        if not self.sample:
            return RuntimeError("Must set a sample before calculating UB matrix.")
        if not self.sample.reflections:
            return RuntimeError("Must set at least one reflection.")
        rflct = self.sample.reflections.order[0]
        ub = ahd.ub_from_one_reflection(
            self.sample, self.sample.reflections[rflct], reference_hkl, reference_stage
        )
        if set_ub:
            self.sample.UB = ub.tolist()
        return ub

    def status(self) -> str:
        """Status method to show what has been configured."""
        status_str = "Configuration status:\n"
        green_check = "\U00002705"
        red_x = "\u274c"
        yellow_dash = "\033[33m" + "--" + "\033[0m"

        status = {
            "geometry": None,
            "connected": False,
            "energy": None,
            "sample": None,
            "reflections": {},
            "mode": self.core.mode,  # there is a default
            "extras": None,
        }

        if self.core.geometry:
            status["geometry"] = self.core.geometry
            status_str += f"{green_check} geometry: {self.core.geometry}\n"
        else:
            status_str += f"{red_x} no geometry set\n"
        status["connected"] = self.connected
        if self.connected:
            status_str += f"{green_check} connected\n"
        else:
            status_str += f"{red_x} not connected\n"
        # TODO - check if this can be None natively
        if self.beam.energy:
            status["energy"] = self.beam.energy.read()
            status_str += f"{green_check} energy: {self.beam.energy.get()}\n"
        else:
            status_str += f"{red_x} no energy set"
        # Sample
        if self.sample and not self._is_default_sample(self.sample):
            status["sample"] = self.sample.name
            status_str += f"{green_check} sample: {self.sample}\n"
            status["reflections"] = self.sample.reflections
            if len(self.sample.reflections) == 0:
                status_str += f"{red_x} no reflections set\n"
            elif len(self.sample.reflections) == 1:
                status_str += f"{red_x} one reflection set.\nset one more to calculate UB matrix\n"
            else:
                status_str += (
                    f"{green_check} two reflections set: can calculate UB matrix"
                )
        else:
            status_str += f"{red_x} no sample set\n"
            status_str += f"{red_x} no reflections set (must set sample first)\n"
        status["extras"] = self.core.extras
        if self.core.extras:
            for key, val in self.core.extras.items():
                if val == 0 or val is None:
                    status_str += f"{red_x} {key} not set\n"
                else:
                    status_str += f"{green_check} {key}: {val}\n"
        else:
            status_str += f"{yellow_dash} no extras"

        if (
            all(status.values())
            and status["extras"]
            and all(v != 0 for v in status["extras"].values())
        ):
            status_str += f"\n{green_check} ready to calculate Bragg peaks"
        else:
            status_str += f"\n{red_x} cannot calculate Bragg peaks"

        print(status_str)  # noqa: T201

    def help(self):
        print(CDIDiffractometer.__doc__)  # noqa: T201

    def _is_default_sample(self, sample: Sample) -> bool:
        return (
            sample.name == "sample"
            and (sample.lattice.a == sample.lattice.b == sample.lattice.c == 1)
            and (
                sample.lattice.alpha
                == sample.lattice.beta
                == sample.lattice.gamma
                == 90.0
            )
        )
