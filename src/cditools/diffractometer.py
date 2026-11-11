from __future__ import annotations

import logging
from collections.abc import Callable

import hklpy2
import hklpy2.user
from hklpy2.diffract import DiffractometerBase

PSEUDOS = ["h", "k", "l"]
REALS = ["mu", "omega1", "chi1", "omega2", "chi2", "gamma1", "delta1"]

logger = logging.getLogger(__name__)


# TODO - hklpy2.creator() uses this class as a mixin
# might not need all the init params
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
    >>> diffr.add_sample(name="gold", a=4.008)

    `name` and `a` are required parameters. Other optional parameters can be seen
    by inspecting the docstring of add_sample()

    3. Add orientation reflections.
    This step is necessary to calculate the UB matrix:

    >>> diffr.add_reflection(
            pseudos=(1,1,1),
            reals={"mu": 18.2, "omega1": 0, "chi1": 1.4, "omega2": 0, "chi2": 0, "gamma": 0.47, "delta": 31.84},
        )
    >>> diffr.add_reflection(
            pseudos=(2,0,0),
            reals={ "mu": 3, "omega1": 0, "chi1": 0, "omega2": 0, "chi2": 0, "gamma": 2.39, "delta": 38.5},
        )

    4. Now you can calculate the UB matrix:
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

    def __init__(
        self,
        prefix: str = "XF:09IDC-",
        solver: str | None = "ad_hoc",
        geometry: str | None = "cdi-geometry",
        solver_kwargs: dict = {},  # noqa: B006
        pseudos: list[str] = PSEUDOS,
        reals: list[str] = REALS,
        reals_units: str | None = None,
        forward_solution_function: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            prefix=prefix,
            solver=solver,
            geometry=geometry,
            solver_kwargs=solver_kwargs,
            pseudos=pseudos,
            reals=reals,
            reals_units=reals_units,
            forward_solution_function=forward_solution_function,
            **kwargs,
        )

    def cahkl_table(self, *reflections: list[hklpy2.user.AxesTuple], digits: int = 4):
        hklpy2.user.set_diffractometer(self)
        return hklpy2.user.cahkl_table(*reflections, digits=digits)

    def calc_UB(self):
        """
        A wrapper around core.calc_UB() to simplify the syntax easier.
        """
        if self.sample:
            return self.core.calc_UB(*self.sample.reflections)
        e = "Sample is not defined"
        raise RuntimeError(e)

    def status(self) -> str:
        """
        Status object to show what has been configured.
        """
        status_str = "Status:\n"
        green_check = "\U00002705"
        red_x = "\u274c"

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
            status_str += f"{green_check} energy: f{self.beam.energy.read()}\n"
        else:
            status_str += f"{red_x} no energy set"
        if self.sample:
            status["sample"] = self.sample.name
            status_str += f"{green_check} sample: {self.sample.name}\n"
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
        status["extras"] = self.core.extras
        if self.core.extras:
            for key, val in self.core.extras.items():
                if val == 0 or val is None:
                    status_str += f"{red_x} {key} not set\n"
                else:
                    status_str += f"{green_check} {key}: {val}"

        if all(status.values()):
            status_str += f"\n{green_check} Ready to calculate Bragg peaks"

        print(status_str)  # noqa: T201

    def help(self):
        print(CDIDiffractometer.__doc__)  # noqa: T201

    # def forward(self, pseudos: dict, wavelength: float | None = None) -> NamedTuple:
    #     return super().forward(pseudos, wavelength)

    # @pseudo_position_argument
    # def forward(self, pseudos: dict, wavelength: float | None = None) -> NamedTuple:
    #     """Compute real-space coordinates from pseudos (hkl -> angles)."""
    #     logger.debug("forward: pseudos=%r", pseudos)
    #     solutions = self._forward(pseudos, wavelength=wavelength)
    #     return self._forward_solution(self.real_position, solutions)

    # def _forward(self, pseudos: AnyAxesType, wavelength: float | None = None) -> list:
    #     """
    #     Compute [{names:reals}] from {names: pseudos} (hkl -> angles).

    #     Delegates to :meth:`SolverBase.forward() <hklpy2.backends.base.SolverBase.forward>`
    #     which returns all solutions the backend engine can find (one or
    #     more).  Each solution is then filtered against the active
    #     :attr:`constraints`; only solutions that satisfy all constraints
    #     are included in the returned list.
    #     """
    #     logger.debug("(%s) forward(): pseudos=%r", self.__class__.__name__, pseudos)
    #     self._warn_if_ub_stale()

    #     pdict = self.standardize_pseudos(pseudos)
    #     base_reals = self.diffractometer.real_position._asdict()  # Original values.

    #     self.update_solver(wavelength=wavelength)

    #     # Filter just the solutions that fit the constraints.
    #     solutions = []
    #     try:
    #         # Unit conversions for presets and new values.
    #         angle_units_solver = self.solver.ANGLE_UNITS
    #         angle_units_core = self.diffractometer.reals_units

    #         # Get the constant (read-only) axis names for this mode.
    #         constant_axes = self._constant_axis_names()

    #         # Get presets for current mode.
    #         mode_presets = self.presets

    #         # Build the presets dict:
    #         # - For constant axes: use preset value if set, otherwise use current motor position
    #         # - For written axes: always use current motor position
    #         solver_reals = {}
    #         for axis in self.solver.real_axis_names:
    #             local_axis = self.axes_xref_reversed[axis]
    #             if axis in constant_axes:
    #                 if local_axis in mode_presets:
    #                     solver_reals[axis] = hklpy2.ops.convert_units(
    #                         mode_presets[local_axis],
    #                         angle_units_core,
    #                         angle_units_solver,
    #                     )
    #                 else:
    #                     solver_reals[axis] = hklpy2.ops.convert_units(
    #                         base_reals[local_axis], angle_units_core, angle_units_solver
    #                     )
    #             else:
    #                 solver_reals[axis] = hklpy2.ops.convert_units(
    #                     base_reals[local_axis], angle_units_core, angle_units_solver
    #                 )

    #         self.solver.set_reals(solver_reals)
    #         for solution in self.solver.forward(self._axes_names_d2s(pdict)):
    #             reals = base_reals.copy()  # Fresh copy per solution.
    #             new_reals = self._axes_names_s2d(solution)
    #             for axis, value in new_reals.items():
    #                 # Update with converted new value.
    #                 reals[axis] = hklpy2.ops.convert_units(
    #                     value, angle_units_solver, angle_units_core
    #                 )
    #             # Apply cut-point wrapping before constraint checking.
    #             # Each axis angle is mapped into [cut_point, cut_point+360)
    #             # so that limit checks operate on the wrapped representation.
    #             assert self.constraints is not None
    #             for name, constraint in self.constraints.items():
    #                 if name in reals:
    #                     reals[name] = constraint.apply_cut(reals[name])

    #             if self.constraints.valid(**reals):
    #                 solutions.append(self.diffractometer.RealPosition(**reals))
    #             else:
    #                 # Log which constraint(s) rejected this solution.
    #                 for name, constraint in self.constraints.items():
    #                     if not constraint.valid(**reals):
    #                         logger.info(
    #                             "Solution discarded: %s=%.4f"
    #                             " (after cut_point=%.4f wrapping)"
    #                             " outside limits [%.4f, %.4f]."
    #                             "  pseudos=%r",
    #                             name,
    #                             reals[name],
    #                             constraint.cut_point,
    #                             constraint.low_limit,
    #                             constraint.high_limit,
    #                             pdict,
    #                         )
    #     except hklpy2.ops.NoForwardSolutions as exc:
    #         logger.info(
    #             "No forward solutions from solver for pseudos=%r: %s", pdict, exc
    #         )

    #     return solutions


# def add_sample(self, name: str | xu.materials.Material, a: float | None = None, **kwargs) -> Sample:
#     if isinstance(name, xu.materials.Material):
#         if a is None:
#             a = name.a
#         name = str(name.name)
#     super().add_sample(name, float(a), **kwargs)
