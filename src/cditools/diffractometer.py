from __future__ import annotations

import logging
from collections.abc import Callable

import hklpy2
import hklpy2.ops
from hklpy2.diffract import DiffractometerBase

PSEUDOS = ["h", "k", "l"]
REALS = ["mu", "omega1", "chi1", "omega2", "chi2", "gamma1", "delta1"]
hklpy2.creator()

logger = logging.getLogger(__name__)


class CDIDiffractometer(DiffractometerBase):
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

    def status(self) -> str:
        """
        Status object to show what has been configured.

        Possible helpers:

        g.modes["mode"].is_fully_constrained(g)
        g.modes["mode"].is_implemented(g)

        g.modes["mode"] = g.modes["mode"].with_constraint_values(**kwargs)
        """
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
        status["connected"] = self.connected
        # TODO - check if this can be None natively
        if self.beam.energy:
            status["energy"] = self.beam.energy
        if self.sample:
            status["sample"] = self.sample.name
            status["reflections"] = self.sample.reflections

        # Check if can calc UB
        if self.sample and len(self.sample.reflections) >= 2:
            pass
        return status

    def help(self):
        return self.__doc__


# class CDISolver(hklpy2_solvers.ad_hoc_solver.AdHocSolver):
#     pass


# class TmpCore(hklpy2.ops.Core):

#     def forward(self, pseudos: AnyAxesType, wavelength: float | None = None) -> list:
#         """
#         Compute [{names:reals}] from {names: pseudos} (hkl -> angles).

#         Delegates to :meth:`SolverBase.forward() <hklpy2.backends.base.SolverBase.forward>`
#         which returns all solutions the backend engine can find (one or
#         more).  Each solution is then filtered against the active
#         :attr:`constraints`; only solutions that satisfy all constraints
#         are included in the returned list.
#         """
#         logger.debug("(%s) forward(): pseudos=%r", self.__class__.__name__, pseudos)
#         self._warn_if_ub_stale()

#         pdict = self.standardize_pseudos(pseudos)
#         base_reals = self.diffractometer.real_position._asdict()  # Original values.

#         self.update_solver(wavelength=wavelength)

#         # Filter just the solutions that fit the constraints.
#         solutions = []
#         try:
#             # Unit conversions for presets and new values.
#             angle_units_solver = self.solver.ANGLE_UNITS
#             angle_units_core = self.diffractometer.reals_units

#             # Get the constant (read-only) axis names for this mode.
#             constant_axes = self._constant_axis_names()

#             # Get presets for current mode.
#             mode_presets = self.presets

#             # Build the presets dict:
#             # - For constant axes: use preset value if set, otherwise use current motor position
#             # - For written axes: always use current motor position
#             solver_reals = {}
#             for axis in self.solver.real_axis_names:
#                 local_axis = self.axes_xref_reversed[axis]
#                 if axis in constant_axes:
#                     if local_axis in mode_presets:
#                         solver_reals[axis] = hklpy2.ops.convert_units(
#                             mode_presets[local_axis],
#                             angle_units_core,
#                             angle_units_solver,
#                         )
#                     else:
#                         solver_reals[axis] = hklpy2.ops.convert_units(
#                             base_reals[local_axis], angle_units_core, angle_units_solver
#                         )
#                 else:
#                     solver_reals[axis] = hklpy2.ops.convert_units(
#                         base_reals[local_axis], angle_units_core, angle_units_solver
#                     )

#             self.solver.set_reals(solver_reals)
#             for solution in self.solver.forward(self._axes_names_d2s(pdict)):
#                 reals = base_reals.copy()  # Fresh copy per solution.
#                 new_reals = self._axes_names_s2d(solution)
#                 for axis, value in new_reals.items():
#                     # Update with converted new value.
#                     reals[axis] = hklpy2.ops.convert_units(
#                         value, angle_units_solver, angle_units_core
#                     )
#                 # Apply cut-point wrapping before constraint checking.
#                 # Each axis angle is mapped into [cut_point, cut_point+360)
#                 # so that limit checks operate on the wrapped representation.
#                 assert self.constraints is not None
#                 for name, constraint in self.constraints.items():
#                     if name in reals:
#                         reals[name] = constraint.apply_cut(reals[name])

#                 if self.constraints.valid(**reals):
#                     solutions.append(self.diffractometer.RealPosition(**reals))
#                 else:
#                     # Log which constraint(s) rejected this solution.
#                     for name, constraint in self.constraints.items():
#                         if not constraint.valid(**reals):
#                             logger.info(
#                                 "Solution discarded: %s=%.4f"
#                                 " (after cut_point=%.4f wrapping)"
#                                 " outside limits [%.4f, %.4f]."
#                                 "  pseudos=%r",
#                                 name,
#                                 reals[name],
#                                 constraint.cut_point,
#                                 constraint.low_limit,
#                                 constraint.high_limit,
#                                 pdict,
#                             )
#         except hklpy2.ops.NoForwardSolutions as exc:
#             logger.info(
#                 "No forward solutions from solver for pseudos=%r: %s", pdict, exc
#             )

#         return solutions

# def add_sample(self, name: str | xu.materials.Material, a: float | None = None, **kwargs) -> Sample:
#     if isinstance(name, xu.materials.Material):
#         if a is None:
#             a = name.a
#         name = str(name.name)
#     super().add_sample(name, float(a), **kwargs)
