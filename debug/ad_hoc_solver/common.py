"""Shared setup for the ad_hoc_diffractometer solver debug scripts.

Every script builds the same simulated CDI diffractometer used in
``demo_hkl.ipynb``: gold (a = 4.08 Å) at 10.002 keV, with UB computed from the
(111) and (200) reflections measured on the beamline.

Line numbers quoted in the scripts refer to
``ad_hoc_diffractometer/forward.py`` from ad_hoc_diffractometer 1.0.0
(``.pixi/envs/default/lib/python3.13/site-packages/``).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import ad_hoc_diffractometer as ahd
import hklpy2
import hklpy2.user
import numpy as np
from ad_hoc_diffractometer import forward as ahd_forward
from scipy.optimize import least_squares

GEOMETRIES = Path(__file__).resolve().parent / "geometries"

REALS = ["mu", "omega1", "chi1", "omega2", "chi2", "gamma", "delta"]
"""Diffractometer axis names, in solver stage order."""


def make_diffractometer(geometry_file: str = "cdi-geometry.yml", solver: str = "ad_hoc"):
    """Register a geometry YAML from ./geometries and return an oriented diffractometer.

    The diffractometer always uses the axis names in ``REALS``, even when the
    solver geometry names its stages differently (hklpy2 maps them by position).
    ``solver="cdi_ad_hoc"`` selects ``cditools.hkl_solver.CDIAdHocSolver``.
    """
    path = GEOMETRIES / geometry_file
    name = path.stem
    if name not in ahd.list_geometries():
        ahd.register_geometry_file(str(path), name=name)

    diff = hklpy2.creator(name="cdi", geometry=name, solver=solver, reals=REALS)
    diff.beam.energy.put(10.002)  # keV
    hklpy2.user.set_diffractometer(diff)
    hklpy2.user.add_sample("gold", a=4.08)
    r1 = hklpy2.user.setor(
        1, 1, 1, mu=18.2, omega1=0, chi1=1.4, omega2=0, chi2=0, gamma=0.47, delta=31.84
    )
    r2 = hklpy2.user.setor(
        2, 0, 0, mu=3, omega1=0, chi1=0, omega2=0, chi2=0, gamma=2.39, delta=38.5
    )
    hklpy2.user.calc_UB(r1, r2)
    return diff


def geom(diff):
    """The underlying ``AdHocDiffractometer`` (private hklpy2 adapter attribute)."""
    return diff.core.solver._geom


def to_solver(diff, position) -> dict[str, float]:
    """Convert a RealPosition (or dict) from diffractometer to solver axis names."""
    values = position._asdict() if hasattr(position, "_asdict") else dict(position)
    xref = diff.core.axes_xref
    return {xref[axis]: float(value) for axis, value in values.items()}


def hkl_of(diff, position) -> np.ndarray:
    """Independent check: run the solver's inverse on a returned position."""
    return np.asarray(geom(diff).inverse(to_solver(diff, position)))


def fmt(position, axes) -> str:
    """Format the named axes of a RealPosition (or dict) on one line."""
    values = position._asdict() if hasattr(position, "_asdict") else position
    return "  ".join(f"{axis}={values[axis]:9.3f}" for axis in axes)


def dispatched_solver(diff) -> str:
    """Name of the forward.py function the active mode is routed to.

    Mirrors the if-chain in ``forward.py::_solve_constraint_set`` (lines
    566-635).  Call it after a ``forward()`` so hklpy2 has pushed extras such
    as ``n_hat`` into the solver.
    """
    g = geom(diff)
    checks = [
        (ahd_forward._is_psi_mode, "_solve_psi_mode"),
        (ahd_forward._is_double_diffraction_mode, "_solve_double_diffraction"),
        (ahd_forward._is_zone_mode, "_solve_zone"),
        (lambda _g, mode: mode.has_bisect, "_solve_bisecting"),
        (ahd_forward._is_kappa_virtual_mode, "_solve_kappa_virtual"),
        (ahd_forward._is_surface_mode, "_solve_surface"),
        (ahd_forward._is_omega_mode, "_solve_omega_mode"),
        (ahd_forward._is_qaz_mode, "_solve_qaz_mode"),
        (ahd_forward._is_free_detectors_mode, "_solve_free_detectors"),
    ]
    for predicate, solver_name in checks:
        if predicate(g, g.mode):
            return solver_name
    return "_solve_fixed_sample"


def brute_force(diff, hkl, fixed, free, extra_residual=None, step=45.0):
    """Find every solution by least squares from a grid of starting points.

    Independent of the library's forward solvers: only its ``inverse()`` is
    used, to compute the hkl a set of angles diffracts.

    fixed:
        ``{diffractometer axis: angle}`` for the constrained axes.
    free:
        Diffractometer axis names to solve for.  ``fixed`` plus ``free``
        should cover all axes.
    extra_residual:
        Optional ``f(solver_angles) -> float`` adding one more equation,
        e.g. ``incidence - target``.

    Returns a list of ``{axis: angle}`` dicts, angles wrapped to [-180, 180).
    """
    g = geom(diff)
    target = np.asarray(hkl, dtype=float)
    xref = diff.core.axes_xref
    base = to_solver(diff, fixed)

    def residual(x):
        angles = dict(base)
        angles.update({xref[axis]: v for axis, v in zip(free, x, strict=True)})
        r = list(np.asarray(g.inverse(angles)) - target)
        if extra_residual is not None:
            r.append(extra_residual(angles) / 10)  # degrees -> comparable scale
        return r

    found: list[np.ndarray] = []
    for seed in itertools.product(np.arange(-180, 180, step), repeat=len(free)):
        fit = least_squares(residual, seed, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.max(np.abs(fit.fun)) > 1e-8:
            continue
        x = (fit.x + 180) % 360 - 180
        if not any(np.allclose(x, y, atol=1e-3) for y in found):
            found.append(x)
    found.sort(key=tuple)
    return [dict(zip(free, x, strict=True)) for x in found]
