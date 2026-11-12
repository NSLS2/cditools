"""hklpy2 solver for the CDI diffractometer that returns every forward solution.

``ad_hoc_diffractometer`` 1.0.0 puts the detector in a single place before it
solves the sample stages: the last detector stage at +2theta and every other
detector stage wherever its motor currently is.  Solutions that need any other
detector position are never found, so most modes return at most two.  See
``debug/ad_hoc_solver/`` for the analysis.

:class:`CDIAdHocSolver` replaces ``forward()`` for modes made of fixed sample
angles, at most one fixed detector angle, and at most one surface reference
constraint (``incidence``, ``emergence`` or ``incidence_equals_emergence``).
It enumerates every detector position that matches the Bragg angle, solves the
sample stages for each one, and keeps only candidates that pass the library's
own ``inverse()`` and constraint checks.  Any other mode is passed to the
library unchanged.

Select it with ``hklpy2.creator(..., solver="cdi_ad_hoc")``.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from ad_hoc_diffractometer import forward as ahd_forward
from ad_hoc_diffractometer.mode import (
    DetectorConstraint,
    EwaldSphereViolation,
    ReferenceConstraint,
    SampleConstraint,
)
from ad_hoc_diffractometer.rotation import rotation_matrix
from hklpy2.exceptions import SolverError
from hklpy2.typing import NamedFloatDict
from hklpy2_solvers.ad_hoc_solver import AdHocSolver

logger = logging.getLogger(__name__)

Angles = dict[str, float]
Branches = dict[int, tuple[float, Angles]]

SURFACE_REFERENCES = frozenset({"incidence", "emergence", "incidence_equals_emergence"})
"""Reference constraints handled here; modes with any other go to the library."""

SCAN_STEP = 1.0
"""Grid spacing (degrees) used to bracket reference-constraint roots."""

TOLERANCE = 1e-6
"""Largest accepted hkl error and constraint residual (degrees) for a solution."""


class CDIAdHocSolver(AdHocSolver):
    """``ad_hoc`` solver whose ``forward()`` returns every solution.

    Solutions are sorted by total rotation (sum of absolute axis angles), so
    the least-rotated setting comes first.

    When a mode leaves more axes free than its constraints determine, the
    surplus axes stay at their current motor positions, as in
    ``ad_hoc_diffractometer``: outer detector stages first, then outer sample
    stages.  A mode that leaves too few axes free raises ``SolverError``.
    """

    name = "cdi_ad_hoc"

    def forward(self, pseudos: NamedFloatDict) -> list[NamedFloatDict]:
        """Compute every set of motor angles that reaches ``pseudos`` (hkl -> angles)."""
        if not isinstance(pseudos, dict):
            raise TypeError(f"Must supply dict, received {pseudos!r}")
        self._ensure_ready()
        self._raise_if_reference_vector_unset()

        hkl = np.array([float(pseudos[axis]) for axis in ("h", "k", "l")])
        problem = self._problem(hkl)
        if problem is None:
            logger.debug(
                "mode %r is not handled here; using ad_hoc_diffractometer", self.mode
            )
            return super().forward(pseudos)

        solutions = _finish(problem, hkl, _candidates(problem))
        populate = getattr(ahd_forward, "_populate_output_extras", None)
        if populate is not None:
            populate(self._geom, problem.mode, solutions)
        return list(solutions)

    def _problem(self, hkl: np.ndarray) -> _Problem | None:
        """Describe the forward problem, or return None if this solver can't handle the mode."""
        g = self._geom
        mode = g.mode
        if mode is None:
            return None
        extras = mode.extras or {}
        if {"h2", "k2", "l2"} <= extras.keys() or {"z0", "z1"} <= extras.keys():
            return None  # double-diffraction and zone modes
        if not np.allclose(g.inclination_matrix, np.eye(3)):
            return None

        sample = list(g.sample_stages)
        detector = list(g.detector_stages)
        stage_names = {s.name for s in sample + detector}
        fixed: Angles = {}
        reference: ReferenceConstraint | None = None
        for c in mode.constraints:
            if isinstance(c, SampleConstraint) or (
                isinstance(c, DetectorConstraint) and not c.is_qaz
            ):
                if c.name not in stage_names:
                    return None
                fixed[c.name] = float(c.value)
            elif (
                isinstance(c, ReferenceConstraint)
                and c.name in SURFACE_REFERENCES
                and reference is None
            ):
                reference = c
            else:
                return None

        if np.allclose(hkl, 0.0):
            msg = "forward(): (h, k, l) = (0, 0, 0) is not a valid reflection."
            raise SolverError(msg)
        q_phi = g.sample.UB @ hkl
        q_mag = float(np.linalg.norm(q_phi))
        wavelength = float(g.wavelength)
        sin_theta = q_mag * wavelength / (4.0 * math.pi)
        if sin_theta > 1.0:
            exc = EwaldSphereViolation(
                q_mag=q_mag, q_max=4.0 * math.pi / wavelength, wavelength=wavelength
            )
            raise SolverError(str(exc)) from exc

        base: Angles = {s.name: float(s.angle) for s in sample + detector}
        base.update(fixed)
        free_sample = [i for i, s in enumerate(sample) if s.name not in fixed]
        free_detector = [i for i, s in enumerate(detector) if s.name not in fixed]
        n_ref = 0 if reference is None else 1

        free_names = [sample[i].name for i in free_sample]
        free_names += [detector[i].name for i in free_detector]
        if len(free_names) < 3 + n_ref:
            # The library would return settings that ignore some constraints.
            equations = "Bragg" if reference is None else f"Bragg + {reference.name}"
            msg = (
                f"mode {self.mode!r} fixes too many axes: {equations} needs "
                f"{3 + n_ref} free axes, but only {free_names} are free.  "
                "Remove one of the mode's constraints."
            )
            raise SolverError(msg)

        held: list[str] = []
        while len(free_sample) + len(free_detector) > 3 + n_ref:
            if len(free_detector) > 1:
                held.append(detector[free_detector.pop(0)].name)
            else:
                held.append(sample[free_sample.pop(0)].name)
        if held:
            logger.debug(
                "mode %r is underdetermined; holding %s at current positions",
                self.mode,
                held,
            )
        if (len(free_detector), len(free_sample)) not in {
            (1, 2 + n_ref),
            (2, 1 + n_ref),
        }:
            return None

        n_phi = None
        if reference is not None:
            n_hkl = g.surface_normal if g.surface_normal is not None else g.azimuth
            n_phi = _unit(g.sample.UB @ np.asarray(n_hkl, dtype=float))

        return _Problem(
            geometry=g,
            mode=mode,
            q_phi=q_phi,
            k=2.0 * math.pi / wavelength,
            cos_2theta=1.0 - 2.0 * sin_theta**2,
            beam=_unit(g.basis["longitudinal"]),
            sample=sample,
            detector=detector,
            base=base,
            free_sample=free_sample,
            free_detector=free_detector,
            reference=reference,
            n_phi=n_phi,
        )


@dataclass
class _Problem:
    """Everything the solve needs, in the library's lab-frame conventions."""

    geometry: Any
    mode: Any
    q_phi: np.ndarray
    k: float  # 2 pi / wavelength
    cos_2theta: float
    beam: np.ndarray  # unit incident-beam direction
    sample: list[Any]
    detector: list[Any]
    base: Angles  # every stage: constraint value, or current position
    free_sample: list[int]  # indices into ``sample``
    free_detector: list[int]  # indices into ``detector``
    reference: ReferenceConstraint | None
    n_phi: np.ndarray | None  # unit surface normal, phi frame


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------


def _candidates(p: _Problem) -> list[Angles]:
    """Unchecked candidate solutions (all stages) for the problem."""
    if len(p.free_detector) == 1:
        return _detector_first(p)
    return _sample_first(p)


def _detector_first(p: _Problem) -> list[Angles]:
    """One free detector stage: its angle is set by 2theta, then solve the sample.

    Both roots of beam . D beam = cos(2theta) are used, e.g. delta = +/-2theta
    when gamma is fixed at 0.
    """
    j = p.free_detector[0]
    name = p.detector[j].name
    xs = _solve_one_in_chain(p.detector, p.base, j, p.beam, p.beam, p.cos_2theta)
    if xs is None:
        xs = [p.base[name]]

    candidates: list[Angles] = []
    for x in xs:
        angles = {**p.base, name: x}
        kf = _chain(p.detector, angles) @ p.beam
        q_lab = p.k * (kf - p.beam)
        candidates += _sample_for_q_lab(p, angles, q_lab, kf)
    return candidates


def _sample_for_q_lab(
    p: _Problem, angles: Angles, q_lab: np.ndarray, kf: np.ndarray
) -> list[Angles]:
    """Sample settings with Z @ q_phi == q_lab (plus the reference constraint, if any)."""
    if p.reference is None:
        i, j = p.free_sample
        return [
            {**angles, **pair}
            for pair in _solve_two_in_chain(p.sample, angles, i, j, p.q_phi, q_lab)
        ]

    t, i, j = p.free_sample
    t_name = p.sample[t].name

    def inner(value: float) -> Branches:
        trial = {**angles, t_name: value}
        branches: Branches = {}
        for b, pair in enumerate(
            _solve_two_in_chain(p.sample, trial, i, j, p.q_phi, q_lab)
        ):
            full = {**trial, **pair}
            branches[b] = (_reference_residual(p, _chain(p.sample, full), kf), full)
        return branches

    return _scan_roots(inner)


def _sample_first(p: _Problem) -> list[Angles]:
    """Two free detector stages: solve the sample, then point the detector along kf.

    Any sample orientation with beam . (Z @ q_phi) = k (cos 2theta - 1) puts
    Q on the Ewald sphere; the detector then has two settings for each
    resulting kf direction.
    """
    ewald = p.k * (p.cos_2theta - 1.0)

    if p.reference is None:
        (i,) = p.free_sample
        name = p.sample[i].name
        xs = _solve_one_in_chain(p.sample, p.base, i, p.beam, p.q_phi, ewald)
        if xs is None:
            xs = [p.base[name]]
        samples = [{**p.base, name: x} for x in xs]
    else:
        t, i = p.free_sample
        t_name, i_name = p.sample[t].name, p.sample[i].name

        def inner(value: float) -> Branches:
            trial = {**p.base, t_name: value}
            xs = _solve_one_in_chain(p.sample, trial, i, p.beam, p.q_phi, ewald)
            if xs is None:
                xs = [trial[i_name]]
            branches: Branches = {}
            for b, x in enumerate(xs):
                full = {**trial, i_name: x}
                z = _chain(p.sample, full)
                kf = p.beam + z @ p.q_phi / p.k
                branches[b] = (_reference_residual(p, z, kf), full)
            return branches

        samples = _scan_roots(inner)

    d0, d1 = p.free_detector
    candidates: list[Angles] = []
    for angles in samples:
        kf = _unit(p.beam + _chain(p.sample, angles) @ p.q_phi / p.k)
        for pair in _solve_two_in_chain(p.detector, angles, d0, d1, p.beam, kf):
            candidates.append({**angles, **pair})
    return candidates


def _reference_residual(p: _Problem, z: np.ndarray, kf: np.ndarray) -> float:
    """Reference-constraint residual (degrees), with the library's sign conventions."""
    assert p.reference is not None
    assert p.n_phi is not None
    n_lab = z @ p.n_phi
    incidence = math.degrees(math.asin(np.clip(-(p.beam @ n_lab), -1.0, 1.0)))
    emergence = math.degrees(math.asin(np.clip(_unit(kf) @ n_lab, -1.0, 1.0)))
    if p.reference.name == "incidence":
        return incidence - float(p.reference.value)
    if p.reference.name == "emergence":
        return emergence - float(p.reference.value)
    return incidence - emergence


# ---------------------------------------------------------------------------
# Checking and ordering
# ---------------------------------------------------------------------------


def _finish(p: _Problem, hkl: np.ndarray, candidates: list[Angles]) -> list[Angles]:
    """Keep valid, in-limits, distinct candidates, least-rotated first."""
    solutions: list[Angles] = []
    for candidate in candidates:
        if not _is_valid(p, hkl, candidate):
            continue
        placed = _place_in_range(p, candidate)
        if placed is None:
            continue
        if any(_same(placed, s) for s in solutions):
            continue
        solutions.append(placed)
    solutions.sort(key=lambda angles: sum(abs(v) for v in angles.values()))
    return solutions


def _is_valid(p: _Problem, hkl: np.ndarray, angles: Angles) -> bool:
    """Check a candidate with the library's own inverse and constraint residuals."""
    g = p.geometry
    if not np.allclose(g.inverse(angles), hkl, rtol=0.0, atol=TOLERANCE):
        return False
    return all(abs(c.evaluate(angles, g)) < TOLERANCE for c in p.mode.constraints)


def _place_in_range(p: _Problem, angles: Angles) -> Angles | None:
    """Apply cut points (mode first, then geometry) or wrap to [-180, 180), then check limits.

    Without a cut point, an angle outside the stage limits is moved by 360
    degrees if that brings it inside.  Returns None when any stage is out of limits.
    """
    g = p.geometry
    placed: Angles = {}
    for name, raw in angles.items():
        stage = g.stage(name)
        if name in p.mode.cut_points:
            angle = p.mode.apply_cut_point(name, raw)
        elif name in g.cut_points:
            cut = g.cut_points[name]
            angle = cut + (raw - cut) % 360.0
        else:
            angle = _wrap(raw)
            if not stage.in_limits(angle):
                angle = next(
                    (a for a in (angle + 360.0, angle - 360.0) if stage.in_limits(a)),
                    angle,
                )
        if not stage.in_limits(angle):
            return None
        placed[name] = angle
    return placed


def _same(a: Angles, b: Angles) -> bool:
    return all(abs(_wrap(a[name] - b[name])) < TOLERANCE for name in a)


def _wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Rotation algebra
# ---------------------------------------------------------------------------


def _unit(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _chain(stages: list[Any], angles: Angles) -> np.ndarray:
    """R_0 @ R_1 @ ... for the stages, outermost leftmost (the library's convention)."""
    m = np.eye(3)
    for s in stages:
        m = m @ _rotation(tuple(s.axis), angles[s.name])
    return m


@functools.lru_cache(maxsize=4096)
def _rotation(axis: tuple[float, ...], angle: float) -> np.ndarray:
    """Cached library rotation matrix; most stages sit at the same angle throughout a solve."""
    matrix = rotation_matrix(np.asarray(axis), angle)
    matrix.flags.writeable = False
    return matrix


def _one_rotation(
    a: np.ndarray, n: np.ndarray, b: np.ndarray, c: float
) -> list[float] | None:
    """Angles x (degrees) with a . R(n, x) b = c, for a unit axis n.

    Returns both roots (branch 0 and branch 1; equal at a tangency), [] when
    there is no root, or None when the left side does not depend on x.
    """
    an = float(a @ n)
    nb = float(n @ b)
    cos_coef = float(a @ b) - an * nb
    sin_coef = float(a @ np.cross(n, b))
    amplitude = math.hypot(cos_coef, sin_coef)
    if amplitude < 1e-12:
        return None
    ratio = (c - an * nb) / amplitude
    if abs(ratio) > 1.0 + 1e-9:
        return []
    x0 = math.atan2(sin_coef, cos_coef)
    dx = math.acos(max(-1.0, min(1.0, ratio)))
    return [math.degrees(x0 + dx), math.degrees(x0 - dx)]


def _angle_about(n: np.ndarray, u: np.ndarray, w: np.ndarray) -> float | None:
    """Rotation (degrees) about unit n taking u's part perpendicular to n onto w's."""
    u_perp = u - (u @ n) * n
    w_perp = w - (w @ n) * n
    if np.linalg.norm(u_perp) < 1e-12 or np.linalg.norm(w_perp) < 1e-12:
        return None
    return math.degrees(math.atan2(n @ np.cross(u_perp, w_perp), u_perp @ w_perp))


def _solve_one_in_chain(
    stages: list[Any], angles: Angles, i: int, a: np.ndarray, v: np.ndarray, c: float
) -> list[float] | None:
    """Angles of ``stages[i]`` with a . (chain @ v) = c, other stages at ``angles``."""
    before = _chain(stages[:i], angles)
    after = _chain(stages[i + 1 :], angles)
    return _one_rotation(before.T @ a, _unit(stages[i].axis), after @ v, c)


def _solve_two_in_chain(
    stages: list[Any], angles: Angles, i: int, j: int, v: np.ndarray, q: np.ndarray
) -> list[Angles]:
    """Angles of ``stages[i]`` and ``stages[j]`` (i < j) with chain @ v = q.

    Other stages stay at ``angles``.  Returns one dict per branch, in branch
    order.  A stage whose angle is indeterminate keeps its value from ``angles``.
    """
    before = _chain(stages[:i], angles)
    between = _chain(stages[i + 1 : j], angles)
    after = _chain(stages[j + 1 :], angles)
    n_i = _unit(stages[i].axis)
    n_j = _unit(stages[j].axis)
    v_j = after @ v
    q_i = before.T @ q
    name_i, name_j = stages[i].name, stages[j].name

    # Rotation about n_i leaves the n_i component alone, which fixes stage j.
    xs = _one_rotation(between.T @ n_i, n_j, v_j, float(n_i @ q_i))
    if xs is None:
        xs = [angles[name_j]]
    pairs: list[Angles] = []
    for x in xs:
        w = between @ rotation_matrix(n_j, x) @ v_j
        angle_i = _angle_about(n_i, w, q_i)
        pairs.append(
            {name_i: angles[name_i] if angle_i is None else angle_i, name_j: x}
        )
    return pairs


# ---------------------------------------------------------------------------
# One-dimensional root finding along a scanned angle
# ---------------------------------------------------------------------------


def _scan_roots(
    inner: Callable[[float], Branches], step: float = SCAN_STEP
) -> list[Angles]:
    """Every root of a residual as one angle is scanned through a full turn.

    ``inner(t)`` returns ``{branch: (residual, angles)}`` for the branches that
    exist at ``t``.  Branch labels must vary continuously with ``t``.  Roots are
    bracketed on a grid, including near the ends of a branch where its two roots
    merge, and refined by bisection.
    """
    grid = np.arange(-180.0, 180.0 + step / 2, step)
    values = [inner(float(t)) for t in grid]
    roots: list[Angles] = []
    for k in range(len(grid) - 1):
        t0, t1 = float(grid[k]), float(grid[k + 1])
        v0, v1 = values[k], values[k + 1]
        for b, (f0, a0) in v0.items():
            if f0 == 0.0:
                roots.append(a0)
            elif b in v1:
                if f0 * v1[b][0] < 0:
                    roots += _bisect(inner, b, t0, t1)
            else:  # branch b ends inside (t0, t1)
                t_end = _branch_edge(inner, b, t0, t1)
                if f0 * inner(t_end)[b][0] < 0:
                    roots += _bisect(inner, b, t0, t_end)
        for b, (f1, _) in v1.items():
            if b not in v0:  # branch b starts inside (t0, t1)
                t_start = _branch_edge(inner, b, t1, t0)
                if f1 * inner(t_start)[b][0] < 0:
                    roots += _bisect(inner, b, t_start, t1)
    return roots


def _branch_edge(
    inner: Callable[[float], Branches], b: int, inside: float, outside: float
) -> float:
    """Last ``t`` from ``inside`` towards ``outside`` where branch b exists (to ~1e-9 deg)."""
    for _ in range(30):
        mid = 0.5 * (inside + outside)
        if b in inner(mid):
            inside = mid
        else:
            outside = mid
    return inside


def _bisect(
    inner: Callable[[float], Branches], b: int, lo: float, hi: float
) -> list[Angles]:
    """Refine a bracketed root of branch b (Illinois method); [] if the branch vanishes."""
    f_lo, f_hi = inner(lo)[b][0], inner(hi)[b][0]
    best = lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(100):
        t = hi - f_hi * (hi - lo) / (f_hi - f_lo)
        if not lo < t < hi and not hi < t < lo:
            t = 0.5 * (lo + hi)
        here = inner(t).get(b)
        if here is None:
            return []
        f, angles = here
        if abs(f) < 1e-12 or abs(hi - lo) < 1e-12:
            return [angles]
        best = t
        if (f < 0) == (f_hi < 0):
            f_lo /= 2.0  # Illinois step: keeps the stale end from stalling convergence
        else:
            lo, f_lo = hi, f_hi
        hi, f_hi = t, f
    return [inner(best)[b][1]]
