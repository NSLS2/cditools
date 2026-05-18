from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, ClassVar, Protocol

import bluesky.plan_stubs as bps
import numpy as np
import skbeam.core.constants.xrf as xrfC
from bluesky import Msg
from bluesky.callbacks import LiveTable
from ophyd import Component as Cpt  # type: ignore[import-not-found]
from ophyd import (
    Device,
    EpicsMotor,
    PseudoPositioner,
    PseudoSingle,
    Signal,
)
from ophyd import DynamicDeviceComponent as DDC
from ophyd.pseudopos import (
    pseudo_position_argument,
    real_position_argument,
)
from scipy.interpolate import make_interp_spline


class EpicsMotorRO(EpicsMotor):
    def __init__(self, prefix: str, *, name: str, **kwargs: Any) -> None:
        """Creates an epics motor that is read only

        Args:
            prefix (str): motor prefix
            name (str): motor name
        """
        super().__init__(prefix=prefix, name=name, **kwargs)

    def move(self, *args: object, **kwargs: object):  # noqa: ARG002
        """Move function to override the EpicsMotor class

        Raises:
            PermissionError: raised when trying to move the motor since it is read-only
        """
        msg = f"{self.name} is read-only and cannot be moved."
        raise PermissionError(msg)

    def stop(self, *args: object, **kwargs: object):  # noqa: ARG002
        """Stop function to override the EpicsMotor class

        Raises:
            PermissionError: raised when trying to stop the motor since it is read-only
        """
        msg = f"{self.name} is read-only and cannot be stopped manually."
        raise PermissionError(msg)

    def set(self, *args: object, **kwargs: object):  # noqa: ARG002
        """Set function to override the EpicsMotor class

        Raises:
            PermissionError: raised when trying to set the motor since it is read-only
        """
        msg = f"{self.name} is read-only and cannot be set."
        raise PermissionError(msg)

    def set_position(self, *args: object, **kwargs: object):  # noqa: ARG002
        """Set position function to override the EpicsMotor class

        Raises:
            PermissionError: raised when trying to set the motor since it is read-only
        """
        msg = f"{self.name} is read-only and its position cannot be set."
        raise PermissionError(msg)

    def _readonly_put(self, *args: object, **kwargs: object):  # noqa: ARG002
        """Read only put function for the epics motor

        Raises:
            PermissionError: raised when trying to write to the motor since it is read-only
        """
        msg = f"{self.name} is read-only and cannot write PVs."
        raise PermissionError(msg)


class DM1(Device):
    """Ophyd Device for DM1"""

    slit = DDC(
        {
            "ib": (EpicsMotor, "Slt:WB1-Ax:I}Mtr", {}),
            "ob": (EpicsMotor, "Slt:WB1-Ax:O}Mtr", {}),
            "bb": (EpicsMotor, "Slt:WB1-Ax:B}Mtr", {}),
            "tb": (EpicsMotor, "Slt:WB1-Ax:T}Mtr", {}),
            "hg": (EpicsMotor, "Slt:WB1-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:WB1-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:WB1-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:WB1-Ax:VC}Mtr", {}),
        }
    )
    filt = Cpt(EpicsMotor, "Fltr:DM1-Ax:Y}Mtr")


class DMM(Device):
    h = Cpt(EpicsMotor, "Mono:DMM-Ax:TX}Mtr")
    v = Cpt(EpicsMotor, "Mono:DMM-Ax:TY}Mtr")
    bragg = Cpt(EpicsMotor, "Mono:DMM-Ax:Bragg}Mtr")
    mlm1 = DDC(
        {
            "r": (EpicsMotor, "Mono:DMM-Ax:Roll}Mtr", {}),
            "fr": (EpicsMotor, "Mono:DMM-Ax:FR}Mtr", {}),
        }
    )
    mgap = Cpt(EpicsMotor, "Mono:DMM-Ax:HG}Mtr")
    mlm2 = DDC(
        {
            "p": (EpicsMotor, "Mono:DMM-Ax:Pitch}Mtr", {}),
            "fp": (EpicsMotor, "Mono:DMM-Ax:FP}Mtr", {}),
        }
    )
    zoff = Cpt(EpicsMotor, "Mono:DMM-Ax:TZ}Mtr")


class DCMBase(Device):
    pitch = Cpt(EpicsMotor, "Mono:HDCM-Ax:Pitch}Mtr")
    fine: ClassVar[dict[str, Cpt[EpicsMotor]]] = {
        "fpitch": Cpt(EpicsMotor, "Mono:HDCM-Ax:FP}Mtr"),
        "roll": Cpt(EpicsMotor, "Mono:HDCM-Ax:Roll}Mtr"),
    }
    h = Cpt(EpicsMotor, "Mono:HDCM-Ax:TX}Mtr")
    v = Cpt(EpicsMotor, "Mono:HDCM-Ax:TY}Mtr")


class _RealPosWithBragg(Protocol):
    bragg: float


class Energy(PseudoPositioner):
    bragg = Cpt(EpicsMotor, "Mono:HDCM-Ax:Bragg}Mtr")
    cgap = Cpt(EpicsMotor, "Mono:HDCM-Ax:HG}Mtr")
    # Synthetic Axis
    energy = Cpt(PseudoSingle, egu="keV")

    energy_egu = Cpt(Signal, None, add_prefix=(), value="keV", kind="config")
    motor_egu = Cpt(Signal, None, add_prefix=(), value="eV", kind="config")

    _u_gap_offset = 0

    # Energy "limits"
    _low = 4.63
    _high = 16.0

    # Set up constants
    Xoffset = 20.0  # mm
    d_111 = 3.135
    ANG_OVER_KEV = 12.3984

    # Motor enable flags
    move_u_gap = Cpt(Signal, None, add_prefix=(), value=True)
    move_c2_x = Cpt(Signal, None, add_prefix=(), value=True)
    harmonic = Cpt(Signal, None, add_prefix=(), value=0, kind="config")
    selected_harmonic = Cpt(Signal, None, add_prefix=(), value=0)

    # Experimental
    detune = Cpt(Signal, None, add_prefix=(), value=0)

    ele_list: ClassVar[list[str]] = [
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Cf",
    ]

    elements: ClassVar[dict[str, xrfC.XrfElement]] = {}
    for i in ele_list:
        elements[i] = xrfC.XrfElement(i)

    def __init__(
        self,
        *args: object,
        delta_bragg: int = 0,
        xoffset: int = 0,
        C2Xcal: int = 0,
        T2cal: int = 0,
        d_111: int = 0,
        **kwargs: object,
    ):
        super().__init__(*args, **kwargs)
        self._delta_bragg = delta_bragg
        self._xoffset = xoffset
        self._C2Xcal = C2Xcal
        self._T2cal = T2cal
        self._d_111 = d_111
        self.energy.readback.name = "energy"
        self.energy.setpoint.name = "energy_setpoint"
        calib_path = Path(__file__).parent
        # this is temporary, we need to figure out if there is a calib file for CDI
        calib_file = "../data/CDIUgapCalibration.txt"

        with Path.open(calib_path / calib_file) as f:
            next(f)
            uposlistIn = []
            elistIn = []
            for line in f:
                num = [float(x) for x in line.split()]
                # Check in case there is an extra line at the end of the calibration file
                if len(num) == 2:
                    uposlistIn.append(num[0])
                    elistIn.append(num[1])

        self.etoulookup = make_interp_spline(elistIn, uposlistIn)
        self.utoelookup = make_interp_spline(uposlistIn, elistIn)

    def energy_to_positions(
        self,
        target_energy: float,
        undulator_harmonic: int = 0,
        u_detune: float = 0.0,
    ):
        """Compute undulator and mono positions given a target energy

        Parameters
        ----------
        target_energy : float
            Target energy in keV
        undulator_harmonic : int
            The harmonic in the undulator to use
        u_detune : float
            Amount to 'mistune' the undulator in keV

        Returns
        -------
        bragg : float
            The angle to set the monocromotor in radians
        gap : float
            The gap position in millimeters
        C2X : float
            The C2X position in millimeters
        ugap : float
            The undulator gap position in microns
        """

        # Calculate Bragg RBV
        bragg_RBV = (
            np.arcsin((self.ANG_OVER_KEV / target_energy) / (2 * self.d_111))
            - self._delta_bragg
        )
        bragg = bragg_RBV + self._delta_bragg
        T2 = self._xoffset + np.sin(bragg * np.pi / 180) / np.sin(
            2 * bragg * np.pi / 180
        )
        dT2 = T2 - self._T2cal
        C2X = self._C2Xcal - dT2

        # Calculate C2X
        gap = self._xoffset / 2 / np.cos(bragg)
        ugap = float(self.etoulookup((target_energy + u_detune) / undulator_harmonic))
        ugap *= 1000
        ugap = ugap + self._u_gap_offset

        return bragg, gap, C2X, ugap

    @pseudo_position_argument
    def forward(self, pseudo_pos: object):
        energy = pseudo_pos.energy  # energy assumed in keV
        bragg, _, _, _ = self.energy_to_positions(energy)
        harmonic_raw = self.harmonic.get()
        if not isinstance(harmonic_raw, (int, float)):
            msg = "Harmonic value is not set"
            raise RuntimeError(msg)
        harmonic = int(harmonic_raw)
        if harmonic < 0 or ((harmonic % 2) == 0 and harmonic != 0):
            msg = f"The harmonic must be 0 or odd and positive, you set {harmonic}. Set `energy.harmonic` to a positive odd integer or 0."
            raise RuntimeError(msg)
        detune_raw = self.detune.get()
        if not isinstance(detune_raw, (int, float)):
            msg = "Detune value is not set"
            raise RuntimeError(msg)
        detune = float(detune_raw)
        if energy <= self._low:
            msg = f"The energy you entered is too low ({energy} keV). "
            msg += f"Minimum energy = {self._low:.1f} keV"
            raise ValueError(msg)
        if energy > self._high:
            if (energy < self._low * 1000) or (energy > self._high * 1000):
                # Energy is invalid
                msg = f"The requested photon energy is invalid ({energy} keV). "
                msg += f"Values must be in the range of {self._low:.1f} - {self._high:.1f} keV"
                raise ValueError(msg)
            # Energy is in eV
            energy = energy / 1000.0

        if harmonic < 3:
            harmonic = 3
            # Choose the right harmonic
            _, _, _, ugapcal = self.energy_to_positions(energy, harmonic, detune)
            # Try higher harmonics until the required gap is too small
            while True:
                _, _, _, ugapcal = self.energy_to_positions(
                    energy, harmonic + 2, detune
                )
                if ugapcal < self.u_gap.low_limit:
                    break
                harmonic += 2

        self.selected_harmonic.put(harmonic)

        # Compute where we would move everything to in a perfect world
        bragg, _, c2_x, u_gap = self.energy_to_positions(energy, harmonic, detune)

        # Sometimes move the crystal gap
        if not self.move_c2_x.get():
            c2_x = self.c2_x.position

        # Sometimes move the undulator
        if not self.move_u_gap.get():
            u_gap = self.u_gap.position

        return self.RealPosition(bragg=np.rad2deg(bragg), c2_x=c2_x, cgap=u_gap)

    @real_position_argument
    def inverse(self, real_pos: _RealPosWithBragg):
        bragg = np.deg2rad(real_pos.bragg)
        e = self.ANG_OVER_KEV / (2 * self.d_111 * np.sin(bragg))
        return self.PseudoPosition(energy=float(e))

    @pseudo_position_argument
    def set(self, position: list[int | float]):
        return super().set([float(_) for _ in position])

    def sync_with_epics(self):
        self.epics_d_spacing.put(self._d_111)
        self.epics_bragg_offset.put(self._delta_bragg)

    def retune_undulator(self):
        self.detune.put(0.0)
        self.move(self.energy.get()[0])

    def banner(self, str_list: list[str] | str, border: str = "-"):
        if not isinstance(str_list, list):
            str_list = [str_list]

        num = 2
        for str_val in str_list:
            num = max(len(str_val), num)

        print(border * (num + 2))  # noqa: T201
        for str_val in str_list:
            print(f" {str_val}")  # noqa: T201
        print(border * (num + 2), end="\n\n")  # noqa: T201

    def peakup(
        self,
        detectors: list[Any] | Any,
        start: float | None = None,
        min_step: float = 0.005,
        max_step: float = 0.5,
        *,
        motor: Any = None,
        target_fields: list[str] = ["bpm_current", "bpm_sum"],  # noqa: B006
        MAX_ITER: int = 100,
        verbose: bool = False,
    ) -> Generator[Msg, Any, None]:
        if motor is None:
            msg = "peakup requires a motor to move. Please provide a motor to optimize."
            raise ValueError(msg)

        detector_list = (
            list(detectors) if isinstance(detectors, list | tuple) else [detectors]
        )
        if not detector_list:
            msg = "peakup requires at least one detector. Please provide a detector or list of detectors to optimize on."
            raise ValueError(msg)

        if verbose:
            print("Additional debugging is enabled.")  # noqa: T201

        if not 0 < min_step < max_step:
            msg = f"Invalid step sizes: min_step={min_step}, max_step={max_step}. "
            raise ValueError(msg)

        # Grab starting point
        if start is None:
            start = motor.readback.get()
            if verbose:
                print(f"Starting position: {start:.4}")  # noqa: T201

        # Check foils
        if "bpm_current" in target_fields:
            E = self.energy.energy.readback.get()  # keV
            bpm = detector_list[
                0
            ]  # assume bpm is the first detector, need to figure out how to identify which detector is the bpm if there are multiple detectors
            y = bpm.y.user_readback.get()
            if np.abs(y - 0) < 1:
                foil = "in"
            else:
                foil = "out"
                self.banner("Unknown foil! Continuing without check!")

            if verbose:
                print(f"Energy: {E:.4}")  # noqa: T201
                print(f"Foil:\n  {y=:.4}\n  {foil=}")  # noqa: T201

        # Visualization
        livecb = []
        if verbose is False:
            livecb.append(LiveTable([motor.readback.name, *target_fields]))

        # Optimize on a given detector
        def optimize_on_det(target_field: str, x0: float) -> Generator[Msg, Any, float]:
            past_pos = x0
            next_pos = x0
            step = max_step
            past_I = None
            cur_I = 0
            cur_det = {}

            for _ in range(MAX_ITER):
                yield Msg("checkpoint")
                if verbose:
                    print(f"Moving {motor.name} to {next_pos:.4f}")  # noqa: T201
                yield from bps.mv(motor, next_pos)
                yield from bps.sleep(0.500)
                yield Msg("create", None, name="primary")
                for det in detector_list:
                    yield Msg("trigger", det, group="B")
                yield Msg("trigger", motor, group="B")
                yield Msg("wait", None, "B")
                for det in [*detector_list, motor]:
                    cur_det = yield Msg("read", det)
                    if target_field in cur_det:
                        cur_I = cur_det[target_field]["value"]
                        if verbose:
                            print(f"New measurement on {target_field}: {cur_I:.4}")  # noqa: T201
                yield Msg("save")
                # special case first first loop
            if past_I is None:
                past_I = cur_I
                next_pos += step
                if verbose:
                    print("past_I is None. Continuing...")  # noqa: T201

            dI = cur_I - past_I
            if verbose:
                print(f"{dI=:.4f}")  # noqa: T201
            if dI < 0:
                step = -0.6 * step
            else:
                past_pos = next_pos
                past_I = cur_I
            next_pos = past_pos + step
            if verbose:
                print(f"{next_pos=:.4f}")  # noqa: T201

            # Maximum found
            if np.abs(step) < min_step:
                if verbose:
                    print(  # noqa: T201
                        f"Maximum found for {target_field} at {x0:.4f}!\n  {step=:.4f}"
                    )
                return next_pos
            else:
                msg = "Optimization did not converge!"
                raise Exception(msg)

        # Start optimizing based on each detector field
        x0: float = start if start is not None else 0.0
        for target_field in target_fields:
            if verbose:
                print(f"Optimizing on detector {target_field}")  # noqa: T201
            x0 = yield from optimize_on_det(target_field, x0)
        return x0

    def set_roi(
        self,
        element: Any,
        line: str | None = None,
    ) -> None:
        cur_element = xrfC.XrfElement(element)
        e = ""
        if line is None:
            for e in ["ka1", "la1", "ma1"]:
                if cur_element.emission_line[e] < self.energy.energy.readback.get():
                    break
        elif line.lower() not in [
            "ka1",
            "ka2",
            "kb1",
            "la1",
            "la2",
            "lb1",
            "lb2",
            "lg1",
            "ma1",
        ]:
            print(f"WARNING: line {line} not in allowed lines.")  # noqa: T201
            print("Finding most appropriate line for the current energy.")  # noqa: T201
            line = None
        else:
            e = line.lower()

    def getemissionE(self, element: str, edge: str = "") -> float | None:
        cur_element = xrfC.XrfElement(element)
        if edge == "":
            print("Edge\tEnergy [keV]")  # noqa: T201
            for e in ["ka1", "ka2", "kb1", "la1", "la2", "lb1", "lb2", "lg1", "ma1"]:
                if (
                    cur_element.emission_line[e] < 25.0
                    and cur_element.emission_line[e] > 1.0
                ):
                    # print("{0:s}\t{1:8.2f}".format(e, cur_element.emission_line[e]))
                    print(f"{e}\t{cur_element.emission_line[e]:8.2f}")  # noqa: T201
            return 0.0
        return float(np.round(cur_element.emission_line[edge], 3))

    def getbindingE(self, element: str, edge: str | None = None) -> float | None:
        """
        Return edge energy in eV if edge is specified,
        otherwise return K and L edge energies and yields
        element     <symbol>        element symbol for target
        edge        ['k','l1','l2','l3']    return binding energy of this edge
        """
        if edge is None:
            y = [0.0, "k"]
            print("Edge\tEnergy [eV]\tYield")  # noqa: T201
            for i in ["k", "l1", "l2", "l3"]:
                print(  # noqa: T201
                    f"{i}\t"
                    f"{xrfC.XrayLibWrap(self.elements[element].Z, 'binding_e')[i] * 1000.0:8.2f}\t"
                    f"{xrfC.XrayLibWrap(self.elements[element].Z, 'yield')[i]:5.3f}"
                )
                if (
                    y[0] < xrfC.XrayLibWrap(self.elements[element].Z, "yield")[i]
                    and xrfC.XrayLibWrap(self.elements[element].Z, "binding_e")[i]
                    < 25.0
                ):
                    y[0] = xrfC.XrayLibWrap(self.elements[element].Z, "yield")[i]
                    y[1] = i
            return np.round(
                xrfC.XrayLibWrap(self.elements[element].Z, "binding_e")[y[1]] * 1000.0,
                3,
            )
        return np.round(
            xrfC.XrayLibWrap(self.elements[element].Z, "binding_e")[edge] * 1000.0, 3
        )

    def mono_peakup(
        self,
        element: str = "none",
        peakup: bool = True,
        detectors: list[Device] | None = None,
        motors: list[Device] | None = None,
        start: None = None,
        targets: list[str] | None = None,
    ) -> Generator[Msg, None, None]:
        """
            First draft of the mono peakup scan
            Need more info about the axis to be scanned, the move ID, and which detector will be used for feedback.
        Args:
            element (string): element name
            acquisition_time (float, optional): _description_. Defaults to 1.0.
            peakup (bool, optional): _description_. Defaults to True.
        """

        if element != "none":
            energy_x = self.getbindingE(element)
            yield from bps.mov(self.energy, energy_x)
            self.set_roi(1, element)

        if peakup:
            yield from bps.sleep(5)
            yield from self.peakup(
                detectors=detectors,
                motor=motors,
                start=start,
                target_fields=targets,
                verbose=True,
            )


class VPM(Device):
    fs = DDC(
        {
            "y": (EpicsMotor, "FS:VPM-Ax:Y}Mtr", {}),
        }
    )

    slit = DDC(
        {
            "hg": (EpicsMotor, "Slt:VPM-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:VPM-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:VPM-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:VPM-Ax:VC}Mtr", {}),
        }
    )
    mir = DDC(
        {
            "us_j": (EpicsMotor, "Mir:VPM-Ax:YUC}Mtr", {}),
            "ib_j": (EpicsMotor, "Mir:VPM-Ax:YDI}Mtr", {}),
            "ob_j": (EpicsMotor, "Mir:VPM-Ax:YDO}Mtr", {}),
            "p": (EpicsMotor, "Mir:VPM-Ax:Pitch}Mtr", {}),
            "r": (EpicsMotor, "Mir:VPM-Ax:Roll}Mtr", {}),
            "y": (EpicsMotor, "Mir:VPM-Ax:TY}Mtr", {}),
            "x": (EpicsMotorRO, "Mir:VPM-Ax:TX}Mtr", {}),
            "yaw": (EpicsMotorRO, "Mir:VPM-Ax:Yaw}Mtr", {}),
            "us_lt": (EpicsMotorRO, "Mir:VPM-Ax:XU}Mtr", {}),
            "ds_lt": (EpicsMotorRO, "Mir:VPM-Ax:XD}Mtr", {}),
            "us_b": (EpicsMotorRO, "Mir:VPM-Ax:UB}Mtr", {}),
            "ds_b": (EpicsMotorRO, "Mir:VPM-Ax:DB}Mtr", {}),
            "bend": (EpicsMotorRO, "Mir:VPM-Ax:Bnd}Mtr", {}),
            "bend_off": (EpicsMotorRO, "Mir:VPM-Ax:BndOff}Mtr", {}),
        }
    )


class HPM(Device):
    fs = DDC(
        {
            "y": (EpicsMotor, "FS:HPM-Ax:Y}Mtr", {}),
        }
    )

    slit = DDC(
        {
            "hg": (EpicsMotor, "Slt:HPM-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:HPM-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:HPM-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:HPM-Ax:VC}Mtr", {}),
        }
    )
    mir = DDC(
        {
            "us_j": (EpicsMotor, "Mir:HPM-Ax:YUC}Mtr", {}),
            "ib_j": (EpicsMotor, "Mir:HPM-Ax:YDI}Mtr", {}),
            "ob_j": (EpicsMotor, "Mir:HPM-Ax:YDO}Mtr", {}),
            "p": (EpicsMotor, "Mir:HPM-Ax:Pitch}Mtr", {}),
            "r": (EpicsMotor, "Mir:HPM-Ax:Roll}Mtr", {}),
            "y": (EpicsMotor, "Mir:HPM-Ax:TY}Mtr", {}),
            "bend": (EpicsMotor, "Mir:HPM-Ax:Bnd}Mtr", {}),
            "bend_off": (EpicsMotor, "Mir:HPM-Ax:BndOff}Mtr", {}),
            "us_x": (EpicsMotor, "Mir:HPM-Ax:XU}Mtr", {}),
            "ds_x": (EpicsMotor, "Mir:HPM-Ax:XD}Mtr", {}),
            "x": (EpicsMotor, "Mir:HPM-Ax:TX}Mtr", {}),
            "yaw": (EpicsMotorRO, "Mir:HPM-Ax:Yaw}Mtr", {}),
            "us_lt": (EpicsMotorRO, "Mir:HPM-Ax:XU}Mtr", {}),
            "ds_lt": (EpicsMotorRO, "Mir:HPM-Ax:XD}Mtr", {}),
            "us_b": (EpicsMotorRO, "Mir:HPM-Ax:UB}Mtr", {}),
            "ds_b": (EpicsMotorRO, "Mir:HPM-Ax:DB}Mtr", {}),
        }
    )


class DM2(Device):
    fs = DDC(
        {
            "y": (EpicsMotor, "FS:DM2-Ax:Y}Mtr", {}),
        }
    )
    foil = Cpt(EpicsMotor, "IM:DM2-Ax:Y}Mtr")


class DM3(Device):
    slit = DDC(
        {
            "ib": (EpicsMotor, "Slt:DM3-Ax:I}Mtr", {}),
            "ob": (EpicsMotor, "Slt:DM3-Ax:O}Mtr", {}),
            "bb": (EpicsMotor, "Slt:DM3-Ax:B}Mtr", {}),
            "tb": (EpicsMotor, "Slt:DM3-Ax:T}Mtr", {}),
            "hg": (EpicsMotor, "Slt:DM3-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:DM3-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:DM3-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:DM3-Ax:VC}Mtr", {}),
        }
    )
    bpm = DDC(
        {
            "x": (EpicsMotor, "BPM:DM3-Ax:TX}Mtr", {}),
            "y": (EpicsMotor, "BPM:DM3-Ax:TY}Mtr", {}),
            "foil": (EpicsMotor, "BPM:DM3-Ax:Foil}Mtr", {}),
        }
    )
    fs = DDC(
        {
            "y": (EpicsMotor, "FS:DM3-Ax:FS}Mtr", {}),
        }
    )


class VKB(Device):
    slit = DDC(
        {
            "hg": (EpicsMotor, "Slt:KBv-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:KBv-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:KBv-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:KBv-Ax:VC}Mtr", {}),
        }
    )
    mir = DDC(
        {
            "us_j": (EpicsMotor, "Mir:KBv-Ax:YUC}Mtr", {}),
            "ib_j": (EpicsMotor, "Mir:KBv-Ax:YDI}Mtr", {}),
            "ob_j": (EpicsMotor, "Mir:KBv-Ax:YDO}Mtr", {}),
            "yaw": (EpicsMotor, "Mir:KBv-Ax:Yaw}Mtr", {}),
            "r": (EpicsMotor, "Mir:KBv-Ax:Roll}Mtr", {}),
            "y": (EpicsMotor, "Mir:KBv-Ax:TY}Mtr", {}),
            "x": (EpicsMotor, "Mir:KBv-Ax:TX}Mtr", {}),
            "z": (EpicsMotor, "Mir:KBv-Ax:TZ}Mtr", {}),
            "p": (EpicsMotor, "Mir:KBv-Ax:Pitch}Mtr", {}),
        }
    )
    fs = DDC(
        {
            "y": (EpicsMotor, "Mir:KBv-Ax:FS}Mtr", {}),
        }
    )


class HKB(Device):
    slit = DDC(
        {
            "hg": (EpicsMotor, "Slt:KBh-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:KBh-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:KBh-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:KBh-Ax:VC}Mtr", {}),
        }
    )
    mir = DDC(
        {
            "us_j": (EpicsMotor, "Mir:KBh-Ax:YUC}Mtr", {}),
            "ib_j": (EpicsMotor, "Mir:KBh-Ax:YDI}Mtr", {}),
            "ob_j": (EpicsMotor, "Mir:KBh-Ax:YDO}Mtr", {}),
            "yaw": (EpicsMotor, "Mir:KBh-Ax:Yaw}Mtr", {}),
            "r": (EpicsMotor, "Mir:KBh-Ax:Roll}Mtr", {}),
            "y": (EpicsMotor, "Mir:KBh-Ax:TY}Mtr", {}),
            "x": (EpicsMotor, "Mir:KBh-Ax:TX}Mtr", {}),
            "z": (EpicsMotor, "Mir:KBh-Ax:TZ}Mtr", {}),
            "p": (EpicsMotor, "Mir:KBh-Ax:Pitch}Mtr", {}),
        }
    )
    fs = DDC(
        {
            "y": (EpicsMotor, "Mir:KBh-Ax:FS}Mtr", {}),
        }
    )


class KB(Device):
    vkb = Cpt(VKB, "")
    hkb = Cpt(HKB, "")
    win = DDC(
        {
            "x": (EpicsMotor, "Wnd:Exit-Ax:TX}Mtr", {}),
            "y": (EpicsMotor, "Wnd:Exit-Ax:TY}Mtr", {}),
        }
    )


class DM4(Device):
    bpm = DDC(
        {
            "x": (EpicsMotor, "BPM:DM4-Ax:TX}Mtr", {}),
            "y": (EpicsMotor, "BPM:DM4-Ax:TY}Mtr", {}),
        }
    )


class SAM(Device):
    c_sm = DDC(
        {
            "lrx": (EpicsMotor, "Gon:1-Ax:Rx1}Mtr", {}),
            "lrz": (EpicsMotor, "Gon:1-Ax:Rz1}Mtr", {}),
        }
    )
    c_lg = DDC(
        {
            "lrx": (EpicsMotor, "Gon:1-Ax:Rx2}Mtr", {}),
            "lrz": (EpicsMotor, "Gon:1-Ax:Rz2}Mtr", {}),
        }
    )
    ly = Cpt(EpicsMotor, "Gon:1-Ax:Y}Mtr")
    ry = Cpt(EpicsMotor, "Gon:1-Ax:Ry}Mtr")
    t_sm = DDC(
        {
            "lx": (EpicsMotor, "Gon:1-Ax:X1}Mtr", {}),
            "lz": (EpicsMotor, "Gon:1-Ax:Z1}Mtr", {}),
        }
    )
    t_lg = DDC(
        {
            "lx": (EpicsMotor, "Gon:1-Ax:X2}Mtr", {}),
            "lz": (EpicsMotor, "Gon:1-Ax:Z2}Mtr", {}),
        }
    )
    lfx = Cpt(EpicsMotor, "Gon:1-Ax:XP}Mtr")
    lfy = Cpt(EpicsMotor, "Gon:1-Ax:YP}Mtr")
    lfz = Cpt(EpicsMotor, "Gon:1-Ax:ZP}Mtr")


class GON(Device):
    sam = Cpt(SAM, "")
    align = DDC(
        {
            "rx": (EpicsMotor, "Gon:1-Ax:Rx3}Mtr", {}),
            "rz": (EpicsMotor, "Gon:1-Ax:Rz3}Mtr", {}),
            "x": (EpicsMotor, "Gon:1-Ax:X3}Mtr", {}),
            "y": (EpicsMotor, "Gon:1-Ax:Y3}Mtr", {}),
            "z": (EpicsMotor, "Gon:1-Ax:Z3}Mtr", {}),
        }
    )
    fs = DDC(
        {
            "y": (EpicsMotor, "Gon:1-Ax:Visual}Mtr", {}),
        }
    )


class BCU(Device):
    slit_us = DDC(
        {
            "ib": (EpicsMotor, "Slt:BCUU-Ax:I}Mtr", {}),
            "ob": (EpicsMotor, "Slt:BCUU-Ax:O}Mtr", {}),
            "bb": (EpicsMotor, "Slt:BCUU-Ax:B}Mtr", {}),
            "tb": (EpicsMotor, "Slt:BCUU-Ax:T}Mtr", {}),
            "hg": (EpicsMotor, "Slt:BCUU-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:BCUU-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:BCUU-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:BCUU-Ax:VC}Mtr", {}),
        }
    )
    slit_ds = DDC(
        {
            "ib": (EpicsMotor, "Slt:BCUD-Ax:I}Mtr", {}),
            "ob": (EpicsMotor, "Slt:BCUD-Ax:O}Mtr", {}),
            "bb": (EpicsMotor, "Slt:BCUD-Ax:B}Mtr", {}),
            "tb": (EpicsMotor, "Slt:BCUD-Ax:T}Mtr", {}),
            "hg": (EpicsMotor, "Slt:BCUD-Ax:HG}Mtr", {}),
            "hc": (EpicsMotor, "Slt:BCUD-Ax:HC}Mtr", {}),
            "vg": (EpicsMotor, "Slt:BCUD-Ax:VG}Mtr", {}),
            "vc": (EpicsMotor, "Slt:BCUD-Ax:VC}Mtr", {}),
        }
    )
    ilcam = DDC(
        {
            "x": (EpicsMotor, "Qstar:1-Ax:1}Mtr", {}),
            "y": (EpicsMotor, "Qstar:1-Ax:2}Mtr", {}),
            "z": (EpicsMotor, "Qstar:1-Ax:3}Mtr", {}),
        }
    )
