from pathlib import Path
from typing import Any

from hklpy2.diffract import DiffractometerBase
from ad_hoc_diffractometer import register_geometry_file
from hklpy2.diffract import DiffractometerBase
from hklpy2.incident import WavelengthXray

from cditools.motors import Energy, GON

CUSTOM_GEOMETRY_YAML_PATH = Path(__file__).resolve().parent / "config/cdi-geometry.yml"

class PseudoMonochromator(WavelengthXray):
    """Incident beam whose energy tracks an ophyd ``PseudoSingle``.

    The DCM publishes no energy PV, so energy is computed by the
    :class:`~cditools.motors.Energy` pseudo-positioner.  Subscribing to its
    readback keeps ``self.energy`` (and hence ``self.wavelength``) live, which
    in turn drives hklpy2's solver-update machinery.
    """

    def __init__(
        self,
        prefix: str = "",
        *,
        energy_device: Energy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(prefix, energy_units="keV", **kwargs)
        self._energy_device = energy_device
        if energy_device is not None:
            energy_device.energy.readback.subscribe(self._energy_changed)

    def _energy_changed(self, value: float, **_kwargs: Any) -> None:
        self.energy.put(value)

# class CDIDiffractometer(DiffractometerBase):

#     def __init__(self, gon: GON, energy: Energy, **kwargs):
#         # configure goniometer ophyd signals
#         self.mu = gon.sam.ry
#         self.chi = gon.sam.c_lg.lrx
#         self.phi = gon.sam.c_lg.lrz
#         self.omega2 = gon.sam.c_sm.lrx
#         self.chi2 = gon.sam.c_sm.lrz

#         # configure detector ophyd signals
#         # self.gamma1 = ...
#         # self.delta1 = ...

#         # configure energy readout from Energy device
#         # self.beam = {'class': PseudoMonochromator,
#         #              'energy_device': energy}

#         super().__init__(
#             prefix='',
#             name='cdi',
#             geometry=register_geometry_file(CUSTOM_GEOMETRY_YAML_PATH),
#             solver='ad_hoc',
#             reals=["mu", "chi", "phi", "omega2", "chi2", "gamma1", "delta1"],
#             pseudos=["h", "k", "l"],
#             **kwargs
#         )

# def initialize_hkl_diffractometer(name: str = "cdi") -> DiffractometerBase:
#     # initialize hklpy2 diffractometer
#     cdi = hklpy2.creator(prefix = "XF:09IDC-OP:1{", 
#                          name=name, 
#                          geometry=register_geometry_file(CUSTOM_GEOMETRY_YAML_PATH), 
#                          solver='ad_hoc',
#                          reals={
#                              "mu": ,
#                              "chi": ,
#                              "phi": ,
#                              "omega2": ,
#                              "chi2": ,
#                              "gamma1": ,
#                              "delta1": ,
#                          })

#     # configure energy in keV
#     cdi.beam.energy.put(energy_pseudo.energy)
    
    
#     return cdi

from typing import ClassVar

from ophyd import Component as Cpt
from ophyd import EpicsMotor
from ophyd import Kind
from ophyd import SoftPositioner

import hklpy2
from hklpy2.diffract import Hklpy2PseudoAxis
from hklpy2.incident import EpicsMonochromatorRO

NORMAL_HINTED = Kind.hinted | Kind.normal


class CDIDiffractometer(DiffractometerBase):

    beam = Cpt(
        EpicsMonochromatorRO,
        "",
        source_type="Simulated read-only EPICS Monochromator",
        pv_energy="BraggERdbkAO",  # the energy readback PV
        energy_units="keV",
        pv_wavelength="BraggLambdaRdbkAO",  # the wavelength readback PV
        wavelength_units="angstrom",
        wavelength_deadband=0.000_150,
        kind=NORMAL_HINTED,
    )

    # Pseudo-space axes, in order expected by hkl_soleil E4CV, engine="hkl"
    h = Cpt(Hklpy2PseudoAxis, "", kind=NORMAL_HINTED)
    k = Cpt(Hklpy2PseudoAxis, "", kind=NORMAL_HINTED)
    l = Cpt(Hklpy2PseudoAxis, "", kind=NORMAL_HINTED)

    # Real-space axes, in our own order..
    # Use different names than the solver for some axes
    mu = Cpt(EpicsMotor, "Gon:1-Ax:Ry}Mtr", kind=NORMAL_HINTED)
    chi = Cpt(EpicsMotor, "Gon:1-Ax:Rx2}Mtr", kind=NORMAL_HINTED)
    phi = Cpt(EpicsMotor, "Gon:1-Ax:Rz2}Mtr", kind=NORMAL_HINTED)
    omega2 = Cpt(EpicsMotor, "Gon:1-Ax:Rx1}Mtr", kind=NORMAL_HINTED)
    chi2 = Cpt(EpicsMotor, "Gon:1-Ax:Rz1}Mtr", kind=NORMAL_HINTED)
    gamma1 = Cpt(EpicsMotor, "", kind=NORMAL_HINTED)
    delta1 = Cpt(EpicsMotor, "", kind=NORMAL_HINTED)

    # Just the axes in expected order by hkl_soleil E4CV.
    _pseudo: ClassVar[list[str]] = ["h", "k", "l"]
    _real: ClassVar[list[str]] = ["mu", "chi", "phi", "omega2", "chi2", "gamma1", "delta1"]


    def __init__(self, **kwargs):
        # kwargs["prefix"] = prefix
        super().__init__(
            prefix="XF:09IDC-OP:1{",
            solver="ad_hoc",
            name='cdi',
            geometry=register_geometry_file(CUSTOM_GEOMETRY_YAML_PATH),
            pseudos=["h", "k", "l"],
            reals=["mu", "chi", "phi", "omega2", "chi2", "gamma1", "delta1"],
            **kwargs,
        )


