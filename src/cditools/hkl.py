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

class CDIDiffractometer(DiffractometerBase):

    def __init__(self, gon: GON, energy: Energy, **kwargs):
        # configure goniometer ophyd signals
        self.mu = gon.sam.ry
        self.chi = gon.sam.c_lg.lrx
        self.phi = gon.sam.c_lg.lrz
        self.omega2 = gon.sam.c_sm.lrx
        self.chi2 = gon.sam.c_sm.lrz

        # configure detector ophyd signals
        # self.gamma1 = ...
        # self.delta1 = ...

        # configure energy readout from Energy device
        # self.beam = {'class': PseudoMonochromator,
        #              'energy_device': energy}

        super().__init__(
            prefix='',
            name='cdi',
            geometry=register_geometry_file(CUSTOM_GEOMETRY_YAML_PATH),
            solver='ad_hoc',
            **kwargs
        )

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


