from __future__ import annotations

from ophyd.device import Component as Cpt
from ophyd.device import Device
from ophyd.device import FormattedComponent as FCpt
from ophyd.pv_positioner import PVPositioner
from ophyd.signal import EpicsSignal, EpicsSignalRO


class TDMSAxis(PVPositioner):
    setpoint = Cpt(EpicsSignal, "MTR:VAL-SP")
    readback = Cpt(EpicsSignalRO, "MTR:RBV-RB0", kind="hinted")
    actuate = Cpt(EpicsSignal, "MTR:GO-CMD", kind="omitted")
    done = Cpt(EpicsSignalRO, "MTR:INPOS-STS", kind="omitted")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.readback.name = self.name


class TDMSTower(Device):
    tx = FCpt(TDMSAxis, "XF:09IDC-ES:1{{TDMS:T{self._num}-Ax:TX}}")
    ty = FCpt(TDMSAxis, "XF:09IDC-ES:1{{TDMS:T{self._num}-Ax:TY}}")
    tz = FCpt(TDMSAxis, "XF:09IDC-ES:1{{TDMS:A{self._num}-Ax:TZ}}")

    camay = FCpt(TDMSAxis, "XF:09IDC-ES:1{{TDMS:T{self._num}-Ax:CAM_AY}}")

    # TODO replace with TDMSAxis when motions get comissioned
    ax = FCpt(EpicsSignalRO, "XF:09IDC-ES:1{{TDMS:T{self._num}-Ax:AX}}MTR:RBV-RB0")
    ay = FCpt(EpicsSignalRO, "XF:09IDC-ES:1{{TDMS:A{self._num}-Ax:AY}}MTR:RBV-RB0")
    az = FCpt(EpicsSignalRO, "XF:09IDC-ES:1{{TDMS:T{self._num}-Ax:AZ}}MTR:RBV-RB0")

    def __init__(self, *, num, **kwargs):
        self._num = num
        super().__init__(**kwargs)


T1 = TDMSTower(name="T1", num=1)
T2 = TDMSTower(name="T2", num=2)
