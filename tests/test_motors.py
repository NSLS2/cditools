from __future__ import annotations

import os
import time
from collections.abc import Generator
from subprocess import PIPE, Popen

import pytest

from cditools.motors import (
    BPMDM3,
    BPMDM4,
    FSDM2,
    FSDM3,
    FSHPM,
    FSVPM,
    IMDM2,
    FltrDM1,
    Gon1,
    MirHPM,
    MirKBh,
    MirKBv,
    MirVPM,
    MonoDMM,
    MonoHDCM,
    Qstar1,
    SltBCUD,
    SltBCUU,
    SltDM3,
    SltHPM,
    SltVPM,
    SltWB1,
    WndExit,
)


@pytest.fixture(scope="session")
def black_hole_ioc() -> Generator[None, None, None]:
    os.environ["EPICS_CA_ADDR_LIST"] = "127.0.0.1"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    p = Popen(["black-hole-ioc", "--interfaces", "127.0.0.1"], stdout=PIPE)
    if not p.stdout:
        msg = "Failed to start black-hole-ioc"
        raise RuntimeError(msg)
    start_time = time.time()
    timeout = 30  # seconds
    while True:
        line = p.stdout.readline().decode("utf-8")
        if line.strip().endswith("Server startup complete."):
            break
        if time.time() - start_time > timeout:
            p.terminate()
            msg = "Timeout waiting for black-hole-ioc to start"
            raise RuntimeError(msg)
        if not line:  # Process ended without expected output
            msg = "black-hole-ioc process ended unexpectedly"
            raise RuntimeError(msg)
        time.sleep(0.1)  # Small delay to prevent CPU spinning
    yield
    p.terminate()
    p.wait()


def test_motors_can_connect(black_hole_ioc: None) -> None:
    slt_wb1 = SltWB1(prefix="XF:09IDA-OP:1{Slt:WB1", name="slt_wb1")
    slt_wb1.wait_for_connection(timeout=10.0)

    fltr_dm1 = FltrDM1(prefix="XF:09IDA-OP:1{Fltr:DM1", name="fltr_dm1")
    fltr_dm1.wait_for_connection(timeout=10.0)

    fsvpm = FSVPM(prefix="XF:09IDA-OP:1{FS:VPM", name="fsvpm")
    fsvpm.wait_for_connection(timeout=10.0)

    fshpm = FSHPM(prefix="XF:09IDA-OP:1{FS:HPM", name="fshpm")
    fshpm.wait_for_connection(timeout=10.0)

    fsdm2 = FSDM2(prefix="XF:09IDA-OP:1{FS:DM2", name="fsdm2")
    fsdm2.wait_for_connection(timeout=10.0)

    mono_dmm = MonoDMM(prefix="XF:09IDA-OP:1{Mono:DMM", name="mono_dmm")
    mono_dmm.wait_for_connection(timeout=10.0)

    mono_hdcm = MonoHDCM(prefix="XF:09IDA-OP:1{Mono:HDCM", name="mono_hdcm")
    mono_hdcm.wait_for_connection(timeout=10.0)

    slt_vpm = SltVPM(prefix="XF:09IDA-OP:1{Slt:VPM", name="slt_vpm")
    slt_vpm.wait_for_connection(timeout=10.0)

    slt_hpm = SltHPM(prefix="XF:09IDA-OP:1{Slt:HPM", name="slt_hpm")
    slt_hpm.wait_for_connection(timeout=10.0)

    mir_vpm = MirVPM(prefix="XF:09IDA-OP:1{Mir:VPM", name="mir_vpm")
    mir_vpm.wait_for_connection(timeout=10.0)

    imdm2 = IMDM2(prefix="XF:09IDA-OP:1{IM:DM2", name="imdm2")
    imdm2.wait_for_connection(timeout=10.0)

    mir_hpm = MirHPM(prefix="XF:09IDA-OP:1{Mir:HPM", name="mir_hpm")
    mir_hpm.wait_for_connection(timeout=10.0)

    slt_dm3 = SltDM3(prefix="XF:09IDB-OP:1{Slt:DM3", name="slt_dm3")
    slt_dm3.wait_for_connection(timeout=10.0)

    bpm_dm3 = BPMDM3(prefix="XF:09IDB-OP:1{BPM:DM3", name="bpm_dm3")
    bpm_dm3.wait_for_connection(timeout=10.0)

    fsdm3 = FSDM3(prefix="XF:09IDB-OP:1{FS:DM3", name="fsdm3")
    fsdm3.wait_for_connection(timeout=10.0)

    mir_kbv = MirKBv(prefix="XF:09IDC-OP:1{Mir:KBv", name="mir_kbv")
    mir_kbv.wait_for_connection(timeout=10.0)

    mir_kbh = MirKBh(prefix="XF:09IDC-OP:1{Mir:KBh", name="mir_kbh")
    mir_kbh.wait_for_connection(timeout=10.0)

    bpm_dm4 = BPMDM4(prefix="XF:09IDC-OP:1{BPM:DM4", name="bpm_dm4")
    bpm_dm4.wait_for_connection(timeout=10.0)

    wnd_exit = WndExit(prefix="XF:09IDC-OP:1{Wnd:Exit", name="wnd_exit")
    wnd_exit.wait_for_connection(timeout=10.0)

    gon1 = Gon1(prefix="XF:09IDC-OP:1{Gon:1", name="gon1")
    gon1.wait_for_connection(timeout=10.0)

    slt_bcuu = SltBCUU(prefix="XF:09IDC-OP:1{Slt:BCUU", name="slt_bcuu")
    slt_bcuu.wait_for_connection(timeout=10.0)

    slt_bcud = SltBCUD(prefix="XF:09IDC-OP:1{Slt:BCUD", name="slt_bcud")
    slt_bcud.wait_for_connection(timeout=10.0)

    qstar1 = Qstar1(prefix="XF:09IDC-OP:1{Qstar:1", name="qstar1")
    qstar1.wait_for_connection(timeout=10.0)
