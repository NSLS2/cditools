from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import xrayutilities as xu
from event_model import DataKey  # type: ignore[import-untyped]
from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    DeviceVector,
    StandardReadable,
    StrictEnum,
)
from ophyd_async.epics.core import EpicsDevice, epics_signal_r, epics_signal_rw

from cditools.motors import Energy


@dataclass
class AttenuatorCombination:
    transmission: float
    attenuators: list[int]

    @property
    def attenuation(self):
        return 1 - self.transmission


class AttenuatorStatusEnum(StrictEnum):
    LOW = "Low"  # off / not obstructing
    HIGH = "High"  # on / obstructing


class Attenuator(EpicsDevice, AsyncMovable[AttenuatorStatusEnum]):
    def __init__(self, prefix: str, num: int, material: str, thickness: float):
        """
        Parameters
        ----------
        prefix : str
            The common prefix for the attenuator bank
        num : int
            An integer denoting which attenuator within the bank this is
        thickness : float
            The thickness of the attenuator in microns

        Attributes
        ----------
        position : SignalRW[AttenuatorStatusEnum]
            The read / write PV to open and close the attenuator and get
            the current state of the attenuator
        mode : SignalRW[bool]
        in_status : SignalR[AttenuatorStatusEnum]
        """
        self.prefix = prefix
        self.num = num
        self.filter_material = getattr(xu.materials, material)
        self.thickness = thickness  # microns

        self.position = epics_signal_rw(
            AttenuatorStatusEnum,
            f"{self.prefix}:DO{self.num}-Sts",
            write_pv=f"{self.prefix}:DO{self.num}-Cmd",
        )
        self.mode = epics_signal_rw(bool, f"{self.prefix}:DIO{self.num}-Mode")
        self.in_status = epics_signal_r(
            AttenuatorStatusEnum, f"{self.prefix}:DI{self.num}-Sts"
        )

        super().__init__(prefix=self.prefix)

    def __repr__(self):
        return f"{self.thickness!s} microns, {self.filter_material}"

    @property
    def name(self):
        return f"attenuator_{self.num}"

    @AsyncStatus.wrap
    async def set(self, value: AttenuatorStatusEnum):
        await self.position.set(value)

    async def open(self):
        """Open means open to allowing the beam to pass unobstructed"""
        await self.position.set(AttenuatorStatusEnum.LOW)

    async def close(self):
        """Closed means obstructing the beam"""
        await self.position.set(AttenuatorStatusEnum.HIGH)

    def attenuation(self, photon_energy: float, egu: str = "KeV"):
        """Attenuation is the fraction of the beam removed"""
        return 1 - self.transmission(photon_energy, egu=egu)

    def transmission(self, photon_energy: float, egu: str = "KeV"):
        """Transmission is the fraction of beam remaining"""
        abs_len = self._absorption_length(photon_energy, egu=egu)
        return np.exp(-self.thickness / abs_len)

    def _absorption_length(self, photon_energy: float, egu: str = "KeV") -> float:
        """
        Calculates L, the characteristic absorption length of this material,
        at this beam energy.

        Parameters
        ----------
        photon energy : float
            The beam energy
        egu : {'KeV', 'eV'}
            The engineering units of the beam energy

        Returns
        -------
        float
            The characteristic absorption length of the filter material (microns)
        """
        if egu == "KeV":
            photon_energy = photon_energy * 1e3
        elif egu != "eV":
            msg = f"Photon energy units must be eV or KeV (not {egu=})"
            raise RuntimeError(msg)
        return self.filter_material.absorption_length(photon_energy)  # type: ignore[reportArgumentType]


class AttenuatorBank(StandardReadable, EpicsDevice, AsyncMovable[float]):
    """
    The ioc for the iologik1 lives on xf09id1-inst-ioc1.nsls2.bnl.gov
    """

    def __init__(
        self, prefix: str, atten_configs: list[tuple[str, float]], energy: Energy
    ):
        self.prefix = prefix
        self.energy = energy

        with self.add_children_as_readables():
            self.attenuators = DeviceVector(
                {
                    i: Attenuator(self.prefix, i, material, thickness)
                    for i, (material, thickness) in enumerate(atten_configs, start=1)
                }
            )
        super().__init__(prefix=self.prefix)

    def get_photon_energy(self):
        return self.energy.energy.readback.get()

    def get_egu(self):
        return self.energy.egu

    async def read(self):  # type: ignore[reportUnknownParameterType]
        """
        Polls the bluesky energy object for the current beam energy, and
        returns that energy, each filter position, each transmission, and
        the total transmission.
        """
        status = OrderedDict()
        active_attens = []
        energy = self.get_photon_energy()
        egu = self.get_egu()
        positions = await asyncio.gather(
            *(a.position.get_value() for _, a in self.attenuators.items())
        )
        for i, pos in zip(self.attenuators, positions):
            atten = self.attenuators[i]
            is_active = pos == AttenuatorStatusEnum.HIGH
            if is_active:
                active_attens.append(atten)
            transmission = atten.transmission(energy, egu) if is_active else 0
            status[atten.name] = {"active": is_active, "transmission": transmission}
        status["photon_energy"] = energy
        status["egu"] = egu
        status["total_transmission"] = self._calculate_total_transmission(
            energy, *active_attens
        )
        return status

    async def describe(self) -> OrderedDict[str, DataKey]:
        """Describe the structure of values returned by read()."""

        description = OrderedDict()

        for atten in self.attenuators.values():
            description[atten.name] = DataKey(
                source=atten.position.source,
                dtype="string",
                shape=[],
            )
        energy_source = getattr(
            self.energy.energy.readback,
            "source",
            f"ca://{self.prefix}:photon_energy",
        )
        description["photon_energy"] = DataKey(
            source=energy_source,
            dtype="number",
            shape=[],
        )
        description["egu"] = DataKey(
            source=f"ca://{self.prefix}:egu",
            dtype="string",
            shape=[],
        )
        description["total_transmission"] = DataKey(
            source=f"ca://{self.prefix}:total_transmission",
            dtype="number",
            shape=[],
        )

        return description

    @AsyncStatus.wrap
    async def set(self, value: float):
        """Set the transmission for the attenuator bank"""
        attenuation_combination = self.find_closest_transmission(
            self.get_photon_energy(), value
        )
        coros = []
        for (
            num,
            atten,
        ) in self.attenuators.items():
            if num in attenuation_combination.attenuators:
                coros.append(atten.close())
            else:
                coros.append(atten.open())
        await asyncio.gather(*coros)

    def find_closest_transmission(
        self, photon_energy: float, target_transmission: float
    ) -> AttenuatorCombination:
        available_attenuations = self._calculate_available_transmissions(photon_energy)
        best_idx = np.argmin(
            [abs(target_transmission - _.transmission) for _ in available_attenuations]
        )
        return available_attenuations[best_idx]

    def _calculate_available_transmissions(
        self, photon_energy: float
    ) -> list[AttenuatorCombination]:
        """
        Calculates all possible transmissions for the attenuator bank, using
        the powerset of the available attenuators. The result is not sorted,
        as it is more efficient to scan linearly each time for the closest
        match.
        """
        available_transmissions = []
        for combination in self._powerset():
            attens = [self.attenuators[a] for a in self.attenuators if a in combination]
            total_transmission = self._calculate_total_transmission(
                photon_energy, *attens
            )
            available_transmissions.append(
                AttenuatorCombination(total_transmission, combination)
            )
        return available_transmissions

    def _calculate_total_transmission(
        self, photon_energy: float, *attenuators: Attenuator
    ) -> float:
        transmissions = [
            a.transmission(photon_energy, self.get_egu()) for a in attenuators
        ]
        return round(float(math.prod(transmissions)), 3)

    def _powerset(self) -> list[list[int]]:
        """
        This is a famously O(n*2^n) problem.
        """
        powerset = []
        for i in range(1 << len(self.attenuators)):
            combination = []
            for j in range(len(self.attenuators)):
                if i & (1 << j):
                    combination.append(j + 1)  # +1 because attenuators are 1-indexed
            powerset.append(combination)
        return powerset
