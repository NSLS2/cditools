from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import xrayutilities as xu
from bluesky.protocols import Hints
from event_model import DataKey  # type: ignore[import-untyped]
from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    DeviceVector,
    StandardReadable,
    StrictEnum,
    set_and_wait_for_other_value,
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


class Attenuator(StandardReadable, EpicsDevice, AsyncMovable[AttenuatorStatusEnum]):
    def __init__(
        self,
        prefix: str,
        num: int,
        material: str,
        thickness: float,
        energy: Energy,
        **kwargs,
    ):
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
        self.energy = energy

        self.position = epics_signal_rw(
            AttenuatorStatusEnum,
            f"{self.prefix}:DO{self.num}-Sts",
            write_pv=f"{self.prefix}:DO{self.num}-Cmd",
        )
        self.mode = epics_signal_rw(bool, f"{self.prefix}:DIO{self.num}-Mode")
        self.in_status = epics_signal_r(
            AttenuatorStatusEnum, f"{self.prefix}:DI{self.num}-Sts"
        )

        super().__init__(prefix=self.prefix, **kwargs)

    def __repr__(self):
        return f"{self.thickness!s} microns, {self.filter_material}"

    @AsyncStatus.wrap
    async def set(self, value: AttenuatorStatusEnum):
        await set_and_wait_for_other_value(
            set_signal=self.position,
            set_value=value,
            match_signal=self.position,
            match_value=value,
        )

    async def open(self):
        """Open means open to allowing the beam to pass unobstructed"""
        await self.set(AttenuatorStatusEnum.LOW)

    async def close(self):
        """Closed means obstructing the beam"""
        await self.set(AttenuatorStatusEnum.HIGH)

    async def is_active(self):
        sts = await self.position.get_value()
        return sts == AttenuatorStatusEnum.HIGH

    async def read(self):  # type: ignore[reportUnknownParameterType]
        """Implements StandardReadable.read() according to the BlueskyInterface"""
        status = OrderedDict()
        is_active = await self.is_active()
        status.update(
            {
                self.name + "_transmission": {
                    "value": self.transmission() if is_active else 1.0,
                    "timestamp": time.time(),
                }
            }
        )

        status.update(
            {
                self.name + "_active": {
                    "value": is_active,
                    "timestamp": time.time(),
                }
            }
        )
        return status

    async def describe(self):  # type: ignore[reportUnknownParameterType]
        """Implements StandardReadable.describe() according to BlueskyInterface"""
        description = OrderedDict()

        transmission_info = DataKey(
            source="method",
            dtype="number",
            shape=[],
        )
        description.update({self.name + "_transmission": transmission_info})

        active_info = DataKey(source=self.position.source, dtype="boolean", shape=[])
        description.update({self.name + "_active": active_info})
        return description

    def attenuation(self):
        """Attenuation is the fraction of the beam removed"""
        return 1 - self.transmission()

    def transmission(self):
        """Transmission is the fraction of beam remaining"""
        return np.exp(-self.thickness / self._absorption_length())

    def _get_photon_energy(self):
        return self.energy.energy.readback.get()

    def _get_egu(self):
        return self.energy.egu

    def _absorption_length(self) -> float:
        """
        Calculates L, the characteristic absorption length of this material,
        at this beam energy.

        Returns
        -------
        float
            The characteristic absorption length of the filter material (microns)
        """
        mult = 1
        if self._get_egu() == "KeV":
            mult = 1e3
        elif self._get_egu() != "eV":
            msg = f"Photon energy units must be eV or KeV (not {self._get_egu()=})"
            raise RuntimeError(msg)
        return self.filter_material.absorption_length(self._get_photon_energy() * mult)  # type: ignore[reportArgumentType]


class AttenuatorBank(StandardReadable, EpicsDevice, AsyncMovable[float]):
    def __init__(
        self,
        prefix: str,
        atten_configs: list[tuple[str, float]],
        energy: Energy,
        **kwargs,
    ):
        self.prefix = prefix
        self.energy = energy

        with self.add_children_as_readables():
            self.attenuators = DeviceVector(
                {
                    i: Attenuator(self.prefix, i, material, thickness, energy)
                    for i, (material, thickness) in enumerate(atten_configs, start=1)
                }
            )
        super().__init__(prefix=self.prefix, **kwargs)

    @property
    def hints(self) -> Hints:
        # Expose the computed signal's fields so plot routines can use it
        return {"fields": [self.name + "-total_transmission"]}

    def get_photon_energy(self):
        return self.energy.energy.readback.get()

    def get_egu(self):
        return self.energy.egu

    async def read(self):  # type: ignore[reportUnknownParameterType]
        """Returns each filter position, each transmission, and the total transmission."""
        status = OrderedDict()
        active_attens = []
        for _, atten in self.attenuators.items():
            status.update(await atten.read())
            if await atten.is_active():
                active_attens.append(atten)
        total_transmission = self._calculate_total_transmission(*active_attens)
        status.update(
            {
                f"{self.name}-total_transmission": {
                    "value": total_transmission,
                    "timestamp": time.time(),
                }
            }
        )
        return status

    async def describe(self) -> OrderedDict[str, DataKey]:
        """Describe the structure of values returned by read()."""

        description = OrderedDict()

        transmission_info = DataKey(
            source=f"ca://{self.prefix}:total_transmission",
            dtype="number",
            shape=[],
        )
        description.update({f"{self.name}-total_transmission": transmission_info})

        for atten in self.attenuators.values():
            description.update(await atten.describe())

        return description

    @AsyncStatus.wrap
    async def set(self, value: float):
        """Set the transmission for the attenuator bank"""
        attenuation_combination = self.find_closest_transmission(value)
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
        self, target_transmission: float
    ) -> AttenuatorCombination:
        available_attenuations = self._calculate_available_transmissions()
        best_idx = np.argmin(
            [abs(target_transmission - _.transmission) for _ in available_attenuations]
        )
        return available_attenuations[best_idx]

    def _calculate_available_transmissions(self) -> list[AttenuatorCombination]:
        """
        Calculates all possible transmissions for the attenuator bank, using
        the powerset of the available attenuators. The result is not sorted,
        as it is more efficient to scan linearly each time for the closest
        match.
        """
        available_transmissions = []
        for combination in self._powerset():
            attens = [self.attenuators[a] for a in self.attenuators if a in combination]
            total_transmission = self._calculate_total_transmission(*attens)
            available_transmissions.append(
                AttenuatorCombination(total_transmission, combination)
            )
        return available_transmissions

    def _calculate_total_transmission(self, *attenuators: Attenuator) -> float:
        transmissions = [a.transmission() for a in attenuators]
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
