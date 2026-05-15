from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

import numpy as np
import xrayutilities as xu
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


THICKNESSES = (16, 24, 66, 124)  # microns


class AttenuatorStatusEnum(StrictEnum):
    LOW = "Low"  # off / not obstructing
    HIGH = "High"  # on / obstructing


class Attenuator(EpicsDevice, AsyncMovable[AttenuatorStatusEnum]):
    filter_material = xu.materials.Al

    def __init__(self, prefix: str, num: int, thickness: int):
        """
        prefix - the common prefix for the attenuator bank
        num - an integer denoting which attenuator within the bank this is
        thickness - the thickness of the attenuator in microns

        position - the read / write PV to open and close the attenuator
        """
        self.prefix = prefix
        self.num = num
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

    def transmission(self, photon_energy: float, egu: str = "KeV"):
        """Transmission is the fraction of beam remaining"""
        abs_len = self._absorption_length(photon_energy, egu=egu)
        return np.exp(-self.thickness / abs_len)

    def attenuation(self, photon_energy: float, egu: str = "KeV"):
        """Attenuation is the fraction of the beam removed"""
        return 1 - self.transmission(photon_energy, egu=egu)

    def _absorption_length(self, photon_energy: float, egu: str = "KeV") -> float:
        """
        Calculates L, the characteristic absorption length of this material,
        at this beam energy.

        photon energy: the beam energy
        egu: the engineering units of the beam energy (KeV or eV)
        absorption length: the characteristic absorption length of the
            filter material (microns)
        """
        if egu == "KeV":
            photon_energy = photon_energy * 1e3
        elif egu != "eV":
            msg = "Photon energy units must be eV or KeV"
            raise RuntimeError(msg)
        return self.filter_material.absorption_length(photon_energy)  # type: ignore[reportArgumentType]


class AttenuatorBank(StandardReadable, EpicsDevice, AsyncMovable[float]):
    """
    The ioc for the iologik1 lives on xf09id1-inst-ioc1.nsls2.bnl.gov
    """

    prefix = "XF:09ID1-ES{IOLOGIK1:E1212}"
    thicknesses = THICKNESSES

    def __init__(self, energy: Energy):
        self.energy = energy

        with self.add_children_as_readables():
            self.attenuators = DeviceVector(
                {
                    i: Attenuator(self.prefix, i, self.thicknesses[i - 1])
                    for i in range(1, len(self.thicknesses) + 1)
                }
            )
        super().__init__(prefix=self.prefix)

    @property
    def photon_energy(self):
        return self.energy.energy.readback.get()

    @property
    def egu(self):
        return self.energy.egu

    async def get_status(self):  # type: ignore[reportUnknownParameterType]
        """
        Status polls the bluesky energy object for the current beam energy, and
        returns that energy, each filter position, each transmission, and
        the total transmission.
        """
        status = {}
        active_attens = []
        en = self.photon_energy
        egu = self.egu
        positions = await asyncio.gather(
            *(a.position.get_value() for _, a in self.attenuators.items())
        )
        for i, pos in zip(self.attenuators, positions):
            atten = self.attenuators[i]
            is_active = pos == AttenuatorStatusEnum.HIGH
            if is_active:
                active_attens.append(atten)
            transmission = atten.transmission(en, egu) if is_active else 0
            status[atten.name] = {"active": is_active, "transmission": transmission}
        status["active_attenuators"] = [a.num for a in active_attens]
        status["photon_energy"] = en
        status["egu"] = egu
        status["total_transmission"] = self._calculate_total_attenuation(*active_attens)
        return status

    @AsyncStatus.wrap
    async def set(self, value: float):
        attenuation_combination = self.find_closest_attenuation(value)
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

    # TODO - fix the language here (transmission / attenuation)
    def find_closest_attenuation(
        self, target_attenuation: float
    ) -> AttenuatorCombination:
        """
        This could be faster if we implemented binary search,
        but that seems like overkill for our use case. The search space
        is small, so we start in the middle, and work up or down.
        """
        available_attenuations = self._calculate_available_attentuations()
        best_idx = len(available_attenuations) // 2
        atten = available_attenuations[best_idx].transmission
        diff = float("inf")
        new_diff = abs(target_attenuation - atten)
        inc = 1 if target_attenuation > atten else -1

        while new_diff < diff:
            diff = new_diff
            # break if we are about to check outside the list
            if best_idx + inc >= len(available_attenuations) or best_idx + inc < 0:
                break
            atten = available_attenuations[best_idx + inc].transmission
            new_diff = abs(target_attenuation - atten)
            if new_diff < diff:
                best_idx += inc
            else:  # if diff did not change, then we have found the best option
                break
        # TODO - should return just the found attentuation? or also the
        # requested attenuation and/or the difference?
        return available_attenuations[best_idx]

    def _calculate_available_attentuations(self) -> list[AttenuatorCombination]:
        """
        It is more efficient to precompute all possible total
        attenuations, and simply look up the closest one.
        """
        available_attenuations = []
        for combination in self._powerset():
            attens = [self.attenuators[a] for a in self.attenuators if a in combination]
            total_atten = self._calculate_total_attenuation(*attens)
            available_attenuations.append(
                AttenuatorCombination(total_atten, combination)
            )
        # We want the available attenuations sorted so we can efficiently search through them
        available_attenuations.sort(key=lambda a: a.transmission)  # type: ignore[attr-defined]
        return available_attenuations

    def _calculate_total_attenuation(self, *attenuators: Attenuator) -> float:
        transmissions = [
            a.transmission(self.photon_energy, self.egu) for a in attenuators
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
