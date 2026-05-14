from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

import numpy as np
import xraylib
from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    DeviceVector,
    StandardReadable,
    StrictEnum,
)
from ophyd_async.epics.core import EpicsDevice, epics_signal_r, epics_signal_rw


@dataclass
class AttenuatorCombination:
    transmission: float
    attenuators: list[int]


THICKNESSES = (16, 24, 66, 124)  # microns

# The available attenuations can be calculated with the utility
# methods below, but they do not change often,
# so we hardcode them here
# TODO - there will eventually be eight filters
AVAILABLE_ATTENUATIONS = [
    AttenuatorCombination(transmission=0.08, attenuators=[1, 2, 3, 4]),
    AttenuatorCombination(transmission=0.095, attenuators=[2, 3, 4]),
    AttenuatorCombination(transmission=0.104, attenuators=[1, 3, 4]),
    AttenuatorCombination(transmission=0.124, attenuators=[3, 4]),
    AttenuatorCombination(transmission=0.165, attenuators=[1, 2, 4]),
    AttenuatorCombination(transmission=0.196, attenuators=[2, 4]),
    AttenuatorCombination(transmission=0.214, attenuators=[1, 4]),
    AttenuatorCombination(transmission=0.256, attenuators=[4]),
    AttenuatorCombination(transmission=0.312, attenuators=[1, 2, 3]),
    AttenuatorCombination(transmission=0.372, attenuators=[2, 3]),
    AttenuatorCombination(transmission=0.406, attenuators=[1, 3]),
    AttenuatorCombination(transmission=0.484, attenuators=[3]),
    AttenuatorCombination(transmission=0.644, attenuators=[1, 2]),
    AttenuatorCombination(transmission=0.768, attenuators=[2]),
    AttenuatorCombination(transmission=0.839, attenuators=[1]),
    AttenuatorCombination(transmission=1.0, attenuators=[]),
]


class AttenuatorStatusEnum(StrictEnum):
    LOW = "Low"  # off / not obstructing
    HIGH = "High"  # on / obstructing


class Attenuator(EpicsDevice, AsyncMovable[AttenuatorStatusEnum]):
    filter_material = "Al"
    filter_material_z = 13
    filter_density = xraylib.ElementDensity(filter_material_z)  # g/cm³

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

    @AsyncStatus.wrap
    async def set(self, value: AttenuatorStatusEnum):
        await self.position.set(value)

    async def open(self):
        """Open means open to allowing the beam to pass unobstructed"""
        await self.position.set(AttenuatorStatusEnum.LOW)

    async def close(self):
        """Closed means obstructing the beam"""
        await self.position.set(AttenuatorStatusEnum.HIGH)

    @property
    def thickness_cm(self):
        # Thickness is in microns, so convert to cm
        return self.thickness * 1e-4

    @property
    def linear_atten_coefficient(self) -> float:
        """
        Calculates µ, the linear attenuation coefficient of this material,
        at this thickness, and this beam energy.

        photon energy in KeV
        xraylib.CS_Total in cm²/g
        linear_atten_coefficient in cm⁻¹
        """
        photon_energy = 8.6  # KeV TODO - get the right number; this is taken from bmm
        mass_atten_cross_section = xraylib.CS_Total(
            self.filter_material_z, photon_energy
        )
        return mass_atten_cross_section * self.filter_density

    @property
    def transmission(self):
        """Transmission is the fraction of remaining beam"""
        return np.exp(-self.linear_atten_coefficient * self.thickness_cm)

    @property
    def attenuation(self):
        """Attenuation is the fraction of the beam removed"""
        return 1 - self.transmission


class AttenuatorBank(StandardReadable, EpicsDevice, AsyncMovable[float]):
    """
    The ioc for the iologik1 lives on xf09id1-inst-ioc1.nsls2.bnl.gov
    """

    prefix = "XF:09ID1-ES{IOLOGIK1:E1212}"
    thicknesses = THICKNESSES
    available_attenuations = AVAILABLE_ATTENUATIONS

    def __init__(self):
        with self.add_children_as_readables():
            self.attenuators = DeviceVector(
                {
                    i: Attenuator(self.prefix, i, self.thicknesses[i - 1])
                    for i in range(1, len(self.thicknesses) + 1)
                }
            )
        super().__init__(prefix=self.prefix)

    async def get_status(self):
        return await asyncio.gather(
            *(a.position.get_value() for _, a in self.attenuators.items())
        )

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

    def find_closest_attenuation(
        self, target_attenuation: float
    ) -> AttenuatorCombination:
        """
        This could be faster if we implemented binary search,
        but that seems like overkill for our use case. The search space
        is small, so we start in the middle, and work up or down.
        """
        best_idx = len(self.available_attenuations) // 2
        atten = self.available_attenuations[best_idx].transmission
        diff = float("inf")
        new_diff = abs(target_attenuation - atten)
        inc = 1 if target_attenuation > atten else -1

        while new_diff < diff:
            diff = new_diff
            # break if we are about to check oustide the list
            if best_idx + inc >= len(self.available_attenuations) or best_idx + inc < 0:
                break
            atten = self.available_attenuations[best_idx + inc].transmission
            new_diff = abs(target_attenuation - atten)
            if new_diff < diff:
                best_idx += inc
            else:  # if diff did not change, then we have found the best option
                break
        # TODO - should return just the found attentuation? or also the
        # requested attenuation and/or the difference?
        return self.available_attenuations[best_idx]

    """
    These are utility methods that should not be called during production.
    They are used to calculate the available attenuations from all
    combinations of attenuators. The result is then used as the
    AttenuationBank()._available_attenuations attribute.
    """

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
        return round(float(math.prod([a.transmission for a in attenuators])), 3)

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


"""
from cditools.attenuator import AttenuatorBank

bank = AttenuatorBank()
atten = bank.attenuators[1]

"""
