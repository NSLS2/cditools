from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import xraylib
from ophyd_async.core import DeviceVector, StandardReadable
from ophyd_async.epics.core import EpicsDevice


@dataclass
class AttenuatorCombination:
    attenuation: float
    attenuators: list[int]


# The available attenuations can be calculated with the utility
# methods below, but they do not change often,
# so we hardcode them here
# TODO - there will eventually be eight filters
AVAILABLE_ATTENUATIONS = [
    AttenuatorCombination(attenuation=0.08, attenuators=[0, 1, 2, 3]),
    AttenuatorCombination(attenuation=0.095, attenuators=[1, 2, 3]),
    AttenuatorCombination(attenuation=0.104, attenuators=[0, 2, 3]),
    AttenuatorCombination(attenuation=0.124, attenuators=[2, 3]),
    AttenuatorCombination(attenuation=0.165, attenuators=[0, 1, 3]),
    AttenuatorCombination(attenuation=0.196, attenuators=[1, 3]),
    AttenuatorCombination(attenuation=0.214, attenuators=[0, 3]),
    AttenuatorCombination(attenuation=0.256, attenuators=[3]),
    AttenuatorCombination(attenuation=0.312, attenuators=[0, 1, 2]),
    AttenuatorCombination(attenuation=0.372, attenuators=[1, 2]),
    AttenuatorCombination(attenuation=0.406, attenuators=[0, 2]),
    AttenuatorCombination(attenuation=0.484, attenuators=[2]),
    AttenuatorCombination(attenuation=0.644, attenuators=[0, 1]),
    AttenuatorCombination(attenuation=0.768, attenuators=[1]),
    AttenuatorCombination(attenuation=0.839, attenuators=[0]),
    AttenuatorCombination(attenuation=1.0, attenuators=[]),
]


class Attenuator(EpicsDevice):
    filter_material = "Al"
    filter_material_z = 13
    filter_density = xraylib.ElementDensity(filter_material_z)  # g/cm³

    def __init__(self, prefix: str, num: int, thickness: int):
        """
        pv - the PV for this filter
        """
        self.prefix = prefix
        self.num = num
        self.pv = f"{self.prefix}:D0{self.num + 1}-Cmd"
        self.thickness = thickness  # microns
        super().__init__(prefix=self.prefix)

    def __repr__(self):
        return str(self.thickness)

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
    def attenuation(self):
        """
        Attenuation is the fraction of remaining beam
        """
        return np.exp(-self.linear_atten_coefficient * self.thickness_cm)


class AttenuatorBank(StandardReadable, EpicsDevice):
    """
    The ioc for the iologik1 lives on xf09id1-inst-ioc1
    """

    prefix = "XF:09ID1-ES{IOLOGIK1:E1212}:"
    thicknesses = (16, 24, 66, 124)  # microns
    available_attenuations = AVAILABLE_ATTENUATIONS

    def __init__(self):
        with self.add_children_as_readables():
            self.attenuators = DeviceVector(
                {
                    i: Attenuator(self.prefix, i, self.thicknesses[i])
                    for i in range(len(self.thicknesses))
                }
            )
        super().__init__(prefix=self.prefix)

    def set_attenuation(self, target_attenuation: float):
        pass

    def find_closest_attenuation(
        self, target_attenuation: float
    ) -> AttenuatorCombination:
        """
        This could be faster if we implemented binary search,
        but that seems like overkill for our use case. The search space
        is small, so we start in the middle, and work up or down.
        """
        best_idx = len(self.available_attenuations) // 2
        atten = self.available_attenuations[best_idx].attenuation
        diff = float("inf")
        new_diff = abs(target_attenuation - atten)

        while new_diff < diff:
            diff = new_diff
            # TODO - the (in|de)crement can surely be combined into a clever one liner
            if target_attenuation > atten:
                atten = self.available_attenuations[best_idx + 1].attenuation
                new_diff = abs(target_attenuation - atten)
                if new_diff < diff:
                    best_idx += 1
            else:
                atten = self.available_attenuations[best_idx - 1].attenuation
                new_diff = abs(target_attenuation - atten)
                if new_diff < diff:
                    best_idx -= 1
            # Break if we have reached the end of the list
            if best_idx >= len(self.available_attenuations) or best_idx < 0:
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
        available_attenuations.sort(key=lambda a: a.attenuation)  # type: ignore[attr-defined]
        return available_attenuations

    def _calculate_total_attenuation(self, *attenuators: Attenuator) -> float:
        return round(float(math.prod([a.attenuation for a in attenuators])), 3)

    def _powerset(self) -> list[list[int]]:
        """
        This is a famously O(n*2^n) problem.
        """
        powerset = []
        for i in range(1 << len(self.attenuators)):
            combination = []
            for j in range(len(self.attenuators)):
                if i & (1 << j):
                    combination.append(j)
            powerset.append(combination)
        return powerset
