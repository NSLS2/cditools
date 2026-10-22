from __future__ import annotations

from collections.abc import Callable

import hklpy2
from hklpy2.diffract import DiffractometerBase

PSEUDOS = ["h", "k", "l"]
REALS = ["mu", "omega1", "chi1", "omega2", "chi2", "gamma1", "delta1"]
hklpy2.creator()


class CDIDiffractometer(DiffractometerBase):
    def __init__(
        self,
        prefix: str = "XF:09IDC-",
        solver: str | None = "ad_hoc",
        geometry: str | None = "cdi-geometry",
        solver_kwargs: dict = {},  # noqa: B006
        pseudos: list[str] = PSEUDOS,
        reals: list[str] = REALS,
        reals_units: str | None = None,
        forward_solution_function: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            prefix=prefix,
            solver=solver,
            geometry=geometry,
            solver_kwargs=solver_kwargs,
            pseudos=pseudos,
            reals=reals,
            reals_units=reals_units,
            forward_solution_function=forward_solution_function,
            **kwargs,
        )
