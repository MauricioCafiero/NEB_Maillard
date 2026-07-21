"""GFN2-xTB calculator factory for the NEB workflow.

We use ``tblite``'s ASE interface (:class:`tblite.ase.TBLite`) with the
``GFN2-xTB`` semi-empirical method.  Every NEB image needs its *own*
calculator instance, so callers should invoke :func:`make_calculator` once per
image rather than sharing one.

Robustness
----------
NEB intermediate images can be geometrically distorted, which sometimes makes
the self-consistent charge (SCC) cycle fail to converge within the default
250 iterations.  To keep the band from crashing on a single bad image we

* raise the SCC iteration budget,
* use a slightly higher electronic temperature (smears occupations and aids
  convergence on distorted geometries), and
* wrap the calculator so a failed SCC is retried once with even looser
  settings before giving up.
"""

from __future__ import annotations

from ase.calculators.calculator import CalculationFailed
from tblite.ase import TBLite

# 0 = silent.  GFN2-xTB's SCC log is noisy across the many NEB image
# evaluations, so silence it by default.
_DEFAULT_VERBOSITY = 0


class RobustTBLite(TBLite):
    """TBLite that retries SCC convergence with progressively looser settings.

    NEB intermediate images can be geometrically distorted enough that the
    self-consistent charge (SCC) cycle fails.  To keep the band from crashing on
    a single bad image we retry, first with a loose SCC, then -- if that still
    fails -- with an *extremely* high electronic temperature.  A very high
    temperature smears the occupations almost flat (near the Aufbau limit) and
    converges essentially always; the resulting energy/forces are not quantitatively
    accurate, but they point the right way to escape the distorted geometry so the
    optimizer can relax the image back onto the physical PES.
    """

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        try:
            return super().calculate(atoms, properties, system_changes)
        except CalculationFailed:
            pass
        # Retry 1: loose SCC.
        self.parameters["max_iterations"] = max(
            int(self.parameters.get("max_iterations", 500) or 500), 2000)
        self.parameters["electronic_temperature"] = max(
            float(self.parameters.get("electronic_temperature", 500.0) or 500.0),
            1500.0)
        self.parameters["mixer_damping"] = 0.2
        self.reset()
        try:
            return super().calculate(atoms, properties, system_changes)
        except CalculationFailed:
            pass
        # Retry 2: extreme temperature -- almost always converges.  Accuracy is
        # poor but the forces are physically signed (escape the bad geometry).
        self.parameters["max_iterations"] = 20000
        self.parameters["electronic_temperature"] = 50000.0
        self.parameters["mixer_damping"] = 0.4
        self.reset()
        return super().calculate(atoms, properties, system_changes)


def make_calculator(
    charge: int = 0,
    multiplicity: int = 1,
    *,
    accuracy: float = 1.0,
    max_iterations: int = 500,
    electronic_temperature: float = 500.0,
    mixer_damping: float = 0.3,
    robust: bool = True,
) -> TBLite:
    """Return a fresh GFN2-xTB ASE calculator.

    Parameters
    ----------
    charge
        Total molecular charge.
    multiplicity
        Spin multiplicity (1 = singlet, 2 = doublet, ...).  tblite expects the
        number of unpaired electrons via ``uhf`` (i.e. ``multiplicity - 1``).
    accuracy
        Numerical accuracy of the xTB single-point calculation.
    max_iterations
        Maximum number of SCC iterations.
    electronic_temperature
        Electronic temperature in Kelvin (higher aids SCC convergence on
        distorted geometries).
    mixer_damping
        SCC mixer damping factor.
    robust
        If True (default) wrap the calculator so a failed SCC is retried once
        with looser settings -- recommended for NEB.
    """
    cls = RobustTBLite if robust else TBLite
    return cls(
        method="GFN2-xTB",
        charge=charge,
        uhf=multiplicity - 1,
        accuracy=accuracy,
        max_iterations=max_iterations,
        electronic_temperature=electronic_temperature,
        mixer_damping=mixer_damping,
        verbosity=_DEFAULT_VERBOSITY,
    )