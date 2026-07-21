"""UMA (Fairchem) ASE calculator factory for the NEB workflow.

This is the machine-learned interatomic potential counterpart of
:mod:`neb_ts.calculator` (GFN2-xTB).  We load the small UMA model
``uma-s-1p1`` from the gated `facebook/UMA` HuggingFace repo (needs
``HF_TOKEN`` in the environment) and wrap it with Fairchem's ASE
interface, :class:`fairchem.core.FAIRChemCalculator`, on the **omol**
task (molecules).

Notes
-----
* **CPU only on this machine.**  Fairchem's ASE calculator does not
  support Apple MPS; the UMA predictor is therefore created with
  ``device="cpu"``.  A single 44-atom omol evaluation takes ~1 s on the
  A18 Pro, so a 17-image CI-NEB for a few hundred steps is tractable but
  not instant — keep the image count modest.
* The ``omol`` head requires the total charge and spin multiplicity to
  be set in ``atoms.info`` (``charge`` and ``spin``).  ``spin`` is the
  multiplicity (1 = singlet).
* A *single* predictor object is shared by every image; only the
  :class:`FAIRChemCalculator` wrapper is per-image (matching the UMA
  tutorial's NEB example).  This keeps memory low.
"""

from __future__ import annotations

from typing import Literal

from ase import Atoms

# Task head names accepted by FAIRChemCalculator.  We default to "omol"
# (molecules) for the Maillard system; the others are surface/bulk/MOF/
# molecular-crystal heads and are kept here only so the CLI can validate.
TaskName = Literal["omol", "oc20", "omat", "odac", "omc"]

_predictor_cache: dict[tuple[str, str], object] = {}


def _get_predictor(model: str, device: str):
    """Load (and cache) a UMA predictor.

    The predictor is expensive to build (it downloads ~150 M weights and
    moves them onto the device), so we build it once and reuse it for
    every image's calculator wrapper.
    """
    key = (model, device)
    if key not in _predictor_cache:
        from fairchem.core import pretrained_mlip  # imported lazily

        _predictor_cache[key] = pretrained_mlip.get_predict_unit(model, device=device)
    return _predictor_cache[key]


def set_charge_spin(atoms: Atoms, charge: int, multiplicity: int) -> None:
    """Attach the omol-required ``charge`` / ``spin`` metadata.

    ``spin`` is the spin multiplicity (1 = singlet, 2 = doublet, ...).
    """
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(multiplicity)


def make_uma_calculator(
    charge: int = 0,
    multiplicity: int = 1,
    *,
    model: str = "uma-s-1p1",
    task_name: TaskName = "omol",
    device: str = "cpu",
) -> "FAIRChemCalculator":  # type: ignore[name-defined]
    """Return a fresh UMA ASE calculator for one image.

    Parameters
    ----------
    charge, multiplicity
        Total molecular charge and spin multiplicity (1 = singlet).
        These are *not* constructor arguments of FAIRChemCalculator; the
        ``omol`` head reads them from ``atoms.info`` at evaluation time,
        so callers must also run :func:`set_charge_spin` on the atoms.
    model
        UMA model identifier.  ``uma-s-1p1`` is the 150 M-parameter small
        model.
    task_name
        UMA task head.  ``"omol"`` for molecules (the Maillard system).
    device
        Compute device.  ``"cpu"`` on this Mac (MPS is not supported by
        the Fairchem ASE calculator).
    """
    from fairchem.core import FAIRChemCalculator  # imported lazily

    predictor = _get_predictor(model, device)
    calc = FAIRChemCalculator(predictor, task_name=task_name)
    # Stash the requested charge/spin on the calculator so the NEB
    # plumbing can propagate them to every atoms.info when attaching.
    calc._neb_charge = int(charge)  # type: ignore[attr-defined]
    calc._neb_spin = int(multiplicity)  # type: ignore[attr-defined]
    return calc