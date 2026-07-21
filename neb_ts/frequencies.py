"""Vibrational analysis to verify a transition-state candidate.

A true transition state is a first-order saddle point: exactly **one** negative
(imaginary) Hessian eigenvalue.  This module uses ASE's finite-difference
vibrational analysis (:class:`ase.vibrations.Vibrations`) with the GFN2-xTB
calculator and reports the number of imaginary modes and the imaginary
frequency, which corresponds to the reaction coordinate.

Note: finite-difference frequencies require ``6N`` force evaluations for a
molecule of ``N`` atoms, which is the expensive part of the workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.vibrations import Vibrations

from .calculator import make_calculator


CM1_PER_EV = 8065.544  # eV -> cm^-1


def compute_frequencies(atoms: Atoms, charge: int = 0, multiplicity: int = 1, *, delta: float = 0.01) -> tuple[list[float], int]:
    """Return vibrational frequencies (cm^-1) and the number of imaginary modes.

    Frequencies are returned sorted ascending; imaginary modes appear as
    negative numbers.  ``atoms`` is left with a calculator attached.
    """
    atoms = atoms.copy()
    atoms.calc = make_calculator(charge, multiplicity)
    vib = Vibrations(atoms, delta=delta, nfree=2)
    vib.run()
    freqs = vib.get_frequencies()
    # ASE returns complex numbers for imaginary modes; convert to signed reals.
    real_freqs = [float(f.real) * (1.0 if abs(f.imag) < 1e-10 else -1.0) if isinstance(f, complex) else float(f) for f in freqs]
    real_freqs = sorted(real_freqs)
    n_imag = sum(1 for f in real_freqs if f < -10.0)  # ignore numerically-tiny negative values
    return real_freqs, n_imag


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a TS candidate by vibrational analysis (GFN2-xTB).")
    ap.add_argument("structure", help="Structure to analyse (e.g. output/ts_structure.xyz)")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--multiplicity", type=int, default=1)
    ap.add_argument("--delta", type=float, default=0.01, help="Displacement in Å (default 0.01)")
    ap.add_argument("--out", default="output/frequencies.txt")
    args = ap.parse_args(argv)

    atoms = read(args.structure)
    print(f"Computing vibrational frequencies for {len(atoms)} atoms "
          f"({6 * len(atoms)} force evaluations) ...")
    freqs, n_imag = compute_frequencies(atoms, args.charge, args.multiplicity, delta=args.delta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# frequency (cm^-1)\n")
        for nu in freqs:
            f.write(f"{nu: .4f}\n")
    print(f"Imaginary modes (< -10 cm^-1): {n_imag}")
    if n_imag == 1:
        print("=> Exactly one imaginary mode: confirmed transition state.")
    elif n_imag == 0:
        print("=> No imaginary modes: this is a minimum, not a transition state.")
    else:
        print(f"=> {n_imag} imaginary modes: higher-order saddle (not a true TS).")
    print(f"\nImaginary / lowest frequencies (cm^-1):")
    for nu in freqs[:max(6, n_imag)]:
        print(f"  {nu: .2f}")
    print(f"\nFull spectrum written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())