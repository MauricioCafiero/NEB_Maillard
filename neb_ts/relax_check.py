"""Relax a structure at GFN2-xTB and report whether it stays pre-reactive.

Used to verify the built pre-reactive complex does not collapse to the
carbinolamine product on relaxation -- if it did, there would be no barrier
and no TS to find for the concerted step.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from .calculator import make_calculator
from .neb_run import relax
from .build_precomplex import C_ALDEHYDE, O_ALDEHYDE, N_AMINE, H_TRANSFER, _report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="Input structure (e.g. output/prereactive_complex.xyz)")
    ap.add_argument("--out", default="output/prereactive_complex_relaxed.xyz")
    ap.add_argument("--fmax", type=float, default=0.05)
    args = ap.parse_args(argv)

    atoms = read(args.structure)
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("multiplicity", 1)
    print("Before relaxation:")
    _report(atoms)
    relaxed = relax(atoms, 0, 1, args.fmax, "pre-reactive complex",
                    Path(args.out).with_suffix(".traj"), log=[])
    print("\nAfter GFN2-xTB relaxation:")
    _report(relaxed)
    write(args.out, relaxed)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())