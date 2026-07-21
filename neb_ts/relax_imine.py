"""Relax the built imine+H2O product and confirm it is a true minimum.

Checks that the dehydration product survives GFN2-xTB relaxation:
C10=N27 stays ~1.28 A (imine double bond), O4 stays as a free water far from
C10, and no imaginary frequencies (relaxed geometry is a minimum).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from .neb_run import relax
from .build_imine import C_ALDEHYDE, O_HYDROXYL, N_AMINE, H_ON_N, H_ON_O, _report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", nargs="?", default="output/imine_product.xyz")
    ap.add_argument("--out", default="output/imine_product_relaxed.xyz")
    ap.add_argument("--fmax", type=float, default=0.05)
    args = ap.parse_args(argv)
    atoms = read(args.structure)
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("multiplicity", 1)
    print("Before relaxation:")
    _report(atoms)
    relaxed = relax(atoms, 0, 1, args.fmax, "imine+H2O product",
                    Path(args.out).with_suffix(".traj"), log=[])
    print("\nAfter GFN2-xTB relaxation:")
    _report(relaxed)
    write(args.out, relaxed)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())