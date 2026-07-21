"""Refine a TS guess to a saddle by a projected (FixedMode) optimization.

Take a good near-saddle guess (e.g. a CI-NEB climbing image) and the NEB tangent
as the reaction-coordinate mode.  Constrain the geometry so it cannot move along
that mode (project the mode out of both forces and steps) and *minimize* in
every perpendicular direction.  The geometry therefore relaxes to the nearest
point on the saddle ridge at the fixed reaction-coordinate value -- a first-order
saddle, provided the mode is the true reaction coordinate.

This is more robust than the Dimer method for a good starting guess: it is a
plain minimization in a constrained subspace, so it cannot "run away" uphill the
way an inverted-force dimer can.  Convergence is then checked by vibrational
analysis (exactly one imaginary frequency = true TS).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.constraints import FixedMode
from ase.io import read, write
from ase.optimize import FIRE

from .calculator import make_calculator


def _tangent_from_path(path_file, ts_index):
    images = read(path_file, ":")
    if ts_index <= 0 or ts_index >= len(images) - 1:
        raise ValueError("ts_index must be an interior image index.")
    return images[ts_index + 1].positions - images[ts_index - 1].positions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="TS guess (e.g. output/dehyd/ts_guess_img4.xyz)")
    ap.add_argument("--path", default="output/dehyd/neb_final_path.xyz",
                    help="NEB path file to derive the reaction-coordinate mode")
    ap.add_argument("--ts-index", type=int, default=4, help="TS image index in --path")
    ap.add_argument("--out", default="output/ts1/ts_fixedmode.xyz")
    ap.add_argument("--fmax", type=float, default=0.02,
                    help="Perpendicular-force convergence (eV/A); tighter than NEB.")
    ap.add_argument("--max-steps", type=int, default=800)
    args = ap.parse_args(argv)

    atoms = read(args.structure)
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("multiplicity", 1)
    atoms.calc = make_calculator(0, 1)
    e0 = float(atoms.get_potential_energy())
    f0 = atoms.get_forces()
    # Perpendicular force magnitude = total force with the mode removed.
    mode = _tangent_from_path(args.path, args.ts_index)
    fm = FixedMode(mode)
    fm.adjust_forces(atoms, f0)
    fmax0 = float(np.sqrt((f0 ** 2).sum(axis=1)).max())
    print(f"Start: E={e0:.4f} eV, max|F_perp|={fmax0:.4f} eV/A")

    atoms.set_constraint(fm)
    opt = FIRE(atoms, trajectory=str(Path(args.out).with_suffix(".traj")), logfile=None)
    print(f"Fixed-mode optimization (FIRE, fmax={args.fmax}, max {args.max_steps} steps) ...")
    opt.run(fmax=args.fmax, steps=args.max_steps)

    e1 = float(atoms.get_potential_energy())
    f1 = atoms.get_forces()
    fm.adjust_forces(atoms, f1)
    fmax1 = float(np.sqrt((f1 ** 2).sum(axis=1)).max())
    print(f"End:   E={e1:.4f} eV, max|F_perp|={fmax1:.4f} eV/A")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atoms.constraints = []        # drop constraint before writing
    write(out, atoms)
    print(f"\nRefined TS written to {out}")
    print("Verify with: uv run python -m neb_ts.frequencies", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())