"""Refine a NEB TS guess to a first-order saddle with a fixed-mode (projected)
minimization at the **UMA** level of theory.

This is the robust counterpart to :mod:`neb_ts.refine_ts_fixedmode` (which is
hard-wired to GFN2).  The CI-NEB climbing image on the UMA PES ran away into a
dissociative channel (+388 kcal, O-C ripped to 12 Å, a proton flung 29 Å out),
because the inverted climbing-image force is free to slide downhill along any
dissociative direction.  The *climb-off* NEB band, by contrast, stayed physical
(barrier +10.8 kcal at image 7, intact 41+3 connectivity) -- the springs pinned
each image to the path.

So we take the clean climb-off TS image, constrain motion along the NEB tangent
(the reaction coordinate) with :class:`ase.constraints.FixedMode`, and *minimize*
in every perpendicular direction.  This is a plain minimization in a constrained
subspace -- it cannot run away uphill.  The result is the nearest point on the
saddle ridge at the fixed reaction-coordinate value: a first-order saddle,
provided the tangent is the true reaction coordinate (the NEB max makes this a
good bet).  Convergence is then checked by vibrational analysis (exactly one
imaginary frequency).

Usage::

    python -m neb_ts.refine_fixedmode_uma output/uma_geodesic/ts_guess_prerelax.xyz \\
        --path output/uma_geodesic/prerelax_band.xyz --ts-index 7 \\
        --uma-model uma-s-1p1 --uma-task omol --uma-device cpu \\
        --out output/uma_geodesic/ts_refined.xyz
    python -m neb_ts.frequencies output/uma_geodesic/ts_refined.xyz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.constraints import FixedMode
from ase.io import read, write
from ase.optimize import FIRE

from .uma import make_uma_calculator, set_charge_spin
from .neb_run import _resolve_calculator_factory


def _tangent_from_path(path_file: str, ts_index: int) -> np.ndarray:
    images = read(path_file, ":")
    if ts_index <= 0 or ts_index >= len(images) - 1:
        raise ValueError(f"ts_index {ts_index} must be an interior image (1..{len(images)-2}).")
    tangent = images[ts_index + 1].positions - images[ts_index - 1].positions
    return tangent / np.linalg.norm(tangent)


def _max_heavy_heavy(atoms) -> float:
    from scipy.spatial.distance import cdist
    d = cdist(atoms.get_positions(), atoms.get_positions())
    sym = atoms.get_chemical_symbols()
    return float(max(
        d[i, j] for i in range(len(atoms)) for j in range(i + 1, len(atoms))
        if sym[i] != "H" and sym[j] != "H"
    ))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="TS guess (e.g. output/uma_geodesic/ts_guess_prerelax.xyz)")
    ap.add_argument("--path", required=True, help="NEB path file to derive the reaction-coordinate mode")
    ap.add_argument("--ts-index", type=int, default=7, help="TS image index in --path (for the tangent seed)")
    ap.add_argument("--out", default="output/uma_geodesic/ts_refined.xyz")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--multiplicity", type=int, default=1)
    ap.add_argument("--fmax", type=float, default=0.02, help="Perpendicular-force convergence (eV/A)")
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="omol")
    ap.add_argument("--uma-device", default="cpu")
    args = ap.parse_args(argv)

    atoms = read(args.structure)
    set_charge_spin(atoms, args.charge, args.multiplicity)
    calc = make_uma_calculator(args.charge, args.multiplicity,
                               model=args.uma_model, task_name=args.uma_task, device=args.uma_device)
    atoms.calc = calc

    e0 = float(atoms.get_potential_energy())
    f0 = atoms.get_forces()
    mode = _tangent_from_path(args.path, args.ts_index)
    fm = FixedMode(mode)
    fm.adjust_forces(atoms, f0)  # project the mode out -> perpendicular force
    fmax0 = float(np.sqrt((f0 ** 2).sum(axis=1)).max())
    print(f"Start: E={e0:.6f} eV ({e0 * 23.0605:.2f} kcal ref not shown), max|F_perp|={fmax0:.4f} eV/A")
    print(f"       max heavy-heavy = {_max_heavy_heavy(atoms):.2f} A (reactant span ~11.3 A)")

    atoms.set_constraint(fm)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    opt = FIRE(atoms, trajectory=str(out.with_suffix(".traj")), logfile=None)
    print(f"Fixed-mode UMA optimization (FIRE, fmax={args.fmax}, max {args.max_steps} steps) ...")
    opt.run(fmax=args.fmax, steps=args.max_steps)

    e1 = float(atoms.get_potential_energy())
    f1 = atoms.get_forces()
    fm.adjust_forces(atoms, f1)
    fmax1 = float(np.sqrt((f1 ** 2).sum(axis=1)).max())
    # full (unprojected) force along the mode -- should be ~0 if we are at the saddle
    f_full = atoms.get_forces()
    mode_flat = mode.ravel() / np.linalg.norm(mode)
    f_along = float(abs(f_full.ravel().dot(mode_flat)))
    print(f"End:   E={e1:.6f} eV, max|F_perp|={fmax1:.4f} eV/A, |F|along mode={f_along:.4f} eV/A")
    print(f"       max heavy-heavy = {_max_heavy_heavy(atoms):.2f} A, dE = {(e1-e0)*23.0605:+.2f} kcal")

    atoms.constraints = []  # drop constraint before writing / freq analysis
    write(out, atoms)
    print(f"\nRefined TS written to {out}")
    print("Verify with: python -m neb_ts.frequencies", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())