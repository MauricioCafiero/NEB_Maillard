"""CI-NEB from a ready-made seed XYZ, with a Hookean restraint keeping the
shuttle water (O_s) near the leaving-group oxygen (O4).

The acid-catalysis band keeps letting the shuttle water drift 14-19 A away from
O4: the shuttle is only weakly (non-H-bonded) engaged at the true endpoints
(Os-O4 ~2.7 A), so the NEB perpendicular force nudges the free water to lower
energy and the weak tangent spring cannot hold it.  A Hookean one-sided
restraint pulls O_s back toward O4 whenever Os-O4 exceeds `rt` (default 3.5 A).
This is consistent with both true endpoints (2.71 / 2.74 A) and exerts NO force
at the TS (shuttle engaged at ~2.7 A), so it does not distort the saddle -- it
only removes the unphysical drift artifact.

Usage:
    uv run python -m neb_ts.seed_neb_hooke output/acid/seed3.xyz \
        --outdir output/acid/neb4 --o4 4 --os 44 --rt 3.5 --kh 5.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE
from ase.constraints import Hookean

from .calculator import make_calculator
from .neb_run import KCAL_MOL_PER_EV, _write_energy_profile


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed", help="Multi-frame XYZ initial path")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--k", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--ci-max-steps", type=int, default=2000)
    ap.add_argument("--outdir", default="output/acid/neb4")
    ap.add_argument("--o4", type=int, default=4, help="Index of leaving-group O")
    ap.add_argument("--os", type=int, default=44, help="Index of shuttle O_s")
    ap.add_argument("--rt", type=float, default=3.5, help="Hookean threshold (A)")
    ap.add_argument("--kh", type=float, default=5.0, help="Hookean spring (eV/A^2)")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    images = read(args.seed, ":")
    for img in images:
        img.calc = make_calculator(0, 1)
        # Hookean: pull O_s back toward O4 when their distance exceeds rt.
        # One-sided (only acts beyond rt), so it does not interfere where the
        # shuttle is properly engaged (~2.7 A).
        img.set_constraint(Hookean(args.o4, args.os, args.kh, args.rt))
    e0 = float(images[0].get_potential_energy())
    e1 = float(images[-1].get_potential_energy())
    print(f"{len(images)} images. endpoint0 E={e0:.4f} endpoint1 E={e1:.4f} "
          f"({(e1-e0)*KCAL_MOL_PER_EV:.1f} kcal apart). "
          f"Hookean O4({args.o4})-Os({args.os}) rt={args.rt} kh={args.kh}")

    neb = NEB(images, k=args.k, climb=False, method="improvedtangent",
              remove_rotation_and_translation=True)
    opt = FIRE(neb, trajectory=str(out / "neb.traj"), logfile=None)
    print(f"NEB pre-relaxation (FIRE, fmax={args.fmax}, max {args.max_steps}) ...")
    opt.run(fmax=args.fmax, steps=args.max_steps)

    neb.climb = True
    opt_ci = FIRE(neb, trajectory=str(out / "neb_ci.traj"), logfile=None)
    print(f"CI-NEB (climbing image on, fmax={args.fmax}, max {args.ci_max_steps}) ...")
    opt_ci.run(fmax=args.fmax, steps=args.ci_max_steps)

    energies = [float(img.get_potential_energy()) for img in images]
    ts_index = int(np.argmax(energies))
    ts_energy = energies[ts_index]
    print("---- Energy profile (eV) ----")
    for i, e in enumerate(energies):
        tag = "  <-- TS" if i == ts_index else ""
        print(f"  image {i:2d}: {e: .6f}  ({(e-e0)*KCAL_MOL_PER_EV:7.2f} kcal){tag}")
    print("---- Summary ----")
    print(f"  Reactant : {e0:.4f} eV")
    print(f"  Product  : {e1:.4f} eV")
    print(f"  TS (img {ts_index}) : {ts_energy:.4f} eV  "
          f"({(ts_energy-e0)*KCAL_MOL_PER_EV:.2f} kcal/mol)")
    print(f"  Forward barrier : {(ts_energy-e0)*KCAL_MOL_PER_EV:.2f} kcal/mol")
    print(f"  Reverse barrier : {(ts_energy-e1)*KCAL_MOL_PER_EV:.2f} kcal/mol")

    ts = images[ts_index].copy(); ts.calc = None; ts.constraints = None
    write(out / "ts_structure.xyz", ts)
    write(out / "neb_final_path.xyz", images)
    _write_energy_profile(out / "energy_profile.dat", energies, e0, ts_index)
    print(f"\nTS written to {out/'ts_structure.xyz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())