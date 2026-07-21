"""Run a CI-NEB from a ready-made initial-path XYZ (no re-interpolation).

Unlike ``refine_ts1_neb`` (which re-subdivides between anchor images) this reads
a multi-frame XYZ *verbatim* as the initial band, attaches the crash-proof
GFN2-xTB calculator to every image, runs plain NEB then CI-NEB, and reports the
climbing image.  Use this with a hand-built collision-free seed (e.g.
``build_water_seed``) so the band starts on the right ridge and the climbing
image converges on the true saddle.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE

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
    ap.add_argument("--outdir", default="output/ts1")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    images = read(args.seed, ":")
    for img in images:
        img.calc = make_calculator(0, 1)
    e0 = float(images[0].get_potential_energy())
    e1 = float(images[-1].get_potential_energy())
    print(f"{len(images)} images.  endpoint0 E={e0:.4f}  endpoint1 E={e1:.4f} "
          f"({(e1-e0)*KCAL_MOL_PER_EV:.1f} kcal/mol apart)")

    neb = NEB(images, k=args.k, climb=False, method="improvedtangent",
              remove_rotation_and_translation=True)
    opt = FIRE(neb, trajectory=str(out / "neb.traj"), logfile=None)
    print(f"NEB pre-relaxation (FIRE, fmax={args.fmax}, max {args.max_steps} steps) ...")
    opt.run(fmax=args.fmax, steps=args.max_steps)

    neb.climb = True
    opt_ci = FIRE(neb, trajectory=str(out / "neb_ci.traj"), logfile=None)
    print(f"CI-NEB (climbing image on, fmax={args.fmax}, max {args.ci_max_steps} steps) ...")
    opt_ci.run(fmax=args.fmax, steps=args.ci_max_steps)

    energies = [float(img.get_potential_energy()) for img in images]
    ts_index = int(np.argmax(energies))
    ts_energy = energies[ts_index]
    print("---- Energy profile (eV) ----")
    for i, e in enumerate(energies):
        tag = "  <-- TS" if i == ts_index else ""
        print(f"  image {i:2d}: {e: .6f}  (d={e-e0: .6f}, {(e-e0)*KCAL_MOL_PER_EV:7.2f} kcal){tag}")
    print("---- Summary ----")
    print(f"  Reactant : {e0:.4f} eV")
    print(f"  Product  : {e1:.4f} eV")
    print(f"  TS (img {ts_index}) : {ts_energy:.4f} eV  ({(ts_energy-e0)*KCAL_MOL_PER_EV:.2f} kcal/mol)")
    print(f"  Forward barrier : {(ts_energy-e0)*KCAL_MOL_PER_EV:.2f} kcal/mol")
    print(f"  Reverse barrier : {(ts_energy-e1)*KCAL_MOL_PER_EV:.2f} kcal/mol")

    ts = images[ts_index].copy(); ts.calc = None
    write(out / "ts_structure.xyz", ts)
    write(out / "neb_final_path.xyz", images)
    _write_energy_profile(out / "energy_profile.dat", energies, e0, ts_index)
    print(f"\nTS written to {out/'ts_structure.xyz'}")
    print(f"Final path written to {out/'neb_final_path.xyz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())