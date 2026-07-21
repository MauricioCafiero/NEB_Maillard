"""Re-evaluate a NEB final path at a single, consistent GFN2-xTB tier.

The RobustTBLite wrapper permanently loosens SCC settings (electronic_temperature
up to 50000 K) once an image fails to converge, so energies stored in the NEB
trajectory can mix tiers and be off by eV-scale (the electronic free energy is
temperature-dependent).  This reads the final multi-frame path, attaches a
*fresh* tier-1 calculator to every image, and prints/writes a clean energy
profile + the climbing-image geometry -- the profile to trust for barriers.

Usage:
    uv run python -m neb_ts.reeval_profile output/acid/neb/neb_final_path.xyz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from .calculator import make_calculator
from .neb_run import KCAL_MOL_PER_EV, _write_energy_profile


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Multi-frame XYZ final path (e.g. neb_final_path.xyz)")
    ap.add_argument("--outdir", default=None,
                    help="Write clean energy_profile.dat + ts_structure.xyz here "
                         "(default: same dir as path)")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    args = ap.parse_args(argv)

    p = Path(args.path)
    out = Path(args.outdir) if args.outdir else p.parent
    out.mkdir(parents=True, exist_ok=True)

    images = read(p, ":")
    calc_kwargs = dict(robust=False)  # plain tier-1, no fallback mutation
    energies = []
    print(f"Re-evaluating {len(images)} images at tier-1 GFN2-xTB ...")
    for i, img in enumerate(images):
        img.calc = make_calculator(args.charge, args.mult, **calc_kwargs)
        e = float(img.get_potential_energy())
        energies.append(e)
        img.calc = None

    e0 = energies[0]
    e1 = energies[-1]
    ts_index = int(np.argmax(energies))
    ts_energy = energies[ts_index]
    print("---- Clean energy profile (eV, tier-1) ----")
    for i, e in enumerate(energies):
        tag = "  <-- TS" if i == ts_index else ""
        print(f"  image {i:2d}: {e: .6f}  (d={e - e0: .6f}, "
              f"{(e - e0) * KCAL_MOL_PER_EV:7.2f} kcal){tag}")
    print("---- Summary ----")
    print(f"  Reactant : {e0:.4f} eV")
    print(f"  Product  : {e1:.4f} eV  ({(e1 - e0) * KCAL_MOL_PER_EV:.2f} kcal)")
    print(f"  TS (img {ts_index}) : {ts_energy:.4f} eV  "
          f"({(ts_energy - e0) * KCAL_MOL_PER_EV:.2f} kcal/mol)")
    print(f"  Forward barrier : {(ts_energy - e0) * KCAL_MOL_PER_EV:.2f} kcal/mol")
    print(f"  Reverse barrier : {(ts_energy - e1) * KCAL_MOL_PER_EV:.2f} kcal/mol")

    _write_energy_profile(out / "energy_profile_clean.dat", energies, e0, ts_index)
    ts = images[ts_index]
    write(out / "ts_structure_clean.xyz", ts)
    print(f"\nClean profile -> {out/'energy_profile_clean.dat'}")
    print(f"TS geometry   -> {out/'ts_structure_clean.xyz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())