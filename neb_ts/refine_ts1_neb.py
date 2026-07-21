"""Refine the dehydration TS1 with a NEB seeded by a known-good path segment.

The first dehydration CI-NEB (output/dehyd) produced a collision-free path with
a clean interior saddle at image 4 (+21 kcal/mol), but its climbing image ran
to the high endothermic product endpoint instead of converging the saddle.  A
fresh NEB between the two flanking minima (carbinolamine and the contact imine*H2O
intermediate, both below the saddle) is the right setup, but IDPP interpolation
between those endpoints sends atoms through each other.

This script instead seeds the band with the *already-good* images 0..5 from the
first NEB path -- no re-interpolation -- so the band starts collision-free and
the climbing image converges on the +21 kcal TS1 (the highest point, above both
endpoints).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.mep import NEB
from ase.optimize import FIRE
from ase.io import read, write

from .calculator import make_calculator
from .neb_run import KCAL_MOL_PER_EV, _write_energy_profile


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default="output/dehyd/neb_final_path.xyz",
                    help="First NEB path (multi-model XYZ) to seed the band")
    ap.add_argument("--i0", type=int, default=0, help="Start image index (carbinolamine)")
    ap.add_argument("--i1", type=int, default=5, help="End image index (contact intermediate)")
    ap.add_argument("--subdivide", type=int, default=1,
                    help="Subdivide each good-path segment into this many sub-images (1 = use as-is)")
    ap.add_argument("--end0", default=None, help="Optional relaxed start minimum XYZ (else path[i0])")
    ap.add_argument("--end1", default=None, help="Optional relaxed end minimum XYZ (else path[i1])")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--k", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--outdir", default="output/ts1")
    args = ap.parse_args(argv)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    src = read(args.path, ":")
    i0, i1 = args.i0, args.i1
    good = src[i0:i1 + 1]                      # collision-free path through the saddle
    end0 = (read(args.end0) if args.end0 else good[0]).copy()
    end1 = (read(args.end1) if args.end1 else good[-1]).copy()
    e0 = float(_energy(end0))
    e1 = float(_energy(end1))
    print(f"endpoint 0: E={e0:.4f} eV   endpoint 1: E={e1:.4f} eV  "
          f"({(e1-e0)*KCAL_MOL_PER_EV:.1f} kcal/mol apart)")

    # Build the band seeded by the good path.  Subdivide each consecutive pair of
    # good images (which are close on the MEP) by linear interpolation -- short,
    # safe segments, never interpolating across the whole reaction.
    images = [end0]
    anchors = good                       # i0..i1 from the source path
    for seg in range(len(anchors) - 1):
        a, b = anchors[seg].positions, anchors[seg + 1].positions
        for s in range(args.subdivide):
            frac = (s + 1) / (args.subdivide + 1)
            m = end0.copy()
            m.positions = (1 - frac) * a + frac * b
            images.append(m)
    images.append(end1)
    for img in images:
        img.calc = make_calculator(0, 1)

    neb = NEB(images, k=args.k, climb=False, method="improvedtangent",
              remove_rotation_and_translation=True)
    opt = FIRE(neb, trajectory=str(out / "neb.traj"), logfile=None)
    print(f"NEB pre-relaxation ({len(images)} images, FIRE, fmax={args.fmax}, "
          f"max {args.max_steps} steps) ...")
    opt.run(fmax=args.fmax, steps=args.max_steps)

    neb.climb = True
    opt_ci = FIRE(neb, trajectory=str(out / "neb_ci.traj"), logfile=None)
    print(f"CI-NEB (climbing image on, fmax={args.fmax}, max {args.max_steps} steps) ...")
    opt_ci.run(fmax=args.fmax, steps=args.max_steps)

    energies = [float(img.get_potential_energy()) for img in images]
    ts_index = int(np.argmax(energies))
    ts_energy = energies[ts_index]
    print("---- Energy profile (eV) ----")
    for i, e in enumerate(energies):
        tag = "  <-- TS" if i == ts_index else ""
        print(f"  image {i:2d}: {e: .6f}  (d={e - e0: .6f}, {(e-e0)*KCAL_MOL_PER_EV:7.2f} kcal){tag}")
    print("---- Summary ----")
    print(f"  Reactant : {e0:.4f} eV")
    print(f"  Product  : {e1:.4f} eV")
    print(f"  TS (img {ts_index}) : {ts_energy:.4f} eV")
    print(f"  Forward barrier : {ts_energy-e0:.4f} eV ({(ts_energy-e0)*KCAL_MOL_PER_EV:.2f} kcal/mol)")
    print(f"  Reverse barrier : {ts_energy-e1:.4f} eV ({(ts_energy-e1)*KCAL_MOL_PER_EV:.2f} kcal/mol)")

    ts = images[ts_index].copy(); ts.calc = None
    write(out / "ts_structure.xyz", ts)
    write(out / "neb_final_path.xyz", images)
    _write_energy_profile(out / "energy_profile.dat", energies, e0, ts_index)
    print(f"\nTS written to {out/'ts_structure.xyz'}")
    return 0


def _energy(atoms):
    a = atoms.copy()
    a.calc = make_calculator(0, 1)
    return a.get_potential_energy()


if __name__ == "__main__":
    raise SystemExit(main())