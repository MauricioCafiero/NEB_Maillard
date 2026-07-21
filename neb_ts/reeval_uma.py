"""Re-evaluate a saved NEB path on the UMA MLIP (single points) and compare
to the GFN2-xTB profile.

This is the cross-check when GFN2 CI-NEB finds no interior saddle: rather than
re-running the whole band on UMA (which can dissociate far from equilibrium),
we take the *clean, GFN2-relaxed* geodesic path images and evaluate each as a
single point on UMA ``omol``.  Single points on near-equilibrium geometries are
where the MLIP is most reliable, so this tests whether UMA agrees the route is
barrierless -- or sees a barrier GFN2 does not -- without re-exploring the PES.

Usage:
    python -m neb_ts.reeval_uma <path.xyz> [--gfn2-profile energy_profile.dat] \
        [--outdir output/geodesic]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from .uma import make_uma_calculator, set_charge_spin

KCAL = 23.0605


def _load_gfn2(path: Path) -> dict[int, float]:
    """Parse an energy_profile.dat -> {image_index: energy_eV}."""
    out = {}
    if not path or not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out[int(parts[0])] = float(parts[1])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Single-point UMA re-evaluation of a NEB path.")
    ap.add_argument("path", help="Multi-model XYZ of the path images (e.g. neb_final_path.xyz)")
    ap.add_argument("--gfn2-profile", default=None, help="GFN2 energy_profile.dat to compare against")
    ap.add_argument("--outdir", default=None, help="Where to write uma_energy_profile.dat (default: path's dir)")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--multiplicity", type=int, default=1)
    args = ap.parse_args(argv)

    images = read(args.path, ":")
    outdir = Path(args.outdir) if args.outdir else Path(args.path).parent
    gfn2 = _load_gfn2(Path(args.gfn2_profile) if args.gfn2_profile else outdir / "energy_profile.dat")

    print(f"Re-evaluating {len(images)} path images on UMA uma-s-1p1 / omol (CPU) ...")
    uma_e, uma_fmax = [], []
    for i, a in enumerate(images):
        set_charge_spin(a, args.charge, args.multiplicity)
        a.calc = make_uma_calculator(args.charge, args.multiplicity)
        e = float(a.get_potential_energy())
        fmax = float(np.linalg.norm(a.get_forces(), axis=1).max())
        uma_e.append(e); uma_fmax.append(fmax)
        a.calc = None  # detach so write() does not try to pickle the predictor

    u0 = uma_e[0]
    print(f"{'img':>3} {'GFN2(eV)':>14} {'UMA(eV)':>14} {'UMA|F|max':>9} "
          f"{'UMA dE':>10} {'GFN2 dE':>10}")
    for i, e in enumerate(uma_e):
        g = gfn2.get(i)
        ud = (e - u0) * KCAL
        gd = ((g - gfn2[0]) * KCAL) if (g is not None and 0 in gfn2) else None
        gs = f"{g:14.6f}" if g is not None else f"{'--':>14}"
        gds = f"{gd:10.3f}" if gd is not None else f"{'--':>10}"
        print(f"{i:3d} {gs} {e:14.6f} {uma_fmax[i]:9.4f} {ud:10.3f} {gds}")

    out = outdir / "uma_energy_profile.dat"
    with open(out, "w") as fh:
        fh.write("# image_index  UMA_energy(eV)  UMA_rel_to_img0(eV)  UMA_rel(kcal/mol)"
                 + ("  GFN2_energy(eV)  GFN2_rel(kcal/mol)" if gfn2 else "") + "\n")
        for i, e in enumerate(uma_e):
            row = f"{i:3d}  {e: .8f}  {e - u0: .8f}  {(e - u0) * KCAL: .4f}"
            if gfn2 and i in gfn2:
                row += f"  {gfn2[i]: .8f}  {(gfn2[i] - gfn2[0]) * KCAL: .4f}"
            fh.write(row + "\n")
    print(f"\nWrote {out}")
    print(f"UMA reaction energy (img0 -> img-1): {(uma_e[-1] - u0) * KCAL:.4f} kcal/mol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())