"""Build a collision-free seed by linear interpolation with O4-H37 rigid.

The only genuine collision in a straight linear interpolation between the
carbinolamine and the contact intermediate is the O4-H37 pair: O4 (the
carbinolamine OH oxygen) and its proton H37 are interpolated independently, so
the two momentarily squash to ~0.79 A at the midpoint (+160 kcal/mol explosion).

Every other atom pair stays well separated along the linear chord.  This
builder therefore does plain linear interpolation of *all* atoms start->end,
except H37, which is carried rigidly on O4 (H37 = O4(t) + 0.965 * unit(start
O4->H37)).  No other geometric assumption is imposed -- in particular the N-H
proton H36 is left to interpolate freely (the real H36 transfer is a swing-
around, not an in-line N->O motion, so any in-line constraint is wrong).  The
NEB's perpendicular relaxation and IDPP re-interpolation handle the rest.
"""
from __future__ import annotations

import argparse

import numpy as np
from ase.io import read, write

C10, O4, N27, H36, H37 = 10, 4, 27, 36, 37


def build_linear_rigid_seed(start, end, n):
    sp, ep = start.positions, end.positions
    u_h37 = sp[H37] - sp[O4]
    u_h37 = u_h37 / np.linalg.norm(u_h37)
    frames = [start.copy()]
    for s in range(1, n):
        frac = s / n
        atoms = start.copy()
        pos = (1 - frac) * sp + frac * ep          # linear for every atom
        pos[H37] = pos[O4] + 0.965 * u_h37          # ...except H37, rigid on O4
        atoms.positions = pos
        frames.append(atoms)
    frames.append(end.copy())
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", nargs="?", default="output/dehyd/carbinolamine_relaxed.xyz")
    ap.add_argument("end", nargs="?", default="output/dehyd/intermediate.xyz")
    ap.add_argument("--n", type=int, default=11)
    ap.add_argument("--out", default="output/ts1/seed_linear.xyz")
    args = ap.parse_args(argv)
    frames = build_linear_rigid_seed(read(args.start), read(args.end), args.n)
    write(args.out, frames)
    print(f"Wrote {len(frames)}-frame seed to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())