"""Build a dense seed from the 5 converged anchor frames of the FIRST acid
CI-NEB (output/acid/neb/neb_final_path.xyz images 2..6), which are the only
part of that band that worked: the shuttle water stays in place (Os-N27
4.4-5.0 A) while C10-O4 cleaves (1.41->3.17) and Hs1 transfers Os->O4.

seed2 (built fresh from r_min/p_min) let the shuttle drift 14-16 A away,
because both endpoints hold the shuttle as a *free* (non-H-bonded) water, so
the NEB perpendicular force nudged it to lower energy and the weak spring
could not hold it.  Re-using the already-converged anchor path (with the
shuttle in place at every anchor) and a stronger spring keeps the shuttle
anchored, so the climbing image converges on the real ~23-25 kcal TS.

Linear interpolation BETWEEN consecutive (close) NEB anchors is smooth
(anchors are neighboring images, so no chord-dip / Hs1-jump artifacts).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write


def _interp(a, b, frac):
    out = a.copy()
    out.positions = (1 - frac) * a.positions + frac * b.positions
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("anchors", nargs="?", default="output/acid/neb/neb_final_path.xyz",
                    help="Multi-frame XYZ; frames [anchor0..anchor4] are taken by index")
    ap.add_argument("--idx", type=int, nargs=5, default=[2, 3, 4, 5, 6],
                    help="Frame indices of the 5 anchors (R, mid1, mid2, mid3, P)")
    ap.add_argument("--sub", type=int, default=3,
                    help="Interpolated frames between consecutive anchors (default 3)")
    ap.add_argument("--out", default="output/acid/seed3.xyz")
    args = ap.parse_args(argv)

    allf = read(args.anchors, ":")
    anchors = [allf[i].copy() for i in args.idx]
    frames = [anchors[0]]
    for seg in range(len(anchors) - 1):
        a, b = anchors[seg], anchors[seg + 1]
        for s in range(1, args.sub + 1):
            frames.append(_interp(a, b, s / (args.sub + 1)))
        frames.append(b)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write(args.out, frames)
    print(f"Wrote {len(frames)}-frame anchor seed to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())