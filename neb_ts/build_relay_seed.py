"""Build a collision-free initial path for the water-assisted (Grotthuss) dehydration.

Endpoints are the relaxed water-assisted reactant (carbinolamine + shuttle water
bridging N27 and O4) and product (imine + departing water + shuttle water).  The
reaction coordinate is a concerted Grotthuss relay:
    N27-H36 -> O_s   (H36 transfers from amine N to the shuttle water O)
    O_s-H_s1 -> O4    (the shuttle donates H_s1 to the departing O4)
    C10-O4  1.41->3.4 (C-O cleavage, water departs)
    C10-N27 1.48->1.28 (C=N formation)

The seed is plain linear interpolation of every atom start->end, except the two
*non-transferring* water protons (H37 on the departing water O4, H_s2 on the
shuttle water O_s) are carried rigidly on their oxygens (so the O-H water bonds
never collapse -- the failure mode of unconstrained linear interpolation).  The
two *transferring* protons (H36, H_s1) are interpolated by Hermite step between
their donor-bonded and acceptor-bonded positions along the relay line, so each
stays bonded to its donor early and reaches its acceptor late, with the
compressed bridging geometry only at the TS frame.
"""
from __future__ import annotations

import argparse

import numpy as np
from ase.io import read, write

C10, O4, N27, H36, H37 = 10, 4, 27, 36, 37


def _shuttle_indices(n_orig):
    # appended shuttle atoms: O_s = n_orig, H_s1 = n_orig+1, H_s2 = n_orig+2
    return n_orig, n_orig + 1, n_orig + 2


def _hermite(frac):
    return 3 * frac ** 2 - 2 * frac ** 3


def _oh_pairs(start, exclude):
    """All O-H pairs (<1.1 A) in `start`, as (O_idx, H_idx, unit O->H vector).

    Used to keep every water/sugar O-H bond rigid along the path (independent
    linear interpolation of O and H collapses each O-H bond, the recurring
    failure mode).  `exclude` holds H indices that *transfer* (H_s1) and so must
    not be rigidified.
    """
    sp = start.positions
    syms = start.get_chemical_symbols()
    pairs = []
    for o, so in enumerate(syms):
        if so != "O":
            continue
        for h, sh in enumerate(syms):
            if sh != "H" or h in exclude:
                continue
            d = np.linalg.norm(sp[h] - sp[o])
            if 0.7 < d < 1.15:
                u = (sp[h] - sp[o]) / d
                pairs.append((o, h, u))
    return pairs


def _frame(start, end, frac, n_orig, oh_pairs):
    sp, ep = start.positions, end.positions
    O_S, H_S1, H_S2 = _shuttle_indices(n_orig)
    pos = (1 - frac) * sp + frac * ep          # linear for every atom

    # keep every non-transferring O-H bond rigid on its O (reactant orientation)
    for o, h, u in oh_pairs:
        pos[h] = pos[o] + 0.965 * u

    # H36: N27 -> O_s relay hop.  Donor-bonded (1.02 from N27 toward O_s) early,
    # acceptor-bonded (0.96 from O_s toward N27) late, Hermite blend.
    n27, o_s = pos[N27], pos[O_S]
    v = o_s - n27; L = np.linalg.norm(v)
    if L < 1e-6:
        pos[H36] = n27 + np.array([0.0, 0.0, 1.0])
    else:
        u = v / L
        h_on_n = n27 + 1.02 * u
        h_on_o = o_s - 0.96 * u
        a = _hermite(frac)
        pos[H36] = (1 - a) * h_on_n + a * h_on_o

    # H_s1: O_s -> O4 relay hop.  Donor-bonded (0.96 from O_s toward O4) early,
    # acceptor-bonded (0.96 from O4 toward O_s) late, Hermite blend.
    o4 = pos[O4]
    v2 = o4 - o_s; L2 = np.linalg.norm(v2)
    if L2 < 1e-6:
        pos[H_S1] = o_s + np.array([0.0, 0.0, 1.0])
    else:
        u2 = v2 / L2
        h_on_os = o_s + 0.96 * u2
        h_on_o4 = o4 - 0.96 * u2
        a2 = _hermite(frac)
        pos[H_S1] = (1 - a2) * h_on_os + a2 * h_on_o4

    atoms = start.copy()
    atoms.positions = pos
    return atoms


def build_relay_seed(start, end, n):
    n_orig = len(start) - 3
    O_S, H_S1, H_S2 = _shuttle_indices(n_orig)
    # H_s1 transfers O_s->O4 (exclude from rigid O-H).  H36 is on N27 in the
    # reactant (N-H, not O-H) so it is not picked up as an O-H pair anyway.
    oh_pairs = _oh_pairs(start, exclude={H_S1})
    frames = [start.copy()]
    for s in range(1, n):
        frames.append(_frame(start, end, s / n, n_orig, oh_pairs))
    frames.append(end.copy())
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", nargs="?", default="output/shuttle/reactant_relaxed.xyz")
    ap.add_argument("end", nargs="?", default="output/shuttle/product_relaxed.xyz")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--out", default="output/shuttle/seed_relay.xyz")
    args = ap.parse_args(argv)
    frames = build_relay_seed(read(args.start), read(args.end), args.n)
    write(args.out, frames)
    print(f"Wrote {len(frames)}-frame relay seed to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())