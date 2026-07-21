"""Concerted, shuttle-restrained initial path for the water-assisted dehydration.

Fixes the two failure modes of the plain relay seed (build_relay_seed):
  1. Shuttle dissociation -- O_s was linearly interpolated and drifted 15-40 A
     away (it is only H-bonded, so the NEB let it go).  Here O_s is held at a
     fixed H-bond distance from N27 (it translates WITH N27), so it cannot
     dissociate.
  2. Zwitterion -- C10-N27 chord-dipped to ~1.25 mid-path (the imine formed
     before the proton relay, giving a +117 kcal iminium+hydroxide).  Here
     C10-N27 is driven *monotonically* by translating the asparagine fragment
     along C10->N27, in phase with C-O cleavage and the two proton hops.

The reaction coordinate is driven concerted by distances:
  C10-O4   1.41 -> 3.35   (cleavage; O4 moves out along C10->O4)
  C10-N27  1.47 -> 1.26   (C=N; asparagine translates toward C10)
  O_s-N27  ~2.7 -> ~3.0   (shuttle stays H-bonded, moves with N27)
  H36   N27 -> O_s        (relay hop 1, Hermite along the moving N27->O_s line)
  H_s1  O_s -> O4         (relay hop 2, Hermite along the moving O_s->O4 line)
All non-transferring O-H bonds (sugar hydroxyls, spectator water, H37, H_s2)
are kept rigid on their oxygen at the reactant orientation so none collapse.
"""
from __future__ import annotations

import argparse

import numpy as np
from ase.io import read, write

C10, O4, N27, H36, H37 = 10, 4, 27, 36, 37


def _shuttle_indices(n_orig):
    return n_orig, n_orig + 1, n_orig + 2  # O_s, H_s1, H_s2


def _hermite(frac):
    return 3 * frac ** 2 - 2 * frac ** 3


def _oh_pairs(start, exclude):
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
                pairs.append((o, h, (sp[h] - sp[o]) / d))
    return pairs


def _asparagine_indices():
    from .build_imine import _asparagine_indices as _ai
    return _ai(read("output/reactant_mapped.xyz"))


def _bow_dir(start, n_orig):
    """Direction (lab frame) to bow H_s1's Os->O4 path away from the sugar's
    O2-H21 proton, which the straight Os->O4 line clips (H21 sits ~1.3 A off
    the line -> H_s1 collides at 0.61 A).  Bow ~up-and-away from H21.  The
    molecule barely translates, so a fixed lab-frame direction is fine.
    """
    p = start.positions
    O_S, _, _ = _shuttle_indices(n_orig)
    os_, o4, h21 = p[O_S], p[O4], 21
    seg = o4 - os_; L = np.linalg.norm(seg)
    if L < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    u = seg / L
    t = np.clip((h21 - os_) @ u / L, 0, 1)
    perp = h21 - (os_ + t * seg)
    n = np.linalg.norm(perp)
    if n < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return perp / n


def _frame(start, end, frac, n_orig, asn_idx, oh_pairs, bow_dir):
    sp, ep = start.positions, end.positions
    O_S, H_S1, H_S2 = _shuttle_indices(n_orig)

    # target distances (linear in fraction)
    d0_CN = float(np.linalg.norm(sp[N27] - sp[C10]))
    d1_CN = float(np.linalg.norm(ep[N27] - ep[C10]))
    d0_CO = float(np.linalg.norm(sp[O4] - sp[C10]))
    d1_CO = float(np.linalg.norm(ep[O4] - ep[C10]))
    d0_sN = float(np.linalg.norm(sp[O_S] - sp[N27]))
    d1_sN = float(np.linalg.norm(ep[O_S] - ep[N27]))
    target_CN = d0_CN + frac * (d1_CN - d0_CN)
    target_CO = d0_CO + frac * (d1_CO - d0_CO)
    target_sN = d0_sN + frac * (d1_sN - d0_sN)

    pos = (1 - frac) * sp + frac * ep   # linear base for every atom (morphs
    #   sp3->sp2 at N27, rotates the leaving water, etc.)

    # asparagine: corrective translation so C10-N27 = target (monotonic, no chord
    # dip) while PRESERVING the linearly-morphed geometry (N27 sp3->sp2).  Pure
    # rigid translation of the start fragment (the first attempt) left N27 stuck
    # in its sp3 geometry at a 1.27 A double-bond distance -> +347 kcal angle
    # strain.  Linear interp + corrective shift removes both the chord dip and
    # the strain.  This moves N27 (and H36, in the fragment) -- H36 overridden
    # below.
    cur_CN = np.linalg.norm(pos[N27] - pos[C10])
    u_CN = (pos[N27] - pos[C10]); u_CN = u_CN / np.linalg.norm(u_CN)
    pos[asn_idx] = pos[asn_idx] + u_CN * (target_CN - cur_CN)
    n27 = pos[N27].copy()

    # shuttle O_s: hold H-bonded to N27 (moves WITH N27), in the reactant
    # N27->O_s direction, at the interpolated O_s-N27 distance -- cannot drift.
    u_sN = sp[O_S] - sp[N27]; u_sN = u_sN / np.linalg.norm(u_sN)
    o_s = n27 + target_sN * u_sN
    pos[O_S] = o_s

    # departing O4: corrective shift so C10-O4 = target (monotonic cleavage, no
    # dip) while preserving the morphed leaving-water orientation.
    cur_CO = np.linalg.norm(pos[O4] - pos[C10])
    u_CO = (pos[O4] - pos[C10]); u_CO = u_CO / np.linalg.norm(u_CO)
    pos[O4] = pos[O4] + u_CO * (target_CO - cur_CO)
    o4 = pos[O4].copy()

    # rigid non-transferring O-H bonds (sugar OHs, spectator water, H37, H_s2)
    for o, h, u in oh_pairs:
        pos[h] = pos[o] + 0.965 * u

    # H36: N27 -> O_s relay hop (Hermite along the moving N27->O_s line).
    v = o_s - n27; L = np.linalg.norm(v)
    if L < 1e-6:
        pos[H36] = n27 + np.array([0.0, 0.0, 1.0])
    else:
        u = v / L
        h_on_n = n27 + 1.02 * u
        h_on_o = o_s - 0.96 * u
        pos[H36] = (1 - _hermite(frac)) * h_on_n + _hermite(frac) * h_on_o

    # H_s1: O_s -> O4 relay hop along a Bezier arc that bows away from the
    # sugar O2-H21 proton (the straight Os->O4 line clips H21 at ~0.6 A).
    v2 = o4 - o_s; L2 = np.linalg.norm(v2)
    if L2 < 1e-6:
        pos[H_S1] = o_s + np.array([0.0, 0.0, 1.0])
    else:
        u2 = v2 / L2
        h_on_os = o_s + 0.96 * u2
        h_on_o4 = o4 - 0.96 * u2
        t = _hermite(frac)
        apex = 0.5 * (h_on_os + h_on_o4) + 0.9 * bow_dir
        pos[H_S1] = (1 - t) ** 2 * h_on_os + 2 * (1 - t) * t * apex + t ** 2 * h_on_o4

    atoms = start.copy()
    atoms.positions = pos
    return atoms


def build(start, end, n):
    n_orig = len(start) - 3
    O_S, H_S1, H_S2 = _shuttle_indices(n_orig)
    asn_idx = _asparagine_indices()
    oh_pairs = _oh_pairs(start, exclude={H_S1})   # H_s1 transfers; H36 is N-H
    bow_dir = _bow_dir(start, n_orig)
    frames = [start.copy()]
    for s in range(1, n):
        frames.append(_frame(start, end, s / n, n_orig, asn_idx, oh_pairs, bow_dir))
    frames.append(end.copy())
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", nargs="?", default="output/shuttle/reactant_relaxed.xyz")
    ap.add_argument("end", nargs="?", default="output/shuttle/product_relaxed.xyz")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="output/shuttle/seed_concerted.xyz")
    args = ap.parse_args(argv)
    frames = build(read(args.start), read(args.end), args.n)
    write(args.out, frames)
    print(f"Wrote {len(frames)}-frame concerted relay seed to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())