"""Concerted seed for the acid-catalysed C-O cleavage step (decoupled from N27).

Reactant: carbinolamine + shuttle water H-bonded to O4 (H_s1...O4).
Product : contact ion pair  iminium+(C10=N27+H36) ... OH-(shuttle), with the
          departing water O4+H37+H_s1 and C10-O4 broken.

Only ONE proton transfers (H_s1: O_s -> O4).  H36 STAYS on N27 (no amine
deprotonation in this band -> no H36/H_s1 swap possible, since the shuttle is
~4-5 A from N27/H36).  The seed drives C10-O4 cleavage and the H_s1 transfer in
phase; C10-N27 simply interpolates (1.49 -> 1.43); the shuttle O_s/H_s2 and H36
are held rigid; every other O-H is kept rigid so none collapse.
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


def _frame(start, end, frac, n_orig, oh_pairs):
    sp, ep = start.positions, end.positions
    O_S, H_S1, H_S2 = _shuttle_indices(n_orig)
    d0_CO = float(np.linalg.norm(sp[O4] - sp[C10]))
    d1_CO = float(np.linalg.norm(ep[O4] - ep[C10]))
    target_CO = d0_CO + frac * (d1_CO - d0_CO)

    pos = (1 - frac) * sp + frac * ep        # linear base (morphs everything)

    # departing O4: corrective shift so C10-O4 = target (monotonic cleavage).
    cur_CO = np.linalg.norm(pos[O4] - pos[C10])
    u_CO = (pos[O4] - pos[C10]); u_CO = u_CO / np.linalg.norm(u_CO)
    pos[O4] = pos[O4] + u_CO * (target_CO - cur_CO)
    o4 = pos[O4].copy()

    # shuttle O_s: hold rigid at its (stationary) reactant position; the band
    # barely moves it.  Use the reactant O_s so it cannot drift toward N27.
    pos[O_S] = sp[O_S]
    o_s = pos[O_S].copy()

    # rigid non-transferring O-H (sugar OHs, spectator water, H37 on O4, H_s2).
    for o, h, u in oh_pairs:
        pos[h] = pos[o] + 0.965 * u

    # H36 rigid on N27 (stays -- amine NOT deprotonated in this step).
    u_nh = sp[H36] - sp[N27]; u_nh = u_nh / np.linalg.norm(u_nh)
    pos[H36] = pos[N27] + 1.02 * u_nh

    # H_s1: O_s -> O4 transfer (Hermite along the moving O_s->O4 line).
    v = o4 - o_s; L = np.linalg.norm(v)
    if L < 1e-6:
        pos[H_S1] = o_s + np.array([0.0, 0.0, 1.0])
    else:
        u = v / L
        h_on_os = o_s + 0.96 * u
        h_on_o4 = o4 - 0.96 * u
        a = _hermite(frac)
        pos[H_S1] = (1 - a) * h_on_os + a * h_on_o4

    atoms = start.copy()
    atoms.positions = pos
    return atoms


def build(start, end, n):
    n_orig = len(start) - 3
    O_S, H_S1, H_S2 = _shuttle_indices(n_orig)
    oh_pairs = _oh_pairs(start, exclude={H_S1})    # only H_s1 transfers
    frames = [start.copy()]
    for s in range(1, n):
        frames.append(_frame(start, end, s / n, n_orig, oh_pairs))
    frames.append(end.copy())
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", nargs="?", default="output/acid/reactant_relaxed.xyz")
    ap.add_argument("end", nargs="?", default="output/acid/product_relaxed.xyz")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="output/acid/seed.xyz")
    args = ap.parse_args(argv)
    frames = build(read(args.start), read(args.end), args.n)
    write(args.out, frames)
    print(f"Wrote {len(frames)}-frame acid-catalysis seed to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())