"""Build a water-assisted carbinolamine dehydration (Grotthuss proton relay).

The direct N27->O4 proton transfer is a strained 4-center process with a >60
kcal gas-phase barrier; the NEB/Dimer collapse into high-energy minima instead
of finding the saddle.  The chemically correct route is a **water-assisted
Grotthuss relay**: an extra shuttle water accepts H36 from N27 and donates one
of its own protons (H_s1) to the departing O4.  This linear H-bond relay
(N27-H36 ... O_s-H_s1 ... O4) replaces the strained 4-center transfer with a
chain of normal ~1 A proton hops, lowering the barrier to a realistic ~20-30
kcal and giving a findable first-order saddle.

Atom labels (mapped order of the carbinolamine, output/dehyd/carbinolamine_relaxed.xyz):
    C10, O4, N27, H36 (N27 proton), H37 (O4 proton), spectator water O_w=41/H42/H43.
Three NEW atoms are appended for the shuttle water: O_s, H_s1 (donated to O4),
H_s2 (stays on the shuttle).  After the step:
    C10=N27 (1.28),  departing water = O4 + H37 + H_s1,
    shuttle water = O_s + H36 + H_s2,  spectator water 41/42/43 unchanged.
"""
from __future__ import annotations

import argparse

import numpy as np
from ase import Atoms
from ase.io import read, write

C10, O4, N27, H36, H37 = 10, 4, 27, 36, 37
O_W = 41   # spectator water oxygen (already present)


def _place_shuttle(atoms):
    """Append a shuttle water bridging N27 (accept H36) and O4 (donate H_s1).

    Geometric requirement: the donor H (H36) and the acceptor O4 sit on
    *opposite* sides of N27 in the N27-C10-O4 plane (H36-N27-O4 ~112 deg), so a
    water placed *in* the plane cannot both accept H36 and donate to O4 -- the
    earlier in-plane placement put O_s opposite H36, forcing a ~180 deg H36
    reorientation (+248 kcal seed strain) and, on relaxation, let H_s1 escape to
    a sugar hydroxyl.  Placing O_s **above the N27-C10-O4 plane** (the side H36
    points to, ~0.95 A up) makes H36 point naturally at O_s (O_s...H36 ~1.8 A
    H-bond) while O_s-O4 ~3.6 A lets H_s1 reach O4.  H_s1 is oriented toward O4
    (donated), H_s2 away.
    """
    p = atoms.positions
    c10, n27, o4 = p[C10], p[N27], p[O4]
    # normal to the C10-N27-O4 plane; choose the side H36 points to.
    normal = np.cross(c10 - n27, o4 - n27)
    normal = normal / np.linalg.norm(normal)
    side = np.sign((p[H36] - n27) @ normal)
    if side == 0:
        side = 1.0
    # O_s above the plane, 2.7 A from N27 (N...O H-bond), tilted slightly toward
    # O4 so H_s1 can reach it.  Gives O_s-H36 ~1.8, O_s-O4 ~3.6.
    o_s = n27 + 2.7 * (side * normal) + 0.3 * (o4 - n27) / np.linalg.norm(o4 - n27)
    # H_s1 toward O4 (donated proton), ~0.96 A from O_s along O_s->O4
    u_to_o4 = o4 - o_s
    u_to_o4 = u_to_o4 / np.linalg.norm(u_to_o4)
    h_s1 = o_s + 0.96 * u_to_o4
    # H_s2 away from O4, tetrahedral-ish (~104 deg from H_s1)
    u_away = -u_to_o4 + 0.4 * (side * normal)
    u_away = u_away / np.linalg.norm(u_away)
    h_s2 = o_s + 0.96 * u_away
    return o_s, h_s1, h_s2


def build_reactant(carbinolamine):
    """Carbinolamine + shuttle water (bridging N27 and O4)."""
    o_s, h_s1, h_s2 = _place_shuttle(carbinolamine)
    syms = list(carbinolamine.get_chemical_symbols()) + ["O", "H", "H"]
    pos = np.vstack([carbinolamine.positions, [o_s, h_s1, h_s2]])
    return Atoms(symbols=syms, positions=pos)


def build_product(carbinolamine=None, *, imine_product="output/imine_product_relaxed.xyz"):
    """Imine + departing water (O4+H37+H_s1) + shuttle water (O_s+H36+H_s2).

    Built from the **already-stable imine product** (C10=N27 ~1.26, the
    departing water O4+H36+H37 already relaxed) by *relocating* H36 from the
    departing water onto a new shuttle water and adding H_s1 to the departing
    water.  This preserves the proven-stable imine geometry rather than
    re-deriving it (the from-carbinolamine derivation relaxed the C=N back to a
    single bond, i.e. fell into a wrong basin).
    """
    imine = read(imine_product)
    n = len(imine)                       # 44
    O_S, H_S1, H_S2 = n, n + 1, n + 2
    p = imine.positions.copy()
    n27, o4 = p[N27], p[O4]

    # place the shuttle water O_s above the N27-C10-O4 plane (same side as in the
    # reactant), ~2.7 A from N27.  In the product O_s carries H36 (accepted)
    # plus H_s2; it has donated H_s1 to O4.
    c10 = p[C10]
    normal = np.cross(c10 - n27, o4 - n27)
    normal = normal / np.linalg.norm(normal)
    o_s = n27 + 2.7 * normal + 0.3 * (o4 - n27) / np.linalg.norm(o4 - n27)

    # move H36 (currently on the departing water O4) onto the shuttle O_s.
    # reposition the shuttle as a proper water O_s + H36 + H_s2, oriented away
    # from N27.
    away_n = (o_s - n27); away_n = away_n / np.linalg.norm(away_n)
    perp3 = np.cross(away_n, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(perp3) < 1e-3:
        perp3 = np.cross(away_n, np.array([0.0, 1.0, 0.0]))
    perp3 = perp3 / np.linalg.norm(perp3)
    a3 = np.radians(52.0)
    h36 = o_s + 0.96 * (np.cos(a3) * away_n + np.sin(a3) * perp3)
    h_s2 = o_s + 0.96 * (np.cos(a3) * away_n - np.sin(a3) * perp3)

    # add H_s1 onto the departing water O4 (replacing the H36 we moved away) so
    # the departing water is a proper H2O = O4 + H37 + H_s1.
    u_h37 = p[H37] - o4; u_h37 = u_h37 / np.linalg.norm(u_h37)
    # H_s1 opposite H37 through O4 (tetrahedral ~104 deg), biased away from C10.
    c10 = p[C10]
    away_c10 = (o4 - c10); away_c10 = away_c10 / np.linalg.norm(away_c10)
    d = away_c10 - np.dot(away_c10, u_h37) * u_h37
    if np.linalg.norm(d) < 1e-6:
        d = np.array([-u_h37[1], u_h37[0], 0.0])
    d = d / np.linalg.norm(d)
    h_s1 = o4 + 0.96 * d

    syms = list(imine.get_chemical_symbols()) + ["O", "H", "H"]
    pos = np.vstack([p, [o_s, h_s1, h_s2]])
    # overwrite H36 (already in `p`) with its new shuttle position
    pos[H36] = h36
    from ase import Atoms
    return Atoms(symbols=syms, positions=pos)


def _report(atoms):
    p = atoms.positions
    n = len(p) - 3
    O_S, H_S1, H_S2 = n, n + 1, n + 2
    def d(i, j):
        return float(np.linalg.norm(p[i] - p[j]))
    print(f"  C10-N27 : {d(C10,N27):.3f}   (imine ~1.28)")
    print(f"  C10-O4  : {d(C10,O4):.3f}   (broken >2.5 / intact ~1.41)")
    print(f"  N27-H36 : {d(N27,H36):.3f}   (on N reactant / on shuttle product)")
    print(f"  O_s-H36 : {d(O_S,H36):.3f}   (shuttle O-H ~0.96 product)")
    print(f"  O4-H_s1 : {d(O4,H_S1):.3f}   (departing water O-H ~0.96 product)")
    print(f"  O4-H37  : {d(O4,H37):.3f}   (departing water O-H ~0.96)")
    print(f"  O_s-N27 : {d(O_S,N27):.3f}   (relay H-bond ~2.0)")
    print(f"  O_s-O4  : {d(O_S,O4):.3f}   (relay H-bond ~2.8)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carbinolamine", nargs="?", default="output/dehyd/carbinolamine_relaxed.xyz")
    ap.add_argument("--out-r", default="output/shuttle/reactant.xyz")
    ap.add_argument("--out-p", default="output/shuttle/product.xyz")
    args = ap.parse_args(argv)
    cab = read(args.carbinolamine)
    r = build_reactant(cab)
    print("Water-assisted REACTANT (carbinolamine + shuttle water):")
    _report(r)
    p = build_product(cab)
    print("\nWater-assisted PRODUCT (imine + departing water + shuttle water):")
    _report(p)
    from pathlib import Path
    Path(args.out_r).parent.mkdir(parents=True, exist_ok=True)
    write(args.out_r, r); write(args.out_p, p)
    print(f"\nWrote {args.out_r} and {args.out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())