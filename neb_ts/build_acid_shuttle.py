"""Acid-catalysis shuttle: the water protonates the leaving O4, decoupled from N27.

The bridging (Grotthuss) shuttle collapsed to a proton-swap side-channel
(H36<->H_s1 across the N27<->O_s H-bond, +49 kcal) that does NOT dehydrate.
The fix (user choice): place the shuttle water near **O4 only** -- H_s1
H-bonded to O4 (poised to protonate the leaving group) -- and **away from
N27/H36**, so the swap is geometrically impossible.  The rate-limiting step is
then the acid-catalysed C-O cleavage:

    carbinolamine + shuttle(H2O @O4)
        ->  iminium(C10=N27+H36) + departing water(O4+H37+H_s1) + shuttle-OH-

i.e. H_s1 transfers O_s->O4 (protonating the hydroxyl -> good leaving group),
C10-O4 cleaves, C10=N27 forms.  H36 STAYS on N27 (iminium) -- the amine
deprotonation is a separate, later fast step, NOT in this band.  O_s sits beyond
O4 along C10->O4 (~4.9 A from N27, ~3.2+ A from every sugar O), so it cannot
compete via the swap or lose H_s1 to a sugar hydroxyl.

Atom labels (mapped carbinolamine order): C10=10, O4=4, N27=27, H36=36, H37=37.
Shuttle appended at n=44: O_s=44, H_s1=45, H_s2=46.
"""
from __future__ import annotations

import argparse

import numpy as np
from ase import Atoms
from ase.io import read, write

C10, O4, N27, H36, H37 = 10, 4, 27, 36, 37


def _shuttle_indices(n_orig):
    return n_orig, n_orig + 1, n_orig + 2  # O_s, H_s1, H_s2


def _best_shuttle_dir(p, anchor, exclude, u, radius=2.8, n=96):
    """Unit direction PERPENDICULAR to `u` (the C10->O4 axis) maximizing the min
    distance from `anchor + radius*dir` to every atom NOT in `exclude`.
    Perpendicular-only ensures the departing water (leaving along +u) always
    clears the shuttle by `radius`.  Searches a circle in the plane perp to u;
    ties broken toward being farther from N27.
    """
    n27 = p[N27]
    # orthonormal basis (e1, e2) of the plane perpendicular to u
    ref = np.array([0.0, 0.0, 1.0])
    if abs(u @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, ref); e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    best_dir, best_min, best_n27 = e1, -1.0, -1.0
    for i in range(n):
        t = 2.0 * np.pi * i / n
        d = np.cos(t) * e1 + np.sin(t) * e2
        cand = anchor + radius * d
        mind = min(np.linalg.norm(cand - p[j]) for j in range(len(p)) if j not in exclude)
        dn27 = np.linalg.norm(cand - n27)
        if (mind > best_min + 1e-3) or (abs(mind - best_min) <= 1e-3 and dn27 > best_n27):
            best_dir, best_min, best_n27 = d, mind, dn27
    return best_dir


def _place_shuttle_at_o4(atoms):
    """O_s in the clearest spot ~2.8 A from O4, PERPENDICULAR to C10->O4 (so the
    departing water clears it); H_s1 H-bonded to O4; H_s2 pointing away."""
    p = atoms.positions
    o4 = p[O4]
    u = (o4 - p[C10]); u = u / np.linalg.norm(u)
    d = _best_shuttle_dir(p, anchor=o4, exclude={O4, H37}, u=u)
    o_s = o4 + 2.8 * d
    w_to_o4 = (o4 - o_s); w_to_o4 = w_to_o4 / np.linalg.norm(w_to_o4)
    h_s1 = o_s + 0.96 * w_to_o4            # toward O4 (H_s1...O4 ~1.84 H-bond)
    away = -w_to_o4
    perp = np.cross(np.cross(away, d), away)
    nrm = np.linalg.norm(perp)
    perp = perp / nrm if nrm > 1e-6 else d
    a = np.radians(52.0)
    h_s2 = o_s + 0.96 * (np.cos(a) * away + np.sin(a) * perp)
    return o_s, h_s1, h_s2


def build_reactant(carbinolamine):
    """Carbinolamine + shuttle water H-bonded to O4 (acid catalyst)."""
    o_s, h_s1, h_s2 = _place_shuttle_at_o4(carbinolamine)
    syms = list(carbinolamine.get_chemical_symbols()) + ["O", "H", "H"]
    pos = np.vstack([carbinolamine.positions, [o_s, h_s1, h_s2]])
    return Atoms(symbols=syms, positions=pos)


def _asparagine_indices():
    from .build_imine import _asparagine_indices as _ai
    return _ai(read("output/reactant_mapped.xyz"))


def build_product(carbinolamine, *, cn_double=1.28, water_dist=3.3):
    """Iminium + departing water (O4+H37+H_s1) + shuttle-OH- (O_s+H_s2).

    H36 stays on N27 (iminium C10=N27+H36); H_s1 has moved onto the departing
    O4; the shuttle is left as O_s+H_s2 (OH-).  The shuttle stays at its
    side-of-O4 position; the departing water leaves along C10->O4 (perpendicular
    to the shuttle, so no collision).
    """
    atoms = carbinolamine.copy()
    pos = atoms.positions.copy()
    syms = list(atoms.get_chemical_symbols())
    n_orig = len(atoms)
    O_S, H_S1, H_S2 = _shuttle_indices(n_orig)

    u = (carbinolamine.positions[O4] - carbinolamine.positions[C10])
    u = u / np.linalg.norm(u)
    c10 = pos[C10].copy()
    o4_orig = carbinolamine.positions[O4].copy()

    # 1. shorten C10-N27 to a double bond (iminium) by translating asparagine
    n27 = pos[N27].copy()
    v = n27 - c10; v = v / np.linalg.norm(v)
    d0 = float(np.linalg.norm(pos[N27] - pos[C10]))
    shift = v * (cn_double - d0)
    asn = _asparagine_indices()
    pos[asn] = pos[asn] + shift

    # 2. move the leaving water O4+H37 out along C10->O4 (departing)
    pos[O4] = c10 + water_dist * u
    u_h37 = (carbinolamine.positions[H37] - o4_orig)
    u_h37 = u_h37 / np.linalg.norm(u_h37)
    pos[H37] = pos[O4] + 0.96 * u_h37
    away_c10 = (pos[O4] - c10); away_c10 = away_c10 / np.linalg.norm(away_c10)
    d = away_c10 - np.dot(away_c10, u_h37) * u_h37
    if np.linalg.norm(d) < 1e-6:
        d = np.array([-u_h37[1], u_h37[0], 0.0])
    d = d / np.linalg.norm(d)
    h_s1_on_o4 = pos[O4] + 0.96 * d

    # 3. shuttle stays in place (catalyst): reuse the REACTANT shuttle O_s/H_s2
    #    spot -- only H_s1 has left it (gone to O4).  This keeps the shuttle
    #    stationary and clear of the departed water.
    o_s_r, _, h_s2_r = _place_shuttle_at_o4(carbinolamine)

    syms = syms + ["O", "H", "H"]
    pos = np.vstack([pos, [o_s_r, h_s1_on_o4, h_s2_r]])
    atoms = Atoms(symbols=syms, positions=pos)
    return atoms


def _report(atoms):
    p = atoms.positions
    n = len(p) - 3
    O_S, H_S1, H_S2 = n, n + 1, n + 2
    def d(i, j):
        return float(np.linalg.norm(p[i] - p[j]))
    print(f"  C10-O4  : {d(C10,O4):.3f}   (intact ~1.41 / broken >2.5)")
    print(f"  C10-N27 : {d(C10,N27):.3f}   (single ~1.49 / iminium ~1.28)")
    print(f"  N27-H36 : {d(N27,H36):.3f}   (STAYS on N27 = iminium)")
    print(f"  O4-H_s1 : {d(O4,H_S1):.3f}   (H-bond ~1.8 reactant / bond ~0.96 product)")
    print(f"  O_s-H_s1: {d(O_S,H_S1):.3f}   (bond ~0.96 reactant / gone product)")
    print(f"  O_s-O4  : {d(O_S,O4):.3f}   (H-bond ~2.8)")
    print(f"  O_s-N27 : {d(O_S,N27):.3f}   (must be >4 -- no swap!)")
    print(f"  O_s-H36 : {d(O_S,H36):.3f}   (must be >4 -- no swap!)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carbinolamine", nargs="?", default="output/dehyd/carbinolamine_relaxed.xyz")
    ap.add_argument("--out-r", default="output/acid/reactant.xyz")
    ap.add_argument("--out-p", default="output/acid/product.xyz")
    args = ap.parse_args(argv)
    from pathlib import Path
    Path(args.out_r).parent.mkdir(parents=True, exist_ok=True)
    cab = read(args.carbinolamine)
    r = build_reactant(cab)
    print("Acid-catalysis REACTANT (carbinolamine + shuttle @O4):")
    _report(r)
    p = build_product(cab)
    print("\nAcid-catalysis PRODUCT (iminium + departing water + shuttle-OH-):")
    _report(p)
    write(args.out_r, r); write(args.out_p, p)
    print(f"\nWrote {args.out_r} and {args.out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())