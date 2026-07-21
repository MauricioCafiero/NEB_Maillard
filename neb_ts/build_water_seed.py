"""Build a collision-free initial path for the dehydration TS1 segment.

The true dehydration saddle sits between the *carbinolamine* minimum (C10-O4
single 1.41 A, C10-N27 single 1.48, H36 on N27, O4 bearing H37) and the *contact
imine*H2O intermediate (C10=N27 1.26, water O4-H36-H37 3.6 A from C10).

Linear/IDPP interpolation between those endpoints collides: the water oxygen O4
is dragged away from its own proton H37 while H37 is interpolated independently,
so the two momentarily squash to ~0.79 A (an unphysical O..H compression) at
the midpoint -- the sole source of the +160 kcal/mol explosion that has blocked
the NEB.

This builder produces a collision-free seed by moving the departing water as a
*rigid body* (O4 carries H37 with it; the O4-H37 bond stays ~0.96 A throughout),
transferring H36 smoothly from N27 onto O4, and rigidly translating the
asparagine fragment toward C10 so C10-N27 shortens 1.48 -> 1.26 (C=N formation).
The sugar stays fixed.  No two atoms ever pass through one another.

Crucially the three reacting events are driven *in phase* (concerted): C10-O4
lengthens, C10-N27 shortens, and H36 transfers N->O all at the same fraction
t along the band.  An out-of-phase seed (C-O breaking before C=N forming) sits
on a high-energy carbocation-like ridge; the concerted seed stays on the low
asynchronous-but-concerted ridge the true TS lives on.

Output: a multi-frame XYZ usable directly as the NEB initial path.
"""
from __future__ import annotations

import argparse

import numpy as np
from ase.io import read, write

C10, O4, N27, H36, H37 = 10, 4, 27, 36, 37
SUGAR = list(range(0, 24))          # sugar atoms (fixed), mapped order
# asparagine fragment = separated-reactant fragment on N27 minus H37 (now on O4)
ASN_FALLBACK = [i for i in range(24, 41) if i != H37]


def _asparagine_indices(reactant):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    syms = reactant.get_chemical_symbols()
    p = reactant.get_positions()
    lines = [str(len(syms)), "Properties=species:S:1:pos:R:3"]
    for s, xy in zip(syms, p):
        lines.append(f"{s} {xy[0]:.6f} {xy[1]:.6f} {xy[2]:.6f}")
    mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    seen = [False] * mol.GetNumAtoms()
    stack = [N27]
    comp = []
    while stack:
        a = stack.pop()
        if seen[a]:
            continue
        seen[a] = True
        comp.append(a)
        for b in mol.GetAtomWithIdx(a).GetBonds():
            o = b.GetOtherAtomIdx(a)
            if not seen[o]:
                stack.append(o)
    return sorted(i for i in comp if i != H37)


def _frame(start, end, frac, asn_idx):
    """One interpolated frame -- concerted, rigid-fragment, collision-free.

    * sugar atoms (SUGAR): held at the start (carbinolamine) positions -- the
      reaction leaves the sugar framework unchanged (C10 stays put).
    * asparagine atoms (asn_idx): translated along the C10->N27 direction so
      that C10-N27 hits the *linearly interpolated target distance* at this
      fraction (1.485 -> 1.263).  Driving the distance (not the absolute end
      position) keeps C=N formation IN PHASE with C-O cleavage.
    * O4 and H37: O4 linearly interpolated by POSITION start->end (both
      endpoints have |O4-N27| ~3.9 A, so a straight chord keeps O4 clear of
      N27 -- driving O4 along the fixed C10->O4 direction instead collides);
      H37 carried rigidly on O4 (the O4-H37 water bond never collapses).
    * H36: transferred along the (moving) N27->O4 line at the same fraction t
      (H36 = N27(t) + t*(O4(t) - N27(t))) -- concerted proton transfer, in
      phase with C-O cleavage and C=N formation.
    """
    sp = start.positions
    ep = end.positions
    d0_CN = float(np.linalg.norm(sp[N27] - sp[C10]))
    d1_CN = float(np.linalg.norm(ep[N27] - ep[C10]))

    pos = sp.copy()
    # asparagine: translate along C10->N27 so C10-N27 = target.
    u_CN = sp[N27] - sp[C10]
    u_CN = u_CN / np.linalg.norm(u_CN)
    target_CN = d0_CN + frac * (d1_CN - d0_CN)
    disp_asn = u_CN * (target_CN - d0_CN)
    pos[asn_idx] = sp[asn_idx] + disp_asn
    n27 = pos[N27].copy()

    # O4: linear POSITION interpolation start->end (keeps it clear of N27).
    pos[O4] = (1 - frac) * sp[O4] + frac * ep[O4]
    o4 = pos[O4].copy()

    # H37: rigid on O4 (start O-H direction, length 0.965).
    u_h37 = sp[H37] - sp[O4]
    u_h37 = u_h37 / np.linalg.norm(u_h37)
    pos[H37] = o4 + 0.965 * u_h37

    # H36: smooth proton transfer along the (moving) N27->O4 line.  Early it
    # stays bonded to N (1.02 A); late it is bonded to O (0.97 A); a Hermite
    # step blends between the two N-bonded and O-bonded positions.  The
    # compressed N-H-O bridging geometry appears only at the blend midpoint --
    # i.e. exactly at the TS frame -- rather than smeared across the band.
    vec = o4 - n27
    L = np.linalg.norm(vec)
    if L < 1e-6:
        pos[H36] = n27 + np.array([0.0, 0.0, 1.0])
    else:
        u = vec / L
        h_on_n = n27 + 1.02 * u          # H bonded to N, pointing toward O
        h_on_o = o4 - 0.97 * u           # H bonded to O, pointing toward N
        alpha = 3 * frac ** 2 - 2 * frac ** 3   # smooth Hermite step 0->1
        pos[H36] = (1 - alpha) * h_on_n + alpha * h_on_o

    atoms = start.copy()
    atoms.positions = pos
    return atoms


def build_seed(start, end, n, reactant=None):
    asn_idx = (_asparagine_indices(reactant) if reactant is not None
               else ASN_FALLBACK)
    frames = [start.copy()]
    for s in range(1, n):
        frames.append(_frame(start, end, s / n, asn_idx))
    frames.append(end.copy())
    return frames


def _report(atoms):
    pos = atoms.positions
    def d(i, j):
        return float(np.linalg.norm(pos[i] - pos[j]))
    print(f"  C10-O4={d(C10,O4):.3f}  C10-N27={d(C10,N27):.3f}  "
          f"O4-H37={d(O4,H37):.3f}  N27-H36={d(N27,H36):.3f}  "
          f"O4-H36={d(O4,H36):.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", nargs="?", default="output/dehyd/carbinolamine_relaxed.xyz")
    ap.add_argument("end", nargs="?", default="output/dehyd/intermediate.xyz")
    ap.add_argument("--reactant", default="output/reactant_mapped.xyz")
    ap.add_argument("--n", type=int, default=11, help="Total frames (incl. endpoints)")
    ap.add_argument("--out", default="output/ts1/seed_water.xyz")
    args = ap.parse_args(argv)

    start = read(args.start)
    end = read(args.end)
    reactant = read(args.reactant) if args.reactant else None
    frames = build_seed(start, end, args.n, reactant)
    print(f"Seed path ({len(frames)} frames):")
    for i, fr in enumerate(frames):
        print(f"  frame {i:2d}:", end=" ")
        _report(fr)
    write(args.out, frames)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())