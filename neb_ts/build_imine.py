"""Build the imine (Schiff base) + H2O product of carbinolamine dehydration.

The provided "product" (output/product_mapped.xyz) is the **carbinolamine**
(hemiaminal): C10 has a single bond to N27 (1.50 A) and a single bond to the
OH oxygen O4 (1.41 A, bearing H37).  N27 still bears one proton (H36).

The dehydration elementary step -- the rate-determining step of Maillard
glycosylamine formation -- is:

    carbinolamine  ->  imine (Schiff base) + H2O

i.e. C10-O4 cleaves (O4 leaves as water), the N27 proton H36 transfers onto
O4 so the leaving group is H2O (O4-H36-H37), and C10=N27 forms (1.50 -> 1.28 A,
single -> double).  This module builds that imine-product geometry from the
carbinolamine by:

1. Rigidly translating the asparagine fragment so C10-N27 shortens to ~1.28 A.
2. Moving O4 (+ its proton H37) away from C10 to ~3.2 A (the departing water).
3. Transferring H36 from N27 onto O4 to complete the water molecule.

Atom indices are the mapped/common order of output/product_mapped.xyz.
"""
from __future__ import annotations

import argparse

import numpy as np
from ase.io import read, write

C_ALDEHYDE = 10   # anomeric / aldehyde C (becomes the imine C)
O_HYDROXYL = 4    # carbinolamine OH oxygen (leaves as water)
N_AMINE = 27      # amine N (becomes the imine N)
H_ON_N = 36       # N27 proton that transfers to O4 (completes the leaving water)
H_ON_O = 37       # O4 proton (stays on the departing water)


def _fragment_containing(atoms, idx):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    syms = atoms.get_chemical_symbols()
    p = atoms.get_positions()
    lines = [str(len(syms)), "Properties=species:S:1:pos:R:3"]
    for s, xy in zip(syms, p):
        lines.append(f"{s} {xy[0]:.6f} {xy[1]:.6f} {xy[2]:.6f}")
    mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    seen = [False] * mol.GetNumAtoms()
    stack = [idx]
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
    return sorted(comp)


def _asparagine_indices(carbinolamine, reactant=None):
    """Indices of the asparagine molecule, excluding the transferring proton H37.

    Connectivity in the carbinolamine merges sugar+asparagine (the C10-N27 bond
    already exists), so we take the asparagine fragment from the *separated*
    reactant geometry and drop H37 (which has moved onto the sugar's O4 and is
    handled separately as part of the departing water).
    """
    if reactant is None:
        # Fallback: hard-coded asparagine atom range (mapped order).
        asn = list(range(24, 41))
    else:
        asn = _fragment_containing(reactant, N_AMINE)
    return [i for i in asn if i != H_ON_O]   # drop H37 (now on O4)


def build_imine(carbinolamine, *, cn_double=1.28, water_dist=3.2, oh=0.96,
                reactant=None):
    atoms = carbinolamine.copy()
    pos = atoms.positions

    # 1. Shorten C10-N27 to a double bond by translating the asparagine atoms
    #    (C10 itself stays put -- it is on the sugar).
    c10 = pos[C_ALDEHYDE].copy()
    n27 = pos[N_AMINE].copy()
    v = n27 - c10
    v = v / np.linalg.norm(v)
    d0 = float(np.linalg.norm(pos[N_AMINE] - pos[C_ALDEHYDE]))
    shift = v * (cn_double - d0)            # negative -> move asparagine toward C10
    frag = _asparagine_indices(carbinolamine, reactant)
    pos[frag] = pos[frag] + shift
    n27 = pos[N_AMINE].copy()

    # 2. Move the leaving water (O4 + H37) away from C10 along C10->O4.
    c10 = pos[C_ALDEHYDE].copy()
    o4 = pos[O_HYDROXYL].copy()
    v2 = o4 - c10
    v2 = v2 / np.linalg.norm(v2)
    new_o4 = c10 + water_dist * v2
    disp = new_o4 - o4
    pos[O_HYDROXYL] = new_o4
    pos[H_ON_O] = pos[H_ON_O] + disp        # keep O4-H37 intact
    o4 = new_o4.copy()

    # 3. Transfer H36 from N27 onto O4 to complete the water.  Place it on O4
    #    pointing away from C10 (the departing water's second O-H), ~0.96 A.
    u_away = (o4 - c10)
    u_away = u_away / np.linalg.norm(u_away)
    # Start H36 opposite the existing H37 through O4 so the two O-H bonds are
    # not on top of each other; relaxation will set the H-O-H angle.
    u_h37 = pos[H_ON_O] - o4
    u_h37 = u_h37 / np.linalg.norm(u_h37)
    # Direction roughly away from H37, biased away from C10.
    d = u_away - np.dot(u_away, u_h37) * u_h37
    if np.linalg.norm(d) < 1e-6:
        d = np.array([-u_h37[1], u_h37[0], 0.0])
    d = d / np.linalg.norm(d)
    pos[H_ON_N] = o4 + oh * d
    return atoms


def _report(atoms):
    pos = atoms.positions
    def d(i, j):
        return float(np.linalg.norm(pos[i] - pos[j]))
    print(f"  C10-N27 : {d(C_ALDEHYDE, N_AMINE):.3f} A   (imine double ~1.28)")
    print(f"  C10-O4  : {d(C_ALDEHYDE, O_HYDROXYL):.3f} A   (broken, >2.5)")
    print(f"  O4-H37  : {d(O_HYDROXYL, H_ON_O):.3f} A   (water O-H ~0.96)")
    print(f"  O4-H36  : {d(O_HYDROXYL, H_ON_N):.3f} A   (water O-H ~0.96)")
    print(f"  N27-H36 : {d(N_AMINE, H_ON_N):.3f} A   (transferred, >2)")
    # connectivity via simple distance check on the reacting atoms
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    syms = atoms.get_chemical_symbols()
    lines = [str(len(syms)), "Properties=species:S:1:pos:R:3"]
    for s, xy in zip(syms, atoms.positions):
        lines.append(f"{s} {xy[0]:.6f} {xy[1]:.6f} {xy[2]:.6f}")
    mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    print(f"  perceived bonds on C10: " + ", ".join(
        f"{b.GetOtherAtomIdx(C_ALDEHYDE)}({mol.GetAtomWithIdx(b.GetOtherAtomIdx(C_ALDEHYDE)).GetSymbol()})"
        for b in mol.GetAtomWithIdx(C_ALDEHYDE).GetBonds()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carbinolamine", nargs="?", default="output/product_mapped.xyz")
    ap.add_argument("--reactant", default="output/reactant_mapped.xyz",
                    help="Separated-reactant geometry (to identify asparagine atoms)")
    ap.add_argument("--out", default="output/imine_product.xyz")
    args = ap.parse_args(argv)
    cab = read(args.carbinolamine)
    reactant = read(args.reactant) if args.reactant else None
    print("Carbinolamine (input) reacting geometry:")
    _report(cab)
    imine = build_imine(cab, reactant=reactant)
    print("\nImine + H2O product (built):")
    _report(imine)
    write(args.out, imine)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())