"""Build a pre-reactive complex for carbinolamine (hemiaminal) formation.

The provided "reactant" is three separated fragments (open-chain sugar +
asparagine + water).  Its amine (N27) sits ~2.9 A from the sugar aldehyde
carbon C10, but the transferring proton H37 points *away* from the aldehyde
oxygen O4 (N27-H37...O4 angle ~105 deg, H37...O4 ~3.4 A).  Because the proton
is not pre-positioned for transfer, the CI-NEB band slides off the reaction
ridge: the C-N bond forms while the proton shuttles via the spectator water,
giving a barrierless path that bypasses the concerted saddle.

This module rebuilds the geometry so the amine is H-bonded to the aldehyde
oxygen -- H37 placed on the N27 -> O4 line, giving a near-linear N27-H37...O4
hydrogen bond.  The sugar aldehyde (C10=O4) and the rest of the asparagine
framework are left untouched; only the transferring proton is repositioned.
After GFN2-xTB relaxation this should settle into a true pre-reactive-complex
minimum in the same basin as the saddle, so a short, stiff CI-NEB to the
carbinolamine product can resolve the concerted C-N-forming / proton-transferring
barrier.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

# Atom indices (in the mapped/common atom order of output/reactant_mapped.xyz).
C_ALDEHYDE = 10   # sugar anomeric / aldehyde carbon (C=O in reactant)
O_ALDEHYDE = 4    # aldehyde oxygen (becomes the carbinolamine OH)
N_AMINE = 27      # asparagine alpha-amine nitrogen
H_TRANSFER = 37   # amine proton that transfers to O4 in the carbinolamine


def build_precomplex(reactant, *, nh=1.02, target_cn=None) -> "reactant":
    """Return a copy of ``reactant`` with H37 repositioned to H-bond N27..O4.

    Parameters
    ----------
    nh
        N27-H37 bond length to enforce (A).
    target_cn
        If given, translate the whole asparagine amine group so the C10-N27
        distance equals this value (kept close to the reactant distance when
        None -- only the proton is moved).
    """
    atoms = reactant.copy()
    pos = atoms.positions

    if target_cn is not None:
        # Rigidly slide the asparagine fragment so N27 sits at target_cn from
        # C10 along the current C10->N27 direction (pre-organize the attack).
        c10 = pos[C_ALDEHYDE]
        n27 = pos[N_AMINE]
        vec = n27 - c10
        vec = vec / np.linalg.norm(vec)
        new_n27 = c10 + target_cn * vec
        shift = new_n27 - n27
        # Move every atom of the asparagine fragment (contains N27) together.
        frag = _fragment_containing(atoms, N_AMINE)
        pos[frag] = pos[frag] + shift
        n27 = pos[N_AMINE]

    # Place H37 on the N27 -> O4 line so N27-H37...O4 is a linear H-bond.
    n27 = pos[N_AMINE]
    o4 = pos[O_ALDEHYDE]
    n_to_o = o4 - n27
    n_to_o = n_to_o / np.linalg.norm(n_to_o)
    pos[H_TRANSFER] = n27 + nh * n_to_o
    return atoms


def _fragment_containing(atoms, idx):
    """Indices of the bonded molecule containing atom ``idx`` (RDKit perceive)."""
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


def _report(atoms):
    pos = atoms.positions
    def d(i, j):
        return float(np.linalg.norm(pos[i] - pos[j]))
    v_nh = pos[H_TRANSFER] - pos[N_AMINE]
    v_ho = pos[O_ALDEHYDE] - pos[H_TRANSFER]
    ang = float(np.degrees(np.arccos(
        np.dot(v_nh, v_ho) / (np.linalg.norm(v_nh) * np.linalg.norm(v_ho)))))
    print(f"  C10-N27 : {d(C_ALDEHYDE, N_AMINE):.3f} A")
    print(f"  N27-H37 : {d(N_AMINE, H_TRANSFER):.3f} A")
    print(f"  H37-O4  : {d(H_TRANSFER, O_ALDEHYDE):.3f} A")
    print(f"  N27...O4: {d(N_AMINE, O_ALDEHYDE):.3f} A")
    print(f"  angle N27-H37-O4 : {ang:.1f} deg  (180 = linear H-bond)")
    print(f"  C10=O4  : {d(C_ALDEHYDE, O_ALDEHYDE):.3f} A  (aldehyde ~1.21)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reactant", nargs="?", default="output/reactant_mapped.xyz")
    ap.add_argument("--out", default="output/prereactive_complex.xyz")
    ap.add_argument("--target-cn", type=float, default=None,
                    help="Optionally set C10-N27 distance (A) by translating the asparagine fragment")
    ap.add_argument("--nh", type=float, default=1.02, help="N27-H37 bond length (A)")
    args = ap.parse_args(argv)

    r = read(args.reactant)
    print("Reactant (input) reacting geometry:")
    _report(r)
    pre = build_precomplex(r, nh=args.nh, target_cn=args.target_cn)
    print("\nPre-reactive complex (built):")
    _report(pre)
    write(args.out, pre)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())