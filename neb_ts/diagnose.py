"""Diagnose the reacting bond in the mapped reactant/product structures.

Prints the RDKit-perceived connectivity of the mapped endpoints, highlights
bonds present in the product but not the reactant (and vice-versa), and reports
key reacting-atom distances so we can decide how to build the pre-reactive
complex.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import read
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def to_rdmol(atoms):
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    lines = [str(len(syms)), "Properties=species:S:1:pos:R:3"]
    for s, p in zip(syms, pos):
        lines.append(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    return mol


def bondset(mol):
    return {(min(a, b), max(a, b)) for a, b in ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())}


def dist(atoms, i, j):
    return float(np.linalg.norm(atoms.positions[i] - atoms.positions[j]))


def main():
    r = read("output/reactant_mapped.xyz")
    p = read("output/product_mapped.xyz")
    rm = to_rdmol(r)
    pm = to_rdmol(p)
    rb = bondset(rm)
    pb = bondset(pm)

    print(f"reactant bonds: {len(rb)}   product bonds: {len(pb)}")
    only_p = sorted(pb - rb)
    only_r = sorted(rb - pb)
    print("\nBonds in PRODUCT but not REACTANT (forming bonds):")
    for a, b in only_p:
        print(f"  {a}({rm.GetAtomWithIdx(a).GetSymbol()}) - {b}({rm.GetAtomWithIdx(b).GetSymbol()})  "
              f"R-dist={dist(r,a,b):.3f}  P-dist={dist(p,a,b):.3f}")
    print("\nBonds in REACTANT but not PRODUCT (breaking bonds):")
    for a, b in only_r:
        print(f"  {a}({rm.GetAtomWithIdx(a).GetSymbol()}) - {b}({rm.GetAtomWithIdx(b).GetSymbol()})  "
              f"R-dist={dist(r,a,b):.3f}  P-dist={dist(p,a,b):.3f}")

    # Identify the amine N (NH2 with two H) and the anomeric C.
    print("\nNitrogen atoms and their H neighbours (reactant):")
    for i, a in enumerate(rm.GetAtoms()):
        if a.GetSymbol() != "N":
            continue
        hs = [b.GetOtherAtomIdx(i) for b in a.GetBonds() if rm.GetAtomWithIdx(b.GetOtherAtomIdx(i)).GetSymbol() == "H"]
        heavy = [b.GetOtherAtomIdx(i) for b in a.GetBonds() if rm.GetAtomWithIdx(b.GetOtherAtomIdx(i)).GetSymbol() != "H"]
        print(f"  N{i} at {r.positions[i]}  H={hs} heavy={heavy}")

    print("\nN27 and N28 distances to every carbon (reactant):")
    for n in (27, 28):
        print(f"  N{n}:")
        for c in range(6, 12):
            print(f"    C{c}: {dist(r, n, c):.3f} Å  (product: {dist(p, n, c):.3f})")

    # C10 neighbours (identify the aldehyde oxygen and the reacting framework)
    print("\nC10 (anomeric/aldehyde C) neighbours and bond orders:")
    for mol, atoms, tag in ((rm, r, "R"), (pm, p, "P")):
        a = mol.GetAtomWithIdx(10)
        print(f"  {tag}: C10 at {atoms.positions[10]}")
        for b in a.GetBonds():
            o = b.GetOtherAtomIdx(10)
            print(f"    - {o}({mol.GetAtomWithIdx(o).GetSymbol()}) order={b.GetBondTypeAsDouble()} dist={dist(atoms,10,o):.3f}")

    # H-bond / proton-transfer geometry: N27-H37...O4
    print("\nProton-transfer geometry N27-H37...O4:")
    for atoms, tag in ((r, "R"), (p, "P")):
        v_nh = atoms.positions[37] - atoms.positions[27]
        v_ho = atoms.positions[4] - atoms.positions[37]
        nh = float(np.linalg.norm(v_nh))
        ho = float(np.linalg.norm(v_ho))
        ang = float(np.degrees(np.arccos(np.dot(v_nh, v_ho) / (nh * ho)))) if nh * ho > 0 else float("nan")
        no4 = dist(atoms, 27, 4)
        print(f"  {tag}: N27-H37={nh:.3f}  H37-O4={ho:.3f}  N27...O4={no4:.3f}  angle N-H...O={ang:.1f} deg")

    # Fragments in reactant (which atoms are bonded together = same molecule)
    print("\nReactant connected components (fragments):")
    n = rm.GetNumAtoms()
    seen = [False] * n
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        comp = []
        while stack:
            a = stack.pop()
            if seen[a]:
                continue
            seen[a] = True
            comp.append(a)
            for b in rm.GetAtomWithIdx(a).GetBonds():
                o = b.GetOtherAtomIdx(a)
                if not seen[o]:
                    stack.append(o)
        comps.append(sorted(comp))
    for k, comp in enumerate(comps):
        syms = [r.get_chemical_symbols()[i] for i in comp]
        from collections import Counter
        print(f"  fragment {k}: {len(comp)} atoms, formula {Counter(syms)}")


if __name__ == "__main__":
    main()