#!/usr/bin/env python
"""Chemically consistent atom mapping between reactant and product XYZ files
for NEB (Maillard glycosylamine formation).

Strategy
--------
The reactant is 3 disconnected fragments (asparagine, open-chain glucose, water).
The product is 2 fragments (glycosylamine = asparagine + C1-N bond to glucose,
water).  Each reactant fragment is a heavy-atom SUBSTRUCTURE of the product: the
only graph change is the new C1-N bond, which adds an extra bond to atoms that
already exist in both.  So:
  1. Fragment the reactant.
  2. For each reactant fragment, run a heavy-atom substructure search against the
     product (preserving perceived bond orders so C=O matches C=O); pick the
     3D-best match (Kabsch RMSD).  This maps every reactant heavy atom to a
     product heavy atom (bijection).
  3. Map hydrogens by parent-heavy-atom proximity after Kabsch; protons that
     change parent (the transferring H) fall into a global pool matched by
     linear_sum_assignment.
  4. Kabsch-align the reordered product onto the reactant; report.
"""

import numpy as np
from ase.io import read, write
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from scipy.optimize import linear_sum_assignment

R_PATH = "output/reactant_extracted.xyz"
P_PATH = "output/product_extracted.xyz"
OUT_R = "output/reactant_mapped.xyz"
OUT_P = "output/product_mapped.xyz"
OUT_MAP = "output/atom_mapping.txt"


def build_rdkit_mol(atoms):
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    lines = [str(len(syms)), "Properties=species:S:1:pos:R:3"]
    for s, p in zip(syms, pos):
        lines.append(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    if mol is None:
        raise RuntimeError("MolFromXYZBlock None")
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    return mol


def kabsch(P, Q):
    Pc = P - P.mean(axis=0); Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t


def apply(R, t, X):
    return (R @ X.T).T + t


def fragments(mol):
    n = mol.GetNumAtoms()
    seen = [False] * n
    fs = []
    for i in range(n):
        if seen[i]:
            continue
        st = [i]; c = []
        while st:
            a = st.pop()
            if seen[a]:
                continue
            seen[a] = True; c.append(a)
            for b in mol.GetAtomWithIdx(a).GetBonds():
                o = b.GetOtherAtomIdx(a)
                if not seen[o]:
                    st.append(o)
        fs.append(sorted(c))
    return fs


def heavy_query_mol(mol, atom_idx):
    """Heavy-atom single-bond query (atom order preserved).  Matched against a
    bond-order-flattened copy of the product so that perception quirks (C=O
    vs C-O, the reacting C1-O) don't defeat the search.  The new C1-N bond
    (product-only) is tolerated as an extra target bond by substructure
    matching.  Returns (query_mol, heavy_index_list)."""
    heavy = [i for i in atom_idx if mol.GetAtomWithIdx(i).GetSymbol() != "H"]
    e = Chem.RWMol()
    im = {}
    for ni, oi in enumerate(heavy):
        a = mol.GetAtomWithIdx(oi)
        na = Chem.Atom(a.GetSymbol()); na.SetNoImplicit(True)
        e.AddAtom(na); im[oi] = ni
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in im and a2 in im:
            e.AddBond(im[a1], im[a2], Chem.BondType.SINGLE)
    return e.GetMol(), heavy


def flatten_bond_orders(mol):
    """Return a copy of mol with every bond set to SINGLE (preserving atom
    indices), so substructure matching is bond-order-agnostic."""
    rw = Chem.RWMol(mol)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
    # clear aromatic flags just in case
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
    for b in rw.GetBonds():
        b.SetIsAromatic(False)
    return rw.GetMol()


def get_bonded_h(mol, idx):
    out = []
    for b in mol.GetAtomWithIdx(idx).GetBonds():
        o = b.GetOtherAtomIdx(idx)
        if mol.GetAtomWithIdx(o).GetSymbol() == "H":
            out.append(o)
    return sorted(out)


def main():
    r_atoms = read(R_PATH)
    p_atoms = read(P_PATH)
    r_pos = r_atoms.get_positions()
    p_pos = p_atoms.get_positions()
    r_syms = r_atoms.get_chemical_symbols()
    p_syms = p_atoms.get_chemical_symbols()
    n = len(r_atoms)
    assert len(p_atoms) == n

    r_mol = build_rdkit_mol(r_atoms)
    p_mol = build_rdkit_mol(p_atoms)

    # --- 1. Connectivity report ---
    r_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in r_mol.GetBonds()]
    p_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in p_mol.GetBonds()]
    print(f"Reactant bonds: {len(r_bonds)}")
    print(f"Product  bonds: {len(p_bonds)}")
    norm = lambda a, b: (min(a, b), max(a, b))
    rbs = {norm(a, b) for a, b in r_bonds}
    pbs = {norm(a, b) for a, b in p_bonds}
    new_bonds = pbs - rbs
    broken = rbs - pbs
    print(f"New bonds in product: {[(p_syms[a],p_syms[b]) for a,b in new_bonds]}")
    print(f"Broken bonds: {[(r_syms[a],r_syms[b]) for a,b in broken]}")

    r_frags = fragments(r_mol)
    p_frags = fragments(p_mol)
    print(f"Reactant fragments: {len(r_frags)} sizes={[len(f) for f in r_frags]}")
    print(f"Product  fragments: {len(p_frags)} sizes={[len(f) for f in p_frags]}")

    # --- 2. Heavy-atom mapping via fragment substructure search ---
    p_flat = flatten_bond_orders(p_mol)
    heavy_map_r2p = {}
    used_p_heavy = set()
    for fi, frag in enumerate(r_frags):
        qmol, qheavy = heavy_query_mol(r_mol, frag)
        matches = p_flat.GetSubstructMatches(qmol, uniquify=False)
        cand = []
        for m in matches:
            if any(p_syms[mi] == "H" for mi in m):
                continue
            if any(mi in used_p_heavy for mi in m):
                continue
            r_pts = np.array([r_pos[h] for h in qheavy])
            p_pts = np.array([p_pos[mi] for mi in m])
            R, t = kabsch(p_pts, r_pts)
            al = apply(R, t, p_pts)
            rmsd = np.sqrt(np.mean(np.sum((al - r_pts) ** 2, axis=1)))
            cand.append((rmsd, m))
        if not cand:
            raise RuntimeError(f"fragment {fi} found no substructure match in product")
        cand.sort()
        best_rmsd, best_m = cand[0]
        for rh, ph in zip(qheavy, best_m):
            heavy_map_r2p[rh] = ph
            used_p_heavy.add(ph)
        print(f"  Fragment {fi} (reactant heavy {qheavy}): best RMSD={best_rmsd:.4f} A -> product heavy {list(best_m)}")

    r_heavy = [i for i, s in enumerate(r_syms) if s != "H"]
    p_heavy = [i for i, s in enumerate(p_syms) if s != "H"]
    assert len(heavy_map_r2p) == len(r_heavy), "heavy map incomplete"
    assert len(set(heavy_map_r2p.values())) == len(heavy_map_r2p), "heavy map not injective"

    # --- 3. Hydrogen mapping ---
    r_pts = np.array([r_pos[i] for i in r_heavy])
    p_pts = np.array([p_pos[heavy_map_r2p[i]] for i in r_heavy])
    R_h, t_h = kabsch(p_pts, r_pts)
    p_aligned = apply(R_h, t_h, p_pos)

    h_map_r2p = {}
    pooled_rH, pooled_pH = [], []
    for rh in r_heavy:
        ph = heavy_map_r2p[rh]
        rHs = get_bonded_h(r_mol, rh)
        pHs = get_bonded_h(p_mol, ph)
        if len(rHs) == len(pHs) and len(rHs) > 0:
            D = np.array([[np.linalg.norm(p_aligned[phh] - r_pos[rhh]) for rhh in rHs] for phh in pHs])
            rows, cols = linear_sum_assignment(D)
            for a, b in zip(rows, cols):
                h_map_r2p[rHs[b]] = pHs[a]
        else:
            pooled_rH.extend(rHs); pooled_pH.extend(pHs)

    all_rH = [i for i, s in enumerate(r_syms) if s == "H"]
    all_pH = [i for i, s in enumerate(p_syms) if s == "H"]
    mapped_rH = set(h_map_r2p.keys())
    mapped_pH = set(h_map_r2p.values())
    pool_r = [h for h in pooled_rH if h not in mapped_rH]
    pool_p = [h for h in pooled_pH if h not in mapped_pH]
    for h in all_rH:
        if h not in h_map_r2p and h not in pool_r:
            pool_r.append(h)
    for h in all_pH:
        if h not in h_map_r2p.values() and h not in pool_p:
            pool_p.append(h)
    if pool_r and pool_p:
        D = np.array([[np.linalg.norm(p_aligned[ph] - r_pos[rh]) for rh in pool_r] for ph in pool_p])
        rows, cols = linear_sum_assignment(D)
        for a, b in zip(rows, cols):
            h_map_r2p[pool_r[b]] = pool_p[a]

    assert len(h_map_r2p) == len(all_rH), f"H map incomplete {len(h_map_r2p)}/{len(all_rH)}"
    assert len(set(h_map_r2p.values())) == len(all_pH), "H map not injective"

    # --- 4. Permutation P[reactant_k] = product_idx ---
    full = {}; full.update(heavy_map_r2p); full.update(h_map_r2p)
    P = [None] * n
    for ri, pi in full.items():
        P[ri] = pi
    assert all(x is not None for x in P) and len(set(P)) == n

    p_reordered = p_pos[P]
    R, t = kabsch(p_reordered, r_pos)
    p_aligned = apply(R, t, p_reordered)
    disp = np.linalg.norm(p_aligned - r_pos, axis=1)
    rmsd = np.sqrt(np.mean(np.sum((p_aligned - r_pos) ** 2, axis=1)))

    print(f"\nFull mapped RMSD (44 atoms): {rmsd:.5f} A")
    mx = int(np.argmax(disp))
    print(f"Max displacement: {disp[mx]:.4f} A at reactant atom {mx} ({r_syms[mx]})")
    order = np.argsort(disp)[::-1]
    print("\nTop 12 movers (reactant idx, element, disp A):")
    for k in order[:12]:
        print(f"  {k:2d}  {r_syms[k]:2s}  {disp[k]:.4f}")

    hd = sorted([(i, r_syms[i], disp[i]) for i in r_heavy], key=lambda x: -x[2])
    print("\nHeavy-atom displacements (top 8):")
    for i, s, d in hd[:8]:
        print(f"  {i:2d} {s}  {d:.4f}")
    print(f"Mean heavy-atom displacement: {np.mean([d for _,_,d in hd]):.4f} A")
    print(f"Mean H displacement: {np.mean([disp[i] for i in all_rH]):.4f} A")

    # Bond-preservation check
    preserved = 0; broken_mapped = []
    for a, b in r_bonds:
        pa, pb = P[a], P[b]
        if norm(pa, pb) in pbs:
            preserved += 1
        else:
            broken_mapped.append((a, b, r_syms[a], r_syms[b]))
    print(f"\nBond preservation: {preserved}/{len(r_bonds)} reactant bonds present in product after mapping")
    print(f"  broken (expected 2): {broken_mapped}")
    r_mapped_bonds = {norm(P[a], P[b]) for a, b in r_bonds}
    new_mapped = pbs - r_mapped_bonds
    print(f"  new bonds after mapping (expected 3): {[(p_syms[a],p_syms[b]) for a,b in new_mapped]}")

    # Reacting atoms
    cn = [eb for eb in new_bonds if {p_syms[eb[0]], p_syms[eb[1]]} == {"C", "N"}]
    if cn:
        a, b = cn[0]
        pN = a if p_syms[a] == "N" else b
        pC = a if p_syms[a] == "C" else b
        rN = P.index(pN); rC = P.index(pC)
        print(f"\nNew C-N bond: product C={pC} N={pN} -> reactant C={rC} N={rN}")
        print(f"  sugar C1 (reactant {rC}) displacement: {disp[rC]:.4f} A")
        print(f"  amine N  (reactant {rN}) displacement: {disp[rN]:.4f} A")
        rN_Hs = get_bonded_h(r_mol, rN)
        print(f"  H's on amine N in reactant: {rN_Hs}")
        for rh in rN_Hs:
            ph = P[rh]
            pars = [b.GetOtherAtomIdx(ph) for b in p_mol.GetAtomWithIdx(ph).GetBonds() if p_syms[b.GetOtherAtomIdx(ph)] != "H"]
            print(f"    reactant H {rh} (disp {disp[rh]:.3f}) -> product H {ph}, product heavy parent(s) = {[(pp,p_syms[pp]) for pp in pars]}")

    # --- Write outputs ---
    r_centered = r_pos - r_pos.mean(axis=0)
    R2, t2 = kabsch(p_reordered, r_centered)
    p_final = apply(R2, t2, p_reordered)
    p_final_syms = [p_syms[pi] for pi in P]

    def write_xyz(path, syms, pos, comment):
        with open(path, "w") as f:
            f.write(f"{len(syms)}\n{comment}\n")
            for s, p in zip(syms, pos):
                f.write(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    write_xyz(OUT_R, r_syms, r_centered, "reactant mapped (centered)")
    write_xyz(OUT_P, p_final_syms, p_final, "product mapped+aligned onto reactant")
    with open(OUT_MAP, "w") as f:
        f.write("# reactant_idx  reactant_el  ->  product_idx  product_el  displacement(A)\n")
        for k in range(n):
            f.write(f"{k:2d}  {r_syms[k]:2s}  ->  {P[k]:2d}  {p_syms[P[k]]:2s}  {disp[k]:.4f}\n")

    print(f"\nWrote {OUT_R}, {OUT_P}, {OUT_MAP}")
    print(f"\nPermutation P (product_reordered[k] = product[P[k]]):")
    print(P)
    return P, disp, r_syms


if __name__ == "__main__":
    main()