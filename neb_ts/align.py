"""Chemically consistent atom mapping between reactant and product for NEB.

NEB requires atom *k* in the reactant to be the same chemical atom as atom *k*
in the product.  When the two structures come from separate calculations
(e.g. two Gaussian optimizations), the file atom ordering is often inconsistent
-- especially for the many equivalent H and O atoms -- and NEB then interpolates
between *mis-mapped* atoms, producing unphysical, low-barrier "permutation"
paths instead of the real reaction coordinate.

This module reorders the product onto the reactant using RDKit:

1. Perceive bonds in both structures (neutral, total charge 0).
2. Fragment the reactant; for each fragment run a heavy-atom substructure
   search against a bond-order-flattened copy of the product (this tolerates
   the one bond that changes in a reaction, e.g. the new C-N bond in
   glycosylamine formation).  Pick the 3D-best match (Kabsch RMSD).
3. Map hydrogens by parent-heavy-atom proximity (Hungarian assignment); protons
   that change parent (a transferring H) fall into a global pool matched by
   proximity.
4. Kabsch-align the reordered product onto the reactant.

The result is a permutation ``P`` such that ``product_reordered[k] =
product[P[k]]`` corresponds to reactant atom ``k``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from ase import Atoms


def _build_rdkit_mol(atoms: Atoms) -> Chem.Mol:
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    lines = [str(len(syms)), "Properties=species:S:1:pos:R:3"]
    for s, p in zip(syms, pos):
        lines.append(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    if mol is None:
        raise RuntimeError("RDKit failed to parse atoms as XYZ.")
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    return mol


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t


def _apply(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (R @ X.T).T + t


def _fragments(mol: Chem.Mol) -> list[list[int]]:
    n = mol.GetNumAtoms()
    seen = [False] * n
    fs: list[list[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        comp: list[int] = []
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
        fs.append(sorted(comp))
    return fs


def _heavy_query_mol(mol: Chem.Mol, atom_idx: list[int]) -> tuple[Chem.Mol, list[int]]:
    heavy = [i for i in atom_idx if mol.GetAtomWithIdx(i).GetSymbol() != "H"]
    e = Chem.RWMol()
    im: dict[int, int] = {}
    for ni, oi in enumerate(heavy):
        a = mol.GetAtomWithIdx(oi)
        na = Chem.Atom(a.GetSymbol())
        na.SetNoImplicit(True)
        e.AddAtom(na)
        im[oi] = ni
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in im and a2 in im:
            e.AddBond(im[a1], im[a2], Chem.BondType.SINGLE)
    return e.GetMol(), heavy


def _flatten_bond_orders(mol: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
    for b in rw.GetBonds():
        b.SetIsAromatic(False)
    return rw.GetMol()


def _get_bonded_h(mol: Chem.Mol, idx: int) -> list[int]:
    out = []
    for b in mol.GetAtomWithIdx(idx).GetBonds():
        o = b.GetOtherAtomIdx(idx)
        if mol.GetAtomWithIdx(o).GetSymbol() == "H":
            out.append(o)
    return sorted(out)


def compute_atom_mapping(reactant: Atoms, product: Atoms) -> list[int]:
    """Return permutation ``P`` with ``product[P[k]]`` ↔ reactant atom ``k``."""
    r_pos = reactant.get_positions()
    p_pos = product.get_positions()
    r_syms = reactant.get_chemical_symbols()
    p_syms = product.get_chemical_symbols()
    n = len(reactant)
    if len(product) != n or sorted(r_syms) != sorted(p_syms):
        raise ValueError("Reactant and product must have identical composition.")

    r_mol = _build_rdkit_mol(reactant)
    p_mol = _build_rdkit_mol(product)
    p_flat = _flatten_bond_orders(p_mol)

    # 1. Heavy-atom mapping via fragment substructure search.
    r_frags = _fragments(r_mol)
    heavy_map_r2p: dict[int, int] = {}
    used_p_heavy: set[int] = set()
    for frag in r_frags:
        qmol, qheavy = _heavy_query_mol(r_mol, frag)
        matches = p_flat.GetSubstructMatches(qmol, uniquify=False)
        cand = []
        for m in matches:
            if any(p_syms[mi] == "H" for mi in m):
                continue
            if any(mi in used_p_heavy for mi in m):
                continue
            r_pts = np.array([r_pos[h] for h in qheavy])
            p_pts = np.array([p_pos[mi] for mi in m])
            R, t = _kabsch(p_pts, r_pts)
            al = _apply(R, t, p_pts)
            rmsd = float(np.sqrt(np.mean(np.sum((al - r_pts) ** 2, axis=1))))
            cand.append((rmsd, m))
        if not cand:
            raise RuntimeError("A reactant fragment found no substructure match in the product.")
        cand.sort()
        _, best_m = cand[0]
        for rh, ph in zip(qheavy, best_m):
            heavy_map_r2p[rh] = ph
            used_p_heavy.add(ph)

    r_heavy = [i for i, s in enumerate(r_syms) if s != "H"]

    # 2. Hydrogen mapping by parent proximity (Hungarian), with a global pool
    #    for protons that change parent (e.g. a transferring H).
    r_pts = np.array([r_pos[i] for i in r_heavy])
    p_pts = np.array([p_pos[heavy_map_r2p[i]] for i in r_heavy])
    R_h, t_h = _kabsch(p_pts, r_pts)
    p_aligned = _apply(R_h, t_h, p_pos)

    h_map_r2p: dict[int, int] = {}
    pooled_rH: list[int] = []
    pooled_pH: list[int] = []
    for rh in r_heavy:
        ph = heavy_map_r2p[rh]
        rHs = _get_bonded_h(r_mol, rh)
        pHs = _get_bonded_h(p_mol, ph)
        if len(rHs) == len(pHs) and len(rHs) > 0:
            D = np.array([[np.linalg.norm(p_aligned[phh] - r_pos[rhh]) for rhh in rHs] for phh in pHs])
            rows, cols = linear_sum_assignment(D)
            for a, b in zip(rows, cols):
                h_map_r2p[rHs[b]] = pHs[a]
        else:
            pooled_rH.extend(rHs)
            pooled_pH.extend(pHs)

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

    full = {**heavy_map_r2p, **h_map_r2p}
    P = [None] * n  # type: ignore[list-item]
    for ri, pi in full.items():
        P[ri] = pi
    if any(x is None for x in P) or len(set(P)) != n:
        raise RuntimeError("Atom mapping is not a valid permutation.")
    return P  # type: ignore[return-value]


def map_product_onto_reactant(reactant: Atoms, product: Atoms, *, align: bool = True) -> Atoms:
    """Return the product reordered (and optionally Kabsch-aligned) to the reactant.

    Atom *k* of the returned atoms corresponds to atom *k* of ``reactant``.
    The permutation is stored in ``returned.info["atom_mapping"]``.
    """
    P = compute_atom_mapping(reactant, product)
    p_pos = product.get_positions()
    p_syms = product.get_chemical_symbols()
    reordered = Atoms(symbols=[p_syms[m] for m in P], positions=p_pos[P])
    reordered.info = dict(product.info)
    if align:
        R, t = _kabsch(reordered.positions, reactant.positions)
        reordered.positions = _apply(R, t, reordered.positions)
    reordered.info["atom_mapping"] = P
    return reordered


def mapped_displacements(reactant: Atoms, product: Atoms) -> np.ndarray:
    """Per-atom displacement (Å) after optimal mapping+alignment."""
    mapped = map_product_onto_reactant(reactant, product)
    return np.linalg.norm(mapped.positions - reactant.positions, axis=1)


if __name__ == "__main__":
    import sys
    from ase.io import read

    r = read(sys.argv[1])
    p = read(sys.argv[2])
    disp = mapped_displacements(r, p)
    syms = r.get_chemical_symbols()
    print(f"mapped RMSD: {float(np.sqrt((disp**2).mean())):.3f} Å")
    print(f"max displacement: {disp.max():.3f} Å (atom {int(disp.argmax())} {syms[disp.argmax()]})")
    print("heavy-atom max displacement:", float(disp[[i for i, s in enumerate(syms) if s != 'H']].max()), "Å")