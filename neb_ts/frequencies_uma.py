"""Vibrational analysis to verify a TS candidate at the **UMA** level.

Mirror of :mod:`neb_ts.frequencies` (GFN2) but using the UMA (Fairchem) ASE
calculator, so the Hessian is computed on the same PES the TS was refined on.
A true transition state has exactly one negative (imaginary) Hessian eigenvalue.

UMA is an ML potential with smooth forces, so a slightly larger finite-difference
step (``delta=0.02`` Å) is used by default to stay above any tiny numerical
roughness.  6N+1 force evaluations for N atoms (~265 for the 44-atom Maillard
system, ~4-5 min on CPU).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ase import Atoms
from ase.io import read
from ase.vibrations import Vibrations

from .uma import make_uma_calculator, set_charge_spin


CM1_PER_EV = 8065.544  # eV -> cm^-1


def compute_frequencies(atoms: Atoms, charge: int = 0, multiplicity: int = 1, *,
                        delta: float = 0.02, model: str = "uma-s-1p1",
                        task_name: str = "omol", device: str = "cpu") -> tuple[list[float], int]:
    atoms = atoms.copy()
    set_charge_spin(atoms, charge, multiplicity)
    atoms.calc = make_uma_calculator(charge, multiplicity, model=model,
                                     task_name=task_name, device=device)
    vib = Vibrations(atoms, delta=delta, nfree=2)
    vib.run()
    freqs = vib.get_frequencies()
    real_freqs = [float(f.real) * (1.0 if abs(f.imag) < 1e-10 else -1.0) if isinstance(f, complex) else float(f) for f in freqs]
    real_freqs = sorted(real_freqs)
    n_imag = sum(1 for f in real_freqs if f < -10.0)
    return real_freqs, n_imag


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a TS candidate by vibrational analysis (UMA).")
    ap.add_argument("structure", help="Structure to analyse (e.g. output/uma_geodesic/ts_refined.xyz)")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--multiplicity", type=int, default=1)
    ap.add_argument("--delta", type=float, default=0.02, help="Displacement in Å (default 0.02 for ML potential)")
    ap.add_argument("--out", default="output/uma_geodesic/frequencies.txt")
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="omol")
    ap.add_argument("--uma-device", default="cpu")
    args = ap.parse_args(argv)

    atoms = read(args.structure)
    n = len(atoms)
    print(f"Computing UMA vibrational frequencies for {n} atoms "
          f"({6 * n + 1} force evaluations, delta={args.delta} Å) ...")
    freqs, n_imag = compute_frequencies(atoms, args.charge, args.multiplicity,
                                        delta=args.delta, model=args.uma_model,
                                        task_name=args.uma_task, device=args.uma_device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("# frequency (cm^-1)\n")
        for nu in freqs:
            f.write(f"{nu: .4f}\n")
    print(f"Imaginary modes (< -10 cm^-1): {n_imag}")
    if n_imag == 1:
        print("=> Exactly one imaginary mode: confirmed transition state.")
    elif n_imag == 0:
        print("=> No imaginary modes: this is a minimum, not a transition state.")
    else:
        print(f"=> {n_imag} imaginary modes: higher-order saddle (not a true TS).")
    print(f"\nLowest frequencies (cm^-1):")
    for nu in freqs[:8]:
        print(f"  {nu: .2f}")
    print(f"\nFull spectrum written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())