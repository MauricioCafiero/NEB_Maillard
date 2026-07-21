"""Refine a TS guess to a true first-order saddle with the **Dimer** method at
the UMA level.

Unlike fixed-mode refinement (which holds the reaction coordinate fixed and
only relaxes perpendicular -- so it slides *down* into a basin if the starting
RC is past the saddle), the dimer *climbs uphill* along the softest curvature
mode while relaxing in every perpendicular direction, so it converges to a
saddle even from a slightly-off starting point.

Use case here: the segmented reactant->intermediate NEB produced an under-
relaxed off-MEP spike at image 10 (+22.6 kcal artifact, |F_perp|=0.157 eV/A);
fixed-mode from image 10 collapsed 16.8 kcal into the carbinolamine basin.
The dimer, started from a nearby on-MEP image (e.g. image 9 or 11) and seeded
with a *clean* reaction-coordinate mode (intermediate - reactant positions,
NOT the polluted image-10 tangent), climbs to the real TS1.

Seed the dimer mode with ``--mode-from <reactant.xyz> --mode-to <intermediate.xyz>``
(the net RC).  Optionally ``--path`` + ``--ts-index`` for the NEB tangent instead.

Usage::

    python -m neb_ts.refine_dimer_uma output/seg1/ts_img11.xyz \\
        --mode-from output/reactant_mapped.xyz --mode-to output/uma_geodesic/intermediate_relaxed.xyz \\
        --uma-model uma-s-1p1 --uma-task omol --uma-device cpu \\
        --out output/ts1/ts1_dimer.xyz --fmax 0.05 --steps 300
    python -m neb_ts.frequencies_uma output/ts1/ts1_dimer.xyz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from .neb_run import refine_dimer, _resolve_calculator_factory
from .uma import set_charge_spin


def _max_heavy_heavy(atoms) -> float:
    from scipy.spatial.distance import cdist
    d = cdist(atoms.get_positions(), atoms.get_positions())
    sym = atoms.get_chemical_symbols()
    return float(max(
        d[i, j] for i in range(len(atoms)) for j in range(i + 1, len(atoms))
        if sym[i] != "H" and sym[j] != "H"
    ))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="TS guess (e.g. a NEB image near the saddle)")
    ap.add_argument("--mode-from", default=None, help="xyz for the start of the RC mode (e.g. reactant)")
    ap.add_argument("--mode-to", default=None, help="xyz for the end of the RC mode (e.g. intermediate)")
    ap.add_argument("--path", default=None, help="NEB path file (alt. mode seed via tangent)")
    ap.add_argument("--ts-index", type=int, default=None, help="image index in --path for the tangent seed")
    ap.add_argument("--out", default="output/ts1/ts1_dimer.xyz")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--multiplicity", type=int, default=1)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="omol")
    ap.add_argument("--uma-device", default="cpu")
    args = ap.parse_args(argv)

    atoms = read(args.structure)
    set_charge_spin(atoms, args.charge, args.multiplicity)

    initial_mode = None
    if args.mode_from and args.mode_to:
        a = read(args.mode_from); b = read(args.mode_to)
        initial_mode = b.positions - a.positions
        print(f"Mode seed: net RC ({args.mode_to} - {args.mode_from}), |mode|={np.linalg.norm(initial_mode):.3f} Å")
    elif args.path and args.ts_index is not None:
        images = read(args.path, ":")
        initial_mode = images[args.ts_index + 1].positions - images[args.ts_index - 1].positions
        print(f"Mode seed: NEB tangent at img {args.ts_index}")
    else:
        print("WARNING: no mode seed given -- dimer will use a random initial mode (run-away risk).")

    calc_factory = _resolve_calculator_factory("uma", model=args.uma_model,
                                               task_name=args.uma_task, device=args.uma_device)
    print(f"Start: E={atoms.get_potential_energy():.6f} eV, max heavy-heavy={_max_heavy_heavy(atoms):.2f} Å"
          if atoms.calc else "Start: (no calc yet)")
    log: list[str] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    refined = refine_dimer(atoms, args.charge, args.multiplicity, args.fmax,
                           out.parent, log, steps=args.steps, initial_mode=initial_mode,
                           calc_factory=calc_factory)
    write(out, refined)
    for line in log:
        print(line)
    print(f"\nRefined TS written to {out}")
    print(f"Refined max heavy-heavy = {_max_heavy_heavy(refined):.2f} Å (reactant span ~11.3 Å; >>13 = dissociation)")
    print("Verify with: python -m neb_ts.frequencies_uma", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())