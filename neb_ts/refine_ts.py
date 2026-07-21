"""Refine a CI-NEB climbing-image guess to a true first-order saddle point.

The climbing image from CI-NEB is the maximum along the minimum-energy path,
but it is only a true transition state once all forces vanish there.  This
script loads a TS guess (e.g. ``output/ts_structure.xyz``) and runs the Dimer
minimum-mode-following method, which climbs uphill along the softest (reaction)
coordinate while relaxing in every other direction, converging to a saddle.

The Dimer is **seeded with the NEB tangent** (the direction from the image
just before the TS to the image just after it), which is an excellent initial
guess for the reaction coordinate.  Pass ``--path`` pointing at the NEB path
file (``output/neb_final_path.xyz``) and ``--ts-index`` for the climbing image.

Usage:
    uv run python -m neb_ts.refine_ts output/ts_structure.xyz \\
        --path output/neb_final_path.xyz --ts-index 11 \\
        --out output/ts_refined.xyz
    uv run python -m neb_ts.frequencies output/ts_refined.xyz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read, write

from .neb_run import refine_dimer


def _tangent_from_path(path_file: str, ts_index: int) -> np.ndarray | None:
    """Return the normalized (image[ts+1] - image[ts-1]) displacement."""
    images = read(path_file, index=":")
    if not images or ts_index <= 0 or ts_index >= len(images) - 1:
        return None
    tangent = images[ts_index + 1].positions - images[ts_index - 1].positions
    return tangent / np.linalg.norm(tangent)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Refine a TS guess to a saddle with the Dimer method.")
    ap.add_argument("structure", help="TS guess (e.g. output/ts_structure.xyz)")
    ap.add_argument("--path", default=None, help="NEB path file (e.g. output/neb_final_path.xyz) to seed the reaction-coordinate mode")
    ap.add_argument("--ts-index", type=int, default=None, help="Index of the TS image in --path (for the tangent seed)")
    ap.add_argument("--out", default="output/ts_refined.xyz")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--multiplicity", type=int, default=1)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args(argv)

    atoms = read(args.structure)
    atoms.info.setdefault("charge", args.charge)
    atoms.info.setdefault("multiplicity", args.multiplicity)

    initial_mode = None
    if args.path and args.ts_index is not None:
        initial_mode = _tangent_from_path(args.path, args.ts_index)
        if initial_mode is None:
            print("WARNING: could not derive a tangent from --path/--ts-index; "
                  "falling back to a random initial dimer mode (less reliable).")

    log: list[str] = []
    refined = refine_dimer(
        atoms,
        args.charge,
        args.multiplicity,
        args.fmax,
        Path(args.out).parent,
        log,
        steps=args.steps,
        initial_mode=initial_mode,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, refined)
    for line in log:
        print(line)
    print(f"\nRefined transition state written to {out}")
    print("Verify with: uv run python -m neb_ts.frequencies", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())