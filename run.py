#!/usr/bin/env python
"""Convenience entry point for the NEB transition-state search.

Run from the project root:

    uv run python run.py                 # full CI-NEB on the Gaussian logs
    uv run python run.py --refine-dimer  # also refine the TS with the Dimer method

Outputs (in ./output):
    ts_structure.xyz      -- the transition-state geometry (open in PyMOL)
    neb_final_path.xyz    -- full NEB path as a multi-model XYZ (animate in PyMOL)
    idpp_path.xyz         -- initial IDPP path
    energy_profile.dat    -- per-image energies and barriers
    neb_run.log           -- text log of the run
"""

from neb_ts.neb_run import main as run_main

if __name__ == "__main__":
    raise SystemExit(run_main())