"""Nudged elastic band (NEB) transition-state search with GFN2-xTB.

Pipeline
--------
1. Read reactant / product structures (e.g. from Gaussian 16 logs).
2. Re-optimize both endpoints at the GFN2-xTB level so the endpoints and the
   band share the same level of theory.
3. Build the initial path with the image-dependent pair potential (IDPP),
   which avoids the unphysical bond stretching of a straight linear path.
4. Relax the band with a nudged elastic band and then turn on the **climbing
   image** (CI-NEB).  The climbing image converges to the highest-energy point
   along the minimum-energy path -- a transition-state estimate.
5. Optionally refine the climbing image to a true first-order saddle point
   with the **Dimer** method.
6. Report the energy profile and write the TS geometry.

The energy barrier is reported relative to the reactant, in eV (and kcal/mol).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import os

# --- Threading guards for the UMA (torch) backend on macOS ----------------
# fairchem-core pulls in torch, which ships its own OpenMP runtime that
# conflicts with the system libomp when numpy/tblite/rdkit are also loaded.
# Force a single OpenMP/MKL thread and allow the duplicate-lib workaround.
# Harmless for the tblite backend.  Set before importing numpy/torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS
from ase.mep import NEB, NEBTools

from .calculator import make_calculator
from .gaussian import read_gaussian_log

KCAL_MOL_PER_EV = 23.0605


def _resolve_calculator_factory(method: str, **uma_kwargs):
    """Return a ``calc_factory(charge, mult) -> Calculator`` for a backend.

    ``method`` is "tblite" (GFN2-xTB, default) or "uma" (Fairchem UMA).  The
    UMA factory additionally relies on ``atoms.info`` carrying the charge/spin
    metadata, which :func:`_attach_calc` sets uniformly for both backends.
    """
    if method == "tblite":
        return lambda charge, mult: make_calculator(charge, mult)
    if method == "uma":
        from .uma import make_uma_calculator

        return lambda charge, mult: make_uma_calculator(
            charge, mult, **uma_kwargs
        )
    raise ValueError(f"unknown method {method!r}; use 'tblite' or 'uma'")


@dataclass
class NebResult:
    reactant: Atoms
    product: Atoms
    images: list[Atoms]
    energies: list[float]  # len(images), in eV
    ts_image: Atoms
    ts_energy: float  # eV
    barrier_forward: float  # eV, TS - reactant
    barrier_reverse: float  # eV, TS - product
    reaction_energy: float  # eV, product - reactant
    ts_index: int
    log: list[str] = field(default_factory=list)


def _attach_calc(atoms: Atoms, charge: int, mult: int, calc_factory: Callable[[int, int], object] | None = None) -> None:
    """Attach a calculator and the charge/spin metadata to ``atoms``.

    The charge/spin go into ``atoms.info``; the GFN2-xTB backend ignores them
    (it takes them as constructor args via the factory), while the UMA ``omol``
    head reads them from ``atoms.info`` at evaluation time.  Setting them on
    every image keeps both backends consistent.
    """
    if calc_factory is None:
        calc_factory = _resolve_calculator_factory("tblite")
    atoms.calc = calc_factory(charge, mult)
    atoms.info["charge"] = int(charge)
    atoms.info["spin"] = int(mult)


def relax(atoms: Atoms, charge: int, mult: int, fmax: float, label: str, traj: Path | None, log: list[str], *, calc_factory: Callable[[int, int], object] | None = None) -> Atoms:
    """Minimize ``atoms`` (FIRE optimizer) at the chosen level of theory."""
    atoms = atoms.copy()
    _attach_calc(atoms, charge, mult, calc_factory)
    optimizer = FIRE(atoms, trajectory=str(traj) if traj else None, logfile=None)
    log.append(f"Relaxing {label} (fmax={fmax} eV/Å) ...")
    optimizer.run(fmax=fmax)
    f = atoms.get_forces()
    fmax_actual = float(np.sqrt((f ** 2).sum(axis=1)).max())
    log.append(f"  {label} done: E={atoms.get_potential_energy():.6f} eV, max|F|={fmax_actual:.4f}")
    return atoms


def _make_images(initial: Atoms, final: Atoms, n_images: int, charge: int, mult: int, *, calc_factory: Callable[[int, int], object] | None = None) -> list[Atoms]:
    """Create ``n_images + 2`` images (endpoints + interiors), each with a calc."""
    images = [initial.copy()]
    for _ in range(n_images):
        images.append(initial.copy())
    images.append(final.copy())
    for image in images:
        _attach_calc(image, charge, mult, calc_factory)
    return images


def _geodesic_path(initial: Atoms, final: Atoms, n_total: int) -> np.ndarray:
    """Build an ``n_total``-image geodesic initial path between the endpoints.

    Uses Zhu, Thompson & Martínez' geodesic interpolation
    (``geodesic-interpolate``): the shortest path on the molecular
    configuration manifold under a Morse-scaled pairwise-distance metric, so
    the path length is the *number of bond changes* and unphysical events
    (ring opening, proton teleportation) are penalised.  This is a far better
    seed than IDPP for reactions with large amplitude motion / bond
    breaking-forming (it is exactly what IDPP gets wrong for fragment
    associations).  Geometry-only -- no energy evaluations.

    Returns an ``(n_total, n_atoms, 3)`` array.  The caller pins the endpoints
    to the relaxed minima, so only the interior images are taken from this.
    """
    from geodesic_interpolate import Geodesic, redistribute

    syms = initial.get_chemical_symbols()
    raw = np.stack([initial.get_positions(), final.get_positions()])
    g = Geodesic(syms, raw)
    # ``sweep`` minimises the path length one image at a time; robust for
    # 44-atom systems.  ``redistribute`` then spaces images evenly by arc
    # length (so the band is well-conditioned for the spring forces).
    g.sweep(tol=1e-3, max_iter=50, micro_iter=20)
    full = np.asarray(redistribute(syms, g.path, n_total))
    if full.shape[0] != n_total:  # defensive: pad/trim by interpolation
        from scipy.interpolate import interp1d
        t = np.linspace(0, 1, full.shape[0])
        ti = np.linspace(0, 1, n_total)
        full = interp1d(t, full, axis=0)(ti)
    return full


def _seed_path(images: list[Atoms], initial: Atoms, final: Atoms, interp: str, log: list[str], traj_dir: Path) -> None:
    """Fill the interior image positions from the chosen interpolation method.

    ``interp`` is "idpp" (default), "geodesic", or "linear".  Endpoints are
    always pinned to the relaxed minima regardless of the method, since the
    geodesic / IDPP endpoints can differ slightly from the relaxed geometry.
    """
    if interp == "geodesic":
        full = _geodesic_path(initial, final, len(images))
        for i, img in enumerate(images):
            img.set_positions(full[i].copy())
        write(traj_dir / "geodesic_path.xyz", images)
        log.append(f"Geodesic initial path written ({len(images)} images).")
    elif interp == "linear":
        for img in images:
            img.calc = img.calc  # no-op; positions set by interpolate below
        # fall through to ASE's linear interpolation via interpolate()
        neb_tmp = NEB(images, climb=False, method="improvedtangent")
        neb_tmp.interpolate(method="linear")
        write(traj_dir / "linear_path.xyz", images)
        log.append(f"Linear initial path written ({len(images)} images).")
    else:  # idpp
        neb_tmp = NEB(images, climb=False, method="improvedtangent")
        neb_tmp.interpolate(method="idpp")
        write(traj_dir / "idpp_path.xyz", images)
        log.append(f"IDPP initial path written ({len(images)} images).")
    # Pin endpoints to the relaxed minima exactly.
    images[0].set_positions(initial.get_positions().copy())
    images[-1].set_positions(final.get_positions().copy())


def run_neb(
    reactant: Atoms,
    product: Atoms,
    *,
    n_images: int = 15,
    fmax: float = 0.05,
    k: float = 0.05,
    relax_endpoints: bool = True,
    refine_dimer: bool = False,
    climb: bool = True,
    max_steps: int = 400,
    traj_dir: Path = Path("output"),
    log: list[str] | None = None,
    method: str = "tblite",
    uma_kwargs: dict | None = None,
    interp: str = "idpp",
    init_path: str | None = None,
) -> NebResult:
    """Run a CI-NEB transition-state search between ``reactant`` and ``product``.

    ``method`` selects the PES backend: ``"tblite"`` (GFN2-xTB, default) or
    ``"uma"`` (Fairchem UMA, task ``omol`` for molecules).  ``uma_kwargs`` is
    forwarded to :func:`neb_ts.uma.make_uma_calculator` (e.g.
    ``{"model": "uma-s-1p1", "task_name": "omol", "device": "cpu"}``).

    ``init_path`` (optional) resumes a prior run: a multi-model XYZ of the
    full band (``n_images + 2`` images, endpoints included).  The endpoints
    are taken FROM the band (they are the prior run's level-of-theory minima;
    their absolute orientation may differ from the input reactant/product
    because of ``remove_rotation_and_translation``), endpoint re-optimization
    and path interpolation are skipped, and the NEB continues from the saved
    band.  This preserves all prior optimization work after an interruption.
    """
    log = [] if log is None else log
    traj_dir = Path(traj_dir)
    traj_dir.mkdir(parents=True, exist_ok=True)

    calc_factory = _resolve_calculator_factory(method, **(uma_kwargs or {}))
    log.append(f"Backend: {method}" + (f" ({uma_kwargs})" if uma_kwargs else ""))

    charge = int(reactant.info.get("charge", 0))
    mult = int(reactant.info.get("multiplicity", reactant.info.get("spin", 1)))

    if init_path is not None:
        # --- Resume from a saved band. ---
        log.append(f"Resuming from {init_path} (endpoints fixed at prior minima).")
        band = read(init_path, ":")
        if len(band) != n_images + 2:
            raise SystemExit(
                f"init_path has {len(band)} images; expected {n_images + 2} "
                f"(n_images={n_images} + 2 endpoints)."
            )
        initial, final = band[0].copy(), band[-1].copy()
        for a in (initial, final):
            _attach_calc(a, charge, mult, calc_factory)
        e_r = float(initial.get_potential_energy())
        e_p = float(final.get_potential_energy())
        images = [a.copy() for a in band]
        for image in images:
            _attach_calc(image, charge, mult, calc_factory)
    else:
        # 1. Re-optimize the endpoints at the chosen level of theory.
        log.append(f"Initial-path interpolation: {interp}")
        if relax_endpoints:
            initial = relax(reactant, charge, mult, fmax, "reactant", traj_dir / "reactant_relax.traj", log, calc_factory=calc_factory)
            final = relax(product, charge, mult, fmax, "product", traj_dir / "product_relax.traj", log, calc_factory=calc_factory)
        else:
            initial, final = reactant.copy(), product.copy()
            for a in (initial, final):
                _attach_calc(a, charge, mult, calc_factory)
        e_r = float(initial.get_potential_energy())
        e_p = float(final.get_potential_energy())
        # 2. Build images and an initial path (idpp / geodesic / linear).
        images = _make_images(initial, final, n_images, charge, mult, calc_factory=calc_factory)
        _seed_path(images, initial, final, interp, log, traj_dir)

    neb = NEB(images, k=k, climb=False, method="improvedtangent", remove_rotation_and_translation=True)

    # 3. NEB pre-relaxation (climb off).  FIRE is the ASE-recommended optimizer
    # for NEB: it is far more robust than BFGS/LBFGS on the high-dimensional
    # band and does not stall the way quasi-Newton methods can here.
    opt = FIRE(neb, trajectory=str(traj_dir / "neb.traj"), logfile=None)
    log.append(f"NEB relaxation (FIRE, fmax={fmax} eV/Å, max {max_steps} steps) ...")
    opt.run(fmax=fmax, steps=max_steps)
    log.append("  NEB pre-relaxation done.")

    # 4. Turn on the climbing image and re-relax to converge the TS image.
    #    Climb is skipped when ``climb=False`` (e.g. segmented NEBs whose TS
    #    image is refined separately by fixed-mode -- CI can dissociate on
    #    loose/bimolecular PESs, see refine_fixedmode_uma.py).
    if climb:
        neb.climb = True
        opt_ci = FIRE(neb, trajectory=str(traj_dir / "neb_ci.traj"), logfile=None)
        log.append(f"CI-NEB refinement (climbing image on, fmax={fmax} eV/Å, max {max_steps} steps) ...")
        opt_ci.run(fmax=fmax, steps=max_steps)
    else:
        log.append("CI-NEB refinement skipped (climb=False).")

    energies = [float(img.get_potential_energy()) for img in images]
    ts_index = int(np.argmax(energies))
    ts_energy = energies[ts_index]
    ts_image = images[ts_index].copy()
    ts_image.calc = None  # detach before pickling / writing

    # 5. Optional Dimer refinement to a true first-order saddle point.
    if refine_dimer:
        # Seed the dimer with the NEB tangent at the TS (direction between the
        # neighbouring images) -- a good guess for the reaction coordinate.
        if 0 < ts_index < len(images) - 1:
            mode = images[ts_index + 1].positions - images[ts_index - 1].positions
        else:
            mode = None
        ts_image = refine_dimer(ts_image, charge, mult, fmax, traj_dir, log, steps=max_steps, initial_mode=mode, calc_factory=calc_factory)

    barrier_forward = ts_energy - e_r
    barrier_reverse = ts_energy - e_p
    reaction_energy = e_p - e_r

    log.append("---- Energy profile (eV) ----")
    for i, e in enumerate(energies):
        tag = "  <-- TS" if i == ts_index else ""
        log.append(f"  image {i:2d}: {e: .6f}  (Δ={e - e_r: .6f}){tag}")
    log.append("---- Summary ----")
    log.append(f"  Reactant energy : {e_r:.6f} eV")
    log.append(f"  Product  energy : {e_p:.6f} eV")
    log.append(f"  TS energy (img {ts_index}) : {ts_energy:.6f} eV")
    log.append(f"  Forward barrier : {barrier_forward:.4f} eV  ({barrier_forward * KCAL_MOL_PER_EV:.2f} kcal/mol)")
    log.append(f"  Reverse barrier : {barrier_reverse:.4f} eV  ({barrier_reverse * KCAL_MOL_PER_EV:.2f} kcal/mol)")
    log.append(f"  Reaction energy : {reaction_energy:.4f} eV  ({reaction_energy * KCAL_MOL_PER_EV:.2f} kcal/mol)")

    write(traj_dir / "ts_structure.xyz", ts_image)
    write(traj_dir / "neb_final_path.xyz", images)
    _write_energy_profile(traj_dir / "energy_profile.dat", energies, e_r, ts_index)

    return NebResult(
        reactant=initial,
        product=final,
        images=images,
        energies=energies,
        ts_image=ts_image,
        ts_energy=ts_energy,
        barrier_forward=barrier_forward,
        barrier_reverse=barrier_reverse,
        reaction_energy=reaction_energy,
        ts_index=ts_index,
        log=log,
    )


def refine_dimer(ts_guess: Atoms, charge: int, mult: int, fmax: float, traj_dir: Path, log: list[str], *, steps: int = 200, initial_mode: np.ndarray | None = None, calc_factory: Callable[[int, int], object] | None = None) -> Atoms:
    """Refine a climbing-image guess to a true first-order saddle.

    Uses the Dimer minimum-mode-following method: it estimates the lowest
    curvature mode, inverts the force along it (so the image climbs uphill in
    that one direction while relaxing in all others), and converges to a saddle
    point -- which should then show exactly one imaginary frequency.

    Parameters
    ----------
    initial_mode
        Optional ``(n_atoms, 3)`` array giving an initial guess for the
        reaction-coordinate eigenmode (e.g. the NEB tangent at the TS).  Seeding
        the dimer with the NEB tangent is strongly recommended -- with a random
        initial mode the dimer tends to run away to a high-energy structure.
    """
    from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

    atoms = ts_guess.copy()
    _attach_calc(atoms, charge, mult, calc_factory)
    d_control = DimerControl(
        logfile=None,
        eigenmode_logfile=None,
        initial_eigenmode_method="displacement" if initial_mode is not None else "gauss",
    )
    eigenmodes = None
    if initial_mode is not None:
        eigenmodes = [initial_mode / np.linalg.norm(initial_mode)]
    min_mode = MinModeAtoms(atoms, control=d_control, logfile=None, eigenmodes=eigenmodes)
    if initial_mode is None:
        min_mode.displace()
    opt = MinModeTranslate(min_mode, logfile=None, trajectory=str(traj_dir / "dimer.traj"))
    log.append(f"Dimer refinement (fmax={fmax} eV/Å, max {steps} steps) ...")
    opt.run(fmax=fmax, steps=steps)
    e = float(atoms.get_potential_energy())
    f = atoms.get_forces()
    fmax_actual = float(np.sqrt((f ** 2).sum(axis=1)).max())
    try:
        curv = float(min_mode.get_curvature())
    except Exception:
        curv = float("nan")
    log.append(f"  Dimer done: E={e:.6f} eV, max|F|={fmax_actual:.4f}, curvature={curv:.4f}")
    out = atoms.copy()
    out.calc = None
    return out


# Backwards-compatible alias used internally by run_neb.
def _refine_dimer(ts_guess: Atoms, charge: int, mult: int, fmax: float, traj_dir: Path, log: list[str]) -> Atoms:
    return refine_dimer(ts_guess, charge, mult, fmax, traj_dir, log, calc_factory=None)


def _write_energy_profile(path: Path, energies: list[float], e_r: float, ts_index: int) -> None:
    with open(path, "w") as f:
        f.write("# image_index  energy(eV)  relative_to_reactant(eV)  rel(kcal/mol)\n")
        for i, e in enumerate(energies):
            rel = e - e_r
            f.write(f"{i:3d}  {e: .8f}  {rel: .8f}  {rel * KCAL_MOL_PER_EV: .4f}\n")


def _load_input(path: str) -> Atoms:
    """Load a structure from a Gaussian log or any format ASE can read."""
    p = Path(path)
    if p.suffix.lower() == ".log":
        return read_gaussian_log(p)
    return read(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Find a transition state with CI-NEB + GFN2-xTB.")
    ap.add_argument("reactant", help="Reactant structure (Gaussian .log or ASE-readable file)")
    ap.add_argument("product", help="Product structure (Gaussian .log or ASE-readable file)")
    ap.add_argument("--images", type=int, default=15, help="Number of interior NEB images (default 15)")
    ap.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold in eV/Å (default 0.05)")
    ap.add_argument("--k", type=float, default=0.05, help="NEB spring constant in eV/Å² (default 0.05)")
    ap.add_argument("--no-relax-endpoints", action="store_true", help="Skip GFN2-xTB re-optimization of endpoints")
    ap.add_argument("--refine-dimer", action="store_true", help="Refine the climbing image with the Dimer method")
    ap.add_argument("--no-climb", action="store_true", help="Skip the CI-NEB phase (climb-off only); the TS image is the max-energy image. Use when CI dissociates (loose/bimolecular PES) and the max image is refined separately by fixed-mode.")
    ap.add_argument("--max-steps", type=int, default=400, help="Max optimizer steps per NEB phase (default 400)")
    ap.add_argument("--no-map", action="store_true", help="Disable automatic RDKit atom mapping of product onto reactant")
    ap.add_argument("--outdir", default="output", help="Output directory")
    ap.add_argument("--method", choices=["tblite", "uma"], default="tblite",
                    help="PES backend: tblite (GFN2-xTB, default) or uma (Fairchem UMA)")
    ap.add_argument("--uma-model", default="uma-s-1p1", help="UMA model id (default uma-s-1p1)")
    ap.add_argument("--uma-task", default="omol", help="UMA task head (omol for molecules)")
    ap.add_argument("--uma-device", default="cpu", help="UMA device (cpu; MPS not supported)")
    ap.add_argument("--interp", choices=["idpp", "geodesic", "linear"], default="idpp",
                    help="Initial-path interpolation: idpp (default), geodesic (Zhu/Thompson/Martínez), or linear")
    ap.add_argument("--init-path", default=None,
                    help="Resume: multi-model XYZ of a saved full band (n_images+2). Skips endpoint relaxation and path interpolation; continues the NEB from the saved band.")
    args = ap.parse_args(argv)

    reactant = _load_input(args.reactant)
    product = _load_input(args.product)
    if reactant.get_chemical_symbols() != product.get_chemical_symbols():
        raise SystemExit(
            "Reactant and product have different compositions -- NEB requires "
            "an atom-preserving (isomeric) transformation."
        )

    # Ensure atom k means the same chemical atom in both endpoints.  This is
    # essential when the two structures come from separate calculations whose
    # file atom orderings differ (especially for equivalent H/O atoms).
    if not args.no_map:
        from .align import map_product_onto_reactant

        product = map_product_onto_reactant(reactant, product)
        # Re-center the reactant to match the aligned product's frame.
        reactant.positions = reactant.positions - reactant.positions.mean(axis=0)
        print("Atom mapping: product reordered onto reactant via RDKit "
              "(use --no-map to disable).")

    uma_kwargs = None
    if args.method == "uma":
        uma_kwargs = {"model": args.uma_model, "task_name": args.uma_task,
                      "device": args.uma_device}

    result = run_neb(
        reactant,
        product,
        n_images=args.images,
        fmax=args.fmax,
        k=args.k,
        relax_endpoints=not args.no_relax_endpoints,
        refine_dimer=args.refine_dimer,
        climb=not args.no_climb,
        max_steps=args.max_steps,
        traj_dir=Path(args.outdir),
        method=args.method,
        uma_kwargs=uma_kwargs,
        interp=args.interp,
        init_path=args.init_path,
    )
    for line in result.log:
        print(line)
    print("\nTransition state written to", Path(args.outdir) / "ts_structure.xyz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())