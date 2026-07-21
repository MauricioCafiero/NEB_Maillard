# NEB transition-state search for Maillard reactions

Given a reactant and a product structure, find a transition state with the
nudged elastic band (NEB) method. The pipeline uses **GFN2-xTB** (via
[`tblite`](https://github.com/tblite/tblite)) as the fast first-pass PES, with
**UMA** (Fairchem `uma-s-1p1`) as a fallback, ASE for the NEB/relaxation, and
RDKit for atom mapping. Independent **DFT** transition states (e.g. Gaussian 16
QST3 / `saddle=1`) can be supplied and cross-checked.

> **What this code will and won't find.** NEB from endpoints finds the
> highest-energy point on the minimum-energy path. On a "normal" PES that point
> *is* a first-order saddle (one imaginary frequency). On a **flat ridge /
> valley-ridge inflection** (common for loose, bimolecular, or charge-separated
> steps — e.g. the Maillard C–N formation and the dehydration) the NEB maximum
> is a high-energy *minimum* (zero imaginary) and **no discrete TS exists at
> that level of theory**. The workflow below detects this and falls back to a
> better PES (UMA) or a DFT saddle search. See `RESULTS.md` for the two worked
> Maillard steps and the cross-method comparison.

## Methods

A brief summary of the techniques the pipeline combines. Each links to where
it is used in the workflow below.

**Nudged Elastic Band (NEB)** — the core method. A chain of replicas ("images")
connecting the relaxed reactant and product is connected by virtual springs and
relaxed on the potential energy surface. The spring force acts only along the
band tangent and the true force only perpendicular to it ("nudging"); this
prevents the springs from pulling images off the MEP and the corner-cutting
that plagues plain elastic bands. The converged band traces the minimum-energy
path (MEP); its highest image is the approximate transition state. Implemented
with ASE (`NEB` + `FIRE`); tight `--fmax 0.02` eV/Å and enough images (18)
matter for loose/bimolecular TSs (looser settings give off-MEP spikes, see
`RESULTS.md`). → steps 1, 4(a).

**Climbing image (CI-NEB)** — once the band is roughly converged, the highest
image is freed from the springs and pushed up the PES along the tangent, which
drives it exactly onto the saddle. CI is great for compact single-bond TSs but
**runs away down dissociative/bimolecular channels**, so the workflow uses
**climb-off** (`--no-climb`) by default and refines the max image separately. →
step 1, failure-mode note in step 2.

**Initial-path interpolation** — NEB needs a starting band. Three options:
- **Geodesic** (Zhu/Thompson/Martínez) — interpolates in internal coordinates
  following the geodesic of the distance metric, keeping bonds/angles physical;
  the only interpolation that consistently landed images near the real MEP here.
- **IDPP** (image-dependent pair potential) — ASE default; a smooth,
  pair-distance-based path; usually fine, occasionally produces spike artifacts
  on loose TSs.
- **Linear** — Cartesian straight-line; cheap baseline, poor for large motions.
→ step 1 (`--interp geodesic`).

**Atom mapping** — NEB requires atom `k` to label the same atom in both
endpoints. RDKit (`map_product_onto_reactant`) reorders the product onto the
reactant by substructure matching (works when the product is a heavy-atom
substructure of the reactant, i.e. bond-forming steps). When atom order is
already identical, or when bonds break and the substructure assumption fails
(e.g. dehydration), use `--no-map`. → step 0.

**TS refinement — fixed-mode** — proven-safe saddle refinement. Take the NEB
tangent at the max image as a fixed reaction coordinate, project it out of the
forces, and minimize only the perpendicular forces (`ase.constraints.FixedMode`).
It *cannot run away* (the RC is frozen), so it either converges to a true saddle
or rolls the structure off the ridge into the adjacent valley — which is itself
diagnostic of a flat ridge. → step 3.

**TS refinement — Dimer method** — rotates to the softest Hessian mode and
translates uphill along it. Can localize saddles NEB misses, but **runs away on
flat ridges** (the softest mode is a backbone torsion, not the reaction
coordinate there). Used only as an optional cross-check. → key-flags
`--refine-dimer`.

**Vibrational verification** — the decisive test for a TS. Finite-difference
Hessian (ASE `Vibrations`); count modes below −10 cm⁻¹: **1 → first-order
saddle (TS)**, 0 → minimum (flat-ridge max or off-MEP spike), >1 → higher-order
saddle. This is what distinguishes "found a TS" from "found a high-energy
minimum on a flat ridge." → step 2, 3.

**Potentials (PES backends):**
- **GFN2-xTB** (via `tblite`) — fast, parameterized semi-empirical method; the
  default first-pass PES. No external weights. Handles normal covalent TSs
  well; **fails on charge-separated / proton-transfer / dehydration TSs**
  (spike, CI runaway, or fixed-mode collapse).
- **UMA** (Fairchem `uma-s-1p1`, task `omol`, CPU) — a learned ML potential;
  fallback when GFN2 fails. Slower (~3–4 h for a full NEB) but reaction energies
  track DFT better. Can still give a flat ridge where a true saddle exists.
- **DFT** (external, e.g. Gaussian 16 M06-2X/def2-TZVP) — the authoritative
  reference for barrier heights and for true saddle searches (QST3 /
  `opt=(calcall,saddle=1)`). Supplied as `.log` and cross-checked.
→ step 1 (GFN2), step 4 (UMA), step 5 (DFT).

**Free-fragment correction** — when an external TS uses a reduced atom model
(e.g. a dehydration TS with water omitted), add the energy of the free fragment
to compare it on the same atom count as the full endpoints. → step 5.

## Setup

```bash
uv sync                       # creates .venv and installs ase, tblite, rdkit, ...
```

For the UMA backend you also need the Fairchem weights for `uma-s-1p1`
available (CPU-only; MPS is not supported). GFN2-xTB needs no extra weights.

## Workflow for a new reactant/product pair

The recommended end-to-end recipe, in the order that worked for the bundled
Maillard steps. Drop in your own Gaussian 16 `.log` files (or any
ASE-readable `.xyz`) as the two endpoints.

### 0. Keep the machine awake (long runs) and check atom order

```bash
caffeinate -dimsu &                       # prevent sleep on long UMA runs
```

The NEB needs atom `k` to mean the same atom in both endpoints. If both
structures come from the same project template (identical atom ordering —
verify with the snippet below), pass `--no-map` and skip the RDKit mapping
(which can fail when bonds break, e.g. dehydrations). Otherwise let the
automatic RDKit mapping run (it assumes the product is a heavy-atom
substructure of the reactant — true for bond-forming steps like C–N formation,
false for bond-breaking/dehydration steps).

```bash
python - <<'PY'
from neb_ts.gaussian import read_gaussian_log
r = read_gaussian_log("REACTANT.log"); p = read_gaussian_log("PRODUCT.log")
print("same atom order:", r.get_chemical_symbols() == p.get_chemical_symbols(),
      "| composition:", dict.fromkeys(r.get_chemical_symbols()))
PY
```

### 1. First pass — GFN2-xTB geodesic NEB (climb-off)

Geodesic interpolation (Zhu/Thompson/Martínez) keeps the band physical and
lands images near the MEP; it was the only interpolation that got close to a
real TS here. Use **climb-off** (`--no-climb`) for dissociative / bimolecular
steps — the climbing image can run away down a dissociative channel on a loose
PES. Tight `--fmax 0.02` avoids off-MEP spike artifacts.

```bash
uv run python -m neb_ts.neb_run REACTANT.log PRODUCT.log \
  --method tblite --interp geodesic --no-climb \
  --fmax 0.02 --images 18 --max-steps 800 \
  --outdir output/step_gfn2
# add --no-map if both logs share atom order (step 0)
```

### 2. Verify the max image — one imaginary frequency?

```bash
uv run python -m neb_ts.frequencies output/step_gfn2/ts_structure.xyz \
  --out output/step_gfn2/ts_freq.txt
```

- **Exactly 1 imaginary mode → first-order saddle found.** Done; the NEB
  `energy_profile.dat` gives the forward/reverse barriers.
- **0 imaginary (softest mode small & positive) → flat ridge / off-MEP spike.**
  Go to step 3.
- **CI-NEB ran away** (max image at implausibly high energy, bond ripped):
  re-run step 1 with `--no-climb` (already recommended) and refine the max
  image separately (step 3) — do **not** chase the runaway.

### 3. Refine the max image to a saddle (if step 2 was inconclusive)

Fixed-mode refinement is **proven-safe**: it projects the reaction coordinate
out and minimizes the perpendicular forces, so it *cannot* run away the way a
dimer can. Seed it with the NEB tangent at the max image.

```bash
# GFN2:
uv run python -m neb_ts.refine_ts_fixedmode output/step_gfn2/ts_structure.xyz \
  --path output/step_gfn2/neb_final_path.xyz --ts-index <MAX_IMG> \
  --out output/step_gfn2/ts_refined.xyz --fmax 0.02 --max-steps 800
uv run python -m neb_ts.frequencies output/step_gfn2/ts_refined.xyz
```

Interpret the refined structure's frequencies:
- **1 imaginary → true TS.** Report its energy as the barrier.
- **0 imaginary and it collapsed to a basin** → no discrete saddle on this PES
  along this coordinate (flat ridge). The NEB max from step 1 is the practical
  barrier (highest point on the MEP), but it is *not* a first-order saddle.
- **0 imaginary and it held** (barely moved, `dE ≈ 0`) → also a flat ridge; the
  max image is a constrained minimum, not a saddle.

### 4. If GFN2 fails to resolve a discrete TS — pivot

This is expected for **charge-separated / proton-transfer / dehydration** TSs,
which semi-empirical GFN2-xTB handles poorly. Two options:

**(a) Re-run the NEB on the UMA PES** (better ML potential; climb-off +
fixed-mode, same recipe as step 1/3). UMA is slower (~3–4 h for 18 images /
800 steps on CPU) but its reaction energies track DFT better than GFN2.

```bash
uv run python -m neb_ts.neb_run REACTANT.log PRODUCT.log \
  --method uma --interp geodesic --no-climb --no-map \
  --fmax 0.02 --images 18 --max-steps 800 --outdir output/step_uma
# refine + verify:
uv run python -m neb_ts.refine_fixedmode_uma output/step_uma/ts_structure.xyz \
  --path output/step_uma/neb_final_path.xyz --ts-index <MAX_IMG> \
  --out output/step_uma/ts_refined.xyz
uv run python -m neb_ts.frequencies_uma output/step_uma/ts_refined.xyz
```

**(b) Supply / compute a DFT TS** as the authoritative reference (see step 5).
For a true saddle that NEB cannot localize from endpoints, a DFT saddle search
(QST3 or `opt=(calcall,saddle=1)`, seeded by a TS guess) is the reliable route.

### 5. Cross-check against an independent DFT transition state

If you have a Gaussian TS `.log`, extract its geometry and test it on the NEB
PES — this validates the NEB barrier without re-running anything:

```bash
# extract the DFT TS geometry (and its imaginary frequency, if the log has a
# freq block — grep "imaginary frequencies"):
python -c "from neb_ts.gaussian import read_gaussian_log; from ase.io import write; write('ts_dft.xyz', read_gaussian_log('TS.log'))"

# single-point + frequencies on your NEB PES (self-consistent, no guessing):
uv run python -m neb_ts.frequencies_uma ts_dft.xyz --out ts_dft_freq.txt   # or neb_ts.frequencies for GFN2
```

Compute barriers consistently. If the DFT TS used a **reduced atom model**
(e.g. water omitted from a dehydration TS), add the energy of a free fragment
to put it on the same atom count as the endpoints:

```
barrier_fwd = [E(TS_reduced) + E(free_fragment)] - E(reactant_full)
```

For the bundled dehydration step, `E(H₂O)` at M06-2X/def2-TZVP = −76.42608277 a.u.

### 6. Multi-step mechanisms: segmented NEBs

If the reactant→product conversion is stepwise through an intermediate, run a
**segmented NEB** for each step (reactant→intermediate, intermediate→product)
and take the **max barrier as rate-limiting**. Use `--no-climb` for each
segment and refine the max image separately (step 3). The bundled Maillard
reaction is stepwise: reactant → TS1 → carbinolamine → TS2 → imine + H₂O.

## Key flags (`neb_ts.neb_run`)

| flag | meaning |
|---|---|
| `--method tblite\|uma` | PES backend (default `tblite` = GFN2-xTB) |
| `--interp geodesic\|idpp\|linear` | initial-path interpolation (default `idpp`; use `geodesic`) |
| `--no-climb` | climb-off only — use for dissociative / bimolecular steps |
| `--no-map` | skip RDKit atom mapping (when both inputs share atom order, or when mapping fails on bond-breaking steps) |
| `--init-path FILE` | resume from a saved full band (`neb_final_path.xyz`); skips endpoint relaxation + interpolation |
| `--images N` | interior NEB images (default 15; 18 recommended) |
| `--fmax F` | convergence threshold eV/Å (default 0.05; use 0.02) |
| `--max-steps N` | max optimizer steps per phase (default 400) |
| `--refine-dimer` | refine the climbing image with the Dimer method (can run away on loose PESs) |
| `--outdir DIR` | output directory (default `output`) |

## Scripts

| module | purpose |
|---|---|
| `neb_ts.neb_run` | NEB (GFN2 or UMA): extract → relax endpoints → interpolate → NEB → (optional CI) → write max image |
| `neb_ts.gaussian` | `read_gaussian_log(path)` — extract the final optimized structure from a Gaussian 16 `.log` archive |
| `neb_ts.frequencies` | GFN2-xTB finite-difference vibrational analysis (counts imaginary modes) |
| `neb_ts.frequencies_uma` | UMA vibrational analysis |
| `neb_ts.refine_ts_fixedmode` | GFN2 fixed-mode TS refinement (proven-safe; seeded by NEB tangent) |
| `neb_ts.refine_fixedmode_uma` | UMA fixed-mode TS refinement |
| `neb_ts.refine_dimer_uma` | UMA Dimer TS refinement (can run away on a flat ridge) |
| `map_atoms.py` | standalone RDKit atom-mapping / alignment utility (used internally by `neb_run`) |

## Outputs (in `--outdir`)

| file | description |
|---|---|
| `reactant_mapped.xyz` / `product_mapped.xyz` | endpoints, atom-mapped & aligned (unless `--no-map`) |
| `ts_structure.xyz` | the NEB maximum-energy image — **open in PyMOL** (refine to a saddle with fixed-mode if needed) |
| `neb_final_path.xyz` | full NEB path, multi-model XYZ (load in PyMOL, step through states to animate the reaction) |
| `geodesic_path.xyz` / `idpp_path.xyz` | initial interpolated path |
| `energy_profile.dat` | per-image energies + forward/reverse barriers (eV and kcal/mol) |
| `neb.log` | text log of the run |

## Viewing the TS in PyMOL

```bash
pymol output/ts_structure.xyz        # the TS / max image
pymol output/neb_final_path.xyz      # whole path: one state per image, animate with mplay
```

## Pipeline summary

1. **Extract** the final optimized structures from the Gaussian 16 `.log`
   archive entries (`neb_ts.gaussian.read_gaussian_log`), or read `.xyz` directly.
2. **Atom-map** the product onto the reactant (RDKit) — unless `--no-map`.
3. **Re-optimize** both endpoints at the chosen level (GFN2-xTB or UMA).
4. **Initial path** via **geodesic** interpolation (best) or IDPP.
5. **NEB** relaxation with **climb off** (`--no-climb`) — robust on
   dissociative / bimolecular PESs.
6. **Verify** the max image with a vibrational analysis. If it is a saddle
   (1 imaginary), done. If not (flat ridge), **refine** with fixed-mode and
   re-verify; if still no saddle, **pivot** to UMA or a DFT saddle search.
7. **Cross-check** any independent DFT TS by extracting it and running a
   single-point + frequencies on the NEB PES, and compare barriers (add a free
   fragment energy if the DFT used a reduced atom model).
8. For **multi-step** mechanisms, run **segmented NEBs** per step and take the
   max barrier as rate-limiting.

See `RESULTS.md` for the two bundled Maillard steps (C–N formation and the
dehydration) worked through this workflow, including the flat-ridge cases and
the DFT cross-check.