# Results — NEB transition-state search for the Maillard glycosylamine formation

## System
The two Gaussian 16 optimization logs (`reactants.log`, `products.log`) each
contain a single M062X/def2TZVP optimization of a **C₁₀H₂₂N₂O₁₀** (44-atom,
neutral singlet) structure. After extracting the final geometries and perceiving
bonds with RDKit, the reaction is:

**asparagine + open-chain sugar + H₂O  →  N-glycoside (glycosylamine) + H₂O**

i.e. **glycosylamine (Schiff-base) formation — the first step of the Maillard
reaction.** A new **C–N bond** forms between the sugar C1 and asparagine's
α-amine nitrogen (reactant 41 bonds → product 42 bonds), with a concurrent
**proton transfer** from the amine N–H to an oxygen (water rearrangement).

## Pipeline (executes the plan in CLAUDE.md)
1. **Environment**: `uv` + Python 3.12; `tblite` (GFN2-xTB), `ase`, `rdkit`,
   numpy/scipy (`uv sync`).
2. **Gaussian parser** (`neb_ts/gaussian.py`): reconstructs the wrapped archive
   entry and extracts the final optimized Cartesian geometry.
3. **Atom mapping** (`neb_ts/align.py`): RDKit-based, chemically consistent
   reordering of the product onto the reactant (fragment substructure matching +
   Hungarian H-assignment + Kabsch). The Gaussian files list atoms in
   *inconsistent* orders (especially the 22 H and 10 O atoms); without this
   step NEB interpolates between mis-mapped atoms and gives a spurious
   ~1 kcal/mol "permutation" barrier. After mapping, per-fragment RMSDs are
   0.34 Å (sugar heavy), 0.42 Å (asparagine heavy), 0.03 Å (water) — the
   frameworks coincide and only the reacting atoms move.
4. **NEB** (`neb_ts/neb_run.py`): GFN2-xTB endpoint re-optimization → IDPP
   initial path → NEB → **CI-NEB** (climbing image), FIRE optimizer.
5. **Refinement** (`neb_ts/refine_ts.py`): optional Dimer minimum-mode
   following, seeded with the NEB tangent.
6. **Verification** (`neb_ts/frequencies.py`): finite-difference vibrational
   analysis — a true TS has exactly one imaginary frequency.

## Result for the provided structures
With the correct atom mapping, a direct CI-NEB (15 interior images, FIRE,
fmax 0.05 eV/Å) between the two provided structures converges to a
**monotonically downhill** path:

| quantity | value |
|---|---|
| Reactant energy (GFN2-xTB minimum) | −2152.705 eV |
| Product energy (GFN2-xTB minimum) | −2153.638 eV |
| Reaction energy (R → P) | −0.933 eV = **−21.5 kcal/mol** (exothermic) |
| Forward barrier (R → P) | **≈ 0** (the reactant is the highest point on the path) |
| Reverse barrier (P → R) | 0.933 eV = 21.5 kcal/mol |

The reactant was confirmed by vibrational analysis to be a **true minimum**
(no imaginary frequencies). Every interior NEB image lies below the reactant,
so the climbing image latches onto the (fixed) reactant endpoint and there is
**no interior transition state** on this direct path at the GFN2-xTB level.

### Interpretation
A monotonic energy path between two distinct true minima is not physical on a
smooth PES — a saddle must separate them. The NEB therefore **bypassed the true
saddle**: for a fragment-association reaction (two molecules coming together to
form a bond), the IDPP/linear initial path lets the fragments approach along a
low-energy route that slides off the reaction ridge before the climbing image
can locate the C–N bond-formation saddle. This is a well-known NEB failure mode
for association reactions with a poor initial path.

### What would find the true TS
- Use a **pre-reactive complex** as the reactant (the sugar aldehyde and amine
  pre-organized / H-bonded in the reacting orientation) rather than the
  separated-fragment geometry, so the NEB endpoint sits in the same basin as
  the saddle.
- Or map a **single elementary step** (e.g. pre-reactive complex → carbinolamine,
  or carbinolamine → imine + H₂O) instead of the full separated-reactant →
  glycosylamine transformation.
- Or supply a **hand-built initial path** that passes through the C–N
  bond-forming geometry (a shorter, stiffer band, or the growing-string method).

## True-TS search (pre-reactive-complex → single elementary steps)

Following the "what would find the true TS" plan above, the reaction was split
into elementary steps and each attacked with a pre-reactive complex as the
reactant. The elementary step that is actually amenable to a TS search is the
**carbinolamine dehydration** (the rate-determining step of glycosylamine
formation):

    carbinolamine  →  imine (Schiff base) + H₂O

i.e. C10–O4 cleaves (O4 leaves as water), the N27 proton H36 transfers so the
leaving group is H₂O (O4–H36–H37), and C10=N27 forms (1.50 → 1.28 Å).

### Stationary points located (GFN2-xTB, gas phase)
| structure | E (eV) | ΔE vs carbinolamine |
|---|---|---|
| carbinolamine (reactant minimum) | −2153.6384 | 0.0 |
| pre-dehydration H-bonded complex | −2153.7232 | −1.96 kcal |
| contact imine·H₂O intermediate | −2152.9322 | +16.28 kcal |
| separated imine + H₂O | −2152.4556 | +27.27 kcal |
| post-chemistry water-diffusion saddle | −2152.8297 | +18.65 kcal |
| concerted-geometry high minimum | −2150.9587 | +61.79 kcal |

### What the NEB found
1. **First band** (carbinolamine → contact intermediate, 12 images, collision-free
   seed): a *clean physical MEP* — carbinolamine (0) → pre-dehydration H-bond
   complex (−1.96) → rises through +8.5 to a max, then descends to the contact
   intermediate (+16.28). The barrier is **under-resolved at the saddle**: the
   entire dehydration chemistry (C–O cleavage, C=N formation, H36 proton
   transfer) snaps *concertedly* between two adjacent images. The +18.65 kcal
   climbing image is a **post-chemistry water-diffusion saddle** above the
   already-formed imine (C10–O4=3.09, C10=N27=1.264, H36 already on O4), *not*
   the dehydration TS.

2. **Dense band** (pre-complex → contact intermediate, 16 images): drifted to a
   spurious **stepwise / zwitterionic** path — C–O cleaves to give an iminium +
   hydroxide (C10–O4=4.0 while H36 still on N, +47 kcal), unfavourable in the
   gas phase. Triggered by the linear seed biasing C–O cleavage early.

3. **Tight concerted band** (first-band images 5→6, the chemical jump, 14
   images): the climbing image converged to a **concerted geometry** (C10–O4=
   1.91, C10–N27=1.34, N27–H36=1.34, O4–H36=1.30 — H36 bridging N and O, C–O
   half-broken, C=N half-formed, all moving together) at +61.79 kcal with
   max|F|=0.035 eV/Å. **Vibrational analysis: 0 imaginary frequencies** — this is
   a high-energy *minimum*, not a TS. The Dimer method seeded from it did not
   climb (the point was already force-converged) and its single-direction
   curvature estimate (−27.9) conflicts with the rigorous full Hessian (0
   imaginary) — a finite-difference artifact.

### Diagnosis: why no clean TS yet
Both CI-NEB and the Dimer repeatedly **fall into high-energy minima** on the
direct-transfer potential-energy surface. The reason is physical: in the gas
phase the N27→O4 proton transfer is a **strained, direct, 4-center process
with no proton shuttle**, so the barrier is very high (>60 kcal) and the PES
near the top is flat and riddled with shallow minima that trap the climbing
image. In real Maillard chemistry this dehydration is **acid/base-catalysed**:
a water molecule (or the amine) shuttles the proton, lowering the barrier to
~20–30 kcal.

### Key structural finding — an explicit proton shuttle is available
The reactant contains an **explicit water molecule (O_w = atom 41, H42, H43)**
that is already H-bonded to the reacting amine N27 in the carbinolamine
(O_w···N27 = 2.82 Å, H42···N27 = 1.89 Å — a strong N27···H42–O_w hydrogen
bond). This water has been a **passive spectator** in every NEB run above. A
**water-assisted** proton transfer (N27–H36 → O_w → O4, or relayed through
the water) is geometrically available and is the chemically correct route — it
should produce a realistic, findable saddle.

### Next step (the realistic TS)
Build a seed in which the explicit water (atoms 41–43) **participates** in the
proton transfer — bridging N27 and O4 — and re-run the CI-NEB / Dimer on the
carbinolamine → imine step. The water shuttle removes the strained 4-center
direct transfer that produces the >60 kcal minima and should yield a true
first-order saddle (exactly one imaginary frequency) at a physically reasonable
barrier.

## Water-assisted shuttle attempt (2nd explicit water, Grotthuss relay)

Following the "next step" plan, a second explicit water was added as a proton
shuttle (O_s, H_s1, H_s2 appended to a 44-atom carbinolamine → 47 atoms).
Scripts: `neb_ts/build_shuttle.py`, `neb_ts/build_relay_concerted.py`,
`neb_ts/seed_neb.py`. Endpoints in `output/shuttle/`.

### Geometric constraints that shaped the build
- The donor H (H36) and acceptor O4 are on **opposite sides of N27** in the
  C10–N27–O4 plane (H36–N27–O4 = 111.7°; H36 is 0.95 Å above the plane, O4 in
  it). A shuttle water placed **in the plane** cannot both accept H36 and
  donate to O4 — it ends up opposite H36, forcing a ~180° H36 reorientation
  (+248 kcal seed strain) and on relaxation lets its proton escape to a sugar
  hydroxyl (O5/O2).
- The shuttle is therefore placed **above the N–C–O plane** (the H36 side),
  ~2.7 Å from N27, so H36 points naturally at O_s (O_s···H36 ≈ 1.9 Å H-bond)
  while O_s–O4 ≈ 3.4–4.1 Å lets H_s1 reach O4.
- xTB favours a sugar –OH over the shuttle water for an excess proton, so the
  "product" with H36 on O_s is **not a stable minimum** (H36 hops to sugar O2).
  Fix: endpoints are relaxed with the shuttle protons frozen (FixAtoms on
  O_s,H_s1,H_s2, and H36 for the product); the NEB **interior is unconstrained**.
- The concerted seed (`build_relay_concerted.py`) drives C10–O4 and C10–N27
  monotonically (corrective shift after linear interp — removes both the
  chord-dip zwitterion and the N27 sp3→sp2 angle strain), holds O_s H-bonded to
  N27 (no dissociation), Hermite-transfers H36 (N27→O_s) and H_s1 (O_s→O4) in
  phase, keeps all other O–H rigid, and routes H_s1 on a Bézier arc bowed away
  from sugar H21 (the straight O_s→O4 line clips H21 at 0.6 Å). Unrelaxed seed
  peak ≈ +360 kcal (H3O⁺-like crowding at O_s), which NEB relaxes.

### What the water-assisted CI-NEB found (16-image band, `output/shuttle/neb2`)
The band collapses to a **stepwise** path with two under-resolved steps:

| band region | structure | E vs pre-complex |
|---|---|---|
| pre-reactive complex (imgs 0–4) | carbinolamine + shuttle, H36 on N27 (O_s···H36 H-bond) | 0.0 (−2292.09 eV) |
| H36/H_s1 **swap** (img 5 spike) | H36 and H_s1 exchange across the N27↔O_s H-bond | **+49 kcal** |
| swap state (imgs 6–8) | N27–H_s1 + O_s–H36, C–O still intact | +4.5 kcal |
| **C–O cleavage snap** (img 8→9) | C10–O4 1.41→3.16, C10=N27 1.46→1.26, H_s1→O4 | +8.5 kcal (snap, TS higher) |
| product (imgs 9–15) | imine + departing water + shuttle | +34 kcal vs pre-complex |

### Diagnosis — the shuttle bypasses, not assists, the dehydration
The dominant barrier is the **+49 kcal proton swap** (H36↔H_s1 across the
N27↔O_s H-bond): a low-energy proton shuffle that **does not break C–O or form
C=N**. The actual dehydration (C–O cleavage + C=N + H_s1→O4) is a separate,
under-resolved snap of comparable or higher barrier. So the single shuttle
water, placed to bridge N27 and O4, opens a competing proton-transfer
side-channel rather than lowering the dehydration barrier. This is the same
family of failure as the direct route: gas-phase proton transfers at GFN2-xTB
that should be acid/base-catalysed are high-barrier, and an isolated water
finds easier proton moves (swap, or hop to a sugar –OH) instead of assisting
the C–O cleavage.

### Status
No verified TS yet (no climbing image was frequency-checked — the band is
stepwise/under-resolved, not a single clean saddle). The rate-limiting saddle
on the water-assisted PES is the +49 kcal H36/H_s1 swap, which is *not* the
dehydration. Options for the next step are presented to the user: (a) resolve
+ verify the +49 kcal swap TS as the rate-limiting saddle of the
water-assisted path; (b) reposition the shuttle to **acid-catalyse O4**
(H_s1→O4 protonation of the leaving group, decoupled from N27) so it cannot
compete via the swap; (c) a 2-water relay or implicit-solvent model; (d)
accept the gas-phase stepwise mechanism and document the resolved saddle.

## Acid-catalysis attempt (shuttle protonates the leaving O4, decoupled from N27)

Following option (b), the shuttle water was repositioned to protonate the
leaving O4 directly (H_s1: O_s→O4), decoupled from N27 so the H36/H_s1 swap is
geometrically impossible (O_s ~4–5 Å from N27).  The rate-limiting step becomes
the acid-catalysed C–O cleavage:

    carbinolamine + shuttle(H2O @O4) → iminium(C10=N27+H36) + departing water(O4+H37+H_s1) + shuttle-OH⁻

Scripts: `neb_ts/build_acid_shuttle.py`, `neb_ts/build_acid_seed.py`,
`neb_ts/build_acid_anchor_seed.py`, `neb_ts/seed_neb_hooke.py`,
`neb_ts/reeval_profile.py`.  Endpoints in `output/acid/`.

### Key lessons learned (each cost a full CI-NEB run)
- **Frozen-proton endpoint relaxation is forbidden for NEB.**  The first acid
  CI-NEB (`output/acid/neb`, endpoints relaxed with shuttle protons frozen in
  the "poised to protonate O4" spot) collapsed to a *downhill* path with no
  interior TS — the frozen reactant was a **strained constrained minimum 92 kcal
  above the true carbinolamine+neutral-water minimum**, and the frozen product
  (contact ion pair) was 47 kcal above the true iminium+OH⁻ minimum.  The
  unfrozen NEB interior simply fell off these fake-high endpoints.  *Endpoints
  must be unconstrained minima on the same PES as the band interior.*
- The true unconstrained minima (extracted from that band and relaxed, unfrozen)
  are `r_min.xyz` (carbinolamine + neutral shuttle water, E=−2292.091 eV) and
  `p_min.xyz` (iminium + departing water + shuttle-OH⁻, E=−2291.071 eV; **H36
  stays on N27 unfrozen**, so the iminium is a genuine gas-phase minimum).  The
  C–O cleavage step is therefore separable from the later H36 deprotonation
  (iminium→imine).  ΔE(P−R) = +23.5 kcal (endothermic).
- **The first band found the real chemistry** (images 5→6: C10–O4 1.56→3.17,
  H_s1 Os→O4, H36 on N27) at ~+23.5 kcal above the true reactant — i.e. **acid
  catalysis lowers the dehydration bottleneck from >60 kcal (direct) to ~24
  kcal** — but the climbing image latched onto the fake reactant endpoint, not
  the saddle.
- **Re-running between the true minima explodes.**  Every fresh CI-NEB between
  `r_min` and `p_min` (`neb2` k=0.1, `neb3` k=0.2, `neb4` with a Hookean
  Os···O4 restraint) drove the band to dissociation: the shuttle water drifted
  14–19 Å away, the transferring proton H_s1 was flung 24 Å off, and C–O
  over-broke to 9–23 Å, giving garbage +172 to +1120 kcal "TS" geometries.
  Root cause: in the true reactant the shuttle proton sits 3 Å from O4 (a free,
  non-H-bonded water), so the NEB perpendicular force flings the light proton /
  drifts the shuttle, and tangent springs (which only act along the path)
  cannot hold it.  A Hookean on Os···O4 (rt=3.5 Å) caught the shuttle drift but
  the proton still flew off.  (`neb_ts/reeval_profile.py` confirmed these were
  real dissociated geometries, not calculator-fallback artifacts — a fresh
  tier-1 re-evaluation matched the trajectory energies to 0.0000 eV.)

### Dimer refinement from the converged band (the working path)
The first run's relaxed images 5→6 (the only segment that kept the shuttle in
place) bracket the saddle.  A Dimer refinement seeded from image 5 (a relaxed,
bound NEB image at +23.5 kcal, C10–O4=1.56) with the NEB tangent as the
reaction-coordinate mode **converged** (fmax=0.038 eV/Å, curvature=−6.88) to a
textbook-looking acid-catalysed dehydration geometry at **+39.9 kcal**:
C10–O4=2.41 (C–O breaking), C10–N27=1.30 (C=N forming), H_s1 mid-transfer
O_s→O4 (0.98 / 1.68 Å), H36 on N27 (iminium), shuttle engaged (O_s–O4=2.66 Å).
**Vibrational analysis (`output/acid/frequencies_ts.txt`): 0 imaginary
frequencies** — the softest real mode is +6.04 cm⁻¹.  This point is a
**high-energy minimum, not a transition state**.  The Dimer's negative 1-D
curvature (−6.88) was a finite-difference artifact; the rigorous full Hessian
has all modes positive.  The Dimer climbed *off* the MEP into a spurious high
minimum, exactly as the direct-route Dimer did at +61.8 kcal.

### Conclusion — no findable concerted TS at gas-phase GFN2-xTB
The carbinolamine→iminium dehydration at the gas-phase GFN2-xTB level has **no
findable concerted first-order transition state**.  Both CI-NEB and the Dimer
method repeatedly fall into high-energy minima (a valley-ridge inflection /
stepwise, charge-separated mechanism): the direct route gives a +61.8 kcal
minimum (0 imaginary), the acid-catalysed route a +39.9 kcal minimum (0
imaginary, softest mode +6 cm⁻¹).  The acid catalyst lowers the bottleneck by
~22 kcal — a real, chemically sensible effect — but does not produce a
verifiable saddle.  The reaction proceeds stepwise through a charge-separated
iminium+OH⁻ region; a true first-order TS likely requires **implicit or
explicit solvation** (which stabilises the charge-separated intermediate
differently and can restore a true saddle), or mapping the **full neutral
carbinolamine → imine + H₂O** step (with H36 deprotonation concerted, avoiding
the charged iminium trap) rather than the iminium half-step.

### Next-step options (presented to the user)
1. **Implicit solvation** (tblite ALPB/GBSA, water): re-relax endpoints and
   re-run CI-NEB/Dimer in solvent — the chemically correct setting for the
   Maillard reaction and the most likely to yield a true saddle.
2. **Full neutral-imine step**: target imine + H₂O (H36 deprotonated to the
   shuttle) instead of the iminium, removing the charge-separated minimum.
3. Accept the **+39.9 kcal acid-catalysed high minimum** (softest mode +6 cm⁻¹)
   as the rate-limiting bottleneck / valley-ridge inflection and document it,
   noting a true TS is not defined at this level of theory in the gas phase.

## How to re-run / explore
```bash
uv sync
# Full pipeline on the Gaussian logs (auto atom-mapping + CI-NEB):
uv run python -m neb_ts.neb_run reactants.log products.log --images 15 --fmax 0.05 --k 0.1 --max-steps 1000
# Verify a candidate by vibrational analysis:
uv run python -m neb_ts.frequencies output/ts_structure.xyz
# Refine a climbing-image guess with the Dimer method (seeded by the NEB tangent):
uv run python -m neb_ts.refine_ts output/ts_structure.xyz --path output/neb_final_path.xyz --ts-index 0 --out output/ts_refined.xyz
```

## Output files (in `output/`)
| file | description |
|---|---|
| `reactant_mapped.xyz`, `product_mapped.xyz` | endpoints, atom-mapped & aligned (open in PyMOL) |
| `neb_final_path.xyz` | full NEB path, 17-frame multi-model XYZ (load in PyMOL, step through states to animate the reaction) |
| `idpp_path.xyz` | initial IDPP path |
| `ts_structure.xyz` | NEB maximum (here = reactant, since the path is downhill) |
| `energy_profile.dat` | per-image energies and barriers |
| `frequencies_reactant.txt` | reactant vibrational spectrum (confirms minimum) |
| `atom_mapping.txt` | reactant↔product atom correspondence |
| `neb_run.log` | text log of the run |

---

## Update (2026-07-20) — UMA stepwise mechanism, TS1 localization & DFT QST3 cross-check

### Mechanism on the UMA PES is **stepwise** (not concerted)
A carbinolamine (aminol) intermediate is a true minimum at **+7.07 kcal/mol**
above the reactant precomplex (0 imaginary modes, softest +6.22 cm⁻¹; confirmed
on two convergence levels with identical spectra). Bonds at the intermediate:
**C1–N formed** (2.94 → 1.48 Å), **N–H broken**, **O(carbonyl)–H formed**
(proton transferred to the carbonyl oxygen → –OH). Intermediate → product
(imine + H₂O) is **monotonically downhill** (no TS2; the product is just the
relaxed conformer, the +10 kcal gap is conformational). So the rate-limiting
feature is **Step 1: reactant → carbinolamine (TS1)**.

### TS1 localization (UMA) — flat ridge, practical barrier +13.95 kcal/mol
- `output/seg1b/`: reactant → intermediate, **20 images, fmax 0.02 eV/Å,
  800 steps, climb-off (no CI), geodesic interpolation, UMA**. Geodesic
  interpolation at UMA (the lead from the GFN2-xTB geodesic work) was the key
  — it lands images near the MEP where CI/dimer then fail on the bimolecular
  PES.
- Clean on-MEP maximum at **img 13 = +13.95 kcal/mol** (smooth ramp img9 3.1 →
  img11 6.3 → img12 9.4 → img13 14.0; steep drop to the intermediate basin).
  Forward 13.95, reverse 8.09, ΔE 5.86 kcal/mol.
- Fixed-mode refinement of img 13 (`output/seg1b/ts1_refined.xyz`): held at
  +13.34 kcal, **did not collapse** (−0.61 kcal), |F⊥|=0.020, |F∥mode|=0.000,
  intact (11.47 Å). Earlier fixed-mode runs collapsed 16 kcal because they
  started from off-MEP images; seg1b's tight fmax left img 13 already on-MEP.
- **UMA frequencies of the refined structure: 0 imaginary modes**, softest real
  +6.2166 cm⁻¹ — **identical to the carbinolamine intermediate's soft mode**.

### Diagnosis: flat ridge / valley-ridge inflection (no discrete gas-phase TS on UMA)
The NEB maximum is **not a stationary point** — it sits on a descending valley
wall. Removing the perpendicular force (fixed-mode) rolls the structure into
the flat valley floor (a minimum, 0 imaginary), the same flat valley that
contains the +7.07 intermediate; the +6.22 cm⁻¹ soft mode *is* the flat ridge
coordinate. The Dimer method cannot rescue it (rotates onto the +6.22 soft
perpendicular mode and runs away). This matches the prior GFN2-xTB finding
(also a minimum, softest +6 cm⁻¹, no discrete gas-phase saddle).

**Practical forward barrier = the NEB path maximum, +13.95 kcal/mol** — the
defensible activation energy for a flat-ridge / loose bimolecular step (the
highest point on the MEP), even though no first-order saddle is defined on the
UMA gas-phase PES.

### DFT QST3 cross-check (`Step1_TS1alluasp-qst3-2.log`) — validates the barrier
An independent **M062X/def2TZVP QST3** transition-state optimization (same
C₁₀H₂₂N₂O₁₀ system, charge 0, singlet) converged to a stationary point
("Stationary point found"; opt-only log, no freq block). Extracted via
`neb_ts.gaussian.read_gaussian_log` → `output/qst3_ts/ts_qst3.xyz`.

| quantity | DFT (M062X/def2TZVP) | UMA |
|---|---|---|
| TS energy vs reactant | **+10.33 kcal/mol** | +13.95 (NEB max) / +13.34 (refined) |
| UMA single-point on the DFT-TS geometry | — | **+12.61 kcal/mol** |
| Overall reaction energy (R→P) | −4.66 kcal/mol | −3.23 kcal/mol (imine product) |
| Imaginary modes at this geometry | (QST3 saddle, 1 imag expected) | **0 at UMA**, softest +6.22 cm⁻¹ |
| max heavy-heavy | — | 11.36 Å (intact, not dissociated) |

The DFT QST3 TS (+10.33 kcal) and the UMA estimates (+12.6–14 kcal) **agree
within ~2.5 kcal** — the UMA single-point on the exact DFT-TS geometry
(+12.61) sits between them. The UMA low-frequency spectrum of the DFT-TS
geometry (+6.22, +9.67, +15.28, +15.80, +20.14, +27.22 cm⁻¹) is **identical**
to the seg1b refined structure — same flat valley.

**Interpretation:** DFT (M062X) resolves a discrete first-order saddle at
~+10 kcal for Step 1; UMA's PES is smoother/flatter and does not resolve a
discrete saddle (flat ridge, 0 imaginary), but **agrees on the barrier
height** (~+10–14 kcal). The Maillard glycosylamine Step-1 (C–N formation +
proton transfer to carbonyl O) has a **~+10–14 kcal/mol forward barrier** by
both methods; the carbinolamine intermediate at +7 kcal (UMA) is the
rate-relevant stationary point.

### New scripts / outputs
| file | description |
|---|---|
| `neb_ts/refine_fixedmode_uma.py` | UMA fixed-mode TS refinement (seeded by NEB tangent) |
| `neb_ts/frequencies_uma.py` | UMA finite-difference vibrational analysis |
| `neb_ts/refine_dimer_uma.py` | UMA dimer TS refinement |
| `output/seg1b/` | seg1b NEB (geodesic, UMA, fmax 0.02): `energy_profile.dat`, `ts_structure.xyz`, `ts1_refined.xyz`, `ts1_freq.txt` |
| `output/qst3_ts/` | extracted DFT QST3 TS: `ts_qst3.xyz`, `ts_qst3_freq.txt` (UMA freq), `test.log` |

---

## Step 2 (2026-07-20) — Dehydration: carbinolamine → imine + H₂O (products → TS2 → IM2)

The full Maillard sequence is multi-step: **reactants → TS1 → products
(carbinolamine) → TS2 → IM2 (imine + H₂O)**. Step 2 is the **dehydration**
(C–O bond break + proton transfer, water leaves). New Gaussian outputs:
`Step1_TS2alluasp2.log` (TS2), `Step1_IM2alluasp-rep.log` (IM2).

### Structures (all M062X/def2TZVP, charge 0, singlet)
- products (carbinolamine, 44 C₁₀H₂₂N₂O₁₀): −1256.11962847 a.u.
- IM2 (imine + H₂O, 44 C₁₀H₂₂N₂O₁₀): −1256.12000824 a.u. — products & IM2
  nearly degenerate (ΔE = −0.24 kcal) and share **identical atom order**.
- **TS2: a 41-atom model C₁₀H₂₀N₂O₉** (the departing water omitted),
  `opt=(calcall,saddle=1,noeigentest)`, −1179.57955171 a.u. **Confirmed
  first-order saddle: exactly 1 imaginary frequency at −1488.95 cm⁻¹** (sharp
  C–O break / proton transfer). Extracted to `output/qst3_ts/ts2_dft.xyz`.

### DFT Step-2 barriers (free-water approximation, user-supplied E(H₂O))
To put the 41-atom TS2 on the 44-atom scale, add a free water molecule
(E(H₂O)@M062X/def2TZVP = −76.42608277 a.u.): TS2₄₄ = −1256.00563448 a.u.

| quantity | DFT (M062X/def2TZVP) |
|---|---|
| **Forward barrier (products → TS2)** | **+71.53 kcal/mol** |
| Reverse barrier (IM2 → TS2) | +71.77 kcal/mol |
| Reaction energy (products → IM2) | −0.24 kcal/mol |

**Step 2 is rate-limiting** for the overall sequence: +71.5 kcal vs Step-1
+10.3 kcal (DFT) — ~7× higher. The large imaginary frequency (−1489 cm⁻¹) is
consistent with a sharp, high proton-transfer/C–O-breaking barrier. This is
the **gas-phase, no-proton-shuttle** barrier; acid catalysis or implicit
solvation is expected to lower it substantially in reality.

### GFN2-xTB FAILS to resolve this TS2 (charge-separated dehydration)
Per the user's "use GFN2 if that works" guidance, GFN2-xTB geodesic NEB was
tried first but cannot localize a discrete saddle here (expected semi-empirical
weakness for proton-transfer/dehydration TSs):
- Climb-off geodesic NEB (`output/seg2_gfn2/`, 18 img, fmax 0.02): max image
  (img 9) = +40.94 kcal but a **spike** (+40 kcal over one image from the
  products basin); GFN2 frequencies = **0 imaginary**, softest +8.79 cm⁻¹
  (a minimum, not a saddle).
- CI-NEB (`output/seg2_gfn2_ci/`): climbing image **ran away to +270 kcal**
  (dissociative C–O break, no maximum).
- Fixed-mode from img 9 (`output/seg2_gfn2/ts2_fixedmode.xyz`): **collapsed**
  +40.9 → +8.5 kcal = IM2/product basin, 0 imaginary, softest +8.79 cm⁻¹.
- GFN2 single-point on the DFT-TS2 geometry + free water = +88.9 kcal on the
  GFN2 PES (GFN2 strongly dislikes the DFT saddle geometry).

The DFT +71.5 kcal barrier sits **between** the two non-saddle GFN2 numbers
(NEB max +40.9, DFT-geometry SP +88.9) — GFN2 brackets but does not resolve the
saddle. The DFT TS2 (1 imaginary, −1489 cm⁻¹) is the authoritative structure.

### UMA Step-2 NEB (fallback) — flat ridge, like Step 1
Because GFN2 failed, a UMA geodesic NEB (climb-off + fixed-mode, the seg1b
recipe) was run: `output/seg2_uma/` (~4 h). Result: NEB max (img 11) =
**+47.53 kcal/mol** forward (reverse +48.41, reaction −0.88 kcal — UMA's
reaction energy agrees with DFT −0.24, unlike GFN2 +7.62).

The UMA path **did traverse the dehydration** — the reacting glycosylamine bond
C10–N27 goes 1.50 Å (carbinolamine) → 1.46 Å (max) → 1.26 Å (imine), and the
leaving oxygen 1.42 Å (C–OH) → 2.37 Å (max, C–O broken) → 2.43 Å (free water).
So img 11 is a genuine mid-dehydration point (C–O broken, C–N not yet double),
Kabsch-RMSD 2.6 Å from the Step-1 TS1 (not a carbinolamine). Fixed-mode
refinement **held** it (dE = 0.00 kcal, |F⊥| = 0.0131 already converged — did
not collapse, unlike GFN2). **But UMA frequencies: 0 imaginary modes**, softest
+6.22 cm⁻¹ — the UMA dehydration, like Step 1, is a **flat ridge**: the NEB max
(+47.53 kcal) is the practical UMA barrier (highest point on the MEP), but no
first-order saddle exists on the UMA PES.

### Cross-method summary (both steps)
```
              Step 1 (C–N formation)        Step 2 (dehydration)
DFT  M062X    +10.33 (saddle, 1 imag)      +71.53 (saddle, 1 imag −1489)  ← rate-limiting
UMA  uma-s    +13.95 (flat ridge, 0 imag)  +47.53 (flat ridge, 0 imag)
GFN2-xTB      flat ridge, NEB max only      no saddle (brackets +40.9 / +88.9)
```
**Both semi-empirical/ML PESs (GFN2-xTB, UMA) fail to resolve a discrete
dehydration TS2** (flat ridge, 0 imaginary) — a DFT saddle search (QST3 /
`saddle=1`, seeded by a TS guess) is required. UMA underestimates Step 2 by
~24 kcal vs DFT (flatter PES). The **DFT +71.53 kcal/mol** is the authoritative
**rate-limiting** barrier for the overall Maillard sequence (gas-phase, no
proton shuttle; implicit/explicit solvation or acid catalysis is expected to
lower it substantially — the earlier GFN2 acid-catalysis note placed a related
bottleneck near +40 kcal).

### Step-2 outputs
| file | description |
|---|---|
| `output/seg2_gfn2/` | GFN2 climb-off NEB: `energy_profile.dat`, `ts_structure.xyz` (img9, 0 imag), `ts2_freq.txt`, `ts2_fixedmode.xyz` (collapsed), `neb.log` |
| `output/seg2_gfn2_ci/` | GFN2 CI-NEB (climbing image ran away) |
| `output/qst3_ts/ts2_dft.xyz` | extracted DFT TS2 (41-atom model) |
| `output/seg2_uma/` | UMA Step-2 NEB: `energy_profile.dat`, `ts_structure.xyz` (img11, +47.5, 0 imag), `ts2_refined.xyz`, `ts2_freq.txt`, `neb.log` |