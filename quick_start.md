# Quick start — Linux (optionally CUDA)

Clone-and-run recipe for the NEB transition-state search. The full
workflow (refinement, flat-ridge handling, DFT cross-check, multi-step
segmented NEBs) is in [`README.md`](README.md); worked Maillard examples
and barrier numbers are in [`RESULTS.md`](RESULTS.md).

## One-time environment

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/MauricioCafiero/NEB_Maillard.git
cd NEB_Maillard
uv sync                       # creates .venv; installs ase, tblite, rdkit, fairchem-core/torch (Python >= 3.12)
```

Notes:
- `tblite` (GFN2-xTB) ships as a self-contained Linux `manylinux` wheel —
  no system libomp or Homebrew needed. Its OpenMP runtime is bundled.
- The threading/OpenMP guards (`OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE`) are set in
  `neb_ts/neb_run.py` at import, so no shell setup is required.

## A. GFN2-xTB path (no token, runs anywhere)

`reactants.log` and `products.log` are the bundled Gaussian-16 endpoints.

```sh
uv run python -m neb_ts.neb_run reactants.log products.log \
  --method tblite --interp geodesic --no-climb --no-map \
  --fmax 0.02 --images 18 --max-steps 800 --outdir output/run_gfn2
# verify the TS (exactly 1 imaginary freq = first-order saddle):
uv run python -m neb_ts.frequencies output/run_gfn2/ts_structure.xyz \
  --out output/run_gfn2/ts_freq.txt
```

Add `--no-map` only when both inputs share atom ordering (check with the
snippet in `README.md` §0); drop it to let RDKit map the product onto the
reactant (bond-forming steps).

## B. UMA path (needs a HuggingFace token)

`uma-s-1p1` lives in the gated `facebook/UMA` repo. Accept the license on
<https://huggingface.co/facebook/UMA>, then create a token at
<https://huggingface.co/settings/tokens>.

```sh
export HF_TOKEN=<your token>
uv run python -m neb_ts.neb_run reactants.log products.log \
  --method uma --interp geodesic --no-climb --no-map \
  --uma-device cuda \          # or: --uma-device cpu  (default)
  --fmax 0.02 --images 18 --max-steps 800 --outdir output/run_uma
uv run python -m neb_ts.frequencies_uma output/run_uma/ts_structure.xyz \
  --out output/run_uma/ts_freq.txt
```

The first UMA run downloads ~150 MB of weights (cached thereafter).

## CUDA notes

- `uv sync` installs the **CPU** build of `torch` by default. To use the
  GPU, pull torch from the PyTorch CUDA index, matching the CUDA toolkit
  on the box:
  ```sh
  UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 uv sync
  ```
  Without this UMA still runs — on CPU, same as on the Mac.
- Every UMA entry point (`neb_ts.neb_run`, `neb_ts.frequencies_uma`,
  `neb_ts.refine_fixedmode_uma`, `neb_ts.refine_dimer_uma`) exposes
  `--uma-device`, so switching to CUDA needs no code change — just pass
  `--uma-device cuda`.

## Long runs

UMA NEBs take hours. On a laptop, hold off sleep (`caffeinate -dimsu &`
on macOS; on Linux use `systemd-inhibit` or run under `tmux`/`nohup`).
On a cluster, use your job scheduler.