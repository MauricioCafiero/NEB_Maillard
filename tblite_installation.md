# How `tblite` was installed in this project

## Summary

`tblite` (the Python bindings for the C library `libtblite`, providing the
GFN2-xTB / GFN1-xTB / GFN-FF semi-empirical methods) was installed as a
**prebuilt wheel from PyPI**, managed by **uv** — not pip, not conda, and
not built from source on this machine.

## Declaration

In `pyproject.toml` it is a project dependency:

```toml
dependencies = [
    "ase>=3.23",
    "tblite>=0.4",
    "rdkit>=2024.3",
    ...
]
```

`uv` resolves this through `uv.lock`, which pins it to **`tblite 0.7.0`**.
The installer field in the installed `.dist-info` reads `uv`.

## Source: prebuilt wheel from PyPI

`uv.lock` records the source as:

```
source = { registry = "https://pypi.org/simple" }
```

So `uv` pulled a ready-made macOS arm64 wheel. No local compilation, and no
Meson / Ninja / Fortran compiler is required on this machine.

## Self-contained (delocated) wheel

The wheel's `WHEEL` metadata lists two generators:

- `meson` — how the upstream project builds `libtblite` and its bindings.
- `delocate 0.13.0` — the macOS tool that rewrites and bundles dylib
  references so the wheel ships every dynamic library it needs.

That is why `tblite` works here without a Homebrew `libtblite` install. The
bundled native libraries, under
`.venv/lib/python3.12/site-packages/tblite/`, are:

```
tblite/_libtblite.cpython-312-darwin.so   # CPython C extension (Python -> C bridge)
tblite/.dylibs/libopenblasp-r0.3.33.dylib  # linear algebra (OpenBLAS)
tblite/.dylibs/libomp.dylib               # OpenMP runtime
tblite/.dylibs/libgfortran.5.dylib        # Fortran runtime
tblite/.dylibs/libquadmath.0.dylib
tblite/.dylibs/libgcc_s.1.1.dylib
tblite/.dylibs/libgomp.1.dylib
```

## What it is

`tblite-python` (the `awvwgk` Python bindings) wrapping the C library
`libtblite`. The ASE interface used in this project
(`from tblite.ase import TBLite`) lives in this same wheel.

## Reproducing the install

From the project root, with `uv` available:

```sh
uv sync          # installs everything in pyproject.toml, incl. tblite, into .venv
```

To inspect the installed package:

```sh
.venv/bin/python -c "from tblite.ase import TBLite; print(TBLite)"
ls .venv/lib/python3.12/site-packages/tblite/.dylibs
```

## Note on the OpenMP conflict with the UMA backend

`tblite` ships its **own `libomp.dylib`** inside `.dylibs/`. When
`fairchem-core` (which pulls in `torch`, itself bundling another OpenMP
runtime) is loaded in the same process, you get the
"multiple copies of the OpenMP runtime" error:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

This is the macOS libomp clash, not a `tblite` bug. `tblite` alone never
needed any workaround. For the UMA backend, `neb_ts/neb_run.py` sets the
following environment variables at import time to make the two runtimes
coexist:

```python
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
```

## Caveat

`tblite` exposes **no `__version__` attribute** on the module —
`tblite.__version__` raises `AttributeError`. Its version lives only in the
`.dist-info` metadata and in `uv.lock`. The `0.7.0` figure above comes from
`uv.lock` / `dist-info`, not from the import.