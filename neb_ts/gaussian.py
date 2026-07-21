"""Extract the final optimized structure from a Gaussian 16 optimization log.

Gaussian writes the final geometry inside the archive entry (the "Punch" block)
that starts with ``1\\1\\GINC-...`` and ends with ``\\\\@``.  The geometry lives in
the atom fields that follow the ``charge,multiplicity`` field, e.g.::

    ...\\0,1\\O,-2.587...,x,y\\C,...\\...\\\\Version=...

This module reconstructs the wrapped archive line and parses those atom fields
into an :class:`ase.Atoms` object.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

from ase import Atoms

# Atomic symbols -> atomic numbers (ASE has these, but keep a local copy so the
# parser does not depend on the periodic table module layout).
from ase.data import atomic_numbers


_ARCHIVE_START_RE = re.compile(r"^\s*1\\1\\GINC-")
_ARCHIVE_END = "@"


def _reconstruct_archive(text: str) -> str:
    """Return the single logical archive line from a Gaussian log.

    Gaussian wraps the archive at 72 columns.  In the file each wrap is a new
    physical line with no continuation marker, so we collect every physical
    line from the one starting with ``1\\1\\GINC-`` up to (and including) the
    line that ends with ``@``, joining them without separators.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if _ARCHIVE_START_RE.match(line):
            start = i
    if start is None:
        raise ValueError("No Gaussian archive entry (1\\1\\GINC-) found in log.")

    end = start
    for j in range(start, len(lines)):
        if lines[j].rstrip().endswith(_ARCHIVE_END):
            end = j
            break
    else:
        raise ValueError("Archive entry has no terminating '@' line.")

    # Gaussian pads every wrapped archive line with a single leading space and
    # hard-wraps at a fixed column.  Strip leading/trailing whitespace from each
    # physical line, then concatenate with no separator to reconstruct the
    # single logical archive record.
    return "".join(lines[k].strip() for k in range(start, end + 1))


def _split_archive_fields(archive: str) -> list[str]:
    """Split the archive on single backslashes.

    The archive uses ``\\`` as the field separator and ``\\\\`` for an escaped
    backslash / the terminator.  Strip the trailing ``\\@`` then split on ``\\``.
    """
    # Remove the trailing terminator "\@" (and any trailing backslashes).
    if archive.endswith("@"):
        archive = archive[:-1]
    archive = archive.rstrip("\\")
    return archive.split("\\")


def parse_archive_atoms(archive: str) -> Tuple[list[str], int, int, list[Tuple[float, float, float]]]:
    """Parse a reconstructed archive line into atoms, charge, multiplicity.

    Returns
    -------
    symbols, charge, multiplicity, positions
    """
    fields = _split_archive_fields(archive)

    # Field layout of a Gaussian archive entry:
    #   0: "1"                         (archive version)
    #   1: "1"                         (energy program / spin handling)
    #   2: "GINC-<host>"
    #   3: job type ("FOpt", "SP", ...)
    #   4: method
    #   5: basis
    #   6: formula
    #   7: point group / label
    #   8: charge,multiplicity  e.g. "0,1"
    #   9..: atom fields "Symbol,x,y,z"
    charge = mult = None
    atom_start = None
    for i, field in enumerate(fields):
        if re.fullmatch(r"-?\d+,\d+", field.strip()):
            parts = field.strip().split(",")
            charge = int(parts[0])
            mult = int(parts[1])
            atom_start = i + 1
            break
    if atom_start is None:
        raise ValueError("Could not locate the 'charge,multiplicity' field.")

    symbols: list[str] = []
    positions: list[Tuple[float, float, float]] = []
    for field in fields[atom_start:]:
        # Stop at the first non-atom field (e.g. "Version=...", "State=...").
        # Atom fields are exactly "Symbol,float,float,float".
        sub = field.split(",")
        if len(sub) != 4:
            break
        sym = sub[0].strip()
        try:
            x, y, z = float(sub[1]), float(sub[2]), float(sub[3])
        except ValueError:
            break
        if sym not in atomic_numbers:
            break
        symbols.append(sym)
        positions.append((x, y, z))

    if not symbols:
        raise ValueError("No atom entries found in archive.")
    return symbols, charge if charge is not None else 0, mult if mult is not None else 1, positions


def read_gaussian_log(path: str | Path) -> Atoms:
    """Read the final optimized structure from a Gaussian 16 .log file.

    The geometry from the archive entry is returned in Angstrom (Gaussian's
    internal unit for the archive coordinates) as an :class:`ase.Atoms` object.
    The overall charge and spin multiplicity are stored in ``atoms.info`` under
    the keys ``"charge"`` and ``"multiplicity"``.
    """
    path = Path(path)
    text = path.read_text()
    archive = _reconstruct_archive(text)
    symbols, charge, mult, positions = parse_archive_atoms(archive)
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.info["charge"] = charge
    atoms.info["multiplicity"] = mult
    atoms.info["source"] = str(path)
    return atoms