# references/

- **Zhu, Thompson & Martínez, "Geodesic interpolation for reaction pathways",
  *J. Chem. Phys.* 150, 164103 (2019), DOI 10.1063/1.5090303.**
  Accepted manuscript (green open access): DOI https://doi.org/10.1063/1.5090303
  (also on DOE PAGES / OSTI, biblio 1528918).

  Used here as the initial-path generator for NEB (the `--interp geodesic`
  option in `neb_ts/neb_run.py`), via the `geodesic-interpolate` package
  (PyPI; author Xiaolei Zhu). The Morse-scaled pairwise-distance metric
  keeps the band physically realizable for bond-forming/breaking reactions,
  fixing the IDPP ring-tearing / proton-fly failure seen on the Maillard
  glycosylamine system.
