This directory contains workflows for processing STL files through the program. `building.ipynb` and `building.py` should both work. `building_timing.py` is the same DiamondCubic export path with per-stage wall-clock timings. Both call `install_fast_branch_connections()` so they use the faster grid-cell connectivity path from `nanoscale3d.branch_connections`.

`validate_speedup.py` compares the original `f3ast` path against that replacement on the example STLs (runtime, peak RSS, and byte-identical `.str` output). Set `FAST_VALIDATE=1` to skip the largest DiamondCubic case.

The included STL files are two different diamond-lattice structures (one taking ~5 minutes and the other ~15 minutes to slice/define growth model), plus a larger `DiamondCubic-5x5x3.stl` used by `building.py`. There are also two of the cylinders+sphere files I showed in the walkthrough.
