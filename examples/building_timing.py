"""Timed version of the building.py stream-export workflow.

Runs the same DiamondCubic example and prints wall-clock time per stage.
Requires f3ast (and a Numba threading layer such as workqueue/tbb).
"""

import os
from time import perf_counter

import f3ast
import matplotlib

from nanoscale3d.branch_connections import install_fast_branch_connections

# One switch for both speedups (branch connections + beam points).
# Set False, or NANOSCALE3D_FAST=0, to compare against original f3ast.
USE_FAST = os.environ.get("NANOSCALE3D_FAST", "1") != "0"
if USE_FAST:
    install_fast_branch_connections()

matplotlib.use("Agg")

file_path = "examples/DiamondCubic-5x5x3.stl"
stream_filename = "DiamondCubic-5x5x3"
GR0 = 0.200  # in um/s, base growth rate
k = 1.0  # in 1/nm?, thermal conductivity
sigma = 4  # in nm, dwell size
scale_factor = 400

settings = {}
settings["structure"] = {"pitch": 3, "fill": False}  # in nm
settings["stream_builder"] = {
    "addressable_pixels": [65536, 56576],
    "max_dwt": 5,  # in ms
    "cutoff_time": 0.01,  # in ms, for faster exporting: remove dwells below cutoff time
    "screen_width": 10.2e3,  # in nm, horizontal screen width / field of view
    # 'serpentine' or 'serial', scanning order between slices
    "scanning_order": "serpentine",
}
# pixel size for thermal resistance
settings["dd_model"] = {"single_pixel_width": 50}

rotation_axis, rotation_angle = (1, 0, 0), 0
tilt_string = "90t100"
out_filename = (
    f"{stream_filename}_{GR0 * 1000:.0f}gr_{k * 100:.0f}k_{sigma:.0f}sig_"
    f"{tilt_string}_{scale_factor}x"
)

# On macOS, pip often cannot install tbb; workqueue is a usable fallback.
try:
    import numba

    numba.config.THREADING_LAYER = "workqueue"
except Exception:
    pass

t_total = perf_counter()

t0 = perf_counter()
struct = f3ast.Structure.from_file(file_path, **settings["structure"])
print(f"Load STL: {perf_counter() - t0:.2f}s")

t0 = perf_counter()
struct.rotate(rotation_axis, rotation_angle)
struct.centre()
struct.rescale(scale_factor)
print(f"Scale geometry (rotate/centre/rescale): {perf_counter() - t0:.2f}s")

# DDModel triggers slicing, branch connectivity, and thermal-resistance setup.
t0 = perf_counter()
model = f3ast.DDModel(struct, GR0, k, sigma, **settings["dd_model"])
n_slices = len(struct.slices)
n_pts = sum(s.shape[0] for s in struct.slices)
print(
    f"Slice + model init: {perf_counter() - t0:.2f}s "
    f"(slices={n_slices}, points={n_pts})"
)

t0 = perf_counter()
stream_builder, dwell_solver = f3ast.StreamBuilder.from_model(
    model, **settings["stream_builder"]
)
dwell_solver.print_total_time()
print(f"Solve dwells / build stream: {perf_counter() - t0:.2f}s")

t0 = perf_counter()
strm = stream_builder.get_stream()
print(f"Get stream: {perf_counter() - t0:.2f}s")

t0 = perf_counter()
strm.write(f"{out_filename}.str")
print(f"Write stream: {perf_counter() - t0:.2f}s")

print(f"Total wall time: {perf_counter() - t_total:.2f}s")
print(f"Wrote {out_filename}.str")
