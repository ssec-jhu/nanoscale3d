"""Validate the fast branch-connection path on available example STLs.

For each case, runs the original f3ast path and the nanoscale3d replacement,
then reports wall time, peak RSS, stream-file identity, and total dwell time.

Also times resistance setup vs dwell solving when the same sliced structure is
rebuilt at several growth rates (GR0), matching the Oct 14 meeting action item.
"""

from __future__ import annotations

import hashlib
import os
import resource
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import f3ast
import matplotlib
import numba
import numpy as np

matplotlib.use("Agg")
numba.config.THREADING_LAYER = "workqueue"

from nanoscale3d.branch_connections import install_fast_branch_connections

# Pitch / model settings match examples/building.py.
STRUCTURE = {"pitch": 3, "fill": False}
STREAM_BUILDER = {
    "addressable_pixels": [65536, 56576],
    "max_dwt": 5,
    "cutoff_time": 0.01,
    "screen_width": 10.2e3,
    "scanning_order": "serpentine",
}
DD_MODEL = {"single_pixel_width": 50}
K = 1.0
SIGMA = 4.0
GR0 = 0.200


@dataclass(frozen=True)
class Case:
    label: str
    file_path: str
    scale_factor: int


# Structures already in the repo. Brandon's ~1 hour files are not available yet.
CASES = [
    Case("5MinuteDiamond x200", "examples/5MinuteDiamond.stl", 200),
    Case("15MinuteDiamond x200", "examples/15MinuteDiamond.stl", 200),
    Case("CylindersWithSphere-LowRes x100", "examples/CylindersWithSphere-LowRes.stl", 100),
    Case("CylindersWithSphere-HiRemeshRes x50", "examples/CylindersWithSphere-HiRemeshRes.stl", 50),
    Case("DiamondCubic-5x5x3 x400", "examples/DiamondCubic-5x5x3.stl", 400),
]


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kibibytes.
    if os.uname().sysname == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


@contextmanager
def restored_f3ast():
    """Snapshot and restore the f3ast call sites this package monkey-patches."""
    import f3ast.branches as branches
    import f3ast.slicing as slicing
    import f3ast.structure as structure

    originals = {
        "branches.get_branch_connections": branches.get_branch_connections,
        "structure.get_branch_connections": structure.get_branch_connections,
        "slicing.split_eqd": slicing.split_eqd,
        "structure.split_eqd": structure.split_eqd,
    }
    try:
        yield
    finally:
        branches.get_branch_connections = originals["branches.get_branch_connections"]
        structure.get_branch_connections = originals["structure.get_branch_connections"]
        slicing.split_eqd = originals["slicing.split_eqd"]
        structure.split_eqd = originals["structure.split_eqd"]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(file_path: str, scale_factor: int, gr0: float = GR0):
    """Load → scale → DDModel → stream. Returns timing dict and stream bytes."""
    times: dict[str, float] = {}
    t_total = perf_counter()

    t0 = perf_counter()
    struct = f3ast.Structure.from_file(file_path, **STRUCTURE)
    times["load"] = perf_counter() - t0

    t0 = perf_counter()
    struct.rotate((1, 0, 0), 0)
    struct.centre()
    struct.rescale(scale_factor)
    times["scale"] = perf_counter() - t0

    t0 = perf_counter()
    model = f3ast.DDModel(struct, gr0, K, SIGMA, **DD_MODEL)
    times["slice_model"] = perf_counter() - t0

    t0 = perf_counter()
    stream_builder, dwell_solver = f3ast.StreamBuilder.from_model(model, **STREAM_BUILDER)
    times["solve"] = perf_counter() - t0
    total_dwell = float(dwell_solver.get_total_time().total_seconds())

    t0 = perf_counter()
    strm = stream_builder.get_stream()
    times["get_stream"] = perf_counter() - t0

    with tempfile.NamedTemporaryFile(suffix=".str", delete=False) as tmp:
        out_path = tmp.name
    t0 = perf_counter()
    strm.write(out_path)
    times["write"] = perf_counter() - t0
    times["total"] = perf_counter() - t_total

    digest = file_sha256(out_path)
    size = os.path.getsize(out_path)
    os.unlink(out_path)

    n_slices = len(struct.slices)
    n_pts = int(sum(s.shape[0] for s in struct.slices))
    return {
        "times": times,
        "sha256": digest,
        "bytes": size,
        "total_dwell_s": total_dwell,
        "slices": n_slices,
        "points": n_pts,
        "peak_rss_mb": peak_rss_mb(),
        "struct": struct,
        "model": model,
    }


def compare_case(case: Case) -> None:
    print(f"\n=== {case.label} ===")
    print(f"file={case.file_path}  scale={case.scale_factor}")

    with restored_f3ast():
        print("running original f3ast …")
        slow = run_pipeline(case.file_path, case.scale_factor)
        # Drop heavy objects before the fast run so RSS is comparable.
        del slow["struct"], slow["model"]

    install_fast_branch_connections()
    print("running fast path …")
    fast = run_pipeline(case.file_path, case.scale_factor)
    del fast["struct"], fast["model"]

    identical = slow["sha256"] == fast["sha256"]
    dwell_delta = abs(fast["total_dwell_s"] - slow["total_dwell_s"]) / max(slow["total_dwell_s"], 1e-12)

    print(
        f"slices={fast['slices']}  points={fast['points']}  "
        f"stream={fast['bytes']} bytes"
    )
    print(
        f"time  slow={slow['times']['total']:.2f}s  "
        f"fast={fast['times']['total']:.2f}s  "
        f"speedup={slow['times']['total'] / max(fast['times']['total'], 1e-9):.1f}x"
    )
    print(
        f"  slice+model  slow={slow['times']['slice_model']:.2f}s  "
        f"fast={fast['times']['slice_model']:.2f}s"
    )
    print(
        f"peak RSS after run  slow={slow['peak_rss_mb']:.0f} MB  "
        f"fast={fast['peak_rss_mb']:.0f} MB"
    )
    print(f"stream identical: {identical}")
    print(
        f"total dwell  slow={slow['total_dwell_s']:.3f}s  "
        f"fast={fast['total_dwell_s']:.3f}s  "
        f"rel_diff={dwell_delta:.3e}"
    )
    if not identical:
        raise SystemExit(f"stream mismatch for {case.label}")


def growth_rate_resistance_probe(file_path: str, scale_factor: int) -> None:
    """Time resistance vs dwell solve when only GR0 changes on a fixed structure."""
    from f3ast.resistance import get_resistance

    print(f"\n=== Growth-rate probe ({file_path}, scale={scale_factor}) ===")
    install_fast_branch_connections()

    struct = f3ast.Structure.from_file(file_path, **STRUCTURE)
    struct.rotate((1, 0, 0), 0)
    struct.centre()
    struct.rescale(scale_factor)

    # Slicing is independent of GR0. Force it once up front.
    t0 = perf_counter()
    _ = struct.slices
    t_slices = perf_counter() - t0
    print(f"generate_slices (once): {t_slices:.2f}s  slices={len(struct.slices)}")

    t0 = perf_counter()
    cached_resistance = get_resistance(struct, **DD_MODEL)
    t_resistance = perf_counter() - t0
    print(f"get_resistance (once): {t_resistance:.3f}s")

    growth_rates = [0.100, 0.200, 0.300, 0.500]
    print("rebuild without caching resistance (new DDModel each GR0):")
    for gr0 in growth_rates:
        t0 = perf_counter()
        new_model = f3ast.DDModel(struct, gr0, K, SIGMA, **DD_MODEL)
        t_init = perf_counter() - t0

        t0 = perf_counter()
        _stream_builder, dwell_solver = f3ast.StreamBuilder.from_model(
            new_model, **STREAM_BUILDER
        )
        t_solve = perf_counter() - t0
        total_dwell = float(dwell_solver.get_total_time().total_seconds())
        print(
            f"GR0={gr0:.3f}  DDModel_init={t_init:.3f}s  "
            f"solve={t_solve:.3f}s  total_dwell={total_dwell:.3f}s"
        )

    print("rebuild with resistance reused (get_parameters=False):")
    for gr0 in growth_rates:
        t0 = perf_counter()
        reused = f3ast.DDModel(
            struct, gr0, K, SIGMA, get_parameters=False, **DD_MODEL
        )
        reused._resistance = cached_resistance
        t_init = perf_counter() - t0

        t0 = perf_counter()
        _stream_builder, dwell_solver = f3ast.StreamBuilder.from_model(
            reused, **STREAM_BUILDER
        )
        t_solve = perf_counter() - t0
        total_dwell = float(dwell_solver.get_total_time().total_seconds())
        print(
            f"GR0={gr0:.3f}  DDModel_init={t_init:.3f}s  "
            f"solve={t_solve:.3f}s  total_dwell={total_dwell:.3f}s"
        )


def main() -> None:
    # Skip the largest case when FAST_VALIDATE=1 for a quicker smoke check.
    cases = CASES
    if os.environ.get("FAST_VALIDATE") == "1":
        cases = [c for c in cases if "DiamondCubic" not in c.label]

    for case in cases:
        compare_case(case)

    # Use a mid-size case so the resistance probe finishes quickly.
    growth_rate_resistance_probe("examples/15MinuteDiamond.stl", 200)
    print("\nAll validation cases passed.")


if __name__ == "__main__":
    main()
