"""Fast branch connectivity for the f3ast slicing path.

``f3ast`` decides which branches in consecutive slices are connected by a
Chebyshev nearest-neighbor test. The published implementation builds a KD-tree
for every branch pair. On the DiamondCubic benchmark that pairwise search is
almost the entire multi-minute runtime.

This module answers the same question from a uniform grid. Two points at
Chebyshev distance ``cd`` can fall in cells at most one index apart when the
cell size is ``cd``, so each point only inspects a 3x3 block of cells. The
pruning rule for branches that appear to merge is unchanged, so the connection
indices match ``f3ast.branches.get_branch_connections``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Cell keys are (ix + SHIFT) * STRIDE + (iy + SHIFT). SHIFT keeps indices
# positive; STRIDE is wider than the shifted iy range used below.
_CELL_SHIFT = np.int64(1_000_000_000)
_CELL_STRIDE = np.int64(2_000_000_000)


@njit(cache=True)
def _branch_min_chebyshev(q_pts, q_labels, n_q, b_pts, b_labels, n_b, cd):
    """Minimum Chebyshev distance from each query branch to each base branch.

    Returns ``(min_dist, min_positive)``. Both leave distances greater than
    ``cd`` as ``inf``. ``min_positive`` also ignores exact overlaps.
    """
    # min_dist includes exact overlaps. min_positive ignores them: f3ast's
    # neighbor test is dok_matrix.count_nonzero(), which treats a stored
    # distance of 0 as absent.
    min_dist = np.full((n_q, n_b), np.inf)
    min_positive = np.full((n_q, n_b), np.inf)
    n_base = b_pts.shape[0]
    n_query = q_pts.shape[0]
    if n_base == 0 or n_query == 0 or n_q == 0 or n_b == 0:
        return min_dist, min_positive

    ix = np.floor(b_pts[:, 0] / cd).astype(np.int64)
    iy = np.floor(b_pts[:, 1] / cd).astype(np.int64)
    keys = (ix + _CELL_SHIFT) * _CELL_STRIDE + (iy + _CELL_SHIFT)
    order = np.argsort(keys)
    keys_sorted = keys[order]

    boundary = np.empty(n_base, dtype=np.int64)
    boundary[0] = 0
    n_cells = 1
    for i in range(1, n_base):
        if keys_sorted[i] != keys_sorted[i - 1]:
            boundary[n_cells] = i
            n_cells += 1
    starts = boundary[:n_cells]
    ends = np.empty(n_cells, dtype=np.int64)
    for i in range(n_cells - 1):
        ends[i] = starts[i + 1]
    ends[n_cells - 1] = n_base
    unique_keys = keys_sorted[starts]

    qix = np.floor(q_pts[:, 0] / cd).astype(np.int64)
    qiy = np.floor(q_pts[:, 1] / cd).astype(np.int64)
    for qi in range(n_query):
        qx = q_pts[qi, 0]
        qy = q_pts[qi, 1]
        ql = q_labels[qi]
        base_ix = qix[qi]
        base_iy = qiy[qi]
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                key = (base_ix + dx + _CELL_SHIFT) * _CELL_STRIDE + (base_iy + dy + _CELL_SHIFT)
                pos = np.searchsorted(unique_keys, key)
                if pos >= n_cells or unique_keys[pos] != key:
                    continue
                start = starts[pos]
                end = ends[pos]
                for t in range(start, end):
                    j = order[t]
                    dxv = qx - b_pts[j, 0]
                    dyv = qy - b_pts[j, 1]
                    if dxv < 0.0:
                        dxv = -dxv
                    if dyv < 0.0:
                        dyv = -dyv
                    dist = dxv if dxv >= dyv else dyv
                    if dist <= cd:
                        bl = b_labels[j]
                        if dist < min_dist[ql, bl]:
                            min_dist[ql, bl] = dist
                        if dist > 0.0 and dist < min_positive[ql, bl]:
                            min_positive[ql, bl] = dist
    return min_dist, min_positive


def _pack_branches(branches):
    """Stack ``(n, 2, 2)`` branch polylines into points and branch labels."""
    chunks = []
    label_chunks = []
    for index, branch in enumerate(branches):
        points = np.ascontiguousarray(np.reshape(branch, (-1, 2)), dtype=np.float64)
        if points.shape[0] == 0:
            continue
        chunks.append(points)
        label_chunks.append(np.full(points.shape[0], index, dtype=np.int64))
    n_branches = len(branches)
    if not chunks:
        empty_pts = np.zeros((0, 2), dtype=np.float64)
        empty_labels = np.zeros(0, dtype=np.int64)
        return empty_pts, empty_labels, n_branches
    return np.vstack(chunks), np.concatenate(label_chunks), n_branches


def _validate_extent(points, connection_distance):
    if points.size == 0:
        return
    limit = float(_CELL_SHIFT - 2) * connection_distance
    if float(np.max(np.abs(points))) >= limit:
        raise ValueError("branch coordinates are too large for the connection grid")


def _prune_connections(min_dist, min_positive, same_positive, connection_distance):
    """Apply f3ast's multi-connection pruning and return one array per branch."""
    connections = []
    n_current = min_dist.shape[0]
    for branch_index in range(n_current):
        neighbors = np.flatnonzero(min_positive[branch_index] <= connection_distance)
        n_conn = int(neighbors.size)
        if n_conn > 1:
            in_slice = int(np.count_nonzero(same_positive[branch_index] <= connection_distance))
            if 1 < in_slice <= n_conn:
                n_keep = n_conn - in_slice + 1
                keep = np.argsort(min_dist[branch_index, neighbors])[:n_keep]
                neighbors = neighbors[keep]
        if n_conn == 0:
            connections.append(np.array([]))
        else:
            connections.append(np.asarray(neighbors))
    return connections


def get_branch_connections(branch_intersections_slices, connection_distance):
    """Connections from each branch to branches in the previous slice.

    Args:
        branch_intersections_slices: For each slice, the polyline array of
            each connected component. Matches the lists produced by
            ``f3ast.branches.split_intersection``.
        connection_distance: Chebyshev distance at which two branches are
            treated as touching. ``f3ast`` uses ``pitch + 0.01``.

    Returns:
        ``branch_connections[0]`` is an empty list. For later slices,
        ``branch_connections[i][j]`` is an array of branch indices in slice
        ``i - 1`` that branch ``j`` connects to. Index order follows f3ast,
        including the closest-connection pruning step.
    """
    cd = float(connection_distance)
    if not np.isfinite(cd) or cd <= 0.0:
        raise ValueError("connection_distance must be a positive finite number")

    packed = [_pack_branches(branches) for branches in branch_intersections_slices]
    for points, _labels, _n_branches in packed:
        _validate_extent(points, cd)

    connections = []
    for index, (points, labels, n_branches) in enumerate(packed):
        if index == 0:
            connections.append([])
            continue
        prev_points, prev_labels, prev_n = packed[index - 1]
        min_dist, min_positive = _branch_min_chebyshev(points, labels, n_branches, prev_points, prev_labels, prev_n, cd)
        _same_dist, same_positive = _branch_min_chebyshev(points, labels, n_branches, points, labels, n_branches, cd)
        connections.append(_prune_connections(min_dist, min_positive, same_positive, cd))
    return connections


def split_eqd(branch_intersections_slices, pitch):
    """Equidistant sample points for each branch, matching ``f3ast`` ordering.

    ``f3ast.slicing.split_eqd`` calls a parallel kernel once per branch. The
    kernel launch dominates. This batches every segment in a slice into one
    point-generation pass, then applies the same grid rounding and
    lexicographic unique.
    """
    import numpy.linalg as la

    pitch = float(pitch)
    slices = []
    branch_ids = []
    branch_lengths = []
    for separated in branch_intersections_slices:
        if not separated:
            slices.append(np.zeros((0, 2)))
            branch_ids.append(np.zeros(0))
            branch_lengths.append([])
            continue
        line_counts = np.array([branch.shape[0] for branch in separated], dtype=np.int64)
        lines = np.concatenate(separated, axis=0)
        start = np.ascontiguousarray(lines[:, 0, :], dtype=np.float64)
        end = np.ascontiguousarray(lines[:, 1, :], dtype=np.float64)
        vec = end - start
        n_steps = np.ceil(la.norm(vec, axis=1) / pitch).astype(np.int64)
        total = int(n_steps.sum())
        if total == 0:
            unique_branches = [np.zeros((0, 2)) for _ in separated]
        else:
            line_ids = np.repeat(np.arange(n_steps.shape[0]), n_steps)
            line_start = np.cumsum(n_steps) - n_steps
            local = np.arange(total) - line_start[line_ids]
            # Match f3ast's numba expression, which divides after the multiply:
            # start + (vec * arange(n)) / n
            step = n_steps[line_ids][:, None]
            generated = start[line_ids] + (vec[line_ids] * local[:, None]) / step
            rounded = np.round(generated / pitch) * pitch
            point_ends = np.cumsum(n_steps)
            point_starts = point_ends - n_steps
            line_ends = np.cumsum(line_counts)
            line_starts = line_ends - line_counts
            unique_branches = []
            for begin, stop in zip(line_starts, line_ends, strict=True):
                if begin == stop or n_steps[begin:stop].sum() == 0:
                    unique_branches.append(np.zeros((0, 2)))
                    continue
                group = rounded[point_starts[begin] : point_ends[stop - 1]]
                unique_branches.append(np.unique(group, axis=0))
        slices.append(np.vstack(unique_branches) if unique_branches else np.zeros((0, 2)))
        branch_ids.append(
            np.concatenate([index * np.ones(points.shape[0]) for index, points in enumerate(unique_branches)])
            if unique_branches
            else np.zeros(0)
        )
        branch_lengths.append([_path_length(points) for points in unique_branches])
    return slices, branch_ids, branch_lengths


def _path_length(points):
    """Length of the polyline through ``points``, same reduction as f3ast."""
    import numpy.linalg as la

    return np.sum(la.norm(points[1:] - points[:-1], axis=1))


def install_fast_branch_connections():
    """Use the faster slicing steps inside ``f3ast.Structure.generate_slices``.

    Replaces branch connectivity and equidistant resampling. ``Structure``
    holds its own references, so both the defining module and ``structure``
    are updated.
    """
    import f3ast.branches as branches
    import f3ast.slicing as slicing
    import f3ast.structure as structure

    branches.get_branch_connections = get_branch_connections
    structure.get_branch_connections = get_branch_connections
    slicing.split_eqd = split_eqd
    structure.split_eqd = split_eqd
    return get_branch_connections
