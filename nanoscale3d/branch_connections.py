"""Fast replacements for the slow steps in ``f3ast.Structure.generate_slices``.

Four monkey-patches live here, all turned on together by
``install_fast_branch_connections``:

1. **Cut layers** — only test planes inside each triangle's Z-range.
2. **Split into branches** — vectorized endpoint matching (no Python id list).
3. **Which branches touch** — grid-cell Chebyshev search instead of KD-trees.
4. **Place beam points** — one equidistant pass per layer instead of per branch.

The connection indices, branch arrays, cut segments, and beam points match the
original ``f3ast`` path so stream files stay byte-identical on the validated
meshes.
"""

from __future__ import annotations

from contextlib import contextmanager

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


def split_intersection(intersection):
    """Split one layer's segments into connected branches.

    Matches ``f3ast.branches.split_intersection``: same quantized point
    identity, connected components, branch order, and segment order inside
    each branch. Endpoint→node assignment is done with contiguous NumPy
    arrays instead of a Python list of ``[point_index, node_id]`` pairs.
    """
    from trimesh.graph import connected_component_labels
    from trimesh.grouping import hashable_rows

    intersection = np.asarray(intersection)
    if intersection.size == 0:
        return []

    points = np.ascontiguousarray(intersection.reshape(-1, 2))
    keys = hashable_rows(points)
    order = np.argsort(keys)
    sorted_keys = keys[order]
    change = np.empty(len(keys), dtype=np.bool_)
    change[0] = True
    change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    node_ids = np.cumsum(change) - 1
    node_indices = np.empty(len(keys), dtype=np.int64)
    node_indices[order] = node_ids
    edges = node_indices.reshape(-1, 2)

    conn_labels = connected_component_labels(edges, node_count=int(node_ids[-1]) + 1)
    edge_labels = conn_labels[edges[:, 0]]
    # Stable sort keeps segments in their original order within each label,
    # matching ``intersection[edge_labels == lbl]`` used by f3ast.
    order_e = np.argsort(edge_labels, kind="stable")
    sorted_labels = edge_labels[order_e]
    breaks = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    return [intersection[idx] for idx in np.split(order_e, breaks)]


def get_intersection_lines(self):
    """Cut the mesh on horizontal planes, matching ``Structure.get_intersection_lines``.

    ``trimesh.mesh_multiplane`` tests every triangle against every plane. Most
    triangles only reach a short Z-range, so this builds the same plane list
    then passes only candidate faces into ``mesh_plane``. Returned segments and
    nonempty ``z_levels`` match the original method on the validated meshes.
    """
    import trimesh
    from trimesh.geometry import plane_transform
    from trimesh.intersections import mesh_plane
    from trimesh import transformations as tft

    slice_height = self.pitch / 2
    minz = float(self.bounds[0, 2]) + 1e-3
    maxz = float(self.bounds[1, 2]) - 1e-3
    z_levels = np.arange(minz, maxz, slice_height)
    if z_levels.size == 0:
        return [], z_levels

    plane_normal = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    plane_origin = np.zeros(3, dtype=np.float64)
    tol = float(trimesh.tol.merge)

    face_z = self.vertices[:, 2][self.faces]
    zmin = face_z.min(axis=1) - tol
    zmax = face_z.max(axis=1) + tol
    i0 = np.searchsorted(z_levels, zmin, side="left")
    i1 = np.searchsorted(z_levels, zmax, side="right") - 1
    active = i0 <= i1

    segments = [None] * len(z_levels)
    if np.any(active):
        span = (i1[active] - i0[active] + 1).astype(np.int64)
        face_ids = np.repeat(np.flatnonzero(active), span)
        offsets = np.arange(span.sum(), dtype=np.int64)
        offsets -= np.repeat(np.cumsum(span) - span, span)
        plane_ids = np.repeat(i0[active], span) + offsets
        order = np.argsort(plane_ids, kind="stable")
        plane_ids = plane_ids[order]
        face_ids = face_ids[order]

        change = np.flatnonzero(plane_ids[1:] != plane_ids[:-1]) + 1
        starts = np.concatenate(([0], change))
        stops = np.concatenate((change, [len(plane_ids)]))
        unique_planes = plane_ids[starts]

        vertex_dots = np.dot(plane_normal, (self.vertices - plane_origin).T)
        base_transform = np.linalg.inv(
            plane_transform(origin=plane_origin, normal=plane_normal)
        )
        translation = np.eye(4)
        cursor = 0
        for height_i, height in enumerate(z_levels):
            if cursor < len(unique_planes) and unique_planes[cursor] == height_i:
                local = face_ids[starts[cursor] : stops[cursor]]
                cursor += 1
            else:
                local = None
            if local is None or local.size == 0:
                segments[height_i] = np.zeros((0, 2, 2), dtype=np.float64)
                continue
            lines, _index = mesh_plane(
                mesh=self,
                plane_origin=plane_origin + plane_normal * height,
                plane_normal=plane_normal,
                return_faces=True,
                local_faces=local,
                cached_dots=vertex_dots - height,
            )
            translation[2, 3] = height
            to_2d = np.linalg.inv(np.dot(base_transform, translation))
            if len(lines) == 0:
                segments[height_i] = np.zeros((0, 2, 2), dtype=np.float64)
                continue
            lines_2d = tft.transform_points(lines.reshape((-1, 3)), to_2d)
            segments[height_i] = lines_2d[:, :2].reshape((-1, 2, 2))
    else:
        segments = [np.zeros((0, 2, 2), dtype=np.float64) for _ in z_levels]

    nonempty = [len(seg) != 0 for seg in segments]
    intersection_lines = [seg for seg, keep in zip(segments, nonempty, strict=True) if keep]
    z_levels = z_levels[np.asarray(nonempty, dtype=bool)].flatten()
    return intersection_lines, z_levels


# Snapshot of the f3ast call sites before any monkey-patch. Captured once on
# the first install so uninstall always restores the true originals.
_ORIGINAL_F3AST: dict | None = None


def _f3ast_modules():
    import f3ast.branches as branches
    import f3ast.slicing as slicing
    import f3ast.structure as structure

    return branches, slicing, structure


def _snapshot_original_f3ast():
    global _ORIGINAL_F3AST
    if _ORIGINAL_F3AST is not None:
        return
    branches, slicing, structure = _f3ast_modules()
    _ORIGINAL_F3AST = {
        "branches.get_branch_connections": branches.get_branch_connections,
        "structure.get_branch_connections": structure.get_branch_connections,
        "branches.split_intersection": branches.split_intersection,
        "structure.split_intersection": structure.split_intersection,
        "slicing.split_eqd": slicing.split_eqd,
        "structure.split_eqd": structure.split_eqd,
        "structure.Structure.get_intersection_lines": structure.Structure.get_intersection_lines,
    }


def install_fast_branch_connections():
    """Turn on all slicing speedups inside ``f3ast.Structure.generate_slices``.

    Replaces cut layers, branch splitting, branch connectivity, and
    equidistant beam-point placement. Call
    ``uninstall_fast_branch_connections`` to put the originals back.
    """
    _snapshot_original_f3ast()
    branches, slicing, structure = _f3ast_modules()
    branches.get_branch_connections = get_branch_connections
    structure.get_branch_connections = get_branch_connections
    branches.split_intersection = split_intersection
    structure.split_intersection = split_intersection
    slicing.split_eqd = split_eqd
    structure.split_eqd = split_eqd
    structure.Structure.get_intersection_lines = get_intersection_lines
    return get_branch_connections


def uninstall_fast_branch_connections():
    """Restore f3ast's original slicing functions."""
    _snapshot_original_f3ast()
    assert _ORIGINAL_F3AST is not None
    branches, slicing, structure = _f3ast_modules()
    branches.get_branch_connections = _ORIGINAL_F3AST["branches.get_branch_connections"]
    structure.get_branch_connections = _ORIGINAL_F3AST["structure.get_branch_connections"]
    branches.split_intersection = _ORIGINAL_F3AST["branches.split_intersection"]
    structure.split_intersection = _ORIGINAL_F3AST["structure.split_intersection"]
    slicing.split_eqd = _ORIGINAL_F3AST["slicing.split_eqd"]
    structure.split_eqd = _ORIGINAL_F3AST["structure.split_eqd"]
    structure.Structure.get_intersection_lines = _ORIGINAL_F3AST[
        "structure.Structure.get_intersection_lines"
    ]


@contextmanager
def fast_branch_connections(enabled: bool = True):
    """Temporarily use the fast path (all steps) or the original f3ast path.

    Args:
        enabled: If True, install all speedups for the block. If False, use
            the original f3ast functions for the block.
    """
    _snapshot_original_f3ast()
    branches, slicing, structure = _f3ast_modules()
    previous = {
        "branches.get_branch_connections": branches.get_branch_connections,
        "structure.get_branch_connections": structure.get_branch_connections,
        "branches.split_intersection": branches.split_intersection,
        "structure.split_intersection": structure.split_intersection,
        "slicing.split_eqd": slicing.split_eqd,
        "structure.split_eqd": structure.split_eqd,
        "structure.Structure.get_intersection_lines": structure.Structure.get_intersection_lines,
    }
    try:
        if enabled:
            install_fast_branch_connections()
        else:
            uninstall_fast_branch_connections()
        yield
    finally:
        branches.get_branch_connections = previous["branches.get_branch_connections"]
        structure.get_branch_connections = previous["structure.get_branch_connections"]
        branches.split_intersection = previous["branches.split_intersection"]
        structure.split_intersection = previous["structure.split_intersection"]
        slicing.split_eqd = previous["slicing.split_eqd"]
        structure.split_eqd = previous["structure.split_eqd"]
        structure.Structure.get_intersection_lines = previous[
            "structure.Structure.get_intersection_lines"
        ]
