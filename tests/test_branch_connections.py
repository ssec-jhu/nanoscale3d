"""The fast connectivity search must match f3ast's KD-tree search."""

import numpy as np
import pytest
from scipy.spatial import KDTree

from nanoscale3d.branch_connections import get_branch_connections


def _polyline(points):
    """Turn an (n, 2) point sequence into the (n-1, 2, 2) segment layout f3ast uses."""
    pts = np.asarray(points, dtype=np.float64)
    return np.stack((pts[:-1], pts[1:]), axis=1)


def _oracle(branch_intersections_slices, connection_distance):
    """Direct copy of f3ast.branches.get_branch_connections."""
    branch_connections = []
    separated_pts_below = None
    for i, separated_pts in enumerate(branch_intersections_slices):
        if i == 0:
            branch_connections.append([])
            separated_pts_below = separated_pts
            continue
        this_slice_connections = []
        for br_pts in separated_pts:
            tree = KDTree(br_pts.reshape(-1, 2))
            this_branch_connections = []
            branch_min_distances = []
            for k, branch_pts_below in enumerate(separated_pts_below):
                tree1 = KDTree(branch_pts_below.reshape(-1, 2))
                distance_matrix = tree.sparse_distance_matrix(tree1, connection_distance, p=np.inf)
                if distance_matrix.count_nonzero() > 0:
                    branch_min_distances.append(np.min(list(distance_matrix.values())))
                    this_branch_connections.append(k)
            this_branch_connections = np.array(this_branch_connections)
            n_conn = len(this_branch_connections)
            if n_conn > 1:
                count = 0
                for br_pts2 in separated_pts:
                    tree1 = KDTree(br_pts2.reshape(-1, 2))
                    distance_matrix = tree.sparse_distance_matrix(tree1, connection_distance, p=np.inf)
                    if distance_matrix.count_nonzero() > 0:
                        count += 1
                if count > 1 and count <= n_conn:
                    n_drop = n_conn - count + 1
                    keep_indx = np.argsort(branch_min_distances)[:n_drop]
                    this_branch_connections = this_branch_connections[keep_indx]
            this_slice_connections.append(this_branch_connections)
        branch_connections.append(this_slice_connections)
        separated_pts_below = separated_pts
    return branch_connections


def _assert_same_connections(actual, expected):
    assert len(actual) == len(expected)
    assert actual[0] == []
    assert expected[0] == []
    for actual_slice, expected_slice in zip(actual[1:], expected[1:], strict=True):
        assert len(actual_slice) == len(expected_slice)
        for actual_branch, expected_branch in zip(actual_slice, expected_slice, strict=True):
            np.testing.assert_array_equal(actual_branch, expected_branch)


def test_disconnected_branches_have_no_links():
    below = [_polyline([[0.0, 0.0], [1.0, 0.0]]), _polyline([[50.0, 0.0], [51.0, 0.0]])]
    current = [_polyline([[0.0, 20.0], [1.0, 20.0]]), _polyline([[50.0, -20.0], [51.0, -20.0]])]
    cd = 3.01
    _assert_same_connections(get_branch_connections([below, current], cd), _oracle([below, current], cd))


def test_exact_connection_distance_is_included():
    cd = 3.01
    below = [_polyline([[0.0, 0.0], [0.0, 1.0]])]
    current = [_polyline([[cd, 0.0], [cd, 1.0]])]
    _assert_same_connections(get_branch_connections([below, current], cd), _oracle([below, current], cd))


def test_points_just_outside_the_cutoff_are_rejected():
    cd = 3.01
    below = [_polyline([[0.0, 0.0], [0.0, 1.0]])]
    current = [_polyline([[cd + 1e-6, 0.0], [cd + 1e-6, 1.0]])]
    actual = get_branch_connections([below, current], cd)
    expected = _oracle([below, current], cd)
    _assert_same_connections(actual, expected)
    assert actual[1][0].size == 0


def test_negative_coordinates_match_the_kd_tree():
    cd = 2.5
    below = [_polyline([[-10.0, -4.0], [-8.0, -4.0]]), _polyline([[4.0, 6.0], [6.0, 7.0]])]
    current = [_polyline([[-10.2, -4.1], [-7.5, -3.8]]), _polyline([[20.0, 20.0], [21.0, 20.0]])]
    _assert_same_connections(get_branch_connections([below, current], cd), _oracle([below, current], cd))


def test_exact_overlap_is_not_a_connection():
    """f3ast uses dok.count_nonzero(), so a stored distance of 0 does not count."""
    cd = 3.01
    duplicated = [_polyline([[0.0, 0.0], [0.0, 0.0]])]
    actual = get_branch_connections([duplicated, duplicated], cd)
    _assert_same_connections(actual, _oracle([duplicated, duplicated], cd))
    assert actual[1][0].size == 0


def test_merged_branches_keep_f3ast_pruning_order():
    """A branch near two supports drops the farther one when a sibling is also present."""
    cd = 3.01
    below = [
        _polyline([[0.0, 0.0], [0.4, 0.0]]),
        _polyline([[2.4, 0.0], [2.8, 0.0]]),
        _polyline([[0.0, 30.0], [0.4, 30.0]]),
    ]
    # Branch 0 reaches the first two supports and is long enough to count as its
    # own in-slice neighbor. Branch 1 is the sibling that makes the count 2.
    current = [
        _polyline([[0.1, 0.2], [1.9, 0.2]]),
        _polyline([[0.2, 0.8], [0.6, 1.0]]),
        _polyline([[0.1, 30.2], [0.5, 30.4]]),
    ]
    actual = get_branch_connections([below, current], cd)
    expected = _oracle([below, current], cd)
    _assert_same_connections(actual, expected)
    # Pruning must actually remove a link; otherwise this fixture is too weak.
    assert expected[1][0].size == 1
    np.testing.assert_array_equal(expected[1][0], np.array([0]))


def test_random_slices_match_including_cell_boundaries():
    rng = np.random.default_rng(7)
    cd = 3.01
    slices = []
    for _slice_index in range(4):
        branches = []
        for branch_index in range(6):
            origin = rng.normal(scale=40.0, size=2)
            # Sprinkle samples on cell boundaries as well as in the open cell.
            steps = rng.normal(scale=1.5, size=(8, 2))
            steps[0] = np.array([cd, 0.0])
            points = origin + np.cumsum(steps, axis=0)
            points[1] = np.round(points[1] / cd) * cd
            branches.append(_polyline(points))
        slices.append(branches)
    _assert_same_connections(get_branch_connections(slices, cd), _oracle(slices, cd))


def test_connection_distance_must_be_positive():
    branch = [_polyline([[0.0, 0.0], [1.0, 0.0]])]
    with pytest.raises(ValueError, match="positive"):
        get_branch_connections([branch], 0.0)


def test_split_eqd_matches_f3ast():
    pytest.importorskip("f3ast")
    import numba
    from f3ast.slicing import split_eqd as slow_split_eqd

    numba.config.THREADING_LAYER = "workqueue"
    cd_pitch = 3.0
    slices = [
        [
            _polyline([[0.0, 0.0], [4.0, 0.2], [7.5, 1.0]]),
            _polyline([[20.0, -3.0], [22.0, -3.0], [25.0, -1.0]]),
        ],
        [
            _polyline([[0.2, 0.4], [3.0, 0.4]]),
            _polyline([[-8.0, 5.0], [-5.5, 6.0], [-5.0, 9.0], [-2.0, 9.5]]),
        ],
    ]
    from nanoscale3d.branch_connections import split_eqd as fast_split_eqd

    actual = fast_split_eqd(slices, cd_pitch)
    expected = slow_split_eqd(slices, cd_pitch)
    for actual_slice, expected_slice in zip(actual[0], expected[0], strict=True):
        np.testing.assert_array_equal(actual_slice, expected_slice)
    for actual_ids, expected_ids in zip(actual[1], expected[1], strict=True):
        np.testing.assert_array_equal(actual_ids, expected_ids)
    for actual_lengths, expected_lengths in zip(actual[2], expected[2], strict=True):
        assert len(actual_lengths) == len(expected_lengths)
        for actual_length, expected_length in zip(actual_lengths, expected_lengths, strict=True):
            assert actual_length == expected_length


def test_install_replaces_f3ast_lookup():
    f3ast = pytest.importorskip("f3ast")
    from nanoscale3d.branch_connections import install_fast_branch_connections
    from nanoscale3d.branch_connections import split_eqd as fast_split_eqd

    original_branches = f3ast.branches.get_branch_connections
    original_structure = f3ast.structure.get_branch_connections
    original_slicing = f3ast.slicing.split_eqd
    original_structure_split = f3ast.structure.split_eqd
    try:
        installed = install_fast_branch_connections()
        assert f3ast.branches.get_branch_connections is installed
        assert f3ast.structure.get_branch_connections is installed
        assert f3ast.slicing.split_eqd is fast_split_eqd
        assert f3ast.structure.split_eqd is fast_split_eqd
    finally:
        f3ast.branches.get_branch_connections = original_branches
        f3ast.structure.get_branch_connections = original_structure
        f3ast.slicing.split_eqd = original_slicing
        f3ast.structure.split_eqd = original_structure_split
