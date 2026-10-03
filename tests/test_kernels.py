"""Every ``@export``ed kernel is called here by name and checked on its output.

The parity tests in ``test_parity.py`` drive the same code through the public
API, which is the right way to test a port. This file exists for a narrower
reason: a kernel that the public API happens not to reach, or that is wired up
with the wrong stride, count or length, would otherwise still ship green
because nothing calls it directly. So each test here calls the C ABI symbol
itself and asserts on the buffer the kernel writes.

References are numpy wherever the expected answer is easy to state exactly, and
upstream ``verde`` (or the scipy/sklearn routine it delegates to) wherever that
is the stronger oracle.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from mojoverde._lib import f64, lib

# simd_width_of[DType.float64] on this target, and the kNN block is eight of
# them; both are re-derived from the sizes the kernels are driven at below.
LANES = 4
KNN_BLOCK = 8 * LANES

# The point counts below which the kernels keep their serial loop
# (PARALLEL_MIN = 256 rows), so both branches get exercised.
SMALL = 40
LARGE = 300


def make_rng():
    """A fresh generator, so a test that is re-run on its own sees the same
    numbers as when it runs with the rest of the file."""
    return np.random.default_rng(20240917)


@pytest.fixture
def rng():
    return make_rng()


# ----------------------------------------------------------------- get_region
def test_mv_get_region_returns_the_bounding_box():
    east = np.array([3.0, -1.0, 7.5, 2.0])
    north = np.array([0.0, 4.0, -2.5, 9.0])
    out = np.empty(4)
    lib().mv_get_region((f64(east)), (f64(north)), (out), east.size)
    np.testing.assert_array_equal(out, [-1.0, 7.5, -2.5, 9.0])


def test_mv_get_region_single_point_is_that_point():
    out = np.empty(4)
    lib().mv_get_region(
        (f64(np.array([2.0]))), (f64(np.array([-3.0]))), out, 1
    )
    np.testing.assert_array_equal(out, [2.0, 2.0, -3.0, -3.0])


def test_mv_get_region_of_nothing_is_all_zero():
    out = np.full(4, np.nan)
    lib().mv_get_region((f64(np.zeros(0))), (f64(np.zeros(0))), out, 0)
    np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 0.0])


# ----------------------------------------------------------- line_coordinates
@pytest.mark.parametrize(
    "start, stop, size, pixel_register",
    [
        (0.0, 10.0, 5, 0),
        (-3.0, 7.0, 11, 0),
        (0.0, 10.0, 1, 0),
        (2.5, 2.5, 4, 0),
        (0.0, 10.0, 6, 1),
        (-3.0, 7.0, 11, 1),
    ],
)
def test_mv_line_coordinates(start, stop, size, pixel_register):
    values = np.empty(size)
    lib().mv_line_coordinates(start, stop, size, values, pixel_register)
    expected = np.linspace(start, stop, size)
    if pixel_register:
        # The kernel shifts the nodes in place and leaves the last one on its
        # own, so the check is against the shifted nodes, not against linspace.
        assert values[-1] == stop
        expected = expected[:-1] + (expected[1] - expected[0]) / 2
    else:
        assert values[0] == start
        if size > 1:
            assert values[-1] == stop
    # The interior nodes may differ by an ULP, where this toolchain contracts
    # the multiply-add and numpy rounds twice.
    np.testing.assert_allclose(values[: expected.size], expected, rtol=1e-15)


def test_mv_line_coordinates_pixel_register_uses_the_first_interval():
    # Upstream shifts by the half-interval between the first two nodes, so
    # a region whose spacing is then adjusted shifts every node by the same
    # amount rather than by its own.
    values = np.empty(6)
    lib().mv_line_coordinates(0.0, 10.0, 6, values, 1)
    nodes = np.linspace(0.0, 10.0, 6)
    half = (nodes[1] - nodes[0]) / 2
    np.testing.assert_allclose(values[:-1], nodes[:-1] + half, rtol=0, atol=0)
    # The last node is left on its own, which is what makes the shifted
    # sequence one element shorter than the node count.
    assert values[-1] == 10.0


def test_mv_line_coordinates_empty_and_single():
    empty = np.empty(0)
    lib().mv_line_coordinates(0.0, 1.0, 0, empty, 0)
    assert empty.size == 0
    one = np.empty(1)
    lib().mv_line_coordinates(4.0, 9.0, 1, one, 0)
    np.testing.assert_array_equal(one, [4.0])


# ------------------------------------------------------------------ meshgrid
@pytest.mark.parametrize("ne, nn", [(1, 1), (7, 3), (11, LARGE), (257, LARGE)])
def test_mv_meshgrid_2d_layout(ne, nn):
    east = np.linspace(0.0, 1.0, ne)
    north = np.linspace(-1.0, 1.0, nn)
    east_2d = np.empty(nn * ne)
    north_2d = np.empty(nn * ne)
    lib().mv_meshgrid_2d(
        (f64(east)), (f64(north)), (east_2d), (north_2d), ne, nn, 1
    )
    # Default "xy" indexing: easting varies along the row, northing down it.
    np.testing.assert_allclose(
        east_2d.reshape(nn, ne), np.tile(east, (nn, 1)), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        north_2d.reshape(nn, ne), np.tile(north[:, None], (1, ne)), rtol=0, atol=0
    )


# --------------------------------------------------------------------- inside
@pytest.mark.parametrize("n", [SMALL, LARGE])
def test_mv_inside_counts_the_boundary_as_inside(n):
    east = np.linspace(0.0, 10.0, n)
    north = np.linspace(-5.0, 5.0, n)
    out = np.empty(n, dtype="uint8")
    region = (2.0, 8.0, -4.0, 4.0)
    lib().mv_inside(
        (f64(east)),
        (f64(north)),
        (out),
        n,
        region[0],
        region[1],
        region[2],
        region[3],
        1,
    )
    expected = (
        (east >= region[0])
        & (east <= region[1])
        & (north >= region[2])
        & (north <= region[3])
    )
    np.testing.assert_array_equal(out.view(bool), expected)
    assert 0 < out.sum() < n


def test_mv_inside_of_nothing_is_empty():
    out = np.empty(0, dtype="uint8")
    lib().mv_inside(
        (f64(np.zeros(0))), (f64(np.zeros(0))), out, 0, 0.0, 1.0, 0.0, 1.0, 1
    )
    assert out.size == 0


# ------------------------------------------------------- longitude_continuity
@pytest.mark.parametrize("n", [SMALL, LARGE])
@pytest.mark.parametrize("interval_360", [0, 1])
def test_mv_longitude_continuity(n, interval_360):
    rng = np.random.default_rng(5)
    east = rng.uniform(-200.0, 400.0, n)
    out = np.empty(n)
    lib().mv_longitude_continuity((f64(east)), out, n, interval_360, 1)
    if interval_360:
        expected = east % 360.0
    else:
        expected = ((east + 180.0) % 360.0) - 180.0
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


# ---------------------------------------------------------------- block_split
def test_mv_block_split_labels_the_nearest_centre():
    # Centres at -1.25 and 1.25, a pitch of 2.5, two blocks wide and one
    # block deep, so the label is the block index along easting alone.
    east = np.array([-5.0, -1.3, -1.2, 1.3, 2.0, 5.0])
    north = np.zeros(6)
    labels = np.empty(6, dtype="int64")
    lib().mv_block_split(
        (f64(east)), (f64(north)), labels, 6, -1.25, -1.25, 2.5, 0.0, 2, 1
    )
    # Round to nearest: the two outer points sit outside the lattice and are
    # clamped to the first and the last block.
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])


def test_mv_block_split_labels_are_row_major_over_both_axes():
    east = np.array([0.0, 2.0, 0.0, 2.0, 1.0])
    north = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    labels = np.empty(5, dtype="int64")
    lib().mv_block_split(
        (f64(east)), (f64(north)), labels, 5, 0.0, 0.0, 1.0, 1.0, 2, 2
    )
    # label = north_block * nb_east + east_block
    np.testing.assert_array_equal(labels, [0, 1, 2, 3, 1])


def test_mv_block_split_clamps_the_out_of_range_ends():
    east = np.array([-100.0, 100.0, 0.0])
    north = np.array([0.0, 0.0, 0.0])
    labels = np.empty(3, dtype="int64")
    lib().mv_block_split(
        (f64(east)), (f64(north)), labels, 3, 0.0, 0.0, 1.0, 1.0, 3, 3
    )
    # Clamped to the first and last easting block of the first row.
    np.testing.assert_array_equal(labels, [0, 2, 0])


def test_mv_block_split_survives_a_zero_pitch():
    """A one-block-wide set has no pitch on that axis, and the division by
    it would be a divide by zero; the point belongs to the only block."""
    east = np.array([-3.0, 0.0, 9.0])
    north = np.array([0.0, 0.0, 0.0])
    labels = np.empty(3, dtype="int64")
    lib().mv_block_split(
        (f64(east)), (f64(north)), labels, 3, 0.0, 0.0, 0.0, 0.0, 1, 1
    )
    np.testing.assert_array_equal(labels, [0, 0, 0])


def test_mv_block_split_on_one_point():
    # A lattice of 4 x 4 block centres at 0, 1, 2 and 3: the point (1.5, 2.5)
    # rounds to the centre at (2, 3), and the label is row-major.
    labels = np.empty(1, dtype="int64")
    lib().mv_block_split(
        (f64(np.array([1.5]))), (f64(np.array([2.5]))), labels, 1,
        0.0, 0.0, 1.0, 1.0, 4, 4,
    )
    assert labels[0] == 3 * 4 + 2


# --------------------------------------------------------- rolling window
def _window_reference(east, north, centre_e, centre_n, half):
    """Every window, as the list of point indices a KD-tree would return."""
    nw_e, nw_n = centre_e.size, centre_n.size
    out = []
    for jn in range(nw_n):
        for je in range(nw_e):
            inside = np.flatnonzero(
                (np.abs(east - centre_e[je]) <= half)
                & (np.abs(north - centre_n[jn]) <= half)
            )
            out.append(inside.astype(np.int64))
    return out


def _run_rolling(east, north, centre_e, centre_n, half, nchunks):
    nw_e, nw_n = centre_e.size, centre_n.size
    nwindows = nw_e * nw_n
    n = east.size
    chunk_counts = np.zeros(nchunks * nwindows, dtype="int64")
    lib().mv_rolling_window_count(
        (f64(east)),
        (f64(north)),
        (f64(centre_e)),
        (f64(centre_n)),
        chunk_counts,
        n,
        nw_e,
        nw_n,
        half,
        nchunks,
        1,
    )
    counts = chunk_counts.reshape(nchunks, nwindows)
    offsets = np.zeros(nwindows + 1, dtype="int64")
    offsets[1:] = np.cumsum(counts.sum(axis=0))
    positions = np.zeros((nchunks, nwindows), dtype="int64")
    if nchunks > 1:
        positions[1:] = np.cumsum(counts[:-1], axis=0)
    positions += offsets[:-1]
    flat = np.empty(int(offsets[-1]), dtype="int64")
    lib().mv_rolling_window_fill(
        (f64(east)),
        (f64(north)),
        (f64(centre_e)),
        (f64(centre_n)),
        positions.ravel(),
        flat,
        n,
        nw_e,
        nw_n,
        half,
        nchunks,
        1,
    )
    return [
        flat[start:stop].copy() for start, stop in zip(offsets[:-1], offsets[1:])
    ]


@pytest.mark.parametrize("nchunks", [1, 3, 16])
@pytest.mark.parametrize("half", [0.4, 0.5, 1.0])
def test_rolling_window_count_and_fill_agree_with_the_neighbourhood(
    nchunks, half
):
    rng = np.random.default_rng(9)
    east = rng.uniform(0.0, 10.0, 400)
    north = rng.uniform(-5.0, 5.0, 400)
    centre_e = np.linspace(0.0, 10.0, 7)
    centre_n = np.linspace(-5.0, 5.0, 5)
    got = _run_rolling(east, north, centre_e, centre_n, half, nchunks)
    expected = _window_reference(east, north, centre_e, centre_n, half)
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        # The points of a window come out in increasing index order, which is
        # the order query_ball_point returns them in.
        np.testing.assert_array_equal(a, b)


def test_rolling_window_fill_of_nothing_is_empty():
    got = _run_rolling(
        np.array([0.0]), np.array([0.0]), np.array([100.0, 101.0]),
        np.array([100.0]), 0.5, 4,
    )
    assert [len(g) for g in got] == [0, 0]


def test_rolling_window_count_finds_the_boundaries():
    # Points exactly half a spacing from a centre, and exactly on it.
    centre_e = np.array([0.0, 1.0, 2.0])
    centre_n = np.array([0.0])
    east = np.array([-0.5, 0.0, 0.5, 1.5, 2.5])
    north = np.zeros(5)
    got = _run_rolling(east, north, centre_e, centre_n, 0.5, 2)
    # The boundary counts as inside, so each point lands in the two windows
    # whose centres it is within half of.
    assert [g.tolist() for g in got] == [[0, 1, 2], [2, 3], [3, 4]]


# ------------------------------------------------------- expanding window
@pytest.mark.parametrize("size", [0.2, 0.5, 1.0, 0.0])
def test_mv_expanding_window_select_returns_the_points_in_order(size):
    rng = np.random.default_rng(11)
    east = rng.uniform(0.0, 10.0, 2000)
    north = rng.uniform(-5.0, 5.0, 2000)
    out = np.empty(east.size, dtype="int64")
    centre = (5.0, 0.0)
    found = lib().mv_expanding_window_select(
        (f64(east)), (f64(north)), out, east.size, centre[0], centre[1], size / 2
    )
    half = size / 2
    expected = np.flatnonzero(
        (np.abs(east - centre[0]) <= half) & (np.abs(north - centre[1]) <= half)
    ).astype(np.int64)
    assert found == expected.size
    np.testing.assert_array_equal(out[:found], expected)


def test_mv_expanding_window_select_of_nothing_finds_nothing():
    out = np.empty(0, dtype="int64")
    assert (
        lib().mv_expanding_window_select(
            (f64(np.zeros(0))), (f64(np.zeros(0))), out, 0, 0.0, 0.0, 1.0
        )
        == 0
    )


# ------------------------------------------------------------------ knn
@pytest.mark.parametrize("n", [KNN_BLOCK - 1, KNN_BLOCK, KNN_BLOCK + 1, 257])
@pytest.mark.parametrize("k", [1, 2, 8])
def test_mv_knn_query_matches_a_kdtree(n, k):
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(n * 100 + k)
    east_t = rng.uniform(0.0, 10.0, n)
    north_t = rng.uniform(-5.0, 5.0, n)
    east_q = rng.uniform(0.0, 10.0, 40)
    north_q = rng.uniform(-5.0, 5.0, 40)
    nq, nt = east_q.size, east_t.size
    dist = np.empty(nq * k)
    idx = np.empty(nq * k, dtype="int64")
    lib().mv_knn_query(
        (f64(east_q)),
        (f64(north_q)),
        (f64(east_t)),
        (f64(north_t)),
        dist,
        idx,
        nq,
        nt,
        k,
        1,
    )
    dist = dist.reshape(nq, k)
    idx = idx.reshape(nq, k)
    ref_dist, ref_idx = cKDTree(
        np.column_stack([east_t, north_t])
    ).query(np.column_stack([east_q, north_q]), k=k)
    np.testing.assert_array_equal(idx, np.reshape(ref_idx, idx.shape))
    np.testing.assert_allclose(dist, np.reshape(ref_dist, dist.shape), rtol=1e-12)
    # Sorted and non-negative: the kernel keeps the k-list in order.
    assert (np.diff(dist, axis=1) >= -1e-12).all()
    assert (dist >= 0).all()


def test_mv_knn_query_drops_a_nan_lane_without_poisoning_the_block():
    rng = np.random.default_rng(3)
    east_t = rng.uniform(0.0, 10.0, 300)
    north_t = rng.uniform(0.0, 10.0, 300)
    # Inside the first 32-point block and inside the vector tail, so both the
    # block minimum and the scalar remainder meet a NaN.
    east_t[5] = np.nan
    north_t[100] = np.nan
    east_q = np.array([east_t[6], east_t[101]])
    north_q = np.array([north_t[6], north_t[101]])
    dist = np.empty(2)
    idx = np.empty(2, dtype="int64")
    lib().mv_knn_query(
        (f64(east_q)),
        (f64(north_q)),
        (f64(east_t)),
        (f64(north_t)),
        dist,
        idx,
        2,
        300,
        1,
        1,
    )
    np.testing.assert_array_equal(idx, [6, 101])
    np.testing.assert_allclose(dist, [0.0, 0.0], atol=1e-12)


def test_mv_knn_query_on_a_query_coincident_with_a_training_point():
    train_e = np.linspace(0.0, 1.0, 40)
    train_n = np.zeros(40)
    dist = np.empty(2)
    idx = np.empty(2, dtype="int64")
    lib().mv_knn_query(
        (f64(np.array([0.5, 0.3]))),
        (f64(np.array([0.0, 0.0]))),
        (f64(train_e)),
        (f64(train_n)),
        dist,
        idx,
        2,
        40,
        1,
        1,
    )
    # A query exactly between two training nodes takes the lower index. The
    # reported distance is the Euclidean norm, not the squared distance, so
    # it is tiny rather than zero.
    assert idx[0] == 20
    assert dist[0] == pytest.approx(abs(train_e[20] - 0.5), rel=1e-12)
    assert dist[0] < abs(train_e[19] - 0.5)
    # 0.3 sits just past train_e[11] and well short of train_e[12].
    assert idx[1] == 12
    assert dist[1] == pytest.approx(abs(train_e[12] - 0.3), rel=1e-12)
    assert dist[1] < abs(train_e[11] - 0.3)


# -------------------------------------------------------- Green's functions
@pytest.mark.parametrize("mindist", [0.0, 1e-9, 1.0, 10e3, 1e6])
def test_mv_greens_func_matches_upstream(rng, mindist):
    verde = pytest.importorskip("verde")
    east = (rng.random(5000) - 0.5) * 20
    north = (rng.random(5000) - 0.5) * 20
    out = np.empty(east.size)
    lib().mv_greens_func((f64(east)), (f64(north)), out, east.size, mindist, 1)
    theirs = verde.spline.greens_func_numpy(east, north, mindist)
    finite = np.isfinite(theirs)
    assert finite.any()
    rel = np.abs(out[finite] - theirs[finite]) / np.maximum(
        np.abs(theirs[finite]), 1e-300
    )
    assert rel.max() < 1e-12


def test_mv_greens_func_is_finite_at_a_coincident_point():
    out = np.empty(1)
    lib().mv_greens_func(
        (f64(np.array([0.0]))), (f64(np.array([0.0]))), out, 1, 0.0, 1
    )
    assert np.isfinite(out[0])
    assert out[0] == pytest.approx(0.0, abs=1e-14)


# ------------------------------------------------------------- spline kernels
@pytest.mark.parametrize("m", [1, 5, 6, 7, 13, 33, 65])
def test_mv_jacobian_matches_upstream(rng, m):
    verde = pytest.importorskip("verde")
    n = 37
    east = rng.uniform(0.0, 10.0, n)
    north = rng.uniform(-5.0, 5.0, n)
    force_e = rng.uniform(0.0, 10.0, m)
    force_n = rng.uniform(-5.0, 5.0, m)
    jac = np.empty((n, m))
    lib().mv_jacobian(
        (f64(east)), (f64(north)), (f64(force_e)), (f64(force_n)),
        jac, n, m, 0.0, 1,
    )
    theirs = verde.Spline(mindist=0.0).jacobian(
        (east, north), (force_e, force_n)
    )
    np.testing.assert_allclose(jac, theirs, rtol=1e-12, atol=1e-12)


def test_mv_jacobian_small_distance_branch_is_finite():
    """A force sitting on a data point needs the exp branch, which the
    vector body cannot express; those vectors go back to the scalar code."""
    verde = pytest.importorskip("verde")
    east = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    north = np.zeros(8)
    # One force exactly on a data point, and one just off it.
    force_e = np.concatenate([east, [4.0, 4.0 + 1e-9]])
    force_n = np.concatenate([north, [0.0, -1e-9]])
    jac = np.empty((east.size, force_e.size))
    lib().mv_jacobian(
        (f64(east)), (f64(north)), (f64(force_e)), (f64(force_n)),
        jac, east.size, force_e.size, 0.0, 1,
    )
    theirs = verde.Spline(mindist=0.0).jacobian(
        (east, north), (force_e, force_n)
    )
    assert np.isfinite(jac).all()
    np.testing.assert_allclose(jac, theirs, rtol=1e-12, atol=1e-12)
    assert jac[3, 8] == pytest.approx(0.0, abs=1e-14)


def test_mv_predict_sums_the_forces(rng):
    verde = pytest.importorskip("verde")
    m = 17
    east = rng.uniform(0.0, 10.0, 23)
    north = rng.uniform(-5.0, 5.0, 23)
    force_e = rng.uniform(0.0, 10.0, m)
    force_n = rng.uniform(-5.0, 5.0, m)
    forces = np.linspace(-1.0, 2.0, m)
    out = np.empty(east.size)
    lib().mv_predict(
        (f64(east)), (f64(north)), (f64(force_e)), (f64(force_n)),
        (f64(forces)), out, east.size, m, 0.0, 1,
    )
    theirs = verde.Spline(mindist=0.0)
    theirs.force_coords_ = (force_e, force_n)
    theirs.force_ = forces
    np.testing.assert_allclose(
        out, theirs.predict((east, north)), rtol=1e-12, atol=1e-12
    )
    # Zero forces is zero, which is the one prediction that needs no oracle.
    lib().mv_predict(
        (f64(east)), (f64(north)), (f64(force_e)), (f64(force_n)),
        (f64(np.zeros(m))), out, east.size, m, 0.0, 1,
    )
    np.testing.assert_array_equal(out, np.zeros(east.size))


# ------------------------------------------------------------ vector kernels
@pytest.mark.parametrize("nforces", [1, 5, 9, 33])
def test_mv_jacobian_2d_matches_upstream(rng, nforces):
    verde = pytest.importorskip("verde")
    npoints = 11
    east = rng.uniform(0.0, 10.0, npoints)
    north = rng.uniform(-5.0, 5.0, npoints)
    force_e = rng.uniform(0.0, 10.0, nforces)
    force_n = rng.uniform(-5.0, 5.0, nforces)
    jac = np.empty((2 * npoints, 2 * nforces))
    lib().mv_jacobian_2d(
        (f64(east)), (f64(north)), (f64(force_e)), (f64(force_n)),
        jac, npoints, nforces, 1e3, 0.25, 1,
    )
    theirs = verde.VectorSpline2D(poisson=0.25, mindist=1e3).jacobian(
        (east, north), (force_e, force_n)
    )
    assert jac.shape == theirs.shape
    np.testing.assert_allclose(jac, theirs, rtol=1e-12, atol=1e-12)
    # The Jacobian is (2n_points, 2n_forces) and J_ne is the same block
    # in both halves.
    np.testing.assert_allclose(jac[:npoints, nforces:], jac[npoints:, :nforces])
    # The elastic Green's functions differ in the easting and northing
    # offsets, so the two diagonal blocks J_ee and J_nn are close but not
    # equal, and neither is a transpose of the other.
    ee = jac[:npoints, :nforces]
    nn = jac[npoints:, nforces:]
    assert ee.shape == nn.shape == (npoints, nforces)
    np.testing.assert_allclose(ee, nn, rtol=0.01)


def test_mv_jacobian_2d_zero_distance_hands_back_to_the_scalar_branch():
    """mindist=0 puts a force on a point, which is a log of zero. The
    vector body cannot take it, so those vectors go lane by lane and
    reproduce numpy's NaN."""
    east = np.linspace(0.0, 1.0, 9)
    north = np.zeros(9)
    jac = np.empty((18, 18))
    lib().mv_jacobian_2d(
        (f64(east)), (f64(north)), (f64(east)), (f64(north)),
        jac, 9, 9, 0.0, 0.25, 1,
    )
    de = east[:, None] - east[None, :]
    with np.errstate(all="ignore"):
        distance = np.sqrt(de * de) + 0.0
        ln_r = 2.75 * np.log(distance)
        over_r2 = 1.25 / (distance * distance)
        g_ee = ln_r + over_r2 * 0.0
        g_nn = ln_r + over_r2 * de * de
        g_ne = -over_r2 * de * 0.0
    expected = np.block([[g_ee, g_ne], [g_ne, g_nn]])
    assert np.isnan(jac).any()
    np.testing.assert_allclose(jac, expected, rtol=1e-10, equal_nan=True)


def test_mv_predict_2d_matches_upstream(rng):
    verde = pytest.importorskip("verde")
    nforces = 13
    east = rng.uniform(0.0, 10.0, 29)
    north = rng.uniform(-5.0, 5.0, 29)
    force_e = rng.uniform(0.0, 10.0, nforces)
    force_n = rng.uniform(-5.0, 5.0, nforces)
    forces = np.linspace(-1.0, 1.0, 2 * nforces)
    vec_east = np.empty(east.size)
    vec_north = np.empty(east.size)
    lib().mv_predict_2d(
        (f64(east)), (f64(north)), (f64(force_e)), (f64(force_n)),
        (f64(forces)), vec_east, vec_north, east.size, nforces, 1e3, 0.25, 1,
    )
    theirs = verde.VectorSpline2D(poisson=0.25, mindist=1e3)
    theirs.force_coords = (force_e, force_n)
    theirs.force_ = forces
    for a, b in zip((vec_east, vec_north), theirs.predict((east, north))):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)


# ------------------------------------------------------- variance_to_weights
@pytest.mark.parametrize(
    "variance, expected",
    [
        ([0.0, 2.0, 0.2, 1e-16], [1.0, 0.1, 1.0, 1.0]),
        ([4.0, 4.0, 4.0], [1.0, 1.0, 1.0]),
        ([np.nan, 1.0, 4.0], [1.0, 1.0, 0.25]),
        ([1e-14, 1e-13], [1.0, 0.1]),
    ],
)
def test_mv_variance_to_weights(variance, expected):
    var = f64(np.asarray(variance, dtype="float64"))
    out = np.empty(var.size)
    lib().mv_variance_to_weights(var, out, var.size, 1e-15)
    np.testing.assert_allclose(out, expected, rtol=1e-15)
    assert (out >= 0).all() and (out <= 1).all()


def test_mv_variance_to_weights_of_nothing_is_empty():
    out = np.empty(0)
    lib().mv_variance_to_weights(f64(np.zeros(0)), out, 0, 1e-15)
    assert out.size == 0


# ------------------------------------------------------------- checkerboard
@pytest.mark.parametrize("n", [SMALL, LARGE])
def test_mv_checkerboard_matches_the_formula(n):
    east = np.linspace(0.0, 5000.0, n)
    north = np.linspace(-5000.0, 0.0, n)
    out = np.empty(n)
    lib().mv_checkerboard((f64(east)), (f64(north)), out, n, 1000.0, 5000.0, 5000.0, 1)
    # amplitude * sin(2 pi east / w_east) * cos(2 pi north / w_north)
    expected = (
        1000.0
        * np.sin(2 * np.pi * east / 5000.0)
        * np.cos(2 * np.pi * north / 5000.0)
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)
    assert out[0] == 0.0
    # Northing runs over a full wavelength here, so the extremes of the
    # product are symmetric about zero and stay under the amplitude.
    assert abs(out.max()) == pytest.approx(abs(out.min()), rel=1e-9)
    assert 0 < out.max() < 1000.0


def test_mv_checkerboard_of_nothing_is_empty():
    out = np.empty(0)
    lib().mv_checkerboard(
        (f64(np.zeros(0))), (f64(np.zeros(0))), out, 0, 1.0, 10.0, 10.0, 1
    )
    assert out.size == 0


# ------------------------------------------------------ standard scaling
def _scale_reference(jac):
    """What sklearn's StandardScaler(with_mean=False) produces.

    Called out of its own process. That is not cargo cult: a crash report is
    only worth reading if the thing that crashed is the only moving part in
    it, and a process that has already been through the shared library is not
    that.
    """
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "jac.npz")
        target = os.path.join(tmp, "scaled.npz")
        np.savez(source, jac=jac)
        code = (
            "import sys, numpy as np\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "jac = np.load(sys.argv[1])['jac']\n"
            "scaler = StandardScaler(copy=True, with_mean=False, with_std=True)\n"
            "out = scaler.fit_transform(jac)\n"
            "np.savez(sys.argv[2], scale=scaler.scale_, out=out)\n"
        )
        subprocess.run(
            [sys.executable, "-c", code, source, target],
            check=True, capture_output=True,
        )
        saved = np.load(target)
        return saved["scale"], saved["out"]


def _run_scaler(jac, nrows, ncols, workers):
    scale = np.empty(ncols)
    mean = np.empty(ncols)
    lib().mv_ls_scale_fit(jac, scale, mean, nrows, ncols, workers)
    out = jac.copy()
    lib().mv_ls_scale_transform(out, scale, nrows, ncols, workers)
    return scale, mean, out


@pytest.mark.parametrize(
    "nrows, ncols, workers",
    [
        (3, 1, 1),
        (1, 4, 1),
        (40, 5, 1),
        (40, 5, 8),
        (300, 9, 1),
        (300, 9, 8),
        (300, 300, 1),
        (300, 300, 8),
    ],
)
def test_mv_ls_scale_fit_and_transform_match_standard_scaler(
    nrows, ncols, workers, rng
):
    jac = np.ascontiguousarray(rng.normal(size=(nrows, ncols)))
    scale, mean, out = _run_scaler(jac, nrows, ncols, workers)
    their_scale, theirs = _scale_reference(jac)
    np.testing.assert_allclose(scale, their_scale, rtol=1e-13)
    np.testing.assert_allclose(out, theirs, rtol=1e-13, atol=1e-14)
    # with_mean=False: the mean is computed and reported but never subtracted.
    np.testing.assert_allclose(mean, jac.mean(axis=0), rtol=1e-13)


def test_mv_ls_scale_fit_gives_a_constant_column_a_unit_scale(rng):
    jac = np.ascontiguousarray(rng.normal(size=(50, 4)))
    jac[:, 2] = 7.0
    scale = np.empty(4)
    mean = np.empty(4)
    lib().mv_ls_scale_fit(jac, scale, mean, 50, 4, 1)
    assert scale[2] == 1.0
    assert mean[2] == pytest.approx(7.0)
    out = jac.copy()
    lib().mv_ls_scale_transform(out, scale, 50, 4, 1)
    # A constant column divided by its own scale must stay constant, not
    # become 0/0.
    np.testing.assert_array_equal(out[:, 2], np.full(50, 7.0))


def test_mv_ls_scale_transform_leaves_nothing_to_do_on_an_empty_jacobian():
    jac = np.zeros((0, 3))
    scale = np.full(3, 7.0)
    mean = np.full(3, np.nan)
    lib().mv_ls_scale_fit(jac, scale, mean, 0, 3, 1)
    # 0/0 must not become NaN and then be divided through the columns.
    np.testing.assert_array_equal(scale, [7.0, 7.0, 7.0])
    lib().mv_ls_scale_transform(jac, scale, 0, 3, 1)
    assert jac.shape == (0, 3)


# ------------------------------------------------------ damped least squares
def _ridge_reference(jac, y, weights, damping):
    scale = jac.std(axis=0)
    scale[scale == 0] = 1.0
    x = jac / scale
    if weights is not None:
        root = np.sqrt(weights)
        x = x * root[:, None]
        y = y * root
    normal = x.T @ x + damping * np.eye(x.shape[1])
    return np.linalg.solve(normal, x.T @ y) / scale


def test_mv_ls_least_solves_the_damped_system(rng):
    nrows, ncols = 120, 9
    jac = np.ascontiguousarray(rng.normal(size=(nrows, ncols)))
    y = f64(rng.normal(size=nrows))
    for weights, weighted in ((None, 0), (rng.uniform(0.5, 2.0, nrows), 1)):
        for damping in (1e-3, 1.0, 1e3):
            w = np.ones(nrows) if weights is None else weights
            params = np.empty(ncols)
            scale = np.empty(ncols)
            mean = np.empty(ncols)
            work = np.zeros(ncols * ncols + 2 * ncols)
            assert (
                lib().mv_ls_least_squares(
                    jac.copy(),
                    y,
                    w,
                    params,
                    scale,
                    mean,
                    work,
                    nrows,
                    ncols,
                    damping,
                    weighted,
                    1,
                )
                == 1
            )
            np.testing.assert_allclose(
                params, _ridge_reference(jac, y, weights, damping),
                rtol=1e-8, atol=1e-10,
            )


def test_mv_ls_least_reports_a_non_positive_definite_system(rng):
    # A rank-one Jacobian with no damping: the normal equations are singular,
    # and the kernel reports it through the Int return rather than returning
    # nonsense parameters.
    jac = np.ascontiguousarray(np.outer(rng.normal(size=20), rng.normal(size=6)))
    y = f64(rng.normal(size=20))
    nrows, ncols = jac.shape
    params = np.empty(ncols)
    work = np.zeros(ncols * ncols + 2 * ncols)
    ok = lib().mv_ls_least_squares(
        jac.copy(), y, np.ones(nrows), params, np.empty(ncols), np.empty(ncols),
        work, nrows, ncols, 0.0, 0, 1,
    )
    assert ok == 0


def test_mv_ls_least_matches_upstream(rng):
    verde = pytest.importorskip("verde")
    from verde.base.least_squares import least_squares as verde_least_squares

    nrows, ncols = 40, 6
    jac = np.ascontiguousarray(rng.normal(size=(nrows, ncols)))
    y = f64(rng.normal(size=nrows))
    weights = rng.uniform(0.5, 2.0, nrows)
    for damping in (1e-3, 1e3):
        params = np.empty(ncols)
        ok = lib().mv_ls_least_squares(
            jac.copy(), y, weights, params, np.empty(ncols), np.empty(ncols),
            np.zeros(ncols * ncols + 2 * ncols), nrows, ncols, damping, 1, 1,
        )
        assert ok == 1
        np.testing.assert_allclose(
            params, verde_least_squares(jac.copy(), y, weights, damping),
            rtol=1e-10, atol=1e-12,
        )


# ------------------------------------------------------- blocked reduction
def test_mv_block_compact_lists_the_occupied_blocks():
    count = f64(np.array([2.0, 0.0, 5.0, 0.0, 1.0, 0.0]))
    block_ids = np.empty(6, dtype="int64")
    block_index = np.empty(6, dtype="int64")
    found = lib().mv_block_compact(count, block_ids, block_index, 6)
    assert found == 3
    np.testing.assert_array_equal(block_ids[:found], [0, 2, 4])
    # An empty block maps to -1, which is how upstream drops it.
    np.testing.assert_array_equal(block_index, [0, -1, 1, -1, 2, -1])


def test_mv_block_compact_of_nothing_finds_nothing():
    block_ids = np.empty(3, dtype="int64")
    block_index = np.empty(3, dtype="int64")
    assert (
        lib().mv_block_compact(f64(np.zeros(3)), block_ids, block_index, 3) == 0
    )
    np.testing.assert_array_equal(block_index, [-1, -1, -1])


@pytest.mark.parametrize("nchunks", [1, 3, 16])
@pytest.mark.parametrize("weighted", [0, 1])
def test_mv_binned_mean_and_variance(nchunks, weighted, rng):
    labels = rng.integers(0, 4, 200).astype("int64")
    labels[:2] = -1  # labels outside the block range are ignored
    labels[3] = 99
    values = rng.normal(size=200)
    weights = rng.uniform(0.1, 2.0, 200)
    n, ncomps, nblocks = labels.size, 1, 6
    mean = np.zeros(nblocks)
    sum_w = np.zeros(nblocks)
    count = np.zeros(nblocks)
    work = np.zeros(nchunks * nblocks * 3)
    lib().mv_binned_mean(
        labels, f64(values), f64(weights), mean, sum_w, count, work,
        n, ncomps, nblocks, nchunks, weighted, 1,
    )
    keep = (labels >= 0) & (labels < nblocks)
    for b in range(nblocks):
        chosen = keep & (labels == b)
        k = int(chosen.sum())
        assert count[b] == k
        if k == 0:
            assert sum_w[b] == 0.0 and mean[b] == 0.0
            continue
        w = weights[chosen] if weighted else np.ones(k)
        assert sum_w[b] == pytest.approx(w.sum())
        assert mean[b] == pytest.approx(np.average(values[chosen], weights=w))

    total = np.zeros(nblocks)
    lib().mv_binned_variance(
        labels, f64(values), f64(weights), mean, total,
        np.zeros(nchunks * nblocks * 3), n, ncomps, nblocks, nchunks, weighted,
        1.0, 1,
    )
    for b in range(nblocks):
        chosen = keep & (labels == b)
        k = int(chosen.sum())
        if k == 0:
            continue
        w = weights[chosen] if weighted else np.ones(k)
        expected = np.sum(w * (values[chosen] - mean[b]) ** 2)
        assert total[b] == pytest.approx(expected)


def test_mv_binned_mean_writes_every_block_not_just_the_occupied_ones():
    """The reduction stores to mean/sum_w/count for every block, so a block
    no point names comes out as a zero mean with a zero count rather than as
    whatever the buffer happened to hold. That is what lets BlockMean divide
    by the denominator without a guard."""
    labels = np.array([0, 0, 5, -3], dtype="int64")
    values = np.array([1.0, 3.0, 100.0, 100.0])
    n, ncomps, nblocks = 4, 1, 2
    mean = np.full(nblocks, np.nan)
    sum_w = np.full(nblocks, np.nan)
    count = np.full(nblocks, np.nan)
    lib().mv_binned_mean(
        labels, f64(values), f64(np.ones(n)), mean, sum_w, count,
        np.zeros(2 * nblocks * 3), n, ncomps, nblocks, 2, 0, 1,
    )
    # Labels 5 and -3 are outside [0, nblocks) and are ignored, so block 0
    # holds the two in-range points and block 1 is empty.
    assert mean[0] == pytest.approx(2.0)
    assert count[0] == 2.0 and sum_w[0] == 2.0
    np.testing.assert_array_equal(mean[1], [0.0])
    np.testing.assert_array_equal(count[1], [0.0])


def test_mv_binned_variance_of_nothing_is_zero():
    total = np.full(2, np.nan)
    lib().mv_binned_variance(
        np.zeros(0, dtype="int64"), f64(np.zeros(0)), f64(np.zeros(0)),
        f64(np.zeros(2)), total, np.zeros(1 * 2 * 3), 0, 1, 2, 1, 0, 1.0, 1,
    )
    np.testing.assert_array_equal(total, [0.0, 0.0])
