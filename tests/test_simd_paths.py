"""The vectorised kernels must agree with upstream on the awkward sizes.

Every loop in ``src/ported.mojo`` that runs a vector body also has a
scalar tail, and the Green's function kernels have a per-block escape
back to the scalar kernel for the branches that need an ``exp`` or a
``log`` of a non-positive distance. None of those paths are reached by
the fixtures in ``test_parity.py``, which are all round numbers wide, so
they are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojoverde as mv

vd = pytest.importorskip("verde")

REGION = (0, 10, -5, 5)

# simd_width_of[float64] on this target, and the kNN block is eight of them.
LANES = 4
KNN_BLOCK_POINTS = 8 * LANES


def rel(ours, theirs, floor=1e-9):
    return np.max(np.abs(ours - theirs) / np.maximum(np.abs(theirs), floor))


def forces_with_tail(n: int, seed: int = 0):
    """*n* forces, offset so that n % LANES is never zero."""
    rng = np.random.default_rng(seed)
    east = np.sort(rng.uniform(0, 10, n))
    north = rng.uniform(-5, 5, n)
    return east, north


@pytest.mark.parametrize("n", [5, 6, 7, 9, 13, 17, 33, 65])
def test_spline_jacobian_scalar_tail(n):
    coords = mv.scatter_points(REGION, 37, random_state=1)
    force_coords = forces_with_tail(n)
    ours = mv.Spline(mindist=0.0).jacobian(coords, force_coords)
    theirs = vd.Spline(mindist=0.0).jacobian(coords, force_coords)
    assert ours.shape == (37, n)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [5, 6, 7, 9, 13, 17, 33, 65])
def test_spline_predict_scalar_tail(n):
    grid = mv.grid_coordinates(REGION, shape=(11, 9))
    force_coords = forces_with_tail(n, seed=2)
    values = np.linspace(-1.0, 1.0, n)
    ours = mv.Spline(mindist=0.0)
    ours.force_coords_ = force_coords
    ours.force_ = values
    theirs = vd.Spline(mindist=0.0)
    theirs.force_coords_ = force_coords
    theirs.force_ = values
    assert rel(ours.predict(grid), theirs.predict(grid)) < 1e-12


def test_spline_jacobian_takes_the_small_distance_branch():
    """Distances under 1 need an exp, which sends those vectors back to
    the scalar kernel; a force on top of a data point is the extreme."""
    coords = mv.scatter_points(REGION, 24, random_state=3)
    east, north = np.ravel(coords[0]), np.ravel(coords[1])
    force_east = np.concatenate([east, east + 1e-9, east[:4] + 0.5])
    force_north = np.concatenate([north, north - 1e-9, north[:4]])
    ours = mv.Spline(mindist=0.0).jacobian(coords, (force_east, force_north))
    theirs = vd.Spline(mindist=0.0).jacobian(coords, (force_east, force_north))
    assert np.isfinite(ours).all()
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)
    # The coincident pair is the one the branch exists for.
    assert ours[0, 24] == pytest.approx(0.0, abs=1e-14)


def test_spline_predict_takes_the_small_distance_branch():
    coords = mv.scatter_points(REGION, 24, random_state=4)
    east, north = np.ravel(coords[0]), np.ravel(coords[1])
    force_east = np.concatenate([east, east + 1e-9])
    force_north = np.concatenate([north, north - 1e-9])
    values = np.linspace(-1.0, 1.0, force_east.size)
    grid = mv.grid_coordinates(REGION, shape=(7, 5))
    ours = mv.Spline(mindist=0.0)
    ours.force_coords_ = (force_east, force_north)
    ours.force_ = values
    theirs = vd.Spline(mindist=0.0)
    theirs.force_coords_ = (force_east, force_north)
    theirs.force_ = values
    assert rel(ours.predict(grid), theirs.predict(grid)) < 1e-12


@pytest.mark.parametrize("n", [5, 6, 7, 9, 13, 17, 33])
def test_vector_spline_2d_scalar_tail(n):
    coords = mv.scatter_points(REGION, 21, random_state=5)
    force_coords = forces_with_tail(n, seed=6)
    kwargs = {"poisson": 0.25, "mindist": 1e3}
    ours = mv.VectorSpline2D(**kwargs).jacobian(coords, force_coords)
    theirs = vd.VectorSpline2D(**kwargs).jacobian(coords, force_coords)
    assert ours.shape == (42, 2 * n)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)


def test_vector_spline_2d_predict_scalar_tail():
    coords = mv.scatter_points(REGION, 21, random_state=6)
    force_coords = forces_with_tail(7, seed=7)
    values = np.linspace(-1.0, 1.0, 14)
    grid = mv.grid_coordinates(REGION, shape=(7, 5))
    kwargs = {"poisson": 0.25, "mindist": 1e3}
    ours = mv.VectorSpline2D(**kwargs)
    ours.force_coords = force_coords
    ours.force_ = values
    theirs = vd.VectorSpline2D(**kwargs)
    theirs.force_coords = force_coords
    theirs.force_ = values
    for a, b in zip(ours.predict(grid), theirs.predict(grid)):
        assert rel(a, b) < 1e-11


def _numpy_greens_2d(east, north, mindist, poisson):
    with np.errstate(all="ignore"):
        distance = np.sqrt(east * east + north * north) + mindist
        ln_r = (3.0 - poisson) * np.log(distance)
        over_r2 = (1.0 + poisson) / (distance * distance)
        return (
            ln_r + over_r2 * north * north,
            ln_r + over_r2 * east * east,
            -over_r2 * east * north,
        )


def test_vector_spline_2d_zero_distance_falls_back():
    """A mindist of zero puts a force on top of a point, which is a
    log of zero; the kernel has to hand those vectors back to the scalar
    branch instead of running the range reduction on them. Upstream's
    numba kernel raises there, so this is pinned against numpy."""
    coords = mv.scatter_points(REGION, 6, random_state=7)
    ours = mv.VectorSpline2D(poisson=0.25, mindist=0.0).jacobian(coords, coords)
    east = np.ravel(coords[0])
    north = np.ravel(coords[1])
    de = east[:, None] - east[None, :]
    dn = north[:, None] - north[None, :]
    g_ee, g_nn, g_ne = _numpy_greens_2d(de, dn, 0.0, 0.25)
    expected = np.block(
        [
            [g_ee, g_ne],
            [g_ne, g_nn],
        ]
    )
    assert np.isnan(ours).any()
    np.testing.assert_allclose(ours, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.parametrize(
    "n", [KNN_BLOCK_POINTS - 1, KNN_BLOCK_POINTS, KNN_BLOCK_POINTS + 1, 100, 257]
)
@pytest.mark.parametrize("k", [1, 3, 8])
def test_knn_query_block_and_tail(n, k):
    from scipy.spatial import cKDTree

    from mojoverde.neighbors import knn_query

    rng = np.random.default_rng(n + k)
    training = (rng.random(n) * 10, rng.random(n) * 10)
    query = (rng.random(40) * 10, rng.random(40) * 10)
    dist, idx = knn_query(query, training, k)
    tree = cKDTree(np.stack(training, axis=1))
    ref_dist, ref_idx = tree.query(np.stack(query, axis=1), k=k)
    np.testing.assert_array_equal(idx, np.reshape(ref_idx, idx.shape))
    np.testing.assert_allclose(dist, np.reshape(ref_dist, dist.shape), rtol=1e-12)


def test_knn_query_ignores_nan_training_points():
    """The blocked reduction drops a NaN lane the way the scalar scan
    does, rather than letting it poison the block."""
    from mojoverde.neighbors import knn_query

    training = (np.arange(70.0), np.zeros(70))
    training[0][5] = np.nan
    training[0][40] = np.nan
    dist, idx = knn_query((np.array([6.3]), np.array([0.0])), training, 3)
    np.testing.assert_array_equal(idx, [[6, 7, 8]])
    np.testing.assert_allclose(dist, [[0.3, 0.7, 1.7]])


@pytest.mark.parametrize("ncols", [1, 2, 3, 5, 6, 7, 9, 17, 33])
def test_least_squares_scalar_tail(ncols):
    """The Gram, the Cholesky and both triangular sweeps all carry a
    scalar tail, and a system this small never reaches the parallel
    branch of the column loops."""
    from verde.base.least_squares import least_squares as verde_least_squares

    nrows = 3 * ncols + 5
    rng = np.random.default_rng(ncols)
    jac = np.ascontiguousarray(rng.normal(size=(nrows, ncols)))
    data = rng.normal(size=nrows)
    weights = rng.uniform(0.5, 2.0, nrows)
    for damping in (1e-3, 1.0, 1e3):
        ours = mv.least_squares(jac.copy(), data, weights, damping)
        theirs = verde_least_squares(jac.copy(), data, weights, damping)
        np.testing.assert_allclose(ours, theirs, rtol=1e-7, atol=1e-9)


def test_least_squares_unweighted_damped():
    from verde.base.least_squares import least_squares as verde_least_squares

    rng = np.random.default_rng(0)
    jac = np.ascontiguousarray(rng.normal(size=(29, 7)))
    data = rng.normal(size=29)
    ours = mv.least_squares(jac.copy(), data, None, 1.0)
    theirs = verde_least_squares(jac.copy(), data, None, 1.0)
    np.testing.assert_allclose(ours, theirs, rtol=1e-7, atol=1e-9)
